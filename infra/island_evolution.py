#!/usr/bin/env python3
"""
岛屿进化架构 (Island Evolution Architecture)

核心思想：
1. 不同市场状态(bull/bear/range)是独立的"进化岛屿"
2. 每个岛屿有独立的策略池、Gate、LESSONS
3. 并行采样 + Pareto筛选防止局部最优
4. 周期性跨岛迁移发现普适策略
5. 环境驱动的知识解冻机制

架构：
                    ┌─────────────────────────────────┐
                    │        Meta Controller          │
                    │  - 检测市场风格变化              │
                    │  - 分配进化预算到各岛屿           │
                    │  - 触发跨岛迁移(打破局部最优)     │
                    └─────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
   ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
   │  Island: BEAR │       │  Island: BULL │       │ Island: RANGE │
   │ 独立策略池     │       │ 独立策略池     │       │ 独立策略池     │
   │ 独立 LESSONS  │       │ 独立 LESSONS  │       │ 独立 LESSONS  │
   │ 独立 Gate     │       │ 独立 Gate     │       │ 独立 Gate     │
   └───────────────┘       └───────────────┘       └───────────────┘
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """市场状态枚举"""
    BEAR = "bear"      # 熊市 (20日跌>5%)
    BULL = "bull"      # 牛市 (20日涨>5%)
    RANGE = "range"    # 震荡 (20日涨跌<5%)
    

@dataclass
class StrategyCandidate:
    """策略候选"""
    id: str
    code: str
    hypothesis: Dict
    metrics: Dict = field(default_factory=dict)
    regime_metrics: Dict = field(default_factory=dict)  # 分市场状态的指标
    diversity_score: float = 0.0
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass 
class IslandState:
    """岛屿状态"""
    regime: MarketRegime
    strategies: List[str] = field(default_factory=list)  # 策略ID列表
    rounds_since_new: int = 0  # 连续无新策略轮数
    live_cumulative_return: float = 0.0  # 实盘累计收益
    last_migration: str = ""  # 上次迁移时间
    frozen_paths: Dict[str, float] = field(default_factory=dict)  # 冻结路径及其惩罚系数


class MarketRegimeDetector:
    """
    市场风格检测器 (增强版)
    
    多维度识别当前市场状态：
    1. 收益率维度：短期(5日)、中期(20日)、长期(60日)
    2. 波动率维度：当前波动率 vs 历史波动率
    3. 成交量维度：放量/缩量
    4. 趋势强度：ADX 或类似指标
    5. 市场宽度：上涨家数比例
    """
    
    def __init__(
        self,
        bull_threshold: float = 0.05,   # 20日涨>5%为牛市
        bear_threshold: float = -0.05,  # 20日跌>5%为熊市
        lookback: int = 20,
        change_confirm_days: int = 5,   # 风格变化需持续5天确认
    ):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.lookback = lookback
        self.change_confirm_days = change_confirm_days
        self._regime_history: List[Tuple[str, MarketRegime]] = []
    
    def _compute_market_indicators(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场级别指标
        """
        # 按日期聚合
        market_idx = daily.groupby('date').agg({
            'close': ['mean', 'median', 'count'],
            'open': 'mean',
            'high': 'mean',
            'low': 'mean',
            'volume': 'sum',
            'amount': 'sum',
        }).reset_index()
        market_idx.columns = ['date', 'close_mean', 'close_median', 'stock_count', 
                              'open_mean', 'high_mean', 'low_mean', 'volume_sum', 'amount_sum']
        market_idx = market_idx.sort_values('date').reset_index(drop=True)
        
        # 使用中位数作为指数（更稳健）
        market_idx['close'] = market_idx['close_median']
        
        # 多时间窗口收益率
        for window in [5, 10, 20, 60]:
            market_idx[f'ret_{window}d'] = market_idx['close'].pct_change(window)
        
        # 日收益率
        market_idx['daily_ret'] = market_idx['close'].pct_change()
        
        # 波动率（年化）
        market_idx['volatility_20d'] = market_idx['daily_ret'].rolling(20).std() * np.sqrt(252)
        market_idx['volatility_60d'] = market_idx['daily_ret'].rolling(60).std() * np.sqrt(252)
        
        # 波动率比率（当前 vs 长期）
        market_idx['vol_ratio'] = market_idx['volatility_20d'] / market_idx['volatility_60d'].replace(0, np.nan)
        
        # 成交量变化
        market_idx['volume_ma20'] = market_idx['volume_sum'].rolling(20).mean()
        market_idx['volume_ratio'] = market_idx['volume_sum'] / market_idx['volume_ma20'].replace(0, np.nan)
        
        # 趋势强度 (简化版 ADX - 使用 ATR 比率)
        market_idx['tr'] = np.maximum(
            market_idx['high_mean'] - market_idx['low_mean'],
            np.maximum(
                abs(market_idx['high_mean'] - market_idx['close'].shift(1)),
                abs(market_idx['low_mean'] - market_idx['close'].shift(1))
            )
        )
        market_idx['atr_20'] = market_idx['tr'].rolling(20).mean()
        market_idx['trend_strength'] = market_idx['atr_20'] / market_idx['close'] * 100
        
        return market_idx
    
    def _compute_market_breadth(self, daily: pd.DataFrame, target_date) -> Dict:
        """
        计算市场宽度指标
        """
        target_date = pd.to_datetime(target_date)
        
        # 获取目标日期和前一日
        dates = sorted(daily['date'].unique())
        if target_date not in dates:
            target_date = max([d for d in dates if d <= target_date], default=dates[-1])
        
        target_idx = dates.index(target_date) if target_date in dates else len(dates) - 1
        
        if target_idx == 0:
            return {'advance_ratio': 0.5, 'new_high_ratio': 0, 'above_ma20_ratio': 0.5}
        
        prev_date = dates[target_idx - 1]
        
        # 今日数据
        today_data = daily[daily['date'] == target_date]
        prev_data = daily[daily['date'] == prev_date]
        
        if len(today_data) == 0 or len(prev_data) == 0:
            return {'advance_ratio': 0.5, 'new_high_ratio': 0, 'above_ma20_ratio': 0.5}
        
        # 合并计算涨跌
        merged = today_data.merge(prev_data[['code', 'close']], on='code', suffixes=('', '_prev'))
        if len(merged) == 0:
            return {'advance_ratio': 0.5, 'new_high_ratio': 0, 'above_ma20_ratio': 0.5}
        
        # 上涨家数比例
        advances = (merged['close'] > merged['close_prev']).sum()
        advance_ratio = advances / len(merged)
        
        # 20日新高比例 (简化：今日 close 是否 > 过去20日 high)
        # 这里简化处理，实际应该用 rolling
        new_high_ratio = 0  # 需要更多历史数据才能准确计算
        
        # 站上 MA20 比例 (简化：用当日数据估算)
        above_ma20_ratio = 0.5  # 简化
        
        return {
            'advance_ratio': advance_ratio,
            'new_high_ratio': new_high_ratio,
            'above_ma20_ratio': above_ma20_ratio,
        }
    
    def detect(self, daily: pd.DataFrame, target_date: str = None) -> Dict:
        """
        检测市场状态 (增强版)
        
        Returns:
            {
                'regime': MarketRegime,
                'ret_20d': float,
                'volatility': float,
                'regime_changed': bool,
                'change_direction': Optional[str],
                'confidence': float,  # 判断置信度 0-1
                'indicators': Dict,   # 详细指标
            }
        """
        daily = daily.copy()
        daily['date'] = pd.to_datetime(daily['date'])
        
        if target_date is None:
            target_date = daily['date'].max()
        else:
            target_date = pd.to_datetime(target_date)
        
        # 计算市场指标
        market_idx = self._compute_market_indicators(daily)
        
        # 获取目标日期的数据
        target_row = market_idx[market_idx['date'] == target_date]
        if len(target_row) == 0:
            target_row = market_idx.iloc[-1:]
        
        # 提取指标
        ret_5d = target_row['ret_5d'].values[0] if 'ret_5d' in target_row else 0
        ret_10d = target_row['ret_10d'].values[0] if 'ret_10d' in target_row else 0
        ret_20d = target_row['ret_20d'].values[0] if 'ret_20d' in target_row else 0
        ret_60d = target_row['ret_60d'].values[0] if 'ret_60d' in target_row else 0
        volatility = target_row['volatility_20d'].values[0] if 'volatility_20d' in target_row else 0
        vol_ratio = target_row['vol_ratio'].values[0] if 'vol_ratio' in target_row else 1
        volume_ratio = target_row['volume_ratio'].values[0] if 'volume_ratio' in target_row else 1
        trend_strength = target_row['trend_strength'].values[0] if 'trend_strength' in target_row else 0
        
        # 处理 NaN
        ret_5d = 0 if pd.isna(ret_5d) else ret_5d
        ret_10d = 0 if pd.isna(ret_10d) else ret_10d
        ret_20d = 0 if pd.isna(ret_20d) else ret_20d
        ret_60d = 0 if pd.isna(ret_60d) else ret_60d
        volatility = 0 if pd.isna(volatility) else volatility
        vol_ratio = 1 if pd.isna(vol_ratio) else vol_ratio
        volume_ratio = 1 if pd.isna(volume_ratio) else volume_ratio
        trend_strength = 0 if pd.isna(trend_strength) else trend_strength
        
        # 计算市场宽度
        breadth = self._compute_market_breadth(daily, target_date)
        
        # ===== 多因子判断 =====
        bull_score = 0
        bear_score = 0
        range_score = 0
        
        # 因子1: 20日收益率 (权重 0.35)
        if ret_20d >= self.bull_threshold:
            bull_score += 0.35
        elif ret_20d <= self.bear_threshold:
            bear_score += 0.35
        else:
            range_score += 0.35
        
        # 因子2: 短期动量一致性 (权重 0.20)
        # 5日和10日方向是否一致
        short_trend = (ret_5d > 0.01) + (ret_10d > 0.02)  # 0, 1, 2
        if short_trend == 2:
            bull_score += 0.20
        elif short_trend == 0 and ret_5d < -0.01 and ret_10d < -0.02:
            bear_score += 0.20
        else:
            range_score += 0.10
        
        # 因子3: 长短期趋势一致性 (权重 0.15)
        if ret_20d > 0 and ret_60d > 0:
            bull_score += 0.15
        elif ret_20d < 0 and ret_60d < 0:
            bear_score += 0.15
        else:
            range_score += 0.15
        
        # 因子4: 波动率状态 (权重 0.15)
        # 高波动通常不是牛市
        if vol_ratio > 1.3:
            bear_score += 0.10
            range_score += 0.05
        elif vol_ratio < 0.8:
            bull_score += 0.10
            range_score += 0.05
        else:
            range_score += 0.15
        
        # 因子5: 市场宽度 (权重 0.15)
        advance_ratio = breadth['advance_ratio']
        if advance_ratio > 0.6:
            bull_score += 0.15
        elif advance_ratio < 0.4:
            bear_score += 0.15
        else:
            range_score += 0.15
        
        # 确定最终状态
        scores = {'bull': bull_score, 'bear': bear_score, 'range': range_score}
        max_regime = max(scores, key=scores.get)
        confidence = scores[max_regime]
        
        if max_regime == 'bull':
            regime = MarketRegime.BULL
        elif max_regime == 'bear':
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.RANGE
        
        # 如果没有明显优势，倾向于 RANGE
        if confidence < 0.4:
            regime = MarketRegime.RANGE
            confidence = range_score
        
        # 检测状态变化
        regime_changed = False
        change_direction = None
        
        if len(self._regime_history) >= self.change_confirm_days:
            recent_regimes = [r for _, r in self._regime_history[-self.change_confirm_days:]]
            if all(r == regime for r in recent_regimes):
                if len(self._regime_history) > self.change_confirm_days:
                    old_regime = self._regime_history[-self.change_confirm_days - 1][1]
                    if old_regime != regime:
                        regime_changed = True
                        change_direction = f"{old_regime.value}_to_{regime.value}"
        
        # 记录历史
        self._regime_history.append((str(target_date.date()), regime))
        if len(self._regime_history) > 100:
            self._regime_history = self._regime_history[-100:]
        
        return {
            'regime': regime,
            'ret_20d': ret_20d,
            'volatility': volatility,
            'regime_changed': regime_changed,
            'change_direction': change_direction,
            'date': str(target_date.date()),
            'confidence': confidence,
            'scores': scores,
            'indicators': {
                'ret_5d': ret_5d,
                'ret_10d': ret_10d,
                'ret_20d': ret_20d,
                'ret_60d': ret_60d,
                'volatility_20d': volatility,
                'vol_ratio': vol_ratio,
                'volume_ratio': volume_ratio,
                'trend_strength': trend_strength,
                'advance_ratio': breadth['advance_ratio'],
            }
        }
    
    def get_regime_periods(self, daily: pd.DataFrame) -> Dict[MarketRegime, List[Tuple[str, str]]]:
        """
        获取历史上各市场状态的时间段
        
        Returns:
            {
                MarketRegime.BEAR: [('2022-01-01', '2022-03-15'), ...],
                MarketRegime.BULL: [...],
                MarketRegime.RANGE: [...],
            }
        """
        daily = daily.copy()
        daily['date'] = pd.to_datetime(daily['date'])
        dates = sorted(daily['date'].unique())
        
        periods = {r: [] for r in MarketRegime}
        current_regime = None
        period_start = None
        
        for date in dates:
            result = self.detect(daily, date)
            regime = result['regime']
            
            if current_regime is None:
                current_regime = regime
                period_start = date
            elif regime != current_regime:
                # 状态变化，保存前一个周期
                periods[current_regime].append((
                    str(period_start.date()),
                    str(date.date())
                ))
                current_regime = regime
                period_start = date
        
        # 保存最后一个周期
        if current_regime is not None and period_start is not None:
            periods[current_regime].append((
                str(period_start.date()),
                str(dates[-1].date())
            ))
        
        # 清理历史记录
        self._regime_history = []
        
        return periods


class RegimeGate:
    """
    分市场状态的 Gate
    
    只在特定市场状态的时间段内评估策略
    """
    
    def __init__(
        self,
        regime: MarketRegime,
        min_accuracy: float = 15.0,
        min_trades: int = 7,
        min_avg_return: float = 0.0,
    ):
        self.regime = regime
        self.min_accuracy = min_accuracy
        self.min_trades = min_trades
        self.min_avg_return = min_avg_return
    
    def check(self, metrics: Dict) -> Tuple[bool, str]:
        """
        检查策略是否通过 Gate
        
        Args:
            metrics: 在该 regime 期间的回测指标
            
        Returns:
            (passed, reason)
        """
        acc = metrics.get('accuracy_3pct', 0)
        trades = metrics.get('trades', 0)
        avg_ret = metrics.get('avg_return', 0)
        
        reasons = []
        
        if trades < self.min_trades:
            reasons.append(f"trades={trades}<{self.min_trades}")
        
        if acc < self.min_accuracy:
            reasons.append(f"acc={acc:.1f}%<{self.min_accuracy}%")
        
        if avg_ret <= self.min_avg_return:
            reasons.append(f"avg_ret={avg_ret:.2f}%<={self.min_avg_return}%")
        
        if reasons:
            return False, "; ".join(reasons)
        
        return True, "PASS"


class Island:
    """
    进化岛屿
    
    每个岛屿对应一个市场状态，有独立的：
    - 策略池
    - LESSONS 文件
    - Gate 门槛
    - 进化历史
    """
    
    def __init__(
        self,
        regime: MarketRegime,
        base_dir: Path,
        gate: RegimeGate = None,
    ):
        self.regime = regime
        self.base_dir = Path(base_dir)
        self.gate = gate or RegimeGate(regime)
        
        # 创建岛屿目录
        self.island_dir = self.base_dir / f"island_{regime.value}"
        self.island_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.lessons_file = self.island_dir / "LESSONS.md"
        self.state_file = self.island_dir / "state.json"
        self.strategies_dir = self.island_dir / "strategies"
        self.strategies_dir.mkdir(exist_ok=True)
        
        # 加载状态
        self.state = self._load_state()
    
    def _load_state(self) -> IslandState:
        """加载岛屿状态"""
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text())
            return IslandState(
                regime=self.regime,
                strategies=data.get('strategies', []),
                rounds_since_new=data.get('rounds_since_new', 0),
                live_cumulative_return=data.get('live_cumulative_return', 0.0),
                last_migration=data.get('last_migration', ''),
                frozen_paths=data.get('frozen_paths', {}),
            )
        return IslandState(regime=self.regime)
    
    def _save_state(self):
        """保存岛屿状态"""
        data = {
            'regime': self.regime.value,
            'strategies': self.state.strategies,
            'rounds_since_new': self.state.rounds_since_new,
            'live_cumulative_return': self.state.live_cumulative_return,
            'last_migration': self.state.last_migration,
            'frozen_paths': self.state.frozen_paths,
        }
        self.state_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def get_lessons(self) -> str:
        """获取岛屿的 LESSONS 内容"""
        if self.lessons_file.exists():
            return self.lessons_file.read_text()
        return f"# {self.regime.value.upper()} 市场策略教训\n\n暂无记录。\n"
    
    def append_lesson(self, lesson: str):
        """追加教训"""
        current = self.get_lessons()
        with open(self.lessons_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{lesson}\n")
    
    def add_strategy(self, candidate: StrategyCandidate) -> bool:
        """
        添加策略到岛屿
        
        Returns:
            是否添加成功
        """
        # 检查 Gate
        regime_metrics = candidate.regime_metrics.get(self.regime.value, {})
        passed, reason = self.gate.check(regime_metrics)
        
        if not passed:
            return False
        
        # 保存策略
        strategy_file = self.strategies_dir / f"{candidate.id}.py"
        strategy_file.write_text(candidate.code)
        
        # 保存元数据
        meta_file = self.strategies_dir / f"{candidate.id}.json"
        meta_file.write_text(json.dumps({
            'id': candidate.id,
            'hypothesis': candidate.hypothesis,
            'metrics': candidate.metrics,
            'regime_metrics': candidate.regime_metrics,
            'created_at': candidate.created_at,
        }, indent=2, ensure_ascii=False))
        
        # 更新状态
        if candidate.id not in self.state.strategies:
            self.state.strategies.append(candidate.id)
        self.state.rounds_since_new = 0
        self._save_state()
        
        return True
    
    def increment_no_progress(self):
        """增加无进展轮数"""
        self.state.rounds_since_new += 1
        self._save_state()
    
    def update_live_return(self, return_pct: float):
        """更新实盘收益"""
        self.state.live_cumulative_return += return_pct
        self._save_state()
    
    def should_restart_search(
        self,
        max_no_progress: int = 20,
        live_loss_threshold: float = -10.0,
    ) -> Tuple[bool, str]:
        """
        检查是否应该重启搜索
        
        触发条件 (OR):
        - A: 连续 N 轮无新策略
        - B: 实盘累计亏损超阈值
        
        Returns:
            (should_restart, reason)
        """
        # A: 连续无进展
        if self.state.rounds_since_new >= max_no_progress:
            return True, f"no_progress_{self.state.rounds_since_new}_rounds"
        
        # B: 实盘亏损
        if self.state.live_cumulative_return <= live_loss_threshold:
            return True, f"live_loss_{self.state.live_cumulative_return:.1f}%"
        
        return False, ""
    
    def unfreeze_paths(self, paths: List[str], decay: float = 0.5):
        """
        解冻部分路径（降低惩罚系数）
        
        Args:
            paths: 要解冻的路径列表
            decay: 惩罚衰减系数 (0.5 = 惩罚减半)
        """
        for path in paths:
            if path in self.state.frozen_paths:
                self.state.frozen_paths[path] *= decay
                if self.state.frozen_paths[path] < 0.1:
                    del self.state.frozen_paths[path]
        self._save_state()
    
    def freeze_path(self, path: str, penalty: float = 1.0):
        """冻结路径"""
        self.state.frozen_paths[path] = penalty
        self._save_state()
    
    def get_strategy_count(self) -> int:
        """获取策略数量"""
        return len(self.state.strategies)


class ParetoSelector:
    """
    Pareto 筛选器
    
    多目标筛选，保留 Pareto 前沿的策略
    """
    
    def __init__(self, objectives: List[str] = None):
        """
        Args:
            objectives: 目标列表，格式为 "name" 或 "+name"(最大化) 或 "-name"(最小化)
                       默认: ["+accuracy", "+trades", "+diversity", "-drawdown"]
        """
        self.objectives = objectives or [
            "+accuracy_3pct",
            "+trades", 
            "+diversity_score",
            "-max_drawdown",
        ]
    
    def _get_objective_values(self, candidate: StrategyCandidate, regime: str) -> List[float]:
        """提取候选的目标值"""
        values = []
        metrics = candidate.regime_metrics.get(regime, candidate.metrics)
        
        for obj in self.objectives:
            if obj.startswith('+'):
                name = obj[1:]
                sign = 1
            elif obj.startswith('-'):
                name = obj[1:]
                sign = -1
            else:
                name = obj
                sign = 1
            
            if name == 'diversity_score':
                value = candidate.diversity_score
            else:
                value = metrics.get(name, 0)
            
            values.append(sign * value)
        
        return values
    
    def _dominates(self, a: List[float], b: List[float]) -> bool:
        """检查 a 是否支配 b (所有目标都不差，至少一个更好)"""
        dominated = False
        for va, vb in zip(a, b):
            if va < vb:
                return False  # a 在某个目标上更差
            if va > vb:
                dominated = True  # a 在某个目标上更好
        return dominated
    
    def select(
        self,
        candidates: List[StrategyCandidate],
        regime: str,
        max_select: int = 5,
    ) -> List[StrategyCandidate]:
        """
        筛选 Pareto 前沿的策略
        
        Args:
            candidates: 候选策略列表
            regime: 市场状态
            max_select: 最多选择数量
            
        Returns:
            Pareto 前沿的策略
        """
        if not candidates:
            return []
        
        # 计算目标值
        obj_values = [
            (c, self._get_objective_values(c, regime))
            for c in candidates
        ]
        
        # 找 Pareto 前沿
        pareto_front = []
        for i, (c_i, v_i) in enumerate(obj_values):
            dominated = False
            for j, (c_j, v_j) in enumerate(obj_values):
                if i != j and self._dominates(v_j, v_i):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(c_i)
        
        # 如果前沿太大，按综合得分排序
        if len(pareto_front) > max_select:
            # 计算综合得分
            for c in pareto_front:
                vals = self._get_objective_values(c, regime)
                c._pareto_score = sum(vals)
            pareto_front.sort(key=lambda x: x._pareto_score, reverse=True)
            pareto_front = pareto_front[:max_select]
        
        return pareto_front


class DiversityCalculator:
    """
    多样性计算器
    
    计算策略与现有策略池的差异度
    """
    
    def __init__(self):
        self._code_hashes: Dict[str, str] = {}
    
    def compute_diversity(
        self,
        candidate_code: str,
        existing_codes: List[str],
    ) -> float:
        """
        计算策略代码的多样性分数
        
        使用代码的结构特征计算相似度
        
        Returns:
            diversity_score: 0-1, 越高越多样
        """
        if not existing_codes:
            return 1.0
        
        # 提取特征
        candidate_features = self._extract_features(candidate_code)
        
        similarities = []
        for code in existing_codes:
            features = self._extract_features(code)
            sim = self._compute_similarity(candidate_features, features)
            similarities.append(sim)
        
        # 多样性 = 1 - 最大相似度
        max_similarity = max(similarities) if similarities else 0
        return 1.0 - max_similarity
    
    def _extract_features(self, code: str) -> Dict[str, Any]:
        """提取代码特征"""
        features = {
            'indicators': set(),
            'conditions': set(),
            'parameters': set(),
        }
        
        # 提取常见指标
        indicators = [
            'ret', 'volume', 'close', 'open', 'high', 'low',
            'ma', 'ema', 'rsi', 'macd', 'std', 'volatility',
            'pct_change', 'rolling', 'shift',
        ]
        for ind in indicators:
            if ind in code.lower():
                features['indicators'].add(ind)
        
        # 提取条件模式
        import re
        conditions = re.findall(r'[<>=!]+\s*[\d.]+', code)
        features['conditions'] = set(conditions[:10])  # 取前10个
        
        # 提取参数
        params = re.findall(r'[\d]+(?:\.\d+)?', code)
        features['parameters'] = set(params[:20])  # 取前20个
        
        return features
    
    def _compute_similarity(self, f1: Dict, f2: Dict) -> float:
        """计算特征相似度 (Jaccard)"""
        similarities = []
        
        for key in ['indicators', 'conditions', 'parameters']:
            s1, s2 = f1.get(key, set()), f2.get(key, set())
            if s1 or s2:
                jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
                similarities.append(jaccard)
        
        return np.mean(similarities) if similarities else 0


class MetaController:
    """
    元控制器
    
    管理所有岛屿，协调进化过程
    """
    
    def __init__(
        self,
        base_dir: Path,
        migration_interval: int = 10,  # 每10轮尝试迁移
        restart_no_progress: int = 20,  # 连续20轮无进展触发重启
        restart_live_loss: float = -10.0,  # 累计亏损10%触发重启
    ):
        self.base_dir = Path(base_dir)
        self.islands_dir = self.base_dir / "islands"
        self.islands_dir.mkdir(parents=True, exist_ok=True)
        
        self.migration_interval = migration_interval
        self.restart_no_progress = restart_no_progress
        self.restart_live_loss = restart_live_loss
        
        # 初始化组件
        self.regime_detector = MarketRegimeDetector()
        self.pareto_selector = ParetoSelector()
        self.diversity_calculator = DiversityCalculator()
        
        # 创建岛屿
        self.islands: Dict[MarketRegime, Island] = {}
        for regime in MarketRegime:
            self.islands[regime] = Island(
                regime=regime,
                base_dir=self.islands_dir,
                gate=RegimeGate(regime),
            )
        
        # 加载元状态
        self.meta_state_file = self.islands_dir / "meta_state.json"
        self.meta_state = self._load_meta_state()
    
    def _load_meta_state(self) -> Dict:
        """加载元状态"""
        if self.meta_state_file.exists():
            return json.loads(self.meta_state_file.read_text())
        return {
            'total_rounds': 0,
            'last_migration_round': 0,
            'regime_history': [],
        }
    
    def _save_meta_state(self):
        """保存元状态"""
        self.meta_state_file.write_text(
            json.dumps(self.meta_state, indent=2, ensure_ascii=False)
        )
    
    def detect_current_regime(self, daily: pd.DataFrame) -> Dict:
        """检测当前市场状态"""
        result = self.regime_detector.detect(daily)
        
        # 记录历史
        self.meta_state['regime_history'].append({
            'date': result['date'],
            'regime': result['regime'].value,
            'ret_20d': result['ret_20d'],
        })
        
        # 保留最近100条
        self.meta_state['regime_history'] = self.meta_state['regime_history'][-100:]
        self._save_meta_state()
        
        return result
    
    def get_regime_periods(self, daily: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
        """获取各市场状态的历史时间段"""
        periods = self.regime_detector.get_regime_periods(daily)
        return {r.value: p for r, p in periods.items()}
    
    def select_target_island(self, regime_info: Dict) -> Island:
        """
        选择当前应该进化的岛屿
        
        优先级：
        1. 当前市场状态对应的岛屿
        2. 如果当前岛屿策略充足，轮换到其他岛屿
        """
        current_regime = regime_info['regime']
        return self.islands[current_regime]
    
    def should_migrate(self) -> bool:
        """检查是否应该进行跨岛迁移"""
        rounds = self.meta_state['total_rounds']
        last_migration = self.meta_state['last_migration_round']
        return (rounds - last_migration) >= self.migration_interval
    
    def migrate_strategy(
        self,
        from_island: Island,
        to_island: Island,
        strategy_id: str,
    ) -> bool:
        """
        尝试将策略从一个岛屿迁移到另一个
        
        Returns:
            是否迁移成功
        """
        # 加载策略
        strategy_file = from_island.strategies_dir / f"{strategy_id}.py"
        meta_file = from_island.strategies_dir / f"{strategy_id}.json"
        
        if not strategy_file.exists() or not meta_file.exists():
            return False
        
        code = strategy_file.read_text()
        meta = json.loads(meta_file.read_text())
        
        # 创建候选
        candidate = StrategyCandidate(
            id=f"{strategy_id}_migrated_{to_island.regime.value}",
            code=code,
            hypothesis=meta.get('hypothesis', {}),
            metrics=meta.get('metrics', {}),
            regime_metrics=meta.get('regime_metrics', {}),
        )
        
        # 尝试添加到目标岛屿
        success = to_island.add_strategy(candidate)
        
        if success:
            self.meta_state['last_migration_round'] = self.meta_state['total_rounds']
            self._save_meta_state()
        
        return success
    
    def check_restart_needed(self, regime_info: Dict) -> Dict[MarketRegime, Tuple[bool, str]]:
        """
        检查各岛屿是否需要重启搜索
        
        Returns:
            {regime: (should_restart, reason)}
        """
        results = {}
        
        for regime, island in self.islands.items():
            should_restart, reason = island.should_restart_search(
                max_no_progress=self.restart_no_progress,
                live_loss_threshold=self.restart_live_loss,
            )
            
            # 额外检查：市场状态变化
            if regime_info.get('regime_changed'):
                change_dir = regime_info.get('change_direction', '')
                if regime.value in change_dir:
                    should_restart = True
                    reason = f"regime_changed: {change_dir}"
            
            results[regime] = (should_restart, reason)
        
        return results
    
    def handle_restart(self, island: Island, reason: str):
        """
        处理岛屿重启
        
        - 解冻部分路径
        - 重置无进展计数
        """
        # 根据原因决定解冻哪些路径
        if 'no_progress' in reason:
            # 无进展：解冻所有路径，惩罚减半
            island.unfreeze_paths(list(island.state.frozen_paths.keys()), decay=0.5)
        
        elif 'live_loss' in reason:
            # 实盘亏损：完全解冻
            island.state.frozen_paths = {}
        
        elif 'regime_changed' in reason:
            # 市场变化：完全解冻，并重置收益
            island.state.frozen_paths = {}
            island.state.live_cumulative_return = 0.0
        
        # 重置计数
        island.state.rounds_since_new = 0
        island._save_state()
    
    def increment_round(self):
        """增加轮次计数"""
        self.meta_state['total_rounds'] += 1
        self._save_meta_state()
    
    def get_summary(self) -> Dict:
        """获取所有岛屿的摘要"""
        summary = {
            'total_rounds': self.meta_state['total_rounds'],
            'islands': {},
        }
        
        for regime, island in self.islands.items():
            summary['islands'][regime.value] = {
                'strategies': len(island.state.strategies),
                'rounds_since_new': island.state.rounds_since_new,
                'live_return': island.state.live_cumulative_return,
                'frozen_paths': len(island.state.frozen_paths),
            }
        
        return summary


# ============================================================
# 便捷函数
# ============================================================

def create_meta_controller(base_dir: str = None) -> MetaController:
    """创建元控制器"""
    if base_dir is None:
        base_dir = Path("/root/quant-autoresearch")
    return MetaController(Path(base_dir))


def test_island_evolution():
    """测试岛屿进化架构"""
    print("Testing Island Evolution Architecture...")
    
    # 加载数据
    cache_dir = Path.home() / '.cache/quant-autoresearch'
    daily = pd.read_parquet(cache_dir / 'daily.parquet')
    
    # 创建控制器
    controller = create_meta_controller()
    
    # 检测当前市场状态
    regime_info = controller.detect_current_regime(daily)
    print(f"\n当前市场状态: {regime_info['regime'].value}")
    print(f"20日收益: {regime_info['ret_20d']*100:.1f}%")
    print(f"波动率: {regime_info['volatility']*100:.1f}%")
    
    # 获取历史时间段
    periods = controller.get_regime_periods(daily)
    print("\n历史市场周期:")
    for regime, period_list in periods.items():
        print(f"  {regime}: {len(period_list)} 个周期")
        for start, end in period_list[-3:]:
            print(f"    {start} ~ {end}")
    
    # 摘要
    summary = controller.get_summary()
    print(f"\n岛屿摘要:")
    for regime, info in summary['islands'].items():
        print(f"  {regime}: {info['strategies']} 个策略, 无进展 {info['rounds_since_new']} 轮")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_island_evolution()
