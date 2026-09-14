"""
策略池管理器 - 自适应策略路由系统

核心功能:
1. 管理通过Gate验证的策略池
2. 跟踪每个策略的实时表现
3. 基于Thompson Sampling动态调整权重
4. 市场状态门控激活对应策略

使用方法:
    from infra.strategy_pool import StrategyPool, AdaptiveRouter
    
    pool = StrategyPool()
    pool.load_verified_strategies()
    
    router = AdaptiveRouter(pool)
    signals = router.get_signals(df, date='2026-07-18')
"""

import json
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@dataclass
class Strategy:
    """单个策略的元数据"""
    id: str                      # 策略ID, e.g., "R187"
    path: Path                   # strategy.py路径
    accuracy: float              # 历史准确率(%)
    trades: int                  # 历史交易数
    avg_return: float            # 历史平均收益(%)
    category: str = "unknown"    # 策略类别: reversal, trend, range, defensive
    description: str = ""        # 策略描述
    
    # 运行时状态
    weight: float = 1.0          # 当前权重
    alpha: int = 1               # Thompson Sampling: 成功次数+1
    beta: int = 1                # Thompson Sampling: 失败次数+1
    recent_signals: List = field(default_factory=list)  # 近期信号记录
    
    def __post_init__(self):
        self.recent_signals = []
    
    @property
    def thompson_sample(self) -> float:
        """从Beta分布采样获得当前期望权重"""
        return np.random.beta(self.alpha, self.beta)
    
    @property
    def expected_accuracy(self) -> float:
        """期望准确率 (后验均值)"""
        return self.alpha / (self.alpha + self.beta)
    
    def update(self, success: bool):
        """更新Thompson Sampling参数"""
        if success:
            self.alpha += 1
        else:
            self.beta += 1
    
    def load_predict_function(self) -> Optional[Callable]:
        """动态加载策略的predict_next_day函数"""
        try:
            spec = importlib.util.spec_from_file_location(f"strategy_{self.id}", self.path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.predict_next_day
        except Exception as e:
            print(f"[StrategyPool] 加载策略{self.id}失败: {e}")
            return None


class StrategyPool:
    """
    策略池管理器
    
    功能:
    - 加载和管理通过Gate的策略
    - 按类别分组策略
    - 跟踪策略表现
    """
    
    # 策略类别定义 (含反转子类型)
    CATEGORIES = {
        'reversal': '通用反转',          # 基础超跌反弹
        'reversal_panic': '恐慌反转',    # 跌停/暴跌后反转, 极端恐慌时
        'reversal_volume': '量能反转',   # 缩量后放量反转
        'reversal_rsi': '技术反转',      # RSI超卖反弹
        'trend': '趋势跟随',             # 牛市时有效 (短期不适用)
        'range': '震荡套利',             # 横盘时有效
        'defensive': '防守型',           # 高波动时有效
        'unknown': '未分类',
    }
    
    def __init__(self, rounds_dir: Path = None, state_file: Path = None):
        """
        初始化策略池
        
        Args:
            rounds_dir: rounds目录路径
            state_file: 策略状态持久化文件
        """
        self.rounds_dir = rounds_dir or Path("rounds")
        self.state_file = state_file or Path("data/strategy_pool_state.json")
        self.strategies: Dict[str, Strategy] = {}
        self._category_index: Dict[str, List[str]] = {cat: [] for cat in self.CATEGORIES}
        
    def load_verified_strategies(self, min_accuracy: float = 15.0) -> int:
        """
        加载所有通过Gate的策略
        
        Args:
            min_accuracy: 最低准确率要求(%)
            
        Returns:
            加载的策略数量
        """
        from infra.gate_config import TieredGate
        
        gate = TieredGate()
        loaded = 0
        
        for metrics_file in sorted(self.rounds_dir.glob("*/metrics.json")):
            round_id = metrics_file.parent.name
            strategy_file = metrics_file.parent / "strategy.py"
            
            if not strategy_file.exists():
                continue
                
            try:
                m = json.load(open(metrics_file))
                acc = m.get('accuracy_3pct', 0)
                trades = int(m.get('total_trades', 0))
                avg_ret = m.get('avg_signal_return', 0) * 100  # 转换为百分数
                
                if trades == 0 or acc < min_accuracy:
                    continue
                
                # Gate检查
                passed, reason = gate.check(acc, trades, avg_ret)
                if not passed:
                    continue
                
                # 推断策略类别
                category = self._infer_category(strategy_file, m)
                
                # 获取策略描述
                description = self._extract_description(strategy_file)
                
                strategy = Strategy(
                    id=f"R{round_id}",
                    path=strategy_file,
                    accuracy=acc,
                    trades=trades,
                    avg_return=avg_ret,
                    category=category,
                    description=description,
                )
                
                self.strategies[strategy.id] = strategy
                self._category_index[category].append(strategy.id)
                loaded += 1
                
            except Exception as e:
                print(f"[StrategyPool] 处理{round_id}时出错: {e}")
                continue
        
        # 尝试加载持久化状态
        self._load_state()
        
        return loaded
    
    def _infer_category(self, strategy_file: Path, metrics: dict) -> str:
        """
        根据策略代码和指标推断类别
        
        分类体系:
        - reversal: 通用反转 (超跌反弹)
        - reversal_panic: 恐慌抛售后反转 (跌停、暴跌)
        - reversal_volume: 缩量后放量反转
        - reversal_rsi: RSI/技术指标超卖反弹
        - trend: 趋势跟随 (短期内不适用)
        - range: 震荡套利
        - defensive: 低波动防守
        """
        try:
            code = strategy_file.read_text()
            code_lower = code.lower()
            
            # 先检查是否是反转类策略
            is_reversal = any(kw in code_lower for kw in [
                '跌停', '超跌', '反转', 'reversal', 'oversold', 
                '恐慌', 'panic', '暴跌', '低吸', '抄底',
                'rsi', '超卖', '底背离', '止跌', '企稳'
            ])
            
            if is_reversal:
                # 细分反转子类型
                if any(kw in code_lower for kw in ['跌停', '暴跌', '恐慌', 'panic', '崩盘']):
                    return 'reversal_panic'
                elif any(kw in code_lower for kw in ['缩量', '放量', 'volume', '成交量', '换手']):
                    return 'reversal_volume'
                elif any(kw in code_lower for kw in ['rsi', '超卖', '背离', 'macd', 'kdj']):
                    return 'reversal_rsi'
                else:
                    return 'reversal'
            elif any(kw in code_lower for kw in ['突破', '趋势', '动量', 'trend', 'breakout', 'momentum']):
                return 'trend'
            elif any(kw in code_lower for kw in ['震荡', '区间', '均值回归', 'range', 'mean_revert']):
                return 'range'
            elif any(kw in code_lower for kw in ['防守', '低波动', 'defensive', 'low_vol']):
                return 'defensive'
            else:
                return 'unknown'
        except:
            return 'unknown'
    
    def _extract_description(self, strategy_file: Path) -> str:
        """从策略文件提取描述"""
        try:
            code = strategy_file.read_text()
            # 查找docstring
            for line in code.split('\n'):
                if '"""' in line and ':' in line:
                    # 格式: """HH526-v2: 跌停放量吸筹反转"""
                    start = line.find('"""') + 3
                    end = line.rfind('"""')
                    if end > start:
                        return line[start:end].strip()
                elif '#' in line and '策略' in line:
                    return line.split('#')[1].strip()
            return ""
        except:
            return ""
    
    def get_strategies_by_category(self, category: str) -> List[Strategy]:
        """获取指定类别的所有策略"""
        ids = self._category_index.get(category, [])
        return [self.strategies[id] for id in ids if id in self.strategies]
    
    def get_all_strategies(self) -> List[Strategy]:
        """获取所有策略"""
        return list(self.strategies.values())
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """获取指定策略"""
        return self.strategies.get(strategy_id)
    
    def update_strategy_performance(self, strategy_id: str, date: str, 
                                     signals: List[str], returns: List[float]):
        """
        更新策略表现
        
        Args:
            strategy_id: 策略ID
            date: 信号日期
            signals: 推荐的股票代码列表
            returns: 对应的实际收益率列表
        """
        strategy = self.strategies.get(strategy_id)
        if not strategy:
            return
        
        for code, ret in zip(signals, returns):
            success = ret > 0.03  # >3%视为成功
            strategy.update(success)
            strategy.recent_signals.append({
                'date': date,
                'code': code,
                'return': ret,
                'success': success
            })
        
        # 只保留最近30条记录
        strategy.recent_signals = strategy.recent_signals[-30:]
        
        # 持久化状态
        self._save_state()
    
    def _save_state(self):
        """持久化策略状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        state = {}
        for sid, s in self.strategies.items():
            state[sid] = {
                'alpha': s.alpha,
                'beta': s.beta,
                'recent_signals': s.recent_signals[-30:],
            }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _load_state(self):
        """加载持久化的策略状态"""
        if not self.state_file.exists():
            return
        
        try:
            state = json.load(open(self.state_file))
            for sid, data in state.items():
                if sid in self.strategies:
                    self.strategies[sid].alpha = data.get('alpha', 1)
                    self.strategies[sid].beta = data.get('beta', 1)
                    self.strategies[sid].recent_signals = data.get('recent_signals', [])
        except Exception as e:
            print(f"[StrategyPool] 加载状态失败: {e}")
    
    def format_summary(self) -> str:
        """格式化策略池概览"""
        lines = [
            "=" * 60,
            "策略池概览",
            "=" * 60,
            f"总策略数: {len(self.strategies)}",
            "",
            "按类别分布:",
        ]
        
        for cat, name in self.CATEGORIES.items():
            strategies = self.get_strategies_by_category(cat)
            if strategies:
                lines.append(f"  {name}: {len(strategies)}个")
                for s in sorted(strategies, key=lambda x: x.accuracy, reverse=True)[:3]:
                    lines.append(f"    - {s.id}: {s.accuracy:.1f}%准确率, {s.trades}笔, {s.description[:30]}")
        
        return "\n".join(lines)


class MarketStateClassifier:
    """
    市场状态分类器
    
    识别当前市场处于什么状态,用于策略门控
    """
    
    # 市场状态枚举
    STATES = {
        'panic': '恐慌',      # 急跌, 激活reversal策略
        'fear': '恐惧',       # 下跌但未恐慌
        'neutral': '中性',    # 震荡, 所有策略参与
        'greed': '贪婪',      # 上涨
        'euphoria': '狂热',   # 急涨, 激活trend策略
    }
    
    def __init__(self, lookback_days: int = 20):
        """
        初始化分类器
        
        Args:
            lookback_days: 回看天数
        """
        self.lookback_days = lookback_days
    
    def classify(self, df: pd.DataFrame, target_date: str = None) -> Dict:
        """
        分类当前市场状态
        
        Args:
            df: 包含所有股票数据的DataFrame
            target_date: 目标日期,默认最新
            
        Returns:
            状态信息字典
        """
        if target_date is None:
            target_date = df['date'].max()
        
        # 获取回看窗口数据
        dates = sorted(df['date'].unique())
        target_idx = dates.index(target_date) if target_date in dates else len(dates) - 1
        start_idx = max(0, target_idx - self.lookback_days)
        window_dates = dates[start_idx:target_idx + 1]
        
        window_df = df[df['date'].isin(window_dates)]
        
        # 计算市场指标
        market_stats = window_df.groupby('date').agg({
            'close': 'mean',
            'volume': 'sum',
        }).reset_index()
        
        # 20日涨跌幅
        if len(market_stats) >= 2:
            ret_20d = (market_stats['close'].iloc[-1] / market_stats['close'].iloc[0] - 1) * 100
        else:
            ret_20d = 0
        
        # 5日涨跌幅
        ret_5d = 0
        if len(market_stats) >= 5:
            ret_5d = (market_stats['close'].iloc[-1] / market_stats['close'].iloc[-5] - 1) * 100
        
        # 波动率
        daily_rets = market_stats['close'].pct_change().dropna()
        volatility = daily_rets.std() * np.sqrt(252) * 100 if len(daily_rets) > 0 else 0
        
        # 上涨家数占比
        latest = window_df[window_df['date'] == target_date]
        if len(latest) > 0:
            latest = latest.copy()
            latest['ret'] = latest.groupby('code')['close'].pct_change()
            up_ratio = (latest['ret'] > 0).mean() * 100
        else:
            up_ratio = 50
        
        # 状态判定
        state = self._determine_state(ret_20d, ret_5d, volatility, up_ratio)
        
        return {
            'date': target_date,
            'state': state,
            'state_name': self.STATES[state],
            'ret_20d': ret_20d,
            'ret_5d': ret_5d,
            'volatility': volatility,
            'up_ratio': up_ratio,
        }
    
    def _determine_state(self, ret_20d: float, ret_5d: float, 
                         volatility: float, up_ratio: float) -> str:
        """
        根据指标判定市场状态
        
        简单规则:
        - 20日跌>10% 且 5日跌>5% → panic
        - 20日跌>5% → fear
        - 20日涨>10% 且 5日涨>5% → euphoria
        - 20日涨>5% → greed
        - 其他 → neutral
        """
        if ret_20d < -10 and ret_5d < -5:
            return 'panic'
        elif ret_20d < -5:
            return 'fear'
        elif ret_20d > 10 and ret_5d > 5:
            return 'euphoria'
        elif ret_20d > 5:
            return 'greed'
        else:
            return 'neutral'
    
    def get_active_categories(self, state: str) -> List[str]:
        """
        获取当前状态下应激活的策略类别
        
        Args:
            state: 市场状态
            
        Returns:
            应激活的类别列表
            
        注意: 包含反转子类型 (reversal_panic, reversal_volume, reversal_rsi)
        """
        # 所有反转类型的子类别
        all_reversals = ['reversal', 'reversal_panic', 'reversal_volume', 'reversal_rsi']
        
        activation_map = {
            # 恐慌: 优先恐慌反转, 其他反转也参与
            'panic': ['reversal_panic'] + all_reversals,
            # 恐惧: 所有反转 + 防守
            'fear': all_reversals + ['defensive'],
            # 中性: 全部策略
            'neutral': all_reversals + ['trend', 'range', 'defensive', 'unknown'],
            # 贪婪: 趋势+震荡 (但短期策略趋势不适用, 也加入部分反转)
            'greed': ['trend', 'range', 'reversal_volume'],
            # 狂热: 趋势为主 (短期策略可能应该谨慎)
            'euphoria': ['trend', 'reversal_volume'],
        }
        return activation_map.get(state, ['unknown'])


class AdaptiveRouter:
    """
    自适应策略路由器
    
    核心功能:
    1. 根据市场状态选择策略类别
    2. 用Thompson Sampling在类别内选择具体策略
    3. 聚合多策略信号
    """
    
    def __init__(self, pool: StrategyPool, 
                 classifier: MarketStateClassifier = None,
                 exploration_rate: float = 0.2):
        """
        初始化路由器
        
        Args:
            pool: 策略池
            classifier: 市场状态分类器
            exploration_rate: 探索率(强制尝试低权重策略的概率)
        """
        self.pool = pool
        self.classifier = classifier or MarketStateClassifier()
        self.exploration_rate = exploration_rate
    
    def get_signals(self, df: pd.DataFrame, date: str = None,
                    mode: str = 'weighted') -> Dict:
        """
        获取今日信号
        
        Args:
            df: 市场数据
            date: 目标日期
            mode: 聚合模式
                - 'weighted': 加权投票
                - 'top1': 只用最高权重策略
                - 'union': 所有信号并集
                - 'intersection': 所有信号交集
                
        Returns:
            信号结果字典
        """
        if date is None:
            date = df['date'].max()
        
        # 1. 获取市场状态
        market_state = self.classifier.classify(df, date)
        active_categories = self.classifier.get_active_categories(market_state['state'])
        
        # 2. 获取活跃策略
        active_strategies = []
        for cat in active_categories:
            active_strategies.extend(self.pool.get_strategies_by_category(cat))
        
        if not active_strategies:
            # 没有匹配的策略,使用全部
            active_strategies = self.pool.get_all_strategies()
        
        # 3. Thompson Sampling选择策略权重
        strategy_weights = {}
        for s in active_strategies:
            # 探索: 有概率给低权重策略机会
            if np.random.random() < self.exploration_rate:
                weight = np.random.uniform(0.5, 1.0)
            else:
                weight = s.thompson_sample
            strategy_weights[s.id] = weight
        
        # 4. 运行每个策略获取信号
        all_signals = {}  # stock_code -> [(strategy_id, weight)]
        strategy_results = {}
        
        for strategy in active_strategies:
            predict_fn = strategy.load_predict_function()
            if predict_fn is None:
                continue
            
            try:
                signals = predict_fn(df.copy())
                if signals is None:
                    signals = pd.Series(dtype=float)
                
                weight = strategy_weights[strategy.id]
                strategy_results[strategy.id] = {
                    'signals': signals,
                    'weight': weight,
                    'accuracy': strategy.accuracy,
                }
                
                # 遍历信号 - 注意: pd.Series遍历要用.index或.items()
                if isinstance(signals, pd.Series) and len(signals) > 0:
                    for code in signals.index:
                        code = str(code)  # 确保是字符串
                        if code not in all_signals:
                            all_signals[code] = []
                        all_signals[code].append((strategy.id, weight))
                elif hasattr(signals, '__iter__'):
                    for code in signals:
                        code = str(code)
                        if code not in all_signals:
                            all_signals[code] = []
                        all_signals[code].append((strategy.id, weight))
                    
            except Exception as e:
                print(f"[Router] 运行{strategy.id}出错: {e}")
                continue
        
        # 5. 信号聚合 (传入策略结果用于置信度计算)
        final_signals = self._aggregate_signals(all_signals, mode, strategy_results)
        
        return {
            'date': date,
            'market_state': market_state,
            'active_categories': active_categories,
            'active_strategies': len(active_strategies),
            'strategy_results': strategy_results,
            'all_signals': all_signals,
            'final_signals': final_signals,
            'mode': mode,
        }
    
    def _aggregate_signals(self, all_signals: Dict, mode: str, 
                           strategy_results: Dict = None) -> List[Dict]:
        """
        聚合多策略信号 (增强版)
        
        Args:
            all_signals: {stock_code: [(strategy_id, weight), ...]}
            mode: 聚合模式
            strategy_results: 策略运行结果，用于获取历史准确率等信息
            
        Returns:
            排序后的最终信号列表
        """
        if mode == 'top1':
            # 只用最高权重策略的信号
            max_weight = 0
            top_strategy = None
            for code, votes in all_signals.items():
                for sid, w in votes:
                    if w > max_weight:
                        max_weight = w
                        top_strategy = sid
            
            if top_strategy:
                return [{'code': c, 'weight': 1.0, 'sources': [top_strategy]}
                        for c, votes in all_signals.items()
                        if any(sid == top_strategy for sid, _ in votes)]
            return []
        
        elif mode == 'intersection':
            # 所有策略都推荐的信号
            n_strategies = len(set(sid for votes in all_signals.values() for sid, _ in votes))
            result = []
            for code, votes in all_signals.items():
                if len(votes) == n_strategies:
                    total_weight = sum(w for _, w in votes)
                    result.append({
                        'code': code,
                        'weight': total_weight / n_strategies,
                        'sources': [sid for sid, _ in votes]
                    })
            return sorted(result, key=lambda x: x['weight'], reverse=True)
        
        elif mode == 'union':
            # 所有信号并集
            result = []
            for code, votes in all_signals.items():
                total_weight = sum(w for _, w in votes)
                result.append({
                    'code': code,
                    'weight': total_weight,
                    'sources': [sid for sid, _ in votes]
                })
            return sorted(result, key=lambda x: x['weight'], reverse=True)
        
        elif mode == 'confidence':
            # 置信度加权：考虑策略历史准确率
            result = []
            for code, votes in all_signals.items():
                weighted_sum = 0
                confidence_sum = 0
                
                for sid, w in votes:
                    # 获取策略历史准确率作为置信度
                    if strategy_results and sid in strategy_results:
                        accuracy = strategy_results[sid].get('accuracy', 0.1)
                        confidence = max(0.1, accuracy)  # 最低 10% 置信度
                    else:
                        confidence = 0.15  # 默认置信度
                    
                    weighted_sum += w * confidence
                    confidence_sum += confidence
                
                # 多策略共识加成
                n_sources = len(votes)
                consensus_factor = 1 + 0.2 * (n_sources - 1)  # 每多一个来源+20%
                
                final_weight = (weighted_sum / max(confidence_sum, 0.01)) * consensus_factor
                
                result.append({
                    'code': code,
                    'weight': final_weight,
                    'n_sources': n_sources,
                    'sources': [sid for sid, _ in votes],
                    'avg_confidence': confidence_sum / n_sources if n_sources > 0 else 0,
                })
            
            return sorted(result, key=lambda x: x['weight'], reverse=True)
        
        elif mode == 'majority':
            # 多数投票：只有超过一半策略推荐才入选
            n_total_strategies = len(set(sid for votes in all_signals.values() for sid, _ in votes))
            threshold = max(2, n_total_strategies // 2)  # 至少2个策略同意
            
            result = []
            for code, votes in all_signals.items():
                if len(votes) >= threshold:
                    total_weight = sum(w for _, w in votes)
                    result.append({
                        'code': code,
                        'weight': total_weight,
                        'n_sources': len(votes),
                        'sources': [sid for sid, _ in votes],
                        'vote_ratio': len(votes) / n_total_strategies,
                    })
            
            return sorted(result, key=lambda x: (x['n_sources'], x['weight']), reverse=True)
        
        else:  # weighted (default)
            # 加权投票,多个策略推荐的权重更高
            result = []
            for code, votes in all_signals.items():
                # 权重 = 策略权重之和 * 推荐策略数的奖励
                total_weight = sum(w for _, w in votes)
                consensus_bonus = len(votes) ** 0.5  # 共识奖励
                final_weight = total_weight * consensus_bonus
                
                result.append({
                    'code': code,
                    'weight': final_weight,
                    'n_sources': len(votes),
                    'sources': [sid for sid, _ in votes]
                })
            
            return sorted(result, key=lambda x: x['weight'], reverse=True)
    
    def format_result(self, result: Dict) -> str:
        """格式化输出结果"""
        lines = [
            "=" * 60,
            f"自适应路由器信号 - {result['date']}",
            "=" * 60,
            "",
            f"市场状态: {result['market_state']['state_name']}",
            f"  20日涨跌: {result['market_state']['ret_20d']:.1f}%",
            f"  5日涨跌: {result['market_state']['ret_5d']:.1f}%",
            f"  波动率: {result['market_state']['volatility']:.1f}%",
            "",
            f"激活类别: {', '.join(result['active_categories'])}",
            f"参与策略: {result['active_strategies']}个",
            "",
        ]
        
        if result['strategy_results']:
            lines.append("策略信号:")
            for sid, data in sorted(result['strategy_results'].items(), 
                                   key=lambda x: x[1]['weight'], reverse=True):
                n_signals = len(data['signals'])
                lines.append(f"  {sid}: {n_signals}个信号, 权重={data['weight']:.2f}, "
                           f"历史准确率={data['accuracy']:.1f}%")
        
        lines.append("")
        lines.append(f"最终信号 (mode={result['mode']}):")
        
        if result['final_signals']:
            for s in result['final_signals'][:10]:
                sources = ', '.join(s['sources'][:3])
                if len(s['sources']) > 3:
                    sources += f'...({len(s["sources"])}个)'
                lines.append(f"  {s['code']}: 权重={s['weight']:.2f}, 来源=[{sources}]")
        else:
            lines.append("  无信号")
        
        return "\n".join(lines)


class StrategyMonitor:
    """
    策略有效性监控器
    
    功能:
    1. 跟踪每个策略的实时表现
    2. 检测性能衰退 (使用CUSUM算法)
    3. 生成健康报告和告警
    4. 自动标记需要审查的策略
    
    使用方法:
        monitor = StrategyMonitor(pool)
        monitor.record_outcome(strategy_id, date, code, actual_return)
        report = monitor.health_check()
    """
    
    # 告警阈值
    ALERT_THRESHOLDS = {
        'accuracy_drop': 0.5,      # 准确率下降50%触发告警
        'consecutive_fails': 5,    # 连续失败5次触发告警
        'cusum_threshold': 3.0,    # CUSUM统计量阈值
        'min_samples': 5,          # 最少样本数才进行检测
    }
    
    def __init__(self, pool: StrategyPool, state_file: Path = None):
        """
        初始化监控器
        
        Args:
            pool: 策略池实例
            state_file: 监控状态持久化文件
        """
        self.pool = pool
        self.state_file = state_file or Path("data/strategy_monitor_state.json")
        
        # 每个策略的监控状态
        self.monitor_state: Dict[str, Dict] = {}
        self._load_state()
    
    def _init_strategy_state(self, strategy_id: str) -> Dict:
        """初始化策略监控状态"""
        strategy = self.pool.get_strategy(strategy_id)
        baseline_acc = strategy.accuracy / 100 if strategy else 0.2
        
        return {
            'outcomes': [],           # [(date, code, return, hit), ...]
            'baseline_accuracy': baseline_acc,
            'cusum_pos': 0.0,         # CUSUM正向累积
            'cusum_neg': 0.0,         # CUSUM负向累积
            'consecutive_fails': 0,
            'last_alert': None,
            'status': 'active',       # active, warning, suspended
            'alert_history': [],
        }
    
    def record_outcome(self, strategy_id: str, date: str, code: str, 
                       actual_return: float) -> Dict:
        """
        记录策略信号的实际结果
        
        Args:
            strategy_id: 策略ID
            date: 信号日期
            code: 股票代码
            actual_return: 实际收益率 (小数形式, 如0.05表示5%)
            
        Returns:
            更新后的状态和可能的告警
        """
        if strategy_id not in self.monitor_state:
            self.monitor_state[strategy_id] = self._init_strategy_state(strategy_id)
        
        state = self.monitor_state[strategy_id]
        hit = actual_return > 0.03  # >3%为命中
        
        # 记录结果
        state['outcomes'].append({
            'date': date,
            'code': code,
            'return': actual_return,
            'hit': hit,
        })
        
        # 只保留最近50条
        state['outcomes'] = state['outcomes'][-50:]
        
        # 更新连续失败计数
        if hit:
            state['consecutive_fails'] = 0
        else:
            state['consecutive_fails'] += 1
        
        # 更新CUSUM (累积和控制图)
        # 使用基线准确率作为期望值
        expected = state['baseline_accuracy']
        deviation = (1.0 if hit else 0.0) - expected
        
        # CUSUM累积 (带drift校正)
        k = 0.5 * expected  # 允许的偏移量
        state['cusum_pos'] = max(0, state['cusum_pos'] + deviation - k)
        state['cusum_neg'] = min(0, state['cusum_neg'] + deviation + k)
        
        # 检查告警条件
        alert = self._check_alerts(strategy_id)
        
        # 同步更新到策略池
        self.pool.update_strategy_performance(
            strategy_id, date, [code], [actual_return]
        )
        
        # 持久化
        self._save_state()
        
        return {
            'strategy_id': strategy_id,
            'outcome': {'date': date, 'code': code, 'return': actual_return, 'hit': hit},
            'current_accuracy': self._calc_recent_accuracy(strategy_id),
            'consecutive_fails': state['consecutive_fails'],
            'cusum': {'pos': state['cusum_pos'], 'neg': state['cusum_neg']},
            'alert': alert,
        }
    
    def _check_alerts(self, strategy_id: str) -> Optional[Dict]:
        """检查是否需要触发告警"""
        state = self.monitor_state[strategy_id]
        outcomes = state['outcomes']
        
        if len(outcomes) < self.ALERT_THRESHOLDS['min_samples']:
            return None
        
        alerts = []
        
        # 1. 检查准确率下降
        recent_acc = self._calc_recent_accuracy(strategy_id)
        baseline_acc = state['baseline_accuracy']
        if recent_acc < baseline_acc * (1 - self.ALERT_THRESHOLDS['accuracy_drop']):
            alerts.append({
                'type': 'accuracy_drop',
                'severity': 'warning',
                'message': f"准确率从{baseline_acc*100:.1f}%降至{recent_acc*100:.1f}%",
            })
        
        # 2. 检查连续失败
        if state['consecutive_fails'] >= self.ALERT_THRESHOLDS['consecutive_fails']:
            alerts.append({
                'type': 'consecutive_fails',
                'severity': 'critical',
                'message': f"连续{state['consecutive_fails']}次未命中",
            })
        
        # 3. 检查CUSUM (性能衰退检测)
        if abs(state['cusum_neg']) > self.ALERT_THRESHOLDS['cusum_threshold']:
            alerts.append({
                'type': 'cusum_drift',
                'severity': 'critical',
                'message': f"CUSUM检测到显著性能衰退 (统计量={state['cusum_neg']:.2f})",
            })
        
        if not alerts:
            return None
        
        # 更新状态
        max_severity = max(a['severity'] for a in alerts)
        if max_severity == 'critical':
            state['status'] = 'suspended'
        elif max_severity == 'warning':
            state['status'] = 'warning'
        
        alert = {
            'strategy_id': strategy_id,
            'timestamp': datetime.now().isoformat(),
            'alerts': alerts,
            'action': 'suspend' if max_severity == 'critical' else 'review',
        }
        
        state['alert_history'].append(alert)
        state['last_alert'] = alert['timestamp']
        
        return alert
    
    def _calc_recent_accuracy(self, strategy_id: str, n: int = 10) -> float:
        """计算最近n笔的准确率"""
        outcomes = self.monitor_state.get(strategy_id, {}).get('outcomes', [])
        if not outcomes:
            return 0.0
        
        recent = outcomes[-n:]
        return sum(1 for o in recent if o['hit']) / len(recent)
    
    def health_check(self) -> Dict:
        """
        执行全面健康检查
        
        Returns:
            健康报告字典
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_strategies': len(self.pool.strategies),
            'monitored': len(self.monitor_state),
            'status_summary': {'active': 0, 'warning': 0, 'suspended': 0},
            'strategies': [],
            'alerts': [],
        }
        
        for sid in self.pool.strategies:
            if sid not in self.monitor_state:
                self.monitor_state[sid] = self._init_strategy_state(sid)
            
            state = self.monitor_state[sid]
            strategy = self.pool.get_strategy(sid)
            
            n_outcomes = len(state['outcomes'])
            recent_acc = self._calc_recent_accuracy(sid) if n_outcomes > 0 else None
            
            strategy_report = {
                'id': sid,
                'status': state['status'],
                'baseline_accuracy': state['baseline_accuracy'] * 100,
                'recent_accuracy': recent_acc * 100 if recent_acc else None,
                'n_outcomes': n_outcomes,
                'consecutive_fails': state['consecutive_fails'],
                'cusum_neg': state['cusum_neg'],
                'last_alert': state['last_alert'],
            }
            
            report['strategies'].append(strategy_report)
            report['status_summary'][state['status']] += 1
            
            # 收集最近告警
            if state['alert_history']:
                report['alerts'].extend(state['alert_history'][-3:])
        
        # 按状态排序 (suspended first, then warning, then active)
        status_order = {'suspended': 0, 'warning': 1, 'active': 2}
        report['strategies'].sort(key=lambda x: status_order.get(x['status'], 3))
        
        return report
    
    def reset_strategy(self, strategy_id: str):
        """重置策略监控状态 (在修复问题后调用)"""
        if strategy_id in self.monitor_state:
            self.monitor_state[strategy_id] = self._init_strategy_state(strategy_id)
            self._save_state()
    
    def get_active_strategies(self) -> List[str]:
        """获取所有活跃状态的策略ID"""
        active = []
        for sid in self.pool.strategies:
            if sid not in self.monitor_state:
                active.append(sid)
            elif self.monitor_state[sid]['status'] == 'active':
                active.append(sid)
        return active
    
    def format_report(self, report: Dict = None) -> str:
        """格式化健康报告"""
        if report is None:
            report = self.health_check()
        
        lines = [
            "=" * 60,
            "策略健康报告",
            f"时间: {report['timestamp'][:19]}",
            "=" * 60,
            "",
            f"策略总数: {report['total_strategies']}",
            f"  活跃: {report['status_summary']['active']}",
            f"  警告: {report['status_summary']['warning']}",
            f"  暂停: {report['status_summary']['suspended']}",
            "",
        ]
        
        # 需要关注的策略
        problem_strategies = [s for s in report['strategies'] 
                            if s['status'] != 'active']
        
        if problem_strategies:
            lines.append("需要关注的策略:")
            for s in problem_strategies:
                status_icon = "⚠️" if s['status'] == 'warning' else "🛑"
                acc_info = f"准确率:{s['recent_accuracy']:.1f}%" if s['recent_accuracy'] else "无数据"
                lines.append(f"  {status_icon} {s['id']}: {s['status']} | {acc_info} | "
                           f"连续失败:{s['consecutive_fails']}")
        else:
            lines.append("所有策略状态正常 ✓")
        
        # 最近告警
        if report['alerts']:
            lines.append("")
            lines.append("最近告警:")
            for alert in report['alerts'][-5:]:
                lines.append(f"  [{alert['timestamp'][:10]}] {alert['strategy_id']}: "
                           f"{alert['alerts'][0]['message']}")
        
        return "\n".join(lines)
    
    def _save_state(self):
        """持久化监控状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可JSON序列化的格式
        state_to_save = {}
        for sid, state in self.monitor_state.items():
            state_to_save[sid] = {
                'outcomes': state['outcomes'][-50:],
                'baseline_accuracy': state['baseline_accuracy'],
                'cusum_pos': state['cusum_pos'],
                'cusum_neg': state['cusum_neg'],
                'consecutive_fails': state['consecutive_fails'],
                'last_alert': state['last_alert'],
                'status': state['status'],
                'alert_history': state['alert_history'][-10:],
            }
        
        with open(self.state_file, 'w') as f:
            json.dump(state_to_save, f, indent=2, default=str)
    
    def _load_state(self):
        """加载监控状态"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            for sid, data in state.items():
                self.monitor_state[sid] = {
                    'outcomes': data.get('outcomes', []),
                    'baseline_accuracy': data.get('baseline_accuracy', 0.2),
                    'cusum_pos': data.get('cusum_pos', 0.0),
                    'cusum_neg': data.get('cusum_neg', 0.0),
                    'consecutive_fails': data.get('consecutive_fails', 0),
                    'last_alert': data.get('last_alert'),
                    'status': data.get('status', 'active'),
                    'alert_history': data.get('alert_history', []),
                }
        except Exception as e:
            print(f"[StrategyMonitor] 加载状态失败: {e}")


# 便捷函数
def create_adaptive_router(rounds_dir: Path = None) -> AdaptiveRouter:
    """创建自适应路由器"""
    pool = StrategyPool(rounds_dir)
    n = pool.load_verified_strategies()
    print(f"[StrategyPool] 加载了{n}个通过验证的策略")
    return AdaptiveRouter(pool)


def create_strategy_monitor(rounds_dir: Path = None) -> StrategyMonitor:
    """创建策略监控器"""
    pool = StrategyPool(rounds_dir)
    pool.load_verified_strategies()
    return StrategyMonitor(pool)


if __name__ == "__main__":
    # 演示用法
    import pandas as pd
    from pathlib import Path
    
    print("=== 策略池演示 ===\n")
    
    # 加载策略池
    pool = StrategyPool()
    n = pool.load_verified_strategies()
    print(f"加载了{n}个策略\n")
    print(pool.format_summary())
    
    # 加载数据测试路由器
    print("\n\n=== 路由器演示 ===\n")
    
    cache_dir = Path("~/.cache/quant-autoresearch").expanduser()
    if (cache_dir / "daily.parquet").exists():
        df = pd.read_parquet(cache_dir / "daily.parquet")
        
        router = AdaptiveRouter(pool)
        result = router.get_signals(df)
        print(router.format_result(result))
