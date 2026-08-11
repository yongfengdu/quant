#!/usr/bin/env python3
"""
分市场状态回测器 (Regime-Aware Backtester)

核心功能：
1. 按市场状态(bull/bear/range)分别回测策略
2. 为每个状态计算独立的指标
3. 支持并行回测多个策略候选
"""

import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from infra.island_evolution import (
    MarketRegime, 
    MarketRegimeDetector,
    StrategyCandidate,
)


@dataclass
class RegimeBacktestResult:
    """单个市场状态的回测结果"""
    regime: str
    accuracy_3pct: float  # 3%准确率
    trades: int
    avg_return: float
    median_return: float
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe: float
    
    def to_dict(self) -> Dict:
        return {
            'accuracy_3pct': self.accuracy_3pct,
            'trades': self.trades,
            'avg_return': self.avg_return,
            'median_return': self.median_return,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe': self.sharpe,
        }


class RegimeBacktester:
    """
    分市场状态回测器
    """
    
    def __init__(
        self,
        daily: pd.DataFrame = None,
        cache_dir: Path = None,
    ):
        """
        初始化回测器
        
        Args:
            daily: 日线数据
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir or Path.home() / '.cache/quant-autoresearch'
        
        if daily is not None:
            self.daily = daily
        else:
            self.daily = pd.read_parquet(self.cache_dir / 'daily.parquet')
        
        self.daily['date'] = pd.to_datetime(self.daily['date'])
        
        # 加载 universe
        universe = pd.read_parquet(self.cache_dir / 'universe.parquet')
        self.universe_codes = set(universe['code'].values)
        self.daily = self.daily[self.daily['code'].isin(self.universe_codes)].copy()
        
        # 市场状态检测器
        self.regime_detector = MarketRegimeDetector()
        
        # 缓存市场状态分类
        self._regime_cache: Optional[pd.DataFrame] = None
    
    def _classify_all_dates(self) -> pd.DataFrame:
        """分类所有日期的市场状态"""
        if self._regime_cache is not None:
            return self._regime_cache
        
        dates = sorted(self.daily['date'].unique())
        
        # 计算市场指数
        market_idx = self.daily.groupby('date').agg({
            'close': 'median',
        }).reset_index()
        market_idx = market_idx.sort_values('date').reset_index(drop=True)
        market_idx['ret_20d'] = market_idx['close'].pct_change(20)
        
        # 分类
        def classify(ret_20d):
            if pd.isna(ret_20d):
                return 'range'
            elif ret_20d >= 0.05:
                return 'bull'
            elif ret_20d <= -0.05:
                return 'bear'
            else:
                return 'range'
        
        market_idx['regime'] = market_idx['ret_20d'].apply(classify)
        
        self._regime_cache = market_idx[['date', 'regime']].copy()
        return self._regime_cache
    
    def get_regime_dates(self, regime: str) -> List[pd.Timestamp]:
        """获取指定市场状态的所有日期"""
        classified = self._classify_all_dates()
        dates = classified[classified['regime'] == regime]['date'].tolist()
        return dates
    
    def backtest_on_regime(
        self,
        predict_fn: Callable,
        regime: str,
        min_trades: int = 5,
    ) -> RegimeBacktestResult:
        """
        在指定市场状态下回测策略
        
        Args:
            predict_fn: 预测函数 (df -> Series)
            regime: 市场状态 ('bull', 'bear', 'range')
            min_trades: 最少交易数
            
        Returns:
            回测结果
        """
        regime_dates = self.get_regime_dates(regime)
        
        if len(regime_dates) < 20:
            return RegimeBacktestResult(
                regime=regime,
                accuracy_3pct=0, trades=0, avg_return=0,
                median_return=0, win_rate=0, total_return=0,
                max_drawdown=0, sharpe=0,
            )
        
        trades = []
        
        for date in regime_dates:
            # 获取到该日期为止的数据
            df_until = self.daily[self.daily['date'] <= date].copy()
            
            if len(df_until) < 100:
                continue
            
            try:
                # 运行预测
                signals = predict_fn(df_until)
                
                if signals is None or len(signals) == 0:
                    continue
                
                # 获取有信号的股票
                signal_stocks = signals[signals > 0].index.tolist()
                
                if not signal_stocks:
                    continue
                
                # 计算收益 (T+1 open 买入, T+2 open 卖出)
                for code in signal_stocks:
                    # 找 T+1 和 T+2 的日期
                    all_dates = sorted(self.daily[self.daily['code'] == code]['date'].unique())
                    try:
                        date_idx = all_dates.index(date)
                        if date_idx + 2 >= len(all_dates):
                            continue
                        
                        t1_date = all_dates[date_idx + 1]
                        t2_date = all_dates[date_idx + 2]
                        
                        # 获取价格
                        t1_row = self.daily[
                            (self.daily['code'] == code) & 
                            (self.daily['date'] == t1_date)
                        ]
                        t2_row = self.daily[
                            (self.daily['code'] == code) & 
                            (self.daily['date'] == t2_date)
                        ]
                        
                        if len(t1_row) == 0 or len(t2_row) == 0:
                            continue
                        
                        entry_price = t1_row['open'].values[0]
                        exit_price = t2_row['open'].values[0]
                        
                        if entry_price <= 0:
                            continue
                        
                        ret = (exit_price - entry_price) / entry_price
                        
                        trades.append({
                            'date': str(date.date()),
                            'code': code,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return': ret,
                            'signal_strength': signals.get(code, 0),
                        })
                    except (ValueError, IndexError):
                        continue
                        
            except Exception as e:
                continue
        
        # 计算指标
        n_trades = len(trades)
        
        if n_trades < min_trades:
            return RegimeBacktestResult(
                regime=regime,
                accuracy_3pct=0, trades=n_trades, avg_return=0,
                median_return=0, win_rate=0, total_return=0,
                max_drawdown=0, sharpe=0,
            )
        
        returns = [t['return'] for t in trades]
        returns_arr = np.array(returns)
        
        # 3%准确率
        accuracy_3pct = (returns_arr >= 0.03).mean() * 100
        
        # 平均/中位数收益
        avg_return = np.mean(returns_arr) * 100
        median_return = np.median(returns_arr) * 100
        
        # 胜率
        win_rate = (returns_arr > 0).mean() * 100
        
        # 总收益 (累乘)
        total_return = (np.prod(1 + returns_arr) - 1) * 100
        
        # 最大回撤
        cumulative = np.cumprod(1 + returns_arr)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Sharpe (假设年化)
        if len(returns_arr) > 1 and np.std(returns_arr) > 0:
            sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
        else:
            sharpe = 0
        
        return RegimeBacktestResult(
            regime=regime,
            accuracy_3pct=accuracy_3pct,
            trades=n_trades,
            avg_return=avg_return,
            median_return=median_return,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
        )
    
    def backtest_all_regimes(
        self,
        predict_fn: Callable,
    ) -> Dict[str, RegimeBacktestResult]:
        """
        在所有市场状态下回测策略
        
        Returns:
            {regime: result}
        """
        results = {}
        
        for regime in ['bull', 'bear', 'range']:
            result = self.backtest_on_regime(predict_fn, regime)
            results[regime] = result
        
        return results
    
    def backtest_strategy_code(
        self,
        code: str,
        strategy_id: str = "test",
    ) -> Dict[str, Dict]:
        """
        回测策略代码
        
        Args:
            code: 策略代码字符串
            strategy_id: 策略ID
            
        Returns:
            {
                'overall': {...},
                'regime_metrics': {
                    'bull': {...},
                    'bear': {...},
                    'range': {...},
                }
            }
        """
        # 动态加载策略
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            dir='/tmp'
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"strategy_{strategy_id}", 
                temp_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            predict_fn = module.predict_next_day
            
            # 分状态回测
            regime_results = self.backtest_all_regimes(predict_fn)
            
            # 计算整体指标
            all_trades = sum(r.trades for r in regime_results.values())
            
            if all_trades > 0:
                # 加权平均
                overall_acc = sum(
                    r.accuracy_3pct * r.trades 
                    for r in regime_results.values()
                ) / all_trades
                
                overall_avg_ret = sum(
                    r.avg_return * r.trades 
                    for r in regime_results.values()
                ) / all_trades
            else:
                overall_acc = 0
                overall_avg_ret = 0
            
            return {
                'overall': {
                    'accuracy_3pct': overall_acc,
                    'trades': all_trades,
                    'avg_return': overall_avg_ret,
                },
                'regime_metrics': {
                    regime: result.to_dict()
                    for regime, result in regime_results.items()
                }
            }
            
        finally:
            os.unlink(temp_path)


def backtest_candidate(
    candidate_data: Dict,
    cache_dir: str,
) -> Dict:
    """
    回测单个候选策略（用于并行处理）
    
    Args:
        candidate_data: {
            'id': str,
            'code': str,
        }
        cache_dir: 缓存目录
        
    Returns:
        回测结果
    """
    try:
        backtester = RegimeBacktester(cache_dir=Path(cache_dir))
        result = backtester.backtest_strategy_code(
            candidate_data['code'],
            candidate_data['id'],
        )
        return {
            'id': candidate_data['id'],
            'success': True,
            'result': result,
        }
    except Exception as e:
        return {
            'id': candidate_data['id'],
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


class ParallelRegimeBacktester:
    """
    并行分状态回测器
    
    支持同时回测多个策略候选
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        cache_dir: Path = None,
    ):
        self.max_workers = max_workers
        self.cache_dir = cache_dir or Path.home() / '.cache/quant-autoresearch'
    
    def backtest_candidates(
        self,
        candidates: List[StrategyCandidate],
    ) -> List[Tuple[StrategyCandidate, Dict]]:
        """
        并行回测多个候选策略
        
        Args:
            candidates: 候选策略列表
            
        Returns:
            [(candidate, result), ...]
        """
        results = []
        
        # 准备任务数据
        tasks = [
            {
                'id': c.id,
                'code': c.code,
            }
            for c in candidates
        ]
        
        # 串行执行（更稳定，便于调试）
        # TODO: 改为并行
        backtester = RegimeBacktester(cache_dir=self.cache_dir)
        
        for task, candidate in zip(tasks, candidates):
            try:
                result = backtester.backtest_strategy_code(
                    task['code'],
                    task['id'],
                )
                
                # 更新候选的指标
                candidate.metrics = result['overall']
                candidate.regime_metrics = result['regime_metrics']
                
                results.append((candidate, result))
                
            except Exception as e:
                print(f"回测 {task['id']} 失败: {e}")
                results.append((candidate, {'error': str(e)}))
        
        return results


def test_regime_backtester():
    """测试分状态回测器"""
    print("Testing Regime Backtester...")
    
    backtester = RegimeBacktester()
    
    # 统计各状态的日期数
    for regime in ['bull', 'bear', 'range']:
        dates = backtester.get_regime_dates(regime)
        print(f"{regime}: {len(dates)} 天")
    
    # 测试一个简单策略
    test_code = '''
import pandas as pd
import numpy as np

def predict_next_day(df):
    """简单的超跌反弹策略"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 5日收益
    df['ret_5d'] = g['close'].pct_change(5)
    
    today = df[df['date'] == df['date'].max()].copy()
    
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件: 5日跌幅超过8%
    signal = (today['ret_5d'] < -0.08).astype(float)
    
    return pd.Series(signal.values, index=today['code'])
'''
    
    print("\n回测简单超跌策略...")
    result = backtester.backtest_strategy_code(test_code, "test_oversold")
    
    print(f"\n整体指标:")
    print(f"  准确率: {result['overall']['accuracy_3pct']:.1f}%")
    print(f"  交易数: {result['overall']['trades']}")
    print(f"  均收益: {result['overall']['avg_return']:.2f}%")
    
    print(f"\n分状态指标:")
    for regime, metrics in result['regime_metrics'].items():
        print(f"  {regime}:")
        print(f"    准确率: {metrics['accuracy_3pct']:.1f}%")
        print(f"    交易数: {metrics['trades']}")
        print(f"    均收益: {metrics['avg_return']:.2f}%")
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_regime_backtester()
