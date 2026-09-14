#!/usr/bin/env python3
"""
快速策略评估器 (Fast Strategy Evaluator)

使用采样和并行计算加速评估
"""

import json
import importlib.util
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QuickEvalResult:
    """快速评估结果"""
    strategy_id: str
    total_trades: int = 0
    accuracy_3pct: float = 0.0
    avg_return: float = 0.0
    win_rate: float = 0.0
    p_value: float = 1.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    consistency: float = 0.0
    overall_score: float = 0.0
    grade: str = "F"
    issues: List[str] = field(default_factory=list)
    error: str = None


def load_strategy(path: str):
    """加载策略"""
    spec = importlib.util.spec_from_file_location('strategy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict_next_day


def quick_backtest(daily: pd.DataFrame, predict_fn, sample_rate: float = 0.3) -> pd.DataFrame:
    """
    快速回测 - 使用采样加速
    
    Args:
        daily: 日线数据
        predict_fn: 预测函数
        sample_rate: 采样率 (0.3 = 每3天采样1天)
    """
    daily = daily.copy()
    daily['date'] = pd.to_datetime(daily['date'])
    dates = sorted(daily['date'].unique())
    
    if len(dates) < 60:
        return pd.DataFrame()
    
    # 采样日期
    sample_step = max(1, int(1 / sample_rate))
    sample_dates = dates[40:-2:sample_step]
    
    trades = []
    
    for signal_date in sample_dates:
        # 获取历史数据
        date_idx = dates.index(signal_date)
        history_start = dates[max(0, date_idx - 60)]  # 只用60天历史
        hist = daily[(daily['date'] >= history_start) & (daily['date'] <= signal_date)]
        
        try:
            signals = predict_fn(hist)
            if signals is None or len(signals) == 0:
                continue
            
            if isinstance(signals, pd.Series):
                signals = signals[signals >= 0.3]
            
            if len(signals) == 0:
                continue
                
        except Exception:
            continue
        
        # 计算收益
        buy_date = dates[date_idx + 1]
        sell_date = dates[date_idx + 2]
        
        for code in (signals.index if isinstance(signals, pd.Series) else signals):
            code = str(code)
            
            buy_row = daily[(daily['date'] == buy_date) & (daily['code'] == code)]
            sell_row = daily[(daily['date'] == sell_date) & (daily['code'] == code)]
            
            if len(buy_row) == 0 or len(sell_row) == 0:
                continue
            
            buy_price = buy_row['open'].values[0]
            sell_price = sell_row['open'].values[0]
            
            if buy_price <= 0:
                continue
            
            ret = (sell_price - buy_price) / buy_price
            
            trades.append({
                'date': signal_date,
                'code': code,
                'return': ret,
                'hit_3pct': ret >= 0.03,
            })
    
    return pd.DataFrame(trades)


def evaluate_single(args) -> QuickEvalResult:
    """评估单个策略"""
    strategy_path, daily = args
    strategy_id = Path(strategy_path).stem
    result = QuickEvalResult(strategy_id=strategy_id)
    
    try:
        predict_fn = load_strategy(strategy_path)
        trades_df = quick_backtest(daily, predict_fn, sample_rate=0.25)
        
        if len(trades_df) == 0:
            result.issues.append("无交易")
            return result
        
        returns = trades_df['return'].values
        
        # 基础指标
        result.total_trades = len(trades_df)
        result.accuracy_3pct = trades_df['hit_3pct'].mean()
        result.avg_return = returns.mean()
        result.win_rate = (returns > 0).mean()
        
        # 统计检验
        if len(returns) >= 10:
            t_stat, p_val = stats.ttest_1samp(returns, 0)
            result.p_value = p_val / 2 if t_stat > 0 else 1.0
        
        # 夏普比率
        if returns.std() > 0:
            result.sharpe_ratio = (result.avg_return * 252) / (returns.std() * np.sqrt(252))
        
        # 最大回撤
        cumulative = (1 + pd.Series(returns)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        result.max_drawdown = abs(drawdowns.min())
        
        # 一致性 (月度)
        trades_df['month'] = pd.to_datetime(trades_df['date']).dt.to_period('M')
        monthly = trades_df.groupby('month')['return'].mean()
        if len(monthly) > 0:
            result.consistency = (monthly > 0).mean()
        
        # 问题检测
        if result.total_trades < 15:
            result.issues.append(f"交易不足({result.total_trades})")
        if result.p_value > 0.10:
            result.issues.append(f"不显著(p={result.p_value:.2f})")
        if result.avg_return < 0.001:
            result.issues.append(f"收益低({result.avg_return*100:.2f}%)")
        if result.max_drawdown > 0.25:
            result.issues.append(f"回撤大({result.max_drawdown*100:.0f}%)")
        
        # 评分
        score = 0
        score += min(30, result.accuracy_3pct * 150)  # 准确率 0-30分
        score += min(25, result.avg_return * 2000)     # 收益 0-25分
        score += min(20, (0.10 - result.p_value) * 200) if result.p_value < 0.10 else 0  # 显著性 0-20分
        score += min(15, result.sharpe_ratio * 10)     # 夏普 0-15分
        score += min(10, result.consistency * 10)      # 一致性 0-10分
        
        result.overall_score = max(0, score)
        
        # 评级
        if result.overall_score >= 70 and len(result.issues) == 0:
            result.grade = 'A'
        elif result.overall_score >= 55 and len(result.issues) <= 1:
            result.grade = 'B'
        elif result.overall_score >= 40:
            result.grade = 'C'
        elif result.overall_score >= 25:
            result.grade = 'D'
        else:
            result.grade = 'F'
            
    except Exception as e:
        result.error = str(e)[:50]
        result.issues.append(f"错误: {result.error}")
    
    return result


class FastEvaluator:
    """快速评估器"""
    
    def __init__(self, daily: pd.DataFrame = None):
        if daily is None:
            cache = Path.home() / '.cache/quant-autoresearch'
            daily = pd.read_parquet(cache / 'daily.parquet')
        
        self.daily = daily
        self.daily['date'] = pd.to_datetime(self.daily['date'])
    
    def evaluate_all(self, strategies_dir: str) -> List[QuickEvalResult]:
        """评估目录下所有策略"""
        strategies_dir = Path(strategies_dir)
        strategy_files = sorted(strategies_dir.glob('*.py'))
        strategy_files = [f for f in strategy_files if not f.name.startswith('_')]
        
        results = []
        
        for i, path in enumerate(strategy_files):
            print(f"[{i+1}/{len(strategy_files)}] {path.name}...", end=' ', flush=True)
            result = evaluate_single((str(path), self.daily))
            results.append(result)
            print(f"{result.grade} ({result.overall_score:.0f})")
        
        results.sort(key=lambda x: x.overall_score, reverse=True)
        return results
    
    def print_report(self, results: List[QuickEvalResult]):
        """打印报告"""
        print("\n" + "=" * 90)
        print("策略快速评估报告")
        print("=" * 90)
        
        grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for r in results:
            grades[r.grade] += 1
        
        print(f"\n评级分布: A={grades['A']} B={grades['B']} C={grades['C']} D={grades['D']} F={grades['F']}")
        print("-" * 90)
        
        print(f"{'策略':<15} {'评级':>4} {'分数':>5} {'交易':>5} {'准确率':>7} {'平均收益':>9} {'胜率':>6} {'p值':>6} {'夏普':>6} {'问题'}")
        print("-" * 90)
        
        for r in results:
            issues_str = '; '.join(r.issues[:2]) if r.issues else '-'
            if len(issues_str) > 20:
                issues_str = issues_str[:17] + '...'
            
            print(f"{r.strategy_id:<15} {r.grade:>4} {r.overall_score:>5.0f} "
                  f"{r.total_trades:>5} {r.accuracy_3pct*100:>6.1f}% "
                  f"{r.avg_return*100:>+8.2f}% {r.win_rate*100:>5.1f}% "
                  f"{r.p_value:>5.3f} {r.sharpe_ratio:>5.2f} {issues_str}")
        
        print("=" * 90)
        
        # 保存JSON
        output = {
            'timestamp': datetime.now().isoformat(),
            'total': len(results),
            'grades': grades,
            'results': [
                {
                    'id': r.strategy_id,
                    'grade': r.grade,
                    'score': r.overall_score,
                    'trades': r.total_trades,
                    'accuracy': r.accuracy_3pct,
                    'avg_return': r.avg_return,
                    'win_rate': r.win_rate,
                    'p_value': r.p_value,
                    'sharpe': r.sharpe_ratio,
                    'issues': r.issues,
                }
                for r in results
            ]
        }
        
        with open('strategy_evaluation.json', 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到 strategy_evaluation.json")


if __name__ == '__main__':
    print("加载数据...")
    evaluator = FastEvaluator()
    
    print("\n开始评估...\n")
    results = evaluator.evaluate_all('verified_strategies')
    
    evaluator.print_report(results)
