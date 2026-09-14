#!/usr/bin/env python3
"""
策略评估器 (Strategy Evaluator)

全面评估策略的有效性和合理性，包括：
1. 统计显著性检验
2. 过拟合检测
3. 稳健性分析
4. 风险调整收益
5. 交易成本敏感性
"""

import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EvaluationResult:
    """评估结果"""
    strategy_id: str
    
    # 基础指标
    total_trades: int = 0
    accuracy_3pct: float = 0.0
    avg_return: float = 0.0
    median_return: float = 0.0
    win_rate: float = 0.0
    
    # 统计显著性
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False  # p < 0.05
    
    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 稳健性
    consistency_score: float = 0.0  # 月度正收益比例
    regime_stability: float = 0.0   # 不同市场状态下表现一致性
    
    # 过拟合检测
    in_sample_return: float = 0.0
    out_sample_return: float = 0.0
    overfit_ratio: float = 0.0  # IS/OOS 比值，>2 可能过拟合
    
    # 交易成本敏感性
    return_after_cost: float = 0.0  # 扣除成本后收益
    break_even_cost: float = 0.0    # 盈亏平衡成本
    
    # 综合评分
    overall_score: float = 0.0
    grade: str = "F"  # A/B/C/D/F
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'total_trades': self.total_trades,
            'accuracy_3pct': self.accuracy_3pct,
            'avg_return': self.avg_return,
            'median_return': self.median_return,
            'win_rate': self.win_rate,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'consistency_score': self.consistency_score,
            'regime_stability': self.regime_stability,
            'in_sample_return': self.in_sample_return,
            'out_sample_return': self.out_sample_return,
            'overfit_ratio': self.overfit_ratio,
            'return_after_cost': self.return_after_cost,
            'break_even_cost': self.break_even_cost,
            'overall_score': self.overall_score,
            'grade': self.grade,
            'issues': self.issues,
        }


class StrategyEvaluator:
    """
    策略评估器
    
    评估维度：
    1. 收益能力 (Return)
    2. 统计显著性 (Significance)
    3. 风险控制 (Risk)
    4. 稳健性 (Robustness)
    5. 过拟合风险 (Overfitting)
    """
    
    # 评估标准阈值
    THRESHOLDS = {
        # 最低要求 (Gate)
        'min_trades': 20,           # 最少交易次数
        'min_accuracy': 0.12,       # 最低3%准确率 (随机基线~10%)
        'min_avg_return': 0.001,    # 最低平均收益 0.1%
        'max_p_value': 0.10,        # 最大 p 值 (90% 置信)
        
        # 优秀标准
        'good_accuracy': 0.20,      # 优秀准确率
        'good_sharpe': 0.5,         # 优秀夏普比
        'good_win_rate': 0.55,      # 优秀胜率
        
        # 风险限制
        'max_drawdown': 0.30,       # 最大回撤限制
        'max_overfit_ratio': 2.0,   # 过拟合比率上限
        
        # 交易成本 (单边)
        'trading_cost': 0.002,      # 0.2% (含佣金+滑点)
    }
    
    # 评分权重
    WEIGHTS = {
        'return': 0.25,        # 收益能力
        'significance': 0.20,  # 统计显著性
        'risk': 0.20,          # 风险控制
        'robustness': 0.20,    # 稳健性
        'overfitting': 0.15,   # 过拟合风险
    }
    
    def __init__(self, daily: pd.DataFrame = None, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path.home() / '.cache/quant-autoresearch'
        
        if daily is not None:
            self.daily = daily
        else:
            self.daily = pd.read_parquet(self.cache_dir / 'daily.parquet')
        
        self.daily['date'] = pd.to_datetime(self.daily['date'])
        self.dates = sorted(self.daily['date'].unique())
        
        # 预计算市场状态
        self._compute_market_regimes()
    
    def _compute_market_regimes(self):
        """预计算每日市场状态"""
        mkt = self.daily.groupby('date').agg({'close': 'median'}).reset_index()
        mkt = mkt.sort_values('date')
        mkt['ret_20d'] = mkt['close'].pct_change(20)
        
        def classify(ret):
            if pd.isna(ret):
                return 'range'
            elif ret > 0.05:
                return 'bull'
            elif ret < -0.05:
                return 'bear'
            else:
                return 'range'
        
        mkt['regime'] = mkt['ret_20d'].apply(classify)
        self.market_regimes = dict(zip(mkt['date'], mkt['regime']))
    
    def _load_strategy(self, strategy_path: str):
        """加载策略模块"""
        spec = importlib.util.spec_from_file_location('strategy', strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.predict_next_day
    
    def _run_backtest(self, predict_fn, start_date=None, end_date=None) -> pd.DataFrame:
        """运行回测，返回交易记录"""
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
        
        dates = self.dates
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        
        if len(dates) < 60:
            return pd.DataFrame()
        
        trades = []
        
        for i, signal_date in enumerate(dates[40:-2]):
            # 获取历史数据
            cutoff_idx = self.dates.index(signal_date)
            history_start = self.dates[max(0, cutoff_idx - 120)]
            hist = self.daily[(self.daily['date'] >= history_start) & 
                             (self.daily['date'] <= signal_date)]
            
            # 运行策略
            try:
                signals = predict_fn(hist.copy())
                if signals is None or len(signals) == 0:
                    continue
                
                # 过滤低信号
                if isinstance(signals, pd.Series):
                    signals = signals[signals >= 0.3]
                
                if len(signals) == 0:
                    continue
                
            except Exception as e:
                continue
            
            # 计算收益
            buy_date = dates[dates.index(signal_date) + 1]
            sell_date = dates[dates.index(signal_date) + 2]
            
            for code in (signals.index if isinstance(signals, pd.Series) else signals):
                code = str(code)
                
                # 获取价格
                buy_row = self.daily[(self.daily['date'] == buy_date) & 
                                    (self.daily['code'] == code)]
                sell_row = self.daily[(self.daily['date'] == sell_date) & 
                                     (self.daily['code'] == code)]
                
                if len(buy_row) == 0 or len(sell_row) == 0:
                    continue
                
                buy_price = buy_row['open'].values[0]
                sell_price = sell_row['open'].values[0]
                
                if buy_price <= 0:
                    continue
                
                ret = (sell_price - buy_price) / buy_price
                
                trades.append({
                    'signal_date': signal_date,
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'code': code,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'return': ret,
                    'hit_3pct': ret >= 0.03,
                    'regime': self.market_regimes.get(signal_date, 'range'),
                })
        
        return pd.DataFrame(trades)
    
    def evaluate(self, strategy_path: str) -> EvaluationResult:
        """
        全面评估策略
        """
        strategy_id = Path(strategy_path).stem
        result = EvaluationResult(strategy_id=strategy_id)
        
        # 加载策略
        try:
            predict_fn = self._load_strategy(strategy_path)
        except Exception as e:
            result.issues.append(f"加载失败: {e}")
            return result
        
        # 运行完整回测
        trades = self._run_backtest(predict_fn)
        
        if len(trades) == 0:
            result.issues.append("无交易记录")
            return result
        
        returns = trades['return'].values
        
        # ===== 1. 基础指标 =====
        result.total_trades = len(trades)
        result.accuracy_3pct = trades['hit_3pct'].mean()
        result.avg_return = returns.mean()
        result.median_return = np.median(returns)
        result.win_rate = (returns > 0).mean()
        
        # ===== 2. 统计显著性 =====
        if len(returns) >= 20:
            # t检验: 平均收益是否显著大于0
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            result.t_statistic = t_stat
            result.p_value = p_value / 2 if t_stat > 0 else 1.0  # 单尾检验
            result.is_significant = result.p_value < 0.05 and result.avg_return > 0
        
        # ===== 3. 风险指标 =====
        result.volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # 最大回撤
        cumulative = (1 + pd.Series(returns)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        result.max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        # 夏普比率 (假设无风险利率为0)
        if result.volatility > 0:
            result.sharpe_ratio = (result.avg_return * 252) / result.volatility
        
        # Sortino 比率 (只考虑下行风险)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std > 0:
                result.sortino_ratio = (result.avg_return * 252) / downside_std
        
        # Calmar 比率 (年化收益/最大回撤)
        if result.max_drawdown > 0:
            result.calmar_ratio = (result.avg_return * 252) / result.max_drawdown
        
        # ===== 4. 稳健性分析 =====
        # 月度一致性
        trades['month'] = pd.to_datetime(trades['signal_date']).dt.to_period('M')
        monthly_returns = trades.groupby('month')['return'].mean()
        if len(monthly_returns) > 0:
            result.consistency_score = (monthly_returns > 0).mean()
        
        # 市场状态稳定性
        regime_returns = trades.groupby('regime')['return'].mean()
        if len(regime_returns) > 1:
            result.regime_stability = 1 - (regime_returns.std() / (abs(regime_returns.mean()) + 0.01))
            result.regime_stability = max(0, min(1, result.regime_stability))
        
        # ===== 5. 过拟合检测 (时间分割) =====
        n_dates = len(trades)
        if n_dates >= 40:
            split_idx = int(n_dates * 0.7)
            in_sample = trades.iloc[:split_idx]
            out_sample = trades.iloc[split_idx:]
            
            result.in_sample_return = in_sample['return'].mean()
            result.out_sample_return = out_sample['return'].mean()
            
            if result.out_sample_return > 0:
                result.overfit_ratio = result.in_sample_return / result.out_sample_return
            elif result.in_sample_return > 0:
                result.overfit_ratio = 10  # 样本外亏损，严重过拟合
            else:
                result.overfit_ratio = 1
        
        # ===== 6. 交易成本敏感性 =====
        cost = self.THRESHOLDS['trading_cost'] * 2  # 买卖双向
        result.return_after_cost = result.avg_return - cost
        
        if result.avg_return > 0:
            result.break_even_cost = result.avg_return / 2  # 单边成本
        
        # ===== 7. 问题检测 =====
        issues = []
        
        if result.total_trades < self.THRESHOLDS['min_trades']:
            issues.append(f"交易次数不足 ({result.total_trades}<{self.THRESHOLDS['min_trades']})")
        
        if result.p_value > self.THRESHOLDS['max_p_value']:
            issues.append(f"统计不显著 (p={result.p_value:.3f})")
        
        if result.avg_return < self.THRESHOLDS['min_avg_return']:
            issues.append(f"平均收益过低 ({result.avg_return*100:.2f}%)")
        
        if result.max_drawdown > self.THRESHOLDS['max_drawdown']:
            issues.append(f"最大回撤过大 ({result.max_drawdown*100:.1f}%)")
        
        if result.overfit_ratio > self.THRESHOLDS['max_overfit_ratio']:
            issues.append(f"可能过拟合 (IS/OOS={result.overfit_ratio:.1f})")
        
        if result.return_after_cost < 0:
            issues.append(f"扣除成本后亏损 ({result.return_after_cost*100:.2f}%)")
        
        if result.consistency_score < 0.4:
            issues.append(f"月度一致性差 ({result.consistency_score:.1%})")
        
        result.issues = issues
        
        # ===== 8. 综合评分 =====
        scores = {}
        
        # 收益能力评分 (0-100)
        return_score = min(100, max(0, 
            50 + (result.avg_return - 0.005) / 0.01 * 25 +  # 基于平均收益
            (result.accuracy_3pct - 0.10) / 0.10 * 25      # 基于准确率
        ))
        scores['return'] = return_score
        
        # 统计显著性评分
        if result.is_significant:
            sig_score = min(100, 60 + (0.05 - result.p_value) / 0.05 * 40)
        else:
            sig_score = max(0, 60 - result.p_value * 100)
        scores['significance'] = sig_score
        
        # 风险控制评分
        risk_score = 100
        if result.max_drawdown > 0.10:
            risk_score -= (result.max_drawdown - 0.10) / 0.20 * 50
        if result.sharpe_ratio < 0.5:
            risk_score -= (0.5 - result.sharpe_ratio) / 0.5 * 30
        scores['risk'] = max(0, risk_score)
        
        # 稳健性评分
        robust_score = (
            result.consistency_score * 50 +
            result.regime_stability * 50
        )
        scores['robustness'] = robust_score
        
        # 过拟合风险评分 (越低越好)
        if result.overfit_ratio <= 1.5:
            overfit_score = 100
        elif result.overfit_ratio <= 2.0:
            overfit_score = 70
        elif result.overfit_ratio <= 3.0:
            overfit_score = 40
        else:
            overfit_score = 0
        scores['overfitting'] = overfit_score
        
        # 加权总分
        result.overall_score = sum(
            scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS
        )
        
        # 评级
        if result.overall_score >= 80 and len(issues) == 0:
            result.grade = 'A'
        elif result.overall_score >= 65 and len(issues) <= 1:
            result.grade = 'B'
        elif result.overall_score >= 50 and len(issues) <= 2:
            result.grade = 'C'
        elif result.overall_score >= 35:
            result.grade = 'D'
        else:
            result.grade = 'F'
        
        return result
    
    def evaluate_all(self, strategies_dir: str, output_file: str = None) -> List[EvaluationResult]:
        """批量评估所有策略"""
        strategies_dir = Path(strategies_dir)
        results = []
        
        for py_file in sorted(strategies_dir.glob('*.py')):
            if py_file.name.startswith('_'):
                continue
            
            print(f"评估 {py_file.name}...", end=' ')
            try:
                result = self.evaluate(str(py_file))
                results.append(result)
                print(f"{result.grade} ({result.overall_score:.0f}分)")
            except Exception as e:
                print(f"错误: {e}")
        
        # 排序
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        # 保存结果
        if output_file:
            output = {
                'timestamp': datetime.now().isoformat(),
                'total_strategies': len(results),
                'grade_distribution': {
                    'A': sum(1 for r in results if r.grade == 'A'),
                    'B': sum(1 for r in results if r.grade == 'B'),
                    'C': sum(1 for r in results if r.grade == 'C'),
                    'D': sum(1 for r in results if r.grade == 'D'),
                    'F': sum(1 for r in results if r.grade == 'F'),
                },
                'results': [r.to_dict() for r in results]
            }
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
        
        return results
    
    def print_report(self, results: List[EvaluationResult]):
        """打印评估报告"""
        print("\n" + "=" * 80)
        print("策略评估报告")
        print("=" * 80)
        
        # 评级分布
        grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for r in results:
            grades[r.grade] += 1
        
        print(f"\n评级分布: A={grades['A']} B={grades['B']} C={grades['C']} D={grades['D']} F={grades['F']}")
        print("-" * 80)
        
        # 表头
        print(f"{'策略':15s} {'评级':4s} {'分数':5s} {'交易':5s} {'准确率':7s} {'平均收益':8s} {'p值':6s} {'夏普':5s} {'问题'}")
        print("-" * 80)
        
        for r in results:
            issues_str = '; '.join(r.issues[:2]) if r.issues else '无'
            if len(issues_str) > 25:
                issues_str = issues_str[:22] + '...'
            
            print(f"{r.strategy_id:15s} {r.grade:4s} {r.overall_score:5.0f} "
                  f"{r.total_trades:5d} {r.accuracy_3pct*100:6.1f}% "
                  f"{r.avg_return*100:+7.2f}% {r.p_value:5.3f} "
                  f"{r.sharpe_ratio:5.2f} {issues_str}")
        
        print("=" * 80)


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='策略评估器')
    parser.add_argument('--strategy', '-s', help='单个策略文件路径')
    parser.add_argument('--dir', '-d', help='策略目录')
    parser.add_argument('--output', '-o', help='输出JSON文件')
    
    args = parser.parse_args()
    
    evaluator = StrategyEvaluator()
    
    if args.strategy:
        result = evaluator.evaluate(args.strategy)
        evaluator.print_report([result])
    elif args.dir:
        results = evaluator.evaluate_all(args.dir, args.output)
        evaluator.print_report(results)
    else:
        # 默认评估 verified_strategies
        results = evaluator.evaluate_all('verified_strategies', 'strategy_evaluation.json')
        evaluator.print_report(results)


if __name__ == '__main__':
    main()
