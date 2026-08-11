"""
数据驱动的分层Gate配置模块

基于统计显著性分析(p<0.05)确定不同准确率下的最小交易数要求。
核心思想: 准确率越高,需要的样本量越少即可排除随机运气。

数据基础:
- 基线准确率: 10.14% (随机选股收益>3%的概率)
- 数据集: 46只股票, 1584个交易日, 72K+有效交易机会
- 统计方法: 二项分布检验, H0: 真实准确率=基线

使用方法:
    from infra.gate_config import TieredGate, GateAnalyzer
    
    gate = TieredGate()
    passed, reason = gate.check(accuracy=23.5, trades=30, avg_return=0.5)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import json


@dataclass
class GateTier:
    """单个Gate层级配置"""
    min_accuracy: float      # 最低准确率要求(%)
    min_trades: int          # 最低交易数要求
    p_value: float           # 对应的统计p值
    description: str         # 描述


class TieredGate:
    """
    分层Gate: 基于统计显著性的动态交易数要求
    
    准确率越高,所需样本量越少:
    - 30%准确率 → 只需7笔交易 (与10%基线差距大,容易区分)
    - 15%准确率 → 需要101笔交易 (与10%基线差距小,需大样本)
    """
    
    # 默认基线准确率 (可通过analyze_baseline动态更新)
    DEFAULT_BASELINE = 0.1014  # 10.14%
    
    def __init__(self, baseline_rate: float = None):
        """
        初始化分层Gate
        
        Args:
            baseline_rate: 基线准确率(0-1), 如果None则使用默认值
        """
        self.baseline_rate = baseline_rate or self.DEFAULT_BASELINE
        self.tiers = self._compute_tiers()
        
    def _compute_tiers(self, alpha: float = 0.05) -> List[GateTier]:
        """
        基于统计显著性计算各层级的最小交易数
        
        使用二项分布检验:
        H0: 真实准确率 = baseline_rate
        H1: 真实准确率 > baseline_rate
        """
        tiers = []
        
        # 检查的准确率层级
        accuracy_levels = [0.50, 0.40, 0.30, 0.25, 0.23, 0.20, 0.18, 0.15]
        
        for acc in accuracy_levels:
            min_n = self._find_min_trades(acc, alpha)
            if min_n and min_n <= 300:  # 合理范围内
                tiers.append(GateTier(
                    min_accuracy=acc * 100,
                    min_trades=min_n,
                    p_value=alpha,
                    description=f"{acc*100:.0f}%准确率需≥{min_n}笔交易(p<{alpha})"
                ))
        
        # 按准确率降序排列(高准确率优先匹配)
        tiers.sort(key=lambda t: t.min_accuracy, reverse=True)
        return tiers
    
    def _find_min_trades(self, target_acc: float, alpha: float) -> Optional[int]:
        """
        找到在给定准确率下,拒绝H0所需的最小交易数
        
        Args:
            target_acc: 目标准确率(0-1)
            alpha: 显著性水平
            
        Returns:
            最小交易数,如果>300则返回None
        """
        for n in range(5, 301):
            # 观察到的成功次数
            observed = int(np.ceil(n * target_acc))
            # p值: P(X >= observed | H0)
            p_value = 1 - stats.binom.cdf(observed - 1, n, self.baseline_rate)
            if p_value < alpha:
                return n
        return None
    
    def check(self, accuracy: float, trades: int, avg_return: float = None,
              max_drawdown: float = None) -> Tuple[bool, str]:
        """
        检查策略是否通过Gate
        
        Args:
            accuracy: 准确率(%), e.g., 23.5
            trades: 交易次数
            avg_return: 平均收益率(%), 可选
            max_drawdown: 最大回撤(%), 可选,负数
            
        Returns:
            (passed, reason): 是否通过及原因
        """
        # 找到匹配的层级
        matched_tier = None
        for tier in self.tiers:
            if accuracy >= tier.min_accuracy:
                matched_tier = tier
                break
        
        if matched_tier is None:
            return False, f"准确率{accuracy:.1f}%低于最低要求15%"
        
        # 检查交易数
        if trades < matched_tier.min_trades:
            return False, (f"准确率{accuracy:.1f}%需要≥{matched_tier.min_trades}笔交易"
                          f"(当前{trades}笔,差{matched_tier.min_trades - trades}笔)")
        
        # 检查收益率(如果提供)
        if avg_return is not None and avg_return <= 0:
            return False, f"平均收益率{avg_return:.2f}%≤0"
        
        # 检查最大回撤(如果提供)
        if max_drawdown is not None and max_drawdown < -45:
            return False, f"最大回撤{max_drawdown:.1f}%超过-45%限制"
        
        return True, f"通过: {accuracy:.1f}%准确率+{trades}笔交易(需≥{matched_tier.min_trades}笔)"
    
    def get_requirement(self, accuracy: float) -> Optional[int]:
        """
        获取指定准确率下的最小交易数要求
        
        Args:
            accuracy: 准确率(%)
            
        Returns:
            最小交易数,如果准确率不够则返回None
        """
        for tier in self.tiers:
            if accuracy >= tier.min_accuracy:
                return tier.min_trades
        return None
    
    def format_tiers(self) -> str:
        """格式化输出所有层级配置"""
        lines = [
            "分层Gate配置 (基线准确率={:.2f}%)".format(self.baseline_rate * 100),
            "=" * 50,
            "| 准确率    | 最少交易数 | 统计依据 |",
            "|-----------|-----------|----------|"
        ]
        for tier in self.tiers:
            lines.append(f"| ≥{tier.min_accuracy:5.0f}%   | {tier.min_trades:9} | p<0.05   |")
        return "\n".join(lines)


class GateAnalyzer:
    """
    Gate参数分析器: 基于实际数据计算最优Gate配置
    
    用于:
    1. 分析数据集的基线准确率
    2. 模拟随机策略的准确率分布
    3. 验证Gate参数的统计合理性
    """
    
    def __init__(self, cache_dir: Path = None):
        """
        初始化分析器
        
        Args:
            cache_dir: 数据缓存目录
        """
        self.cache_dir = cache_dir or Path("~/.cache/quant-autoresearch").expanduser()
        self._returns = None
        self._baseline = None
        
    def load_data(self) -> np.ndarray:
        """加载并计算T+1买T+2卖的收益率"""
        if self._returns is not None:
            return self._returns
            
        import pandas as pd
        
        daily = pd.read_parquet(self.cache_dir / "daily.parquet")
        daily = daily.sort_values(['code', 'date'])
        daily['next_open'] = daily.groupby('code')['open'].shift(-1)
        daily['next2_open'] = daily.groupby('code')['open'].shift(-2)
        daily['forward_return'] = (daily['next2_open'] - daily['next_open']) / daily['next_open'] * 100
        
        valid = daily.dropna(subset=['forward_return'])
        self._returns = valid['forward_return'].values
        return self._returns
    
    def analyze_baseline(self, threshold: float = 3.0) -> Dict:
        """
        分析基线准确率
        
        Args:
            threshold: 收益率阈值(%), 默认3%
            
        Returns:
            分析结果字典
        """
        returns = self.load_data()
        
        baseline = (returns > threshold).mean()
        self._baseline = baseline
        
        return {
            "baseline_accuracy": baseline,
            "baseline_pct": baseline * 100,
            "threshold": threshold,
            "total_samples": len(returns),
            "positive_samples": (returns > threshold).sum(),
            "return_mean": returns.mean(),
            "return_std": returns.std(),
            "return_median": np.median(returns),
        }
    
    def simulate_random_strategy(self, n_trades: int, n_simulations: int = 10000,
                                  threshold: float = 3.0) -> Dict:
        """
        模拟随机选股策略的准确率分布
        
        Args:
            n_trades: 交易次数
            n_simulations: 模拟次数
            threshold: 收益率阈值(%)
            
        Returns:
            模拟结果
        """
        returns = self.load_data()
        
        accuracies = []
        for _ in range(n_simulations):
            sample = np.random.choice(returns, size=n_trades, replace=True)
            acc = (sample > threshold).mean() * 100
            accuracies.append(acc)
        
        accs = np.array(accuracies)
        return {
            "n_trades": n_trades,
            "mean_accuracy": accs.mean(),
            "std_accuracy": accs.std(),
            "ci_lower": np.percentile(accs, 2.5),
            "ci_upper": np.percentile(accs, 97.5),
            "p_above_15": (accs >= 15).mean(),
            "p_above_20": (accs >= 20).mean(),
        }
    
    def compute_optimal_tiers(self, alpha: float = 0.05) -> TieredGate:
        """
        基于当前数据计算最优的分层Gate配置
        
        Args:
            alpha: 显著性水平
            
        Returns:
            配置好的TieredGate实例
        """
        if self._baseline is None:
            self.analyze_baseline()
        
        return TieredGate(baseline_rate=self._baseline)
    
    def validate_historical_strategies(self, audit_log_path: Path) -> Dict:
        """
        验证历史策略在新Gate下的通过情况
        
        Args:
            audit_log_path: audit.log路径
            
        Returns:
            验证结果
        """
        import re
        
        strategies = []
        with open(audit_log_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "HAS_ACCURACY" in line and "Accuracy_3%=" in line:
                acc_match = re.search(r"Accuracy_3%=([\d.]+)%", line)
                ret_match = re.search(r"AvgReturn=([-\d.]+)%", line)
                if acc_match:
                    acc = float(acc_match.group(1))
                    ret = float(ret_match.group(1)) if ret_match else 0
                    # 找交易数
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        if "HAS_TRADES" in lines[j]:
                            trades_match = re.search(r"Trades=(\d+)", lines[j])
                            if trades_match:
                                trades = int(trades_match.group(1))
                                strategies.append({
                                    'accuracy': acc,
                                    'trades': trades,
                                    'avg_return': ret
                                })
                                break
        
        # 去重
        seen = set()
        unique = []
        for s in strategies:
            key = (s['accuracy'], s['trades'])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        
        # 用新Gate检验
        gate = self.compute_optimal_tiers()
        
        results = {
            'total': len(unique),
            'high_accuracy': 0,  # >=15%
            'passed_old': 0,     # 旧gate通过(>=15%, >=50trades, >0return)
            'passed_new': 0,     # 新gate通过
            'rescued': [],       # 被新gate救回的策略
            'still_failed': [],  # 仍然失败的策略
        }
        
        for s in unique:
            if s['accuracy'] >= 15:
                results['high_accuracy'] += 1
                
                # 旧gate
                old_pass = (s['trades'] >= 50 and s['avg_return'] > 0)
                if old_pass:
                    results['passed_old'] += 1
                
                # 新gate
                new_pass, reason = gate.check(s['accuracy'], s['trades'], s['avg_return'])
                if new_pass:
                    results['passed_new'] += 1
                    if not old_pass:
                        results['rescued'].append({**s, 'reason': reason})
                else:
                    results['still_failed'].append({**s, 'reason': reason})
        
        return results
    
    def generate_report(self) -> str:
        """生成完整的数据分析报告"""
        baseline = self.analyze_baseline()
        gate = self.compute_optimal_tiers()
        
        lines = [
            "=" * 60,
            "Gate参数数据分析报告",
            "=" * 60,
            "",
            "## 1. 基线准确率分析",
            f"  数据样本: {baseline['total_samples']:,}个交易机会",
            f"  收益>3%的概率: {baseline['baseline_pct']:.2f}%",
            f"  收益率均值: {baseline['return_mean']:.4f}%",
            f"  收益率标准差: {baseline['return_std']:.3f}%",
            "",
            "## 2. 分层Gate配置",
            gate.format_tiers(),
            "",
            "## 3. 随机策略模拟 (10000次)",
        ]
        
        for n in [10, 20, 30, 50, 100]:
            sim = self.simulate_random_strategy(n)
            lines.append(f"  {n}笔交易: 准确率={sim['mean_accuracy']:.1f}%±{sim['std_accuracy']:.1f}%, "
                        f"P(≥20%)={sim['p_above_20']*100:.2f}%")
        
        return "\n".join(lines)


# 便捷函数
def create_gate(baseline_rate: float = None) -> TieredGate:
    """创建分层Gate实例"""
    return TieredGate(baseline_rate)


def check_gate(accuracy: float, trades: int, avg_return: float = None) -> Tuple[bool, str]:
    """快速检查策略是否通过Gate"""
    gate = TieredGate()
    return gate.check(accuracy, trades, avg_return)


if __name__ == "__main__":
    # 演示用法
    print("=== 分层Gate演示 ===\n")
    
    gate = TieredGate()
    print(gate.format_tiers())
    
    print("\n\n=== 测试案例 ===")
    test_cases = [
        (23.7, 38, 0.77, "C3-R008"),
        (23.3, 30, 0.40, "T2-R011"),
        (20.0, 20, 0.14, "T3-R009"),
        (20.0, 10, 0.92, "T3-R020"),
        (15.0, 40, 0.10, "边缘案例"),
        (50.0, 5, 1.00, "高准确率少交易"),
    ]
    
    for acc, trades, ret, name in test_cases:
        passed, reason = gate.check(acc, trades, ret)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {acc}%准确率, {trades}笔 → {status}")
        print(f"    {reason}")
