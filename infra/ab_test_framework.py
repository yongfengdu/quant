#!/usr/bin/env python3
"""
A/B测试框架

用于对比测试改进功能的效果。

设计原则：
1. Control组使用原有pipeline（不含新功能）
2. Treatment组使用改进后的pipeline（含新功能）
3. 每组运行多次实验（建议各3次×20轮）
4. 使用统计检验判断差异是否显著

核心指标：
- 效率：首次通过轮数、无效迭代占比
- 多样性：策略类型数、探索比例
- 质量：最佳准确率、平均收益
"""

import json
import time
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str                      # 实验名称
    description: str               # 实验描述
    feature_flag: str              # 功能标志名
    rounds_per_run: int = 20       # 每次运行的轮数
    runs_per_group: int = 3        # 每组运行次数
    
    # 成功标准
    min_improvement_efficiency: float = 0.25    # 效率提升至少25%
    min_improvement_diversity: float = 0.50     # 多样性提升至少50%
    significance_level: float = 0.05            # 显著性水平
    min_effect_size: float = 0.3                # 最小效应量(Cohen's d)


@dataclass
class RunResult:
    """单次运行结果"""
    group: str                     # "control" or "treatment"
    run_id: int                    # 运行编号
    start_time: datetime           # 开始时间
    end_time: datetime             # 结束时间
    
    # 效率指标
    rounds_to_first_pass: int = 0  # 首次通过轮数
    wasted_rounds: int = 0         # 无效迭代数
    total_rounds: int = 0          # 总轮数
    
    # 多样性指标
    strategy_types: List[str] = field(default_factory=list)
    unique_types: int = 0
    
    # 质量指标
    best_accuracy: float = 0.0
    best_avg_return: float = 0.0
    gate_passed_count: int = 0
    
    # 原始数据
    round_metrics: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'group': self.group,
            'run_id': self.run_id,
            'start_time': str(self.start_time),
            'end_time': str(self.end_time),
            'rounds_to_first_pass': self.rounds_to_first_pass,
            'wasted_rounds': self.wasted_rounds,
            'total_rounds': self.total_rounds,
            'unique_types': self.unique_types,
            'strategy_types': self.strategy_types,
            'best_accuracy': self.best_accuracy,
            'best_avg_return': self.best_avg_return,
            'gate_passed_count': self.gate_passed_count,
        }


@dataclass
class ABTestResult:
    """A/B测试结果"""
    config: ExperimentConfig
    control_runs: List[RunResult]
    treatment_runs: List[RunResult]
    
    # 统计检验结果
    efficiency_pvalue: float = 1.0
    efficiency_effect_size: float = 0.0
    diversity_pvalue: float = 1.0
    diversity_effect_size: float = 0.0
    quality_pvalue: float = 1.0
    quality_effect_size: float = 0.0
    
    # 决策
    decision: str = "NEUTRAL"  # "ACCEPT", "REJECT", "NEUTRAL"
    reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'feature_flag': self.config.feature_flag,
            },
            'control_runs': [r.to_dict() for r in self.control_runs],
            'treatment_runs': [r.to_dict() for r in self.treatment_runs],
            'statistics': {
                'efficiency_pvalue': self.efficiency_pvalue,
                'efficiency_effect_size': self.efficiency_effect_size,
                'diversity_pvalue': self.diversity_pvalue,
                'diversity_effect_size': self.diversity_effect_size,
                'quality_pvalue': self.quality_pvalue,
                'quality_effect_size': self.quality_effect_size,
            },
            'decision': self.decision,
            'reason': self.reason,
        }


class ABTestFramework:
    """
    A/B测试框架
    
    用法:
        framework = ABTestFramework()
        config = ExperimentConfig(
            name="quadrant_feedback",
            description="测试四象限反馈对迭代效率的影响",
            feature_flag="ENABLE_QUADRANT_FEEDBACK",
        )
        result = framework.run_experiment(config)
        print(result.decision, result.reason)
    """
    
    def __init__(
        self,
        project_dir: str = "/root/quant-autoresearch",
        output_dir: str = "/root/quant-autoresearch/experiments/ab_tests",
    ):
        self.project_dir = Path(project_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        dry_run: bool = False,
    ) -> ABTestResult:
        """
        运行完整的A/B测试实验
        
        Args:
            config: 实验配置
            dry_run: 是否只是模拟运行（不实际执行pipeline）
        """
        print(f"=" * 60)
        print(f"A/B Test: {config.name}")
        print(f"Description: {config.description}")
        print(f"Feature: {config.feature_flag}")
        print(f"Rounds per run: {config.rounds_per_run}")
        print(f"Runs per group: {config.runs_per_group}")
        print(f"=" * 60)
        
        # 运行Control组
        print(f"\n[Control Group] Running {config.runs_per_group} experiments...")
        control_runs = []
        for i in range(config.runs_per_group):
            if dry_run:
                result = self._mock_run("control", i, config)
            else:
                result = self._run_pipeline("control", i, config, enable_feature=False)
            control_runs.append(result)
            print(f"  Control run {i+1}/{config.runs_per_group} completed")
        
        # 运行Treatment组
        print(f"\n[Treatment Group] Running {config.runs_per_group} experiments...")
        treatment_runs = []
        for i in range(config.runs_per_group):
            if dry_run:
                result = self._mock_run("treatment", i, config)
            else:
                result = self._run_pipeline("treatment", i, config, enable_feature=True)
            treatment_runs.append(result)
            print(f"  Treatment run {i+1}/{config.runs_per_group} completed")
        
        # 统计分析
        ab_result = ABTestResult(
            config=config,
            control_runs=control_runs,
            treatment_runs=treatment_runs,
        )
        
        self._analyze_results(ab_result)
        self._make_decision(ab_result)
        
        # 保存结果
        self._save_results(ab_result)
        
        return ab_result
    
    def _run_pipeline(
        self,
        group: str,
        run_id: int,
        config: ExperimentConfig,
        enable_feature: bool,
    ) -> RunResult:
        """运行pipeline并收集结果"""
        start_time = datetime.now()
        
        # 设置环境变量控制功能开关
        env = {config.feature_flag: "1" if enable_feature else "0"}
        
        # 运行pipeline
        # 注意：实际实现需要根据pipeline的接口调整
        cmd = [
            str(self.project_dir / ".venv" / "bin" / "python"),
            str(self.project_dir / "pipeline.py"),
            "--rounds", str(config.rounds_per_run),
        ]
        
        try:
            subprocess.run(
                cmd,
                cwd=str(self.project_dir),
                env={**dict(os.environ), **env},
                timeout=3600 * config.rounds_per_run,  # 每轮1小时超时
            )
        except subprocess.TimeoutExpired:
            print(f"Warning: Run {run_id} timed out")
        except Exception as e:
            print(f"Warning: Run {run_id} failed: {e}")
        
        end_time = datetime.now()
        
        # 收集结果
        return self._collect_run_results(group, run_id, start_time, end_time, config)
    
    def _mock_run(
        self,
        group: str,
        run_id: int,
        config: ExperimentConfig,
    ) -> RunResult:
        """模拟运行（用于测试框架）"""
        # 模拟Treatment组有改进
        improvement = 0.2 if group == "treatment" else 0
        
        return RunResult(
            group=group,
            run_id=run_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
            rounds_to_first_pass=int(15 * (1 - improvement) + np.random.randint(-2, 3)),
            wasted_rounds=int(8 * (1 - improvement) + np.random.randint(-1, 2)),
            total_rounds=config.rounds_per_run,
            strategy_types=["oversold"] * int(12 * (1 - improvement * 0.5)),
            unique_types=int(3 + improvement * 2 + np.random.randint(0, 2)),
            best_accuracy=20 + improvement * 5 + np.random.random() * 3,
            best_avg_return=0.5 + improvement * 0.3 + np.random.random() * 0.2,
            gate_passed_count=int(1 + improvement + np.random.randint(0, 2)),
        )
    
    def _collect_run_results(
        self,
        group: str,
        run_id: int,
        start_time: datetime,
        end_time: datetime,
        config: ExperimentConfig,
    ) -> RunResult:
        """从rounds目录收集运行结果"""
        rounds_dir = self.project_dir / "rounds"
        
        result = RunResult(
            group=group,
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            total_rounds=config.rounds_per_run,
        )
        
        # 收集最近N轮的数据
        round_dirs = sorted(rounds_dir.iterdir(), key=lambda x: x.name)[-config.rounds_per_run:]
        
        prev_acc = 0
        for round_dir in round_dirs:
            metrics_file = round_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                result.round_metrics.append(metrics)
                
                acc = metrics.get('accuracy_3pct', 0)
                trades = metrics.get('total_trades', 0)
                avg_ret = metrics.get('avg_signal_return', 0)
                max_dd = metrics.get('max_drawdown', 0)
                
                # 检查是否通过gate
                gate_passed = (
                    acc >= 15 and
                    avg_ret > 0 and
                    trades >= 50 and
                    max_dd > -0.45
                )
                
                if gate_passed:
                    result.gate_passed_count += 1
                    if result.rounds_to_first_pass == 0:
                        result.rounds_to_first_pass = len(result.round_metrics)
                
                # 更新最佳指标
                if acc > result.best_accuracy:
                    result.best_accuracy = acc
                if avg_ret > result.best_avg_return:
                    result.best_avg_return = avg_ret
                
                # 检查无效迭代
                if acc <= prev_acc:
                    result.wasted_rounds += 1
                prev_acc = acc
                
            except Exception as e:
                print(f"Warning: Failed to parse {metrics_file}: {e}")
        
        # 如果没有通过，设为总轮数
        if result.rounds_to_first_pass == 0:
            result.rounds_to_first_pass = result.total_rounds
        
        return result
    
    def _analyze_results(self, result: ABTestResult):
        """统计分析"""
        # 提取指标
        control_efficiency = [r.rounds_to_first_pass for r in result.control_runs]
        treatment_efficiency = [r.rounds_to_first_pass for r in result.treatment_runs]
        
        control_diversity = [r.unique_types for r in result.control_runs]
        treatment_diversity = [r.unique_types for r in result.treatment_runs]
        
        control_quality = [r.best_accuracy for r in result.control_runs]
        treatment_quality = [r.best_accuracy for r in result.treatment_runs]
        
        # t检验
        if len(control_efficiency) >= 2 and len(treatment_efficiency) >= 2:
            # 效率（越小越好，所以control - treatment）
            t_stat, p_value = stats.ttest_ind(control_efficiency, treatment_efficiency)
            result.efficiency_pvalue = p_value
            result.efficiency_effect_size = self._cohens_d(
                control_efficiency, treatment_efficiency
            )
            
            # 多样性（越大越好）
            t_stat, p_value = stats.ttest_ind(treatment_diversity, control_diversity)
            result.diversity_pvalue = p_value
            result.diversity_effect_size = self._cohens_d(
                treatment_diversity, control_diversity
            )
            
            # 质量（越大越好）
            t_stat, p_value = stats.ttest_ind(treatment_quality, control_quality)
            result.quality_pvalue = p_value
            result.quality_effect_size = self._cohens_d(
                treatment_quality, control_quality
            )
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _make_decision(self, result: ABTestResult):
        """根据统计结果做出决策"""
        config = result.config
        
        # 检查是否有显著退化
        # 效率退化：treatment更慢（rounds_to_first_pass更大）
        efficiency_regression = (
            result.efficiency_pvalue < config.significance_level and
            result.efficiency_effect_size < -config.min_effect_size
        )
        
        # 质量退化：treatment更差（accuracy更低）
        quality_regression = (
            result.quality_pvalue < config.significance_level and
            result.quality_effect_size < -config.min_effect_size
        )
        
        if efficiency_regression or quality_regression:
            result.decision = "REJECT"
            reasons = []
            if efficiency_regression:
                reasons.append(f"效率退化(d={result.efficiency_effect_size:.2f})")
            if quality_regression:
                reasons.append(f"质量退化(d={result.quality_effect_size:.2f})")
            result.reason = "检测到显著退化: " + ", ".join(reasons)
            return
        
        # 检查是否有显著改进
        efficiency_improvement = (
            result.efficiency_pvalue < config.significance_level and
            result.efficiency_effect_size > config.min_effect_size
        )
        
        diversity_improvement = (
            result.diversity_pvalue < config.significance_level and
            result.diversity_effect_size > config.min_effect_size
        )
        
        quality_improvement = (
            result.quality_pvalue < config.significance_level and
            result.quality_effect_size > config.min_effect_size
        )
        
        if efficiency_improvement or diversity_improvement or quality_improvement:
            result.decision = "ACCEPT"
            improvements = []
            if efficiency_improvement:
                improvements.append(f"效率提升(d={result.efficiency_effect_size:.2f})")
            if diversity_improvement:
                improvements.append(f"多样性提升(d={result.diversity_effect_size:.2f})")
            if quality_improvement:
                improvements.append(f"质量提升(d={result.quality_effect_size:.2f})")
            result.reason = "检测到显著改进: " + ", ".join(improvements)
            return
        
        # 无显著变化
        result.decision = "NEUTRAL"
        result.reason = "无统计显著差异"
    
    def _save_results(self, result: ABTestResult):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config.name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def compare_with_baseline(
        self,
        baseline_path: str = "experiments/baseline_from_history.json",
        current_runs: List[RunResult] = None,
    ) -> Dict[str, Any]:
        """与历史基线对比"""
        baseline_path = self.project_dir / baseline_path
        
        if not baseline_path.exists():
            return {"error": "Baseline not found"}
        
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        baseline_metrics = baseline.get('metrics', {})
        
        comparison = {}
        
        if current_runs:
            # 计算当前运行的指标
            current_efficiency = np.mean([r.rounds_to_first_pass for r in current_runs])
            current_wasted = np.mean([r.wasted_rounds / r.total_rounds for r in current_runs])
            current_diversity = np.mean([r.unique_types for r in current_runs])
            current_quality = np.mean([r.best_accuracy for r in current_runs])
            
            # 与基线对比
            baseline_efficiency = baseline_metrics.get('rounds_to_first_pass', {}).get('mean', 20)
            baseline_wasted = baseline_metrics.get('wasted_rounds_ratio', {}).get('mean', 0.5)
            baseline_diversity = baseline_metrics.get('unique_strategy_types', {}).get('mean', 4)
            baseline_quality = baseline_metrics.get('best_accuracy', {}).get('mean', 20)
            
            comparison = {
                'efficiency': {
                    'baseline': baseline_efficiency,
                    'current': current_efficiency,
                    'change': (baseline_efficiency - current_efficiency) / baseline_efficiency,
                },
                'wasted_ratio': {
                    'baseline': baseline_wasted,
                    'current': current_wasted,
                    'change': (baseline_wasted - current_wasted) / baseline_wasted,
                },
                'diversity': {
                    'baseline': baseline_diversity,
                    'current': current_diversity,
                    'change': (current_diversity - baseline_diversity) / baseline_diversity,
                },
                'quality': {
                    'baseline': baseline_quality,
                    'current': current_quality,
                    'change': (current_quality - baseline_quality) / baseline_quality,
                },
            }
        
        return comparison
    
    def summary(self, result: ABTestResult) -> str:
        """生成测试结果摘要"""
        lines = [
            f"=" * 60,
            f"A/B Test Summary: {result.config.name}",
            f"=" * 60,
            "",
            f"Decision: **{result.decision}**",
            f"Reason: {result.reason}",
            "",
            "Control Group:",
        ]
        
        for run in result.control_runs:
            lines.append(
                f"  Run {run.run_id}: rounds_to_pass={run.rounds_to_first_pass}, "
                f"types={run.unique_types}, accuracy={run.best_accuracy:.1f}%"
            )
        
        lines.append("")
        lines.append("Treatment Group:")
        
        for run in result.treatment_runs:
            lines.append(
                f"  Run {run.run_id}: rounds_to_pass={run.rounds_to_first_pass}, "
                f"types={run.unique_types}, accuracy={run.best_accuracy:.1f}%"
            )
        
        lines.extend([
            "",
            "Statistical Tests:",
            f"  Efficiency: p={result.efficiency_pvalue:.4f}, d={result.efficiency_effect_size:.2f}",
            f"  Diversity: p={result.diversity_pvalue:.4f}, d={result.diversity_effect_size:.2f}",
            f"  Quality: p={result.quality_pvalue:.4f}, d={result.quality_effect_size:.2f}",
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    import os
    
    print("Testing ABTestFramework with dry run...")
    
    framework = ABTestFramework()
    
    config = ExperimentConfig(
        name="quadrant_feedback_test",
        description="测试四象限反馈对迭代效率的影响",
        feature_flag="ENABLE_QUADRANT_FEEDBACK",
        rounds_per_run=20,
        runs_per_group=3,
    )
    
    result = framework.run_experiment(config, dry_run=True)
    
    print()
    print(framework.summary(result))
