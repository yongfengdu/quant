#!/usr/bin/env python3
"""
四象限反馈报告生成器

生成结构化的四象限反馈报告供Analyst使用，包括：
1. TP/FP/FN/TN统计
2. Near-Miss FN分析（差一点选中的机会）
3. 方向性建议（往哪改）
4. 反事实验证（如果放宽条件会怎样）

这是"可验证改进"功能，需要通过A/B测试验证效果。
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np

from infra.quadrant_stats import QuadrantStats, QuadrantResult
from infra.market_classifier import MarketClassifier, MarketState


@dataclass
class FeedbackReport:
    """四象限反馈报告"""
    
    # 四象限统计
    quadrant: QuadrantResult
    
    # Near-Miss分析
    near_miss_count: int = 0
    near_miss_samples: List[Dict] = None
    near_miss_sectors: Dict[str, int] = None  # 板块分布
    
    # 方向性建议
    suggestions: List[str] = None
    
    # 反事实分析
    counterfactual: Dict[str, Any] = None
    
    # 市场状态分析
    market_state_breakdown: Dict[str, Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'quadrant': self.quadrant.to_dict(),
            'near_miss_count': self.near_miss_count,
            'near_miss_sectors': self.near_miss_sectors,
            'suggestions': self.suggestions,
            'counterfactual': self.counterfactual,
            'market_state_breakdown': self.market_state_breakdown,
        }
    
    def format_for_analyst(self, verbose: bool = False) -> str:
        """
        格式化为Analyst可读的文本
        
        Args:
            verbose: 是否包含详细信息
        """
        lines = []
        
        # 标题
        lines.append("## 四象限反馈分析")
        lines.append("")
        
        # 核心指标
        q = self.quadrant
        lines.append("### 核心指标")
        lines.append(f"- **准确率 (Precision)**: {q.precision:.1%} — 预测的股票中{q.precision:.1%}涨幅≥3%")
        lines.append(f"- **召回率 (Recall)**: {q.recall:.1%} — 所有涨幅≥3%的机会中只捕获了{q.recall:.1%}")
        lines.append(f"- **漏掉机会**: {q.fn_count}个 ({q.missed_opportunity_ratio:.1%})")
        lines.append("")
        
        # 四象限表格
        lines.append("### 四象限分布")
        lines.append("```")
        lines.append(f"                    实际涨幅≥3%    实际涨幅<3%")
        lines.append(f"  预测买入(发信号)    TP={q.tp_count:4d}        FP={q.fp_count:4d}")
        lines.append(f"  未预测(没发信号)    FN={q.fn_count:4d}        TN={q.tn_count:4d}")
        lines.append("```")
        lines.append("")
        
        # 关键洞察
        lines.append("### 关键洞察")
        
        if q.recall < 0.05:
            lines.append(f"- 🔴 **召回率极低({q.recall:.1%})**：策略条件太严，漏掉了{q.missed_opportunity_ratio:.1%}的机会")
            lines.append(f"  → 建议：放宽筛选条件，先保证信号量，再优化准确率")
        elif q.recall < 0.15:
            lines.append(f"- 🟡 **召回率偏低({q.recall:.1%})**：可能错过较多机会")
        else:
            lines.append(f"- 🟢 **召回率合理({q.recall:.1%})**")
        
        if q.precision < 0.10:
            lines.append(f"- 🔴 **准确率过低({q.precision:.1%})**：信号质量差，大部分预测是错的")
        elif q.precision < 0.15:
            lines.append(f"- 🟡 **准确率偏低({q.precision:.1%})**：距离15%目标还有差距")
        else:
            lines.append(f"- 🟢 **准确率达标({q.precision:.1%})**")
        
        lines.append("")
        
        # Near-Miss分析
        if self.near_miss_count > 0:
            lines.append("### Near-Miss分析（差一点选中的FN）")
            lines.append(f"共有 **{self.near_miss_count}** 个Near-Miss FN（信号分数接近阈值但未达到）")
            lines.append("")
            
            if self.near_miss_sectors:
                lines.append("Near-Miss板块分布（前5）:")
                sorted_sectors = sorted(self.near_miss_sectors.items(), key=lambda x: -x[1])[:5]
                for sector, count in sorted_sectors:
                    lines.append(f"  - {sector}: {count}个")
                lines.append("")
            
            if verbose and self.near_miss_samples:
                lines.append("Near-Miss样本（前5）:")
                for sample in self.near_miss_samples[:5]:
                    lines.append(
                        f"  - {sample.get('code')} ({sample.get('signal_date', ''):%Y-%m-%d}): "
                        f"实际涨幅{sample.get('actual_return', 0):.1%}, "
                        f"信号分数{sample.get('signal_score', 0):.3f}"
                    )
                lines.append("")
        
        # 方向性建议
        if self.suggestions:
            lines.append("### 改进建议")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"{i}. {suggestion}")
            lines.append("")
        
        # 反事实分析
        if self.counterfactual:
            lines.append("### 反事实分析（如果放宽条件）")
            cf = self.counterfactual
            if 'relaxed_threshold' in cf:
                lines.append(f"如果将信号阈值从{cf.get('original_threshold', 0.15)}放宽到{cf['relaxed_threshold']}:")
                lines.append(f"  - 预测数量: {cf.get('original_predictions', 0)} → {cf.get('relaxed_predictions', 0)}")
                lines.append(f"  - 预计准确率: {cf.get('estimated_precision', 0):.1%}")
                lines.append(f"  - 预计召回率: {cf.get('estimated_recall', 0):.1%}")
            lines.append("")
        
        # 市场状态分析
        if self.market_state_breakdown:
            lines.append("### 分市场状态表现")
            for state, metrics in self.market_state_breakdown.items():
                if metrics.get('total_opportunities', 0) > 0:
                    lines.append(
                        f"- **{state}**: "
                        f"TP={metrics.get('tp', 0)}, FN={metrics.get('fn', 0)}, "
                        f"召回率={metrics.get('recall', 0):.1%}"
                    )
            lines.append("")
        
        return "\n".join(lines)


class QuadrantFeedbackGenerator:
    """
    四象限反馈生成器
    
    用法:
        generator = QuadrantFeedbackGenerator()
        report = generator.generate(daily_df, trades_df)
        print(report.format_for_analyst())
    """
    
    def __init__(
        self,
        threshold: float = 0.03,
        min_signal_score: float = 0.15,
        near_miss_ratio: float = 0.6,
    ):
        self.threshold = threshold
        self.min_signal_score = min_signal_score
        self.near_miss_ratio = near_miss_ratio
        
        self.quadrant_stats = QuadrantStats(
            threshold=threshold,
            min_signal_score=min_signal_score,
            near_miss_ratio=near_miss_ratio,
        )
        self.market_classifier = MarketClassifier()
    
    def generate(
        self,
        daily: pd.DataFrame,
        trades: pd.DataFrame,
        include_counterfactual: bool = True,
        include_market_breakdown: bool = True,
    ) -> FeedbackReport:
        """
        生成完整的四象限反馈报告
        
        Args:
            daily: 日线数据
            trades: 交易记录
            include_counterfactual: 是否包含反事实分析
            include_market_breakdown: 是否包含市场状态分析
        """
        # 计算四象限
        quadrant = self.quadrant_stats.compute_from_backtest(daily, trades)
        
        # Near-Miss分析
        near_miss = self.quadrant_stats.get_near_miss_fn(quadrant)
        near_miss_sectors = {}
        if near_miss:
            fn_features = self.quadrant_stats.get_fn_features(quadrant, daily)
            if len(fn_features) > 0 and 'sector' in fn_features.columns:
                # 只统计near-miss的板块
                near_miss_codes = set(nm.get('code') for nm in near_miss)
                near_miss_features = fn_features[fn_features['code'].isin(near_miss_codes)]
                if len(near_miss_features) > 0:
                    near_miss_sectors = near_miss_features['sector'].value_counts().to_dict()
        
        # 生成建议
        suggestions = self._generate_suggestions(quadrant, near_miss, near_miss_sectors)
        
        # 反事实分析
        counterfactual = None
        if include_counterfactual:
            counterfactual = self._counterfactual_analysis(daily, trades, quadrant)
        
        # 市场状态分析
        market_breakdown = None
        if include_market_breakdown:
            market_breakdown = self._market_state_breakdown(daily, trades, quadrant)
        
        return FeedbackReport(
            quadrant=quadrant,
            near_miss_count=len(near_miss),
            near_miss_samples=near_miss[:10],
            near_miss_sectors=near_miss_sectors,
            suggestions=suggestions,
            counterfactual=counterfactual,
            market_state_breakdown=market_breakdown,
        )
    
    def _generate_suggestions(
        self,
        quadrant: QuadrantResult,
        near_miss: List[Dict],
        near_miss_sectors: Dict[str, int],
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 基于召回率
        if quadrant.recall < 0.01:
            suggestions.append(
                "**紧急**：召回率几乎为0，条件过于严格。建议大幅放宽筛选条件，"
                "或完全重新设计信号逻辑。"
            )
        elif quadrant.recall < 0.05:
            suggestions.append(
                f"召回率仅{quadrant.recall:.1%}，漏掉了{quadrant.fn_count}个机会。"
                f"考虑放宽最严格的1-2个条件。"
            )
        
        # 基于Near-Miss
        if len(near_miss) > 10:
            suggestions.append(
                f"有{len(near_miss)}个Near-Miss（差一点选中），"
                f"说明当前阈值设置可能过严。尝试将信号阈值降低10-20%。"
            )
        
        # 基于Near-Miss板块分布
        if near_miss_sectors:
            top_sector = max(near_miss_sectors.items(), key=lambda x: x[1])
            if top_sector[1] > len(near_miss) * 0.3:
                suggestions.append(
                    f"Near-Miss集中在{top_sector[0]}板块（{top_sector[1]}个），"
                    f"该板块可能被错误排除或条件过严。"
                )
        
        # 基于准确率
        if quadrant.precision < 0.10 and quadrant.total_predicted > 50:
            suggestions.append(
                f"准确率仅{quadrant.precision:.1%}，信号质量差。"
                f"建议增加过滤条件提高信号质量，而非放宽条件。"
            )
        
        # 基于TP/FP比
        if quadrant.tp_count > 0 and quadrant.fp_count / max(quadrant.tp_count, 1) > 10:
            suggestions.append(
                f"FP/TP比例过高（{quadrant.fp_count}:{quadrant.tp_count}），"
                f"大部分信号是错的。需要更好的过滤机制。"
            )
        
        # 默认建议
        if not suggestions:
            if quadrant.precision >= 0.15 and quadrant.recall < 0.10:
                suggestions.append(
                    "准确率达标但召回率低。可以尝试适度放宽条件，"
                    "在保持准确率的前提下提高召回。"
                )
            else:
                suggestions.append(
                    "当前策略表现一般，建议逐步调整参数观察变化。"
                )
        
        return suggestions
    
    def _counterfactual_analysis(
        self,
        daily: pd.DataFrame,
        trades: pd.DataFrame,
        quadrant: QuadrantResult,
    ) -> Dict[str, Any]:
        """
        反事实分析：如果放宽条件会怎样
        
        注意：这是估算，不是精确计算
        """
        result = {
            'original_threshold': self.min_signal_score,
            'original_predictions': quadrant.total_predicted,
        }
        
        # 计算放宽后的效果
        relaxed_threshold = self.min_signal_score * 0.7  # 放宽30%
        
        # 估算：Near-Miss FN会变成TP/FP
        near_miss = self.quadrant_stats.get_near_miss_fn(quadrant)
        near_miss_returns = [nm.get('actual_return', 0) for nm in near_miss]
        
        if near_miss_returns:
            new_tp = sum(1 for r in near_miss_returns if r >= self.threshold)
            new_fp = len(near_miss_returns) - new_tp
            
            estimated_tp = quadrant.tp_count + new_tp
            estimated_fp = quadrant.fp_count + new_fp
            estimated_fn = quadrant.fn_count - new_tp
            
            total_predicted = estimated_tp + estimated_fp
            total_opportunities = estimated_tp + estimated_fn
            
            result['relaxed_threshold'] = relaxed_threshold
            result['relaxed_predictions'] = total_predicted
            result['estimated_precision'] = estimated_tp / total_predicted if total_predicted > 0 else 0
            result['estimated_recall'] = estimated_tp / total_opportunities if total_opportunities > 0 else 0
            result['new_tp_from_near_miss'] = new_tp
            result['new_fp_from_near_miss'] = new_fp
        
        return result
    
    def _market_state_breakdown(
        self,
        daily: pd.DataFrame,
        trades: pd.DataFrame,
        quadrant: QuadrantResult,
    ) -> Dict[str, Dict]:
        """分市场状态统计"""
        # 获取市场状态分类
        market_states = self.market_classifier.classify(daily)
        market_states['date'] = pd.to_datetime(market_states['date'])
        
        # 将FN记录按市场状态分组
        breakdown = {}
        
        for state in ['bull', 'bear', 'range']:
            state_dates = set(
                market_states[market_states['market_state'] == state]['date']
            )
            
            # 统计该状态下的TP/FN
            tp_in_state = sum(
                1 for r in quadrant.tp_records
                if pd.to_datetime(r.get('signal_date')) in state_dates
            )
            fn_in_state = sum(
                1 for r in quadrant.fn_records
                if pd.to_datetime(r.get('signal_date')) in state_dates
            )
            
            total_opportunities = tp_in_state + fn_in_state
            recall = tp_in_state / total_opportunities if total_opportunities > 0 else 0
            
            breakdown[state] = {
                'tp': tp_in_state,
                'fn': fn_in_state,
                'total_opportunities': total_opportunities,
                'recall': recall,
            }
        
        return breakdown


def generate_feedback_from_files(
    daily_path: str = "~/.cache/quant-autoresearch/daily.parquet",
    trades_path: str = "~/.cache/quant-autoresearch/trades.parquet",
) -> FeedbackReport:
    """从文件生成反馈报告的便捷函数"""
    daily_path = Path(daily_path).expanduser()
    trades_path = Path(trades_path).expanduser()
    
    daily = pd.read_parquet(daily_path)
    trades = pd.read_parquet(trades_path)
    
    generator = QuadrantFeedbackGenerator()
    return generator.generate(daily, trades)


if __name__ == "__main__":
    print("Testing QuadrantFeedbackGenerator...")
    
    try:
        report = generate_feedback_from_files()
        print(report.format_for_analyst(verbose=True))
        print("\n" + "=" * 60)
        print("Report dict:")
        print(json.dumps(report.to_dict(), indent=2, default=str))
    except FileNotFoundError as e:
        print(f"Test skipped: {e}")
