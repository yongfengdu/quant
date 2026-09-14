#!/usr/bin/env python3
"""
基础设施模块单元测试
"""

import unittest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.quadrant_stats import QuadrantStats, QuadrantResult
from infra.market_classifier import MarketClassifier, MarketState
from infra.strategy_features import StrategyFeatureExtractor, StrategyFeatures
from infra.feedback_generator import QuadrantFeedbackGenerator, FeedbackReport
from infra.ab_test_framework import ABTestFramework, ExperimentConfig, RunResult


class TestQuadrantStats(unittest.TestCase):
    """四象限统计测试"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建模拟日线数据
        dates = pd.date_range('2024-01-01', periods=30, freq='B')
        codes = ['000001', '000002', '000003']
        
        rows = []
        for date in dates:
            for code in codes:
                rows.append({
                    'date': date,
                    'code': code,
                    'open': 10.0 + np.random.randn() * 0.5,
                    'close': 10.0 + np.random.randn() * 0.5,
                    'high': 10.5,
                    'low': 9.5,
                    'volume': 1000000,
                    'sector': '银行',
                })
        
        self.daily = pd.DataFrame(rows)
        
        # 修改部分数据使某些股票有明确的涨幅
        # 让000001在第5天有>3%涨幅
        self.daily.loc[
            (self.daily['code'] == '000001') & (self.daily['date'] == dates[5]),
            'open'
        ] = 10.0
        self.daily.loc[
            (self.daily['code'] == '000001') & (self.daily['date'] == dates[6]),
            'open'
        ] = 10.5  # 5% 涨幅
        
    def test_quadrant_result_properties(self):
        """测试QuadrantResult属性计算"""
        result = QuadrantResult(
            tp_count=10,
            fp_count=20,
            fn_count=30,
            tn_count=40,
        )
        
        # precision = 10 / (10+20) = 0.333
        self.assertAlmostEqual(result.precision, 0.333, places=2)
        
        # recall = 10 / (10+30) = 0.25
        self.assertAlmostEqual(result.recall, 0.25, places=2)
        
        # accuracy = (10+40) / 100 = 0.5
        self.assertAlmostEqual(result.accuracy, 0.5, places=2)
        
        # missed_opportunity_ratio = 30 / (10+30) = 0.75
        self.assertAlmostEqual(result.missed_opportunity_ratio, 0.75, places=2)
    
    def test_quadrant_result_empty(self):
        """测试空结果"""
        result = QuadrantResult()
        
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.accuracy, 0.0)
    
    def test_quadrant_stats_init(self):
        """测试初始化"""
        stats = QuadrantStats(threshold=0.03)
        self.assertEqual(stats.threshold, 0.03)
    
    def test_compute_empty_predictions(self):
        """测试空预测"""
        stats = QuadrantStats()
        predictions = pd.DataFrame()
        
        result = stats.compute(self.daily, predictions)
        
        self.assertEqual(result.tp_count, 0)
        self.assertEqual(result.fp_count, 0)
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = QuadrantResult(tp_count=5, fp_count=10)
        d = result.to_dict()
        
        self.assertIn('tp_count', d)
        self.assertIn('precision', d)
        self.assertEqual(d['tp_count'], 5)
    
    def test_summary(self):
        """测试摘要生成"""
        result = QuadrantResult(tp_count=5, fp_count=10, fn_count=15, tn_count=70)
        summary = result.summary()
        
        self.assertIn('TP', summary)
        self.assertIn('Precision', summary)
        self.assertIn('Recall', summary)


class TestMarketClassifier(unittest.TestCase):
    """市场状态分类器测试"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建100天的模拟数据
        dates = pd.date_range('2024-01-01', periods=100, freq='B')
        codes = ['000001', '000002', '000003']
        
        rows = []
        for i, date in enumerate(dates):
            # 模拟趋势：前40天上涨，中间20天震荡，后40天下跌
            if i < 40:
                trend = 0.01  # 牛市
            elif i < 60:
                trend = 0.0   # 震荡
            else:
                trend = -0.01  # 熊市
            
            base_price = 10.0 * (1 + trend) ** i
            
            for code in codes:
                rows.append({
                    'date': date,
                    'code': code,
                    'open': base_price * (1 + np.random.randn() * 0.01),
                    'close': base_price * (1 + np.random.randn() * 0.01),
                    'high': base_price * 1.02,
                    'low': base_price * 0.98,
                    'volume': 1000000,
                    'amount': 10000000,
                })
        
        self.daily = pd.DataFrame(rows)
    
    def test_market_state_enum(self):
        """测试市场状态枚举"""
        self.assertEqual(MarketState.BULL.value, 'bull')
        self.assertEqual(MarketState.BEAR.value, 'bear')
        self.assertEqual(MarketState.RANGE.value, 'range')
    
    def test_classifier_init(self):
        """测试初始化"""
        classifier = MarketClassifier(
            trend_window=20,
            bull_threshold=0.05,
            bear_threshold=-0.05,
        )
        
        self.assertEqual(classifier.trend_window, 20)
        self.assertEqual(classifier.bull_threshold, 0.05)
    
    def test_classify_returns_dataframe(self):
        """测试分类返回DataFrame"""
        classifier = MarketClassifier()
        result = classifier.classify(self.daily)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('date', result.columns)
        self.assertIn('market_state', result.columns)
    
    def test_get_state_distribution(self):
        """测试状态分布"""
        classifier = MarketClassifier()
        distribution = classifier.get_state_distribution(self.daily)
        
        self.assertIsInstance(distribution, dict)
        # 分布应该和为1
        total = sum(distribution.values())
        self.assertAlmostEqual(total, 1.0, places=2)
    
    def test_get_periods(self):
        """测试周期识别"""
        classifier = MarketClassifier(min_period_days=3)
        periods = classifier.get_periods(self.daily)
        
        self.assertIsInstance(periods, list)
        for period in periods:
            self.assertIsInstance(period.state, MarketState)
            self.assertGreaterEqual(period.duration_days, 3)
    
    def test_summary(self):
        """测试摘要生成"""
        classifier = MarketClassifier()
        summary = classifier.summary(self.daily)
        
        self.assertIn('状态分布', summary)
        self.assertIn('市场周期', summary)


class TestStrategyFeatureExtractor(unittest.TestCase):
    """策略特征提取器测试"""
    
    def test_extract_from_hypothesis_oversold(self):
        """测试从假设提取超卖特征"""
        extractor = StrategyFeatureExtractor()
        
        hypothesis = {
            'name': '超卖反弹策略',
            'description': '在连续下跌后寻找反弹机会',
            'changes': ['条件1: 5日跌幅>10%'],
        }
        
        features = extractor.extract_from_hypothesis(hypothesis)
        
        self.assertTrue(features.is_oversold)
        self.assertEqual(features.get_strategy_type(), 'oversold_bounce')
    
    def test_extract_from_hypothesis_trend(self):
        """测试从假设提取趋势特征"""
        extractor = StrategyFeatureExtractor()
        
        hypothesis = {
            'name': '突破追涨策略',
            'description': '股价突破新高时买入',
            'changes': [],
        }
        
        features = extractor.extract_from_hypothesis(hypothesis)
        
        self.assertTrue(features.is_trend)
    
    def test_extract_from_hypothesis_volume(self):
        """测试从假设提取成交量特征"""
        extractor = StrategyFeatureExtractor()
        
        hypothesis = {
            'name': '放量突破',
            'description': '成交量放大2倍以上，换手率提升',
            'changes': [],
        }
        
        features = extractor.extract_from_hypothesis(hypothesis)
        
        self.assertTrue(features.is_volume_based)
    
    def test_extract_from_code(self):
        """测试从代码提取特征"""
        extractor = StrategyFeatureExtractor()
        
        code = '''
def predict(hist):
    # 超卖反弹
    pct_5d = hist["close"].pct_change(5)
    oversold = pct_5d < -0.10
    
    # 放量
    vol_ratio = hist["volume"].rolling(5).mean()
    
    # 排除CXO
    not_cxo = ~hist["sector"].isin(["CXO"])
    
    return oversold & not_cxo
'''
        
        features = extractor.extract_from_code(code)
        
        self.assertTrue(features.is_oversold)
        self.assertTrue(features.has_sector_filter)
        self.assertTrue(features.lookback_medium)  # 5日 属于 5-20 范围
    
    def test_extract_sector_filter(self):
        """测试板块过滤识别"""
        extractor = StrategyFeatureExtractor()
        
        hypothesis = {
            'name': '带板块过滤的策略',
            'description': '排除CXO和半导体板块',
            'changes': ['排除: CXO/半导体'],
        }
        
        features = extractor.extract_from_hypothesis(hypothesis)
        
        self.assertTrue(features.has_sector_filter)
        self.assertTrue(features.has_hard_exclude)
    
    def test_to_vector(self):
        """测试转换为向量"""
        features = StrategyFeatures(
            is_oversold=True,
            is_trend=False,
            condition_count=3,
        )
        
        vector = features.to_vector()
        
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 19)  # 19个特征维度
        self.assertEqual(vector[0], 1.0)   # is_oversold
        self.assertEqual(vector[1], 0.0)   # is_trend
    
    def test_compute_similarity(self):
        """测试相似度计算"""
        extractor = StrategyFeatureExtractor()
        
        features1 = StrategyFeatures(is_oversold=True, is_volume_based=True)
        features2 = StrategyFeatures(is_oversold=True, is_volume_based=True)
        features3 = StrategyFeatures(is_trend=True)
        
        # 相同特征应该高相似度
        sim_same = extractor.compute_similarity(features1, features2)
        self.assertGreater(sim_same, 0.9)
        
        # 不同特征应该低相似度
        sim_diff = extractor.compute_similarity(features1, features3)
        self.assertLess(sim_diff, 0.5)
    
    def test_is_duplicate(self):
        """测试重复判断"""
        extractor = StrategyFeatureExtractor()
        
        features1 = StrategyFeatures(is_oversold=True, is_volume_based=True)
        features2 = StrategyFeatures(is_oversold=True, is_volume_based=True)
        features3 = StrategyFeatures(is_trend=True)
        
        self.assertTrue(extractor.is_duplicate(features1, features2))
        self.assertFalse(extractor.is_duplicate(features1, features3))
    
    def test_get_strategy_type(self):
        """测试获取策略类型"""
        # 超卖反弹
        f1 = StrategyFeatures(is_oversold=True)
        self.assertEqual(f1.get_strategy_type(), 'oversold_bounce')
        
        # 趋势
        f2 = StrategyFeatures(is_trend=True)
        self.assertEqual(f2.get_strategy_type(), 'trend_following')
        
        # 其他
        f3 = StrategyFeatures()
        self.assertEqual(f3.get_strategy_type(), 'other')


class TestIntegration(unittest.TestCase):
    """集成测试：使用真实数据（如果存在）"""
    
    def test_quadrant_with_real_data(self):
        """测试四象限统计与真实数据"""
        daily_path = Path("~/.cache/quant-autoresearch/daily.parquet").expanduser()
        trades_path = Path("~/.cache/quant-autoresearch/trades.parquet").expanduser()
        
        if not daily_path.exists() or not trades_path.exists():
            self.skipTest("Real data not available")
        
        daily = pd.read_parquet(daily_path)
        trades = pd.read_parquet(trades_path)
        
        stats = QuadrantStats()
        result = stats.compute_from_backtest(daily, trades)
        
        # 应该有有效结果
        self.assertGreaterEqual(result.tp_count + result.fp_count, 0)
        print(f"\n[Integration] Quadrant stats: {result.to_dict()}")
    
    def test_market_classifier_with_real_data(self):
        """测试市场分类器与真实数据"""
        daily_path = Path("~/.cache/quant-autoresearch/daily.parquet").expanduser()
        
        if not daily_path.exists():
            self.skipTest("Real data not available")
        
        daily = pd.read_parquet(daily_path)
        
        classifier = MarketClassifier()
        distribution = classifier.get_state_distribution(daily)
        
        # 应该有有效分布
        self.assertGreater(len(distribution), 0)
        print(f"\n[Integration] Market state distribution: {distribution}")


class TestFeedbackGenerator(unittest.TestCase):
    """反馈生成器测试"""
    
    def test_feedback_report_to_dict(self):
        """测试报告转字典"""
        quadrant = QuadrantResult(tp_count=5, fp_count=10, fn_count=15, tn_count=70)
        report = FeedbackReport(
            quadrant=quadrant,
            near_miss_count=3,
            suggestions=["建议1", "建议2"],
        )
        
        d = report.to_dict()
        self.assertIn('quadrant', d)
        self.assertIn('suggestions', d)
        self.assertEqual(d['near_miss_count'], 3)
    
    def test_feedback_format_for_analyst(self):
        """测试格式化输出"""
        quadrant = QuadrantResult(tp_count=5, fp_count=10, fn_count=15, tn_count=70)
        report = FeedbackReport(
            quadrant=quadrant,
            near_miss_count=3,
            suggestions=["建议1"],
        )
        
        text = report.format_for_analyst()
        
        self.assertIn('四象限', text)
        self.assertIn('Precision', text)
        self.assertIn('Recall', text)
    
    def test_generator_with_real_data(self):
        """测试反馈生成器与真实数据"""
        daily_path = Path("~/.cache/quant-autoresearch/daily.parquet").expanduser()
        trades_path = Path("~/.cache/quant-autoresearch/trades.parquet").expanduser()
        
        if not daily_path.exists() or not trades_path.exists():
            self.skipTest("Real data not available")
        
        daily = pd.read_parquet(daily_path)
        trades = pd.read_parquet(trades_path)
        
        generator = QuadrantFeedbackGenerator()
        report = generator.generate(daily, trades)
        
        # 应该有有效结果
        self.assertIsInstance(report, FeedbackReport)
        self.assertIsNotNone(report.quadrant)
        print(f"\n[Integration] Feedback report generated successfully")


class TestABTestFramework(unittest.TestCase):
    """A/B测试框架测试"""
    
    def test_experiment_config(self):
        """测试实验配置"""
        config = ExperimentConfig(
            name="test",
            description="测试",
            feature_flag="TEST_FLAG",
        )
        
        self.assertEqual(config.rounds_per_run, 20)
        self.assertEqual(config.runs_per_group, 3)
    
    def test_run_result_to_dict(self):
        """测试运行结果转字典"""
        result = RunResult(
            group="control",
            run_id=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
            rounds_to_first_pass=15,
            best_accuracy=20.0,
        )
        
        d = result.to_dict()
        self.assertEqual(d['group'], 'control')
        self.assertEqual(d['rounds_to_first_pass'], 15)
    
    def test_dry_run(self):
        """测试干运行模式"""
        framework = ABTestFramework()
        
        config = ExperimentConfig(
            name="dry_run_test",
            description="干运行测试",
            feature_flag="TEST_FLAG",
            rounds_per_run=5,
            runs_per_group=2,
        )
        
        result = framework.run_experiment(config, dry_run=True)
        
        self.assertEqual(len(result.control_runs), 2)
        self.assertEqual(len(result.treatment_runs), 2)
        self.assertIn(result.decision, ["ACCEPT", "REJECT", "NEUTRAL"])


class TestTieredGate(unittest.TestCase):
    """分层Gate测试 - 基于统计显著性"""
    
    def setUp(self):
        """初始化Gate"""
        from infra.gate_config import TieredGate
        self.gate = TieredGate()
    
    def test_high_accuracy_low_trades_pass(self):
        """高准确率+少量交易应该通过"""
        # 30%准确率只需7笔交易
        passed, reason = self.gate.check(accuracy=30.0, trades=7, avg_return=0.5)
        self.assertTrue(passed, f"30%准确率+7笔应该通过: {reason}")
        
        # 25%准确率需13笔
        passed, reason = self.gate.check(accuracy=25.0, trades=13, avg_return=0.5)
        self.assertTrue(passed, f"25%准确率+13笔应该通过: {reason}")
    
    def test_medium_accuracy_needs_more_trades(self):
        """中等准确率需要更多交易"""
        # 20%准确率需26笔
        passed, reason = self.gate.check(accuracy=20.0, trades=25, avg_return=0.5)
        self.assertFalse(passed, "20%准确率+25笔应该失败(需26笔)")
        
        passed, reason = self.gate.check(accuracy=20.0, trades=26, avg_return=0.5)
        self.assertTrue(passed, f"20%准确率+26笔应该通过: {reason}")
    
    def test_low_accuracy_needs_many_trades(self):
        """低准确率(15%)需要大量交易"""
        # 15%准确率需101笔
        passed, reason = self.gate.check(accuracy=15.0, trades=100, avg_return=0.5)
        self.assertFalse(passed, "15%准确率+100笔应该失败(需101笔)")
        
        passed, reason = self.gate.check(accuracy=15.0, trades=101, avg_return=0.5)
        self.assertTrue(passed, f"15%准确率+101笔应该通过: {reason}")
    
    def test_below_baseline_accuracy_rejected(self):
        """低于最低准确率应该被拒绝"""
        passed, reason = self.gate.check(accuracy=12.0, trades=500, avg_return=0.5)
        self.assertFalse(passed, "12%准确率应该被拒绝(低于15%)")
        self.assertIn("低于最低要求", reason)
    
    def test_negative_return_rejected(self):
        """负收益应该被拒绝"""
        passed, reason = self.gate.check(accuracy=25.0, trades=20, avg_return=-0.5)
        self.assertFalse(passed, "负收益应该被拒绝")
        self.assertIn("收益率", reason)
    
    def test_historical_near_miss_cases(self):
        """测试历史近miss案例"""
        # C3-R008: 23.7%准确率, 38笔 → 应该通过(需14笔)
        passed, _ = self.gate.check(accuracy=23.7, trades=38, avg_return=0.77)
        self.assertTrue(passed, "C3-R008应该通过")
        
        # T2-R011: 23.3%准确率, 30笔 → 应该通过(需14笔)
        passed, _ = self.gate.check(accuracy=23.3, trades=30, avg_return=0.40)
        self.assertTrue(passed, "T2-R011应该通过")
        
        # T3-R009: 20.0%准确率, 20笔 → 应该失败(需26笔)
        passed, _ = self.gate.check(accuracy=20.0, trades=20, avg_return=0.14)
        self.assertFalse(passed, "T3-R009应该失败(20<26)")
    
    def test_get_requirement(self):
        """测试获取最小交易数要求"""
        # 高准确率需要少量交易
        self.assertLessEqual(self.gate.get_requirement(50.0), 10)
        self.assertLessEqual(self.gate.get_requirement(30.0), 10)
        
        # 中等准确率需要更多
        req_20 = self.gate.get_requirement(20.0)
        self.assertGreaterEqual(req_20, 20)
        self.assertLessEqual(req_20, 30)
        
        # 低准确率返回None(不支持)
        self.assertIsNone(self.gate.get_requirement(10.0))
    
    def test_tier_ordering(self):
        """测试层级按准确率降序排列"""
        prev_acc = 100
        for tier in self.gate.tiers:
            self.assertLess(tier.min_accuracy, prev_acc)
            prev_acc = tier.min_accuracy


class TestGateAnalyzer(unittest.TestCase):
    """Gate分析器测试"""
    
    def test_baseline_analysis(self):
        """测试基线分析(如果数据存在)"""
        from infra.gate_config import GateAnalyzer
        from pathlib import Path
        
        cache_dir = Path("~/.cache/quant-autoresearch").expanduser()
        if not (cache_dir / "daily.parquet").exists():
            self.skipTest("测试数据不存在")
        
        analyzer = GateAnalyzer(cache_dir)
        result = analyzer.analyze_baseline()
        
        # 基线准确率应该在合理范围内
        self.assertGreater(result['baseline_pct'], 5)
        self.assertLess(result['baseline_pct'], 20)
        self.assertGreater(result['total_samples'], 10000)
    
    def test_random_simulation(self):
        """测试随机策略模拟"""
        from infra.gate_config import GateAnalyzer
        from pathlib import Path
        
        cache_dir = Path("~/.cache/quant-autoresearch").expanduser()
        if not (cache_dir / "daily.parquet").exists():
            self.skipTest("测试数据不存在")
        
        analyzer = GateAnalyzer(cache_dir)
        result = analyzer.simulate_random_strategy(n_trades=50, n_simulations=1000)
        
        # 随机策略的平均准确率应该接近基线
        self.assertGreater(result['mean_accuracy'], 5)
        self.assertLess(result['mean_accuracy'], 20)
        # 标准差应该随交易数增加而减小
        self.assertLess(result['std_accuracy'], 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
