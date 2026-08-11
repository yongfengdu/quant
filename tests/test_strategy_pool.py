"""
策略池和自适应路由器单元测试
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from infra.strategy_pool import (
    Strategy, 
    StrategyPool, 
    MarketStateClassifier, 
    AdaptiveRouter
)


class TestStrategy:
    """测试单个策略类"""
    
    def test_strategy_creation(self):
        """测试策略创建"""
        s = Strategy(
            id="R001",
            path=Path("rounds/001/strategy.py"),
            accuracy=25.0,
            trades=50,
            avg_return=1.5,
            category="reversal"
        )
        assert s.id == "R001"
        assert s.accuracy == 25.0
        assert s.category == "reversal"
        assert s.weight == 1.0
        assert s.alpha == 1
        assert s.beta == 1
    
    def test_thompson_sampling_update(self):
        """测试Thompson Sampling更新"""
        s = Strategy(
            id="R001", path=Path("."), accuracy=20.0, 
            trades=30, avg_return=1.0
        )
        
        # 初始状态
        assert s.alpha == 1
        assert s.beta == 1
        
        # 成功更新
        s.update(success=True)
        assert s.alpha == 2
        assert s.beta == 1
        
        # 失败更新
        s.update(success=False)
        assert s.alpha == 2
        assert s.beta == 2
    
    def test_expected_accuracy(self):
        """测试期望准确率计算"""
        s = Strategy(
            id="R001", path=Path("."), accuracy=20.0,
            trades=30, avg_return=1.0
        )
        
        # 初始: alpha=1, beta=1 → 0.5
        assert s.expected_accuracy == 0.5
        
        # 更新后
        s.alpha = 3
        s.beta = 1
        assert s.expected_accuracy == 0.75
    
    def test_thompson_sample_range(self):
        """测试Thompson采样值在0-1范围内"""
        s = Strategy(
            id="R001", path=Path("."), accuracy=20.0,
            trades=30, avg_return=1.0
        )
        
        samples = [s.thompson_sample for _ in range(100)]
        assert all(0 <= x <= 1 for x in samples)


class TestMarketStateClassifier:
    """测试市场状态分类器"""
    
    def test_classifier_creation(self):
        """测试分类器创建"""
        c = MarketStateClassifier(lookback_days=20)
        assert c.lookback_days == 20
    
    def test_determine_state_panic(self):
        """测试恐慌状态判定"""
        c = MarketStateClassifier()
        state = c._determine_state(ret_20d=-12, ret_5d=-7, volatility=40, up_ratio=20)
        assert state == 'panic'
    
    def test_determine_state_fear(self):
        """测试恐惧状态判定"""
        c = MarketStateClassifier()
        state = c._determine_state(ret_20d=-8, ret_5d=-2, volatility=30, up_ratio=35)
        assert state == 'fear'
    
    def test_determine_state_neutral(self):
        """测试中性状态判定"""
        c = MarketStateClassifier()
        state = c._determine_state(ret_20d=2, ret_5d=1, volatility=20, up_ratio=50)
        assert state == 'neutral'
    
    def test_determine_state_greed(self):
        """测试贪婪状态判定"""
        c = MarketStateClassifier()
        state = c._determine_state(ret_20d=8, ret_5d=2, volatility=25, up_ratio=60)
        assert state == 'greed'
    
    def test_determine_state_euphoria(self):
        """测试狂热状态判定"""
        c = MarketStateClassifier()
        state = c._determine_state(ret_20d=15, ret_5d=8, volatility=35, up_ratio=80)
        assert state == 'euphoria'
    
    def test_get_active_categories_panic(self):
        """测试恐慌状态激活的类别"""
        c = MarketStateClassifier()
        cats = c.get_active_categories('panic')
        assert 'reversal' in cats
        assert 'trend' not in cats
    
    def test_get_active_categories_neutral(self):
        """测试中性状态激活所有类别"""
        c = MarketStateClassifier()
        cats = c.get_active_categories('neutral')
        assert 'reversal' in cats
        assert 'trend' in cats
        assert 'range' in cats


class TestStrategyPool:
    """测试策略池"""
    
    def test_pool_creation(self):
        """测试策略池创建"""
        pool = StrategyPool(rounds_dir=Path("rounds"))
        assert pool.rounds_dir == Path("rounds")
        assert len(pool.strategies) == 0
    
    def test_infer_category_reversal(self, tmp_path):
        """测试反转策略类别推断"""
        pool = StrategyPool()
        
        # 创建临时策略文件
        strategy_file = tmp_path / "strategy.py"
        strategy_file.write_text('"""超跌反转策略"""\ndef predict(): pass')
        
        category = pool._infer_category(strategy_file, {})
        assert category == 'reversal'
    
    def test_infer_category_trend(self, tmp_path):
        """测试趋势策略类别推断"""
        pool = StrategyPool()
        
        strategy_file = tmp_path / "strategy.py"
        strategy_file.write_text('"""突破趋势策略"""\ndef predict(): pass')
        
        category = pool._infer_category(strategy_file, {})
        assert category == 'trend'
    
    def test_get_strategies_by_category(self):
        """测试按类别获取策略"""
        pool = StrategyPool()
        
        # 手动添加策略
        pool.strategies["R001"] = Strategy(
            id="R001", path=Path("."), accuracy=25.0,
            trades=50, avg_return=1.5, category="reversal"
        )
        pool.strategies["R002"] = Strategy(
            id="R002", path=Path("."), accuracy=20.0,
            trades=40, avg_return=1.0, category="trend"
        )
        pool._category_index["reversal"] = ["R001"]
        pool._category_index["trend"] = ["R002"]
        
        reversal = pool.get_strategies_by_category("reversal")
        assert len(reversal) == 1
        assert reversal[0].id == "R001"


class TestAdaptiveRouter:
    """测试自适应路由器"""
    
    def test_router_creation(self):
        """测试路由器创建"""
        pool = StrategyPool()
        router = AdaptiveRouter(pool, exploration_rate=0.2)
        assert router.pool == pool
        assert router.exploration_rate == 0.2
    
    def test_aggregate_signals_weighted(self):
        """测试加权信号聚合"""
        pool = StrategyPool()
        router = AdaptiveRouter(pool)
        
        all_signals = {
            '000001': [('R001', 0.8), ('R002', 0.6)],  # 2个策略推荐
            '000002': [('R001', 0.8)],                  # 1个策略推荐
        }
        
        result = router._aggregate_signals(all_signals, mode='weighted')
        
        # 000001应该排在前面(更多策略推荐,权重更高)
        assert result[0]['code'] == '000001'
        assert result[0]['n_sources'] == 2
        assert len(result) == 2
    
    def test_aggregate_signals_top1(self):
        """测试top1信号聚合"""
        pool = StrategyPool()
        router = AdaptiveRouter(pool)
        
        all_signals = {
            '000001': [('R001', 0.8), ('R002', 0.6)],
            '000002': [('R002', 0.6)],
            '000003': [('R001', 0.8)],
        }
        
        result = router._aggregate_signals(all_signals, mode='top1')
        
        # 只有R001(权重0.8最高)推荐的股票
        codes = [r['code'] for r in result]
        assert '000001' in codes
        assert '000003' in codes
        assert '000002' not in codes  # R002推荐的不在结果中
    
    def test_aggregate_signals_union(self):
        """测试并集信号聚合"""
        pool = StrategyPool()
        router = AdaptiveRouter(pool)
        
        all_signals = {
            '000001': [('R001', 0.8)],
            '000002': [('R002', 0.6)],
        }
        
        result = router._aggregate_signals(all_signals, mode='union')
        
        assert len(result) == 2
        codes = [r['code'] for r in result]
        assert '000001' in codes
        assert '000002' in codes


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def sample_df(self):
        """创建测试用DataFrame"""
        dates = pd.date_range('2026-06-01', '2026-07-17', freq='B')
        codes = ['000001', '000002', '000003']
        
        data = []
        for date in dates:
            for code in codes:
                data.append({
                    'date': date,
                    'code': code,
                    'open': 10 + np.random.randn(),
                    'close': 10 + np.random.randn(),
                    'high': 11 + np.random.randn(),
                    'low': 9 + np.random.randn(),
                    'volume': 1000000 + np.random.randint(100000),
                })
        
        return pd.DataFrame(data)
    
    def test_classifier_with_dataframe(self, sample_df):
        """测试分类器处理实际DataFrame"""
        classifier = MarketStateClassifier()
        result = classifier.classify(sample_df)
        
        assert 'state' in result
        assert 'ret_20d' in result
        assert result['state'] in classifier.STATES
    
    def test_router_get_signals_no_strategies(self, sample_df):
        """测试路由器在没有策略时的行为"""
        pool = StrategyPool()  # 空策略池
        router = AdaptiveRouter(pool)
        
        result = router.get_signals(sample_df)
        
        assert result['active_strategies'] == 0
        assert result['final_signals'] == []


class TestStrategyMonitor:
    """测试策略监控器"""
    
    @pytest.fixture
    def mock_pool(self):
        """创建mock策略池"""
        pool = StrategyPool()
        # 添加测试策略
        pool.strategies = {
            'R001': Strategy(
                id='R001', path=Path('.'), accuracy=25.0,
                trades=50, avg_return=1.0, category='reversal'
            ),
            'R002': Strategy(
                id='R002', path=Path('.'), accuracy=20.0,
                trades=30, avg_return=0.5, category='reversal'
            ),
        }
        return pool
    
    @pytest.fixture
    def monitor(self, mock_pool, tmp_path):
        """创建监控器实例"""
        from infra.strategy_pool import StrategyMonitor
        state_file = tmp_path / "monitor_state.json"
        return StrategyMonitor(mock_pool, state_file=state_file)
    
    def test_record_outcome_hit(self, monitor):
        """测试记录命中结果"""
        result = monitor.record_outcome('R001', '2024-01-01', '000001', 0.05)
        
        assert result['outcome']['hit'] == True
        assert result['consecutive_fails'] == 0
        assert result['current_accuracy'] == 1.0  # 1/1
    
    def test_record_outcome_miss(self, monitor):
        """测试记录未命中结果"""
        result = monitor.record_outcome('R001', '2024-01-01', '000001', 0.01)
        
        assert result['outcome']['hit'] == False
        assert result['consecutive_fails'] == 1
    
    def test_consecutive_fails_alert(self, monitor):
        """测试连续失败告警"""
        # 记录5次连续失败
        for i in range(5):
            result = monitor.record_outcome('R001', f'2024-01-0{i+1}', '000001', -0.02)
        
        assert result['alert'] is not None
        # 检查是否有告警 (可能是 consecutive_fails 或 accuracy_drop)
        alert_types = [a['type'] for a in result['alert']['alerts']]
        assert 'consecutive_fails' in alert_types or 'accuracy_drop' in alert_types
        # 状态应该变为 warning 或 suspended
        assert monitor.monitor_state['R001']['status'] in ['warning', 'suspended']
    
    def test_accuracy_drop_alert(self, monitor):
        """测试准确率下降告警"""
        # 先记录一些数据确保有足够样本
        # 基线准确率25%, 如果下降超过50%到12.5%以下应该告警
        for i in range(10):
            # 全部失败
            monitor.record_outcome('R001', f'2024-01-{i+1:02d}', '000001', 0.01)
        
        state = monitor.monitor_state['R001']
        assert state['status'] in ['warning', 'suspended']
    
    def test_health_check(self, monitor):
        """测试健康检查"""
        # 记录一些结果
        monitor.record_outcome('R001', '2024-01-01', '000001', 0.05)
        monitor.record_outcome('R002', '2024-01-01', '000002', -0.01)
        
        report = monitor.health_check()
        
        assert report['total_strategies'] == 2
        assert report['monitored'] == 2
        assert len(report['strategies']) == 2
    
    def test_get_active_strategies(self, monitor):
        """测试获取活跃策略"""
        # 初始状态都是活跃的
        active = monitor.get_active_strategies()
        assert 'R001' in active
        assert 'R002' in active
        
        # 暂停一个策略
        monitor.monitor_state['R001'] = monitor._init_strategy_state('R001')
        monitor.monitor_state['R001']['status'] = 'suspended'
        
        active = monitor.get_active_strategies()
        assert 'R001' not in active
        assert 'R002' in active
    
    def test_reset_strategy(self, monitor):
        """测试重置策略"""
        # 记录一些失败
        for i in range(5):
            monitor.record_outcome('R001', f'2024-01-{i+1:02d}', '000001', -0.02)
        
        # 确认状态已改变 (warning 或 suspended)
        assert monitor.monitor_state['R001']['status'] in ['warning', 'suspended']
        
        # 重置
        monitor.reset_strategy('R001')
        
        assert monitor.monitor_state['R001']['status'] == 'active'
        assert monitor.monitor_state['R001']['consecutive_fails'] == 0
    
    def test_cusum_detection(self, monitor):
        """测试CUSUM性能衰退检测"""
        # 模拟持续的低于期望表现
        for i in range(15):
            monitor.record_outcome('R001', f'2024-01-{i+1:02d}', '000001', -0.01)
        
        state = monitor.monitor_state['R001']
        # CUSUM负向累积应该显著
        assert state['cusum_neg'] < -1.0
    
    def test_state_persistence(self, monitor, tmp_path):
        """测试状态持久化"""
        from infra.strategy_pool import StrategyMonitor
        
        # 记录一些结果
        monitor.record_outcome('R001', '2024-01-01', '000001', 0.05)
        monitor.record_outcome('R001', '2024-01-02', '000002', -0.01)
        
        # 创建新的监控器实例
        new_monitor = StrategyMonitor(monitor.pool, state_file=monitor.state_file)
        
        # 验证状态被恢复
        assert 'R001' in new_monitor.monitor_state
        assert len(new_monitor.monitor_state['R001']['outcomes']) == 2
    
    def test_format_report(self, monitor):
        """测试报告格式化"""
        monitor.record_outcome('R001', '2024-01-01', '000001', 0.05)
        
        report = monitor.health_check()
        formatted = monitor.format_report(report)
        
        assert "策略健康报告" in formatted
        assert "活跃" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
