#!/usr/bin/env python3
"""
策略特征提取器

从策略代码和假设描述中提取特征向量，用于：
1. 策略差异性约束：确保新策略与已有策略有足够差异
2. 策略聚类分析：发现策略类型的分布
3. 策略相似度计算：判断两个策略是否本质相同

特征维度：
1. 信号类型：超卖反弹 / 趋势跟踪 / 动量 / 均值回归 / 成交量 / 技术形态
2. 时间窗口：使用的回看周期（短期<5 / 中期5-20 / 长期>20）
3. 条件复杂度：条件数量、嵌套层级
4. 风控特征：是否有止损、板块过滤、仓位控制
5. 市场依赖：是否依赖大盘状态、板块强度
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import re
import ast
from pathlib import Path


@dataclass
class StrategyFeatures:
    """策略特征向量"""
    
    # 信号类型（one-hot）
    is_oversold: bool = False       # 超卖反弹
    is_trend: bool = False          # 趋势跟踪
    is_momentum: bool = False       # 动量
    is_mean_reversion: bool = False # 均值回归
    is_volume_based: bool = False   # 成交量相关
    is_pattern: bool = False        # 技术形态
    is_limit_board: bool = False    # 涨跌停相关
    
    # 时间窗口
    lookback_short: bool = False    # <5日
    lookback_medium: bool = False   # 5-20日
    lookback_long: bool = False     # >20日
    
    # 条件复杂度
    condition_count: int = 0        # 条件数量
    uses_and: bool = False          # 使用AND组合
    uses_or: bool = False           # 使用OR组合
    
    # 风控特征
    has_stop_loss: bool = False     # 有止损
    has_sector_filter: bool = False # 有板块过滤
    has_position_control: bool = False # 有仓位控制
    has_hard_exclude: bool = False  # 有硬排除
    
    # 市场依赖
    uses_market_state: bool = False # 使用大盘状态
    uses_sector_strength: bool = False # 使用板块强度
    
    # 使用的指标
    indicators_used: Set[str] = field(default_factory=set)
    
    # 原始信息
    strategy_name: str = ""
    hypothesis_text: str = ""
    
    def to_vector(self) -> List[float]:
        """转换为数值向量"""
        return [
            float(self.is_oversold),
            float(self.is_trend),
            float(self.is_momentum),
            float(self.is_mean_reversion),
            float(self.is_volume_based),
            float(self.is_pattern),
            float(self.is_limit_board),
            float(self.lookback_short),
            float(self.lookback_medium),
            float(self.lookback_long),
            float(self.condition_count) / 10.0,  # 归一化
            float(self.uses_and),
            float(self.uses_or),
            float(self.has_stop_loss),
            float(self.has_sector_filter),
            float(self.has_position_control),
            float(self.has_hard_exclude),
            float(self.uses_market_state),
            float(self.uses_sector_strength),
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_name': self.strategy_name,
            'is_oversold': self.is_oversold,
            'is_trend': self.is_trend,
            'is_momentum': self.is_momentum,
            'is_mean_reversion': self.is_mean_reversion,
            'is_volume_based': self.is_volume_based,
            'is_pattern': self.is_pattern,
            'is_limit_board': self.is_limit_board,
            'lookback_short': self.lookback_short,
            'lookback_medium': self.lookback_medium,
            'lookback_long': self.lookback_long,
            'condition_count': self.condition_count,
            'uses_and': self.uses_and,
            'uses_or': self.uses_or,
            'has_stop_loss': self.has_stop_loss,
            'has_sector_filter': self.has_sector_filter,
            'has_position_control': self.has_position_control,
            'has_hard_exclude': self.has_hard_exclude,
            'uses_market_state': self.uses_market_state,
            'uses_sector_strength': self.uses_sector_strength,
            'indicators_used': list(self.indicators_used),
        }
    
    def get_strategy_type(self) -> str:
        """获取主要策略类型"""
        type_scores = {
            'oversold_bounce': self.is_oversold,
            'trend_following': self.is_trend,
            'momentum': self.is_momentum,
            'mean_reversion': self.is_mean_reversion,
            'volume_based': self.is_volume_based,
            'pattern': self.is_pattern,
            'limit_board': self.is_limit_board,
        }
        
        # 返回得分最高的类型
        max_type = max(type_scores.items(), key=lambda x: x[1])
        if max_type[1]:
            return max_type[0]
        return 'other'


class StrategyFeatureExtractor:
    """
    策略特征提取器
    
    从策略代码和假设描述中提取特征。
    
    用法:
        extractor = StrategyFeatureExtractor()
        features = extractor.extract_from_code(strategy_code)
        features = extractor.extract_from_hypothesis(hypothesis_dict)
    """
    
    # 关键词映射
    OVERSOLD_KEYWORDS = [
        '超卖', '超跌', 'oversold', '连跌', '跌幅', '下跌', '低位',
        '反弹', '反转', '触底', '底部', '低波', '筑底',
    ]
    
    TREND_KEYWORDS = [
        '趋势', 'trend', '突破', '创新高', '新高', '上涨',
        '向上', '多头', '强势', '涨势',
    ]
    
    MOMENTUM_KEYWORDS = [
        '动量', 'momentum', '加速', '惯性', '强度',
    ]
    
    MEAN_REVERSION_KEYWORDS = [
        '均值回归', 'mean reversion', '回归', '偏离', '乖离',
    ]
    
    VOLUME_KEYWORDS = [
        '成交量', '放量', '缩量', '量能', 'volume', '换手',
        '换手率', 'turnover', '量价',
    ]
    
    PATTERN_KEYWORDS = [
        '形态', '锤子', '十字星', 'K线', '阳线', '阴线',
        '吞没', '早晨之星', '黄昏之星', '三连阳',
    ]
    
    LIMIT_KEYWORDS = [
        '涨停', '跌停', 'limit', '涨停板', '跌停板', '打开', '开板',
    ]
    
    STOP_LOSS_KEYWORDS = [
        '止损', 'stop', '止盈', '风控', '最大亏损', '回撤',
    ]
    
    SECTOR_KEYWORDS = [
        '板块', 'sector', '行业', '概念', '题材', '排除',
        '黑名单', 'CXO', '半导体', '医疗', '消费',
    ]
    
    MARKET_KEYWORDS = [
        '大盘', '市场', '指数', 'mkt_', '牛市', '熊市',
        '市场状态', '整体', '全市场',
    ]
    
    # 技术指标关键词
    INDICATOR_PATTERNS = {
        'MA': r'ma\d+|moving_average|均线',
        'RSI': r'rsi|相对强弱',
        'MACD': r'macd',
        'BOLL': r'boll|布林',
        'KDJ': r'kdj',
        'ATR': r'atr|真实波幅',
        'VOLUME': r'volume|amount|成交量|成交额',
        'TURNOVER': r'turnover|换手',
        'AMPLITUDE': r'amplitude|振幅',
    }
    
    def __init__(self):
        pass
    
    def extract_from_code(self, code: str) -> StrategyFeatures:
        """
        从策略代码提取特征
        
        Args:
            code: Python策略代码字符串
            
        Returns:
            StrategyFeatures 对象
        """
        features = StrategyFeatures()
        code_lower = code.lower()
        
        # 提取信号类型
        features.is_oversold = any(kw in code_lower for kw in self.OVERSOLD_KEYWORDS)
        features.is_trend = any(kw in code_lower for kw in self.TREND_KEYWORDS)
        features.is_momentum = any(kw in code_lower for kw in self.MOMENTUM_KEYWORDS)
        features.is_mean_reversion = any(kw in code_lower for kw in self.MEAN_REVERSION_KEYWORDS)
        features.is_volume_based = any(kw in code_lower for kw in self.VOLUME_KEYWORDS)
        features.is_pattern = any(kw in code_lower for kw in self.PATTERN_KEYWORDS)
        features.is_limit_board = any(kw in code_lower for kw in self.LIMIT_KEYWORDS)
        
        # 提取时间窗口
        lookback_values = self._extract_lookback_values(code)
        features.lookback_short = any(v < 5 for v in lookback_values)
        features.lookback_medium = any(5 <= v <= 20 for v in lookback_values)
        features.lookback_long = any(v > 20 for v in lookback_values)
        
        # 提取条件复杂度
        features.condition_count = self._count_conditions(code)
        features.uses_and = ' and ' in code_lower or ' & ' in code
        features.uses_or = ' or ' in code_lower or ' | ' in code
        
        # 提取风控特征
        features.has_stop_loss = any(kw in code_lower for kw in self.STOP_LOSS_KEYWORDS)
        features.has_sector_filter = any(kw in code_lower for kw in self.SECTOR_KEYWORDS)
        features.has_hard_exclude = '排除' in code_lower or 'exclude' in code_lower or '硬' in code_lower
        
        # 提取市场依赖
        features.uses_market_state = any(kw in code_lower for kw in self.MARKET_KEYWORDS)
        features.uses_sector_strength = 'sector_' in code_lower
        
        # 提取使用的指标
        features.indicators_used = self._extract_indicators(code)
        
        return features
    
    def extract_from_hypothesis(self, hypothesis: Dict) -> StrategyFeatures:
        """
        从假设字典提取特征
        
        Args:
            hypothesis: 假设字典，包含 name, description, changes 等字段
            
        Returns:
            StrategyFeatures 对象
        """
        name = hypothesis.get('name', '')
        description = hypothesis.get('description', '')
        changes = hypothesis.get('changes', [])
        
        # 组合所有文本
        all_text = f"{name} {description} {' '.join(changes) if isinstance(changes, list) else changes}"
        all_text_lower = all_text.lower()
        
        features = StrategyFeatures()
        features.strategy_name = name
        features.hypothesis_text = all_text
        
        # 提取信号类型
        features.is_oversold = any(kw in all_text_lower for kw in self.OVERSOLD_KEYWORDS)
        features.is_trend = any(kw in all_text_lower for kw in self.TREND_KEYWORDS)
        features.is_momentum = any(kw in all_text_lower for kw in self.MOMENTUM_KEYWORDS)
        features.is_mean_reversion = any(kw in all_text_lower for kw in self.MEAN_REVERSION_KEYWORDS)
        features.is_volume_based = any(kw in all_text_lower for kw in self.VOLUME_KEYWORDS)
        features.is_pattern = any(kw in all_text_lower for kw in self.PATTERN_KEYWORDS)
        features.is_limit_board = any(kw in all_text_lower for kw in self.LIMIT_KEYWORDS)
        
        # 提取时间窗口（从描述中查找数字+日/天）
        lookback_values = self._extract_lookback_from_text(all_text)
        features.lookback_short = any(v < 5 for v in lookback_values)
        features.lookback_medium = any(5 <= v <= 20 for v in lookback_values)
        features.lookback_long = any(v > 20 for v in lookback_values)
        
        # 提取条件数量（从changes中估计）
        if isinstance(changes, list):
            features.condition_count = len(changes)
        
        # 提取风控特征
        features.has_stop_loss = any(kw in all_text_lower for kw in self.STOP_LOSS_KEYWORDS)
        features.has_sector_filter = any(kw in all_text_lower for kw in self.SECTOR_KEYWORDS)
        features.has_hard_exclude = '排除' in all_text_lower or 'exclude' in all_text_lower or '硬' in all_text_lower
        
        # 提取市场依赖
        features.uses_market_state = any(kw in all_text_lower for kw in self.MARKET_KEYWORDS)
        features.uses_sector_strength = '板块' in all_text_lower and ('强' in all_text_lower or '弱' in all_text_lower)
        
        return features
    
    def extract_from_file(self, strategy_path: str) -> StrategyFeatures:
        """
        从策略文件提取特征
        """
        path = Path(strategy_path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        
        code = path.read_text()
        features = self.extract_from_code(code)
        features.strategy_name = path.stem
        return features
    
    def _extract_lookback_values(self, code: str) -> List[int]:
        """从代码中提取回看窗口值"""
        values = []
        
        # 匹配 rolling(N), shift(N), pct_change(N) 等
        patterns = [
            r'\.rolling\((\d+)\)',
            r'\.shift\((\d+)\)',
            r'\.pct_change\((\d+)\)',
            r'tail\((\d+)\)',
            r'head\((\d+)\)',
            r'last\((\d+)\)',
            r'\[[-:](\d+):\]',  # slice like [-5:]
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            values.extend(int(m) for m in matches)
        
        return values
    
    def _extract_lookback_from_text(self, text: str) -> List[int]:
        """从文本描述中提取回看窗口值"""
        values = []
        
        # 匹配 "N日" "N天" "过去N天"
        patterns = [
            r'(\d+)\s*[日天]',
            r'过去\s*(\d+)',
            r'近\s*(\d+)',
            r'前\s*(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            values.extend(int(m) for m in matches)
        
        return values
    
    def _count_conditions(self, code: str) -> int:
        """估计代码中的条件数量"""
        count = 0
        
        # 计算比较运算符
        count += code.count(' > ')
        count += code.count(' < ')
        count += code.count(' >= ')
        count += code.count(' <= ')
        count += code.count(' == ')
        count += code.count(' != ')
        
        return count
    
    def _extract_indicators(self, code: str) -> Set[str]:
        """提取使用的技术指标"""
        indicators = set()
        code_lower = code.lower()
        
        for indicator, pattern in self.INDICATOR_PATTERNS.items():
            if re.search(pattern, code_lower):
                indicators.add(indicator)
        
        return indicators
    
    def compute_similarity(
        self,
        features1: StrategyFeatures,
        features2: StrategyFeatures,
    ) -> float:
        """
        计算两个策略特征的相似度
        
        Returns:
            0-1之间的相似度分数，1表示完全相同
        """
        v1 = features1.to_vector()
        v2 = features2.to_vector()
        
        # 计算余弦相似度
        import numpy as np
        
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def is_duplicate(
        self,
        features1: StrategyFeatures,
        features2: StrategyFeatures,
        threshold: float = 0.9,
    ) -> bool:
        """
        判断两个策略是否本质相同
        """
        return self.compute_similarity(features1, features2) >= threshold


if __name__ == "__main__":
    # 测试代码
    print("Testing StrategyFeatureExtractor...")
    
    extractor = StrategyFeatureExtractor()
    
    # 测试从假设提取
    hypothesis = {
        'name': '超卖放量高潮反转+质量后置三过滤',
        'description': '在连续下跌后寻找放量反弹信号，使用5日均线确认趋势',
        'changes': [
            '新增条件1: 前5日跌幅>10%（超卖）',
            '新增条件2: 当日放量>前5日均值2倍',
            '排除: CXO/半导体板块',
        ],
    }
    
    features = extractor.extract_from_hypothesis(hypothesis)
    print(f"Strategy: {features.strategy_name}")
    print(f"Type: {features.get_strategy_type()}")
    print(f"Features: {features.to_dict()}")
    print()
    
    # 测试从代码提取
    sample_code = '''
def predict_next_day(hist):
    # 超卖反弹策略
    latest = hist.groupby("code").tail(5)
    
    # 条件1: 5日跌幅>10%
    pct_5d = latest.groupby("code")["close"].pct_change(5)
    oversold = pct_5d < -0.10
    
    # 条件2: 放量
    vol_ratio = latest["volume"] / latest["volume"].rolling(5).mean()
    high_volume = vol_ratio > 2.0
    
    # 排除CXO板块
    not_cxo = ~latest["sector"].isin(["CXO", "半导体"])
    
    signals = oversold & high_volume & not_cxo
    return signals
'''
    
    features2 = extractor.extract_from_code(sample_code)
    print(f"Code features: {features2.to_dict()}")
    print()
    
    # 测试相似度
    similarity = extractor.compute_similarity(features, features2)
    print(f"Similarity: {similarity:.3f}")
    print(f"Is duplicate: {extractor.is_duplicate(features, features2)}")
