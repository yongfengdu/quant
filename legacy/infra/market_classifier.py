#!/usr/bin/env python3
"""
市场状态分类器

将市场划分为不同状态：
- BULL: 牛市（上涨趋势）
- BEAR: 熊市（下跌趋势）
- RANGE: 震荡市（横盘整理）

用途：
1. 分市场状态回测：评估策略在不同市场环境下的表现
2. 市场状态路由：根据当前市场状态选择合适的策略
3. 策略诊断：判断策略失效是因为"策略本身不行"还是"市场不适合"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


class MarketState(Enum):
    """市场状态枚举"""
    BULL = "bull"        # 牛市
    BEAR = "bear"        # 熊市
    RANGE = "range"      # 震荡
    UNKNOWN = "unknown"  # 未知


@dataclass
class MarketPeriod:
    """市场状态时间段"""
    state: MarketState
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    return_pct: float  # 期间涨跌幅
    volatility: float  # 期间波动率
    duration_days: int  # 持续天数
    
    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'start_date': str(self.start_date.date()),
            'end_date': str(self.end_date.date()),
            'return_pct': self.return_pct,
            'volatility': self.volatility,
            'duration_days': self.duration_days,
        }


class MarketClassifier:
    """
    市场状态分类器
    
    基于市场指数（中位数价格）的走势判断市场状态：
    - 使用滚动收益率和波动率
    - 支持多种分类方法
    
    用法:
        classifier = MarketClassifier()
        states = classifier.classify(daily_df)
        periods = classifier.get_periods(daily_df)
    """
    
    def __init__(
        self,
        # 趋势判断参数
        trend_window: int = 20,      # 趋势计算窗口（交易日）
        bull_threshold: float = 0.05,  # 牛市阈值：20日涨幅 > 5%
        bear_threshold: float = -0.05, # 熊市阈值：20日涨幅 < -5%
        
        # 波动率参数
        vol_window: int = 20,        # 波动率计算窗口
        high_vol_threshold: float = 0.02,  # 高波动阈值（日波动率）
        
        # 状态持续性参数
        min_period_days: int = 5,    # 最小状态持续天数
    ):
        self.trend_window = trend_window
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.vol_window = vol_window
        self.high_vol_threshold = high_vol_threshold
        self.min_period_days = min_period_days
    
    def _compute_market_index(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场指数
        
        使用所有股票收盘价的中位数作为市场指数
        """
        daily = daily.copy()
        daily['date'] = pd.to_datetime(daily['date'])
        
        # 计算每日市场指数（中位数）
        market_idx = daily.groupby('date').agg({
            'close': 'median',
            'volume': 'sum',
            'amount': 'sum',
        }).reset_index()
        
        market_idx.columns = ['date', 'mkt_close', 'mkt_volume', 'mkt_amount']
        market_idx = market_idx.sort_values('date').reset_index(drop=True)
        
        # 计算收益率
        market_idx['mkt_return'] = market_idx['mkt_close'].pct_change()
        
        # 计算滚动指标
        market_idx['mkt_return_20d'] = market_idx['mkt_close'].pct_change(self.trend_window)
        market_idx['mkt_volatility'] = market_idx['mkt_return'].rolling(self.vol_window).std()
        
        # 计算均线
        market_idx['mkt_ma20'] = market_idx['mkt_close'].rolling(20).mean()
        market_idx['mkt_ma60'] = market_idx['mkt_close'].rolling(60).mean()
        
        return market_idx
    
    def classify(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        对每个交易日进行市场状态分类
        
        Args:
            daily: 日线数据
            
        Returns:
            DataFrame，包含 date 和 market_state 列
        """
        market_idx = self._compute_market_index(daily)
        
        def classify_day(row) -> str:
            ret_20d = row.get('mkt_return_20d', 0)
            volatility = row.get('mkt_volatility', 0)
            
            # 处理NaN
            if pd.isna(ret_20d):
                return MarketState.UNKNOWN.value
            
            # 趋势判断
            if ret_20d >= self.bull_threshold:
                return MarketState.BULL.value
            elif ret_20d <= self.bear_threshold:
                return MarketState.BEAR.value
            else:
                return MarketState.RANGE.value
        
        market_idx['market_state'] = market_idx.apply(classify_day, axis=1)
        
        return market_idx[['date', 'market_state', 'mkt_close', 'mkt_return_20d', 'mkt_volatility']]
    
    def classify_enhanced(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        增强版分类：结合趋势、波动率、均线等多因素
        
        分类规则：
        1. BULL: 20日涨幅>5% 且 收盘价>MA20>MA60
        2. BEAR: 20日涨幅<-5% 且 收盘价<MA20<MA60
        3. RANGE: 其他情况
        """
        market_idx = self._compute_market_index(daily)
        
        def classify_day(row) -> str:
            ret_20d = row.get('mkt_return_20d', 0)
            close = row.get('mkt_close', 0)
            ma20 = row.get('mkt_ma20', 0)
            ma60 = row.get('mkt_ma60', 0)
            vol = row.get('mkt_volatility', 0)
            
            # 处理NaN
            if pd.isna(ret_20d) or pd.isna(ma20) or pd.isna(ma60):
                return MarketState.UNKNOWN.value
            
            # 趋势 + 均线确认
            is_uptrend = ret_20d >= self.bull_threshold and close > ma20 > ma60
            is_downtrend = ret_20d <= self.bear_threshold and close < ma20 < ma60
            
            if is_uptrend:
                return MarketState.BULL.value
            elif is_downtrend:
                return MarketState.BEAR.value
            else:
                return MarketState.RANGE.value
        
        market_idx['market_state'] = market_idx.apply(classify_day, axis=1)
        
        return market_idx[['date', 'market_state', 'mkt_close', 'mkt_return_20d', 'mkt_volatility', 'mkt_ma20', 'mkt_ma60']]
    
    def get_periods(self, daily: pd.DataFrame, enhanced: bool = False) -> List[MarketPeriod]:
        """
        获取连续的市场状态时间段
        
        Args:
            daily: 日线数据
            enhanced: 是否使用增强版分类
            
        Returns:
            MarketPeriod 列表
        """
        if enhanced:
            classified = self.classify_enhanced(daily)
        else:
            classified = self.classify(daily)
        
        classified = classified[classified['market_state'] != MarketState.UNKNOWN.value]
        classified = classified.sort_values('date').reset_index(drop=True)
        
        if len(classified) == 0:
            return []
        
        periods = []
        current_state = classified.iloc[0]['market_state']
        start_idx = 0
        
        for i in range(1, len(classified)):
            if classified.iloc[i]['market_state'] != current_state:
                # 状态变化，保存前一个时期
                end_idx = i - 1
                period = self._create_period(classified, start_idx, end_idx, current_state)
                if period.duration_days >= self.min_period_days:
                    periods.append(period)
                
                # 开始新时期
                current_state = classified.iloc[i]['market_state']
                start_idx = i
        
        # 保存最后一个时期
        period = self._create_period(classified, start_idx, len(classified) - 1, current_state)
        if period.duration_days >= self.min_period_days:
            periods.append(period)
        
        return periods
    
    def _create_period(
        self,
        classified: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        state: str,
    ) -> MarketPeriod:
        """创建 MarketPeriod 对象"""
        start_date = classified.iloc[start_idx]['date']
        end_date = classified.iloc[end_idx]['date']
        
        start_close = classified.iloc[start_idx]['mkt_close']
        end_close = classified.iloc[end_idx]['mkt_close']
        
        return_pct = (end_close - start_close) / start_close if start_close > 0 else 0
        
        # 计算期间波动率
        period_data = classified.iloc[start_idx:end_idx + 1]
        volatility = period_data['mkt_volatility'].mean() if 'mkt_volatility' in period_data.columns else 0
        
        duration_days = end_idx - start_idx + 1
        
        return MarketPeriod(
            state=MarketState(state),
            start_date=start_date,
            end_date=end_date,
            return_pct=return_pct,
            volatility=volatility if not pd.isna(volatility) else 0,
            duration_days=duration_days,
        )
    
    def get_state_on_date(self, daily: pd.DataFrame, date: pd.Timestamp) -> MarketState:
        """
        获取指定日期的市场状态
        """
        classified = self.classify(daily)
        date = pd.to_datetime(date)
        
        row = classified[classified['date'] == date]
        if len(row) == 0:
            return MarketState.UNKNOWN
        
        state_str = row.iloc[0]['market_state']
        return MarketState(state_str)
    
    def get_state_distribution(self, daily: pd.DataFrame) -> Dict[str, float]:
        """
        获取市场状态分布
        
        Returns:
            Dict: {state: 占比}
        """
        classified = self.classify(daily)
        classified = classified[classified['market_state'] != MarketState.UNKNOWN.value]
        
        if len(classified) == 0:
            return {}
        
        counts = classified['market_state'].value_counts()
        total = len(classified)
        
        return {state: count / total for state, count in counts.items()}
    
    def summary(self, daily: pd.DataFrame) -> str:
        """生成市场状态摘要"""
        distribution = self.get_state_distribution(daily)
        periods = self.get_periods(daily)
        
        lines = [
            "=== 市场状态摘要 ===",
            "",
            "状态分布:",
        ]
        
        for state, ratio in sorted(distribution.items()):
            lines.append(f"  {state}: {ratio:.1%}")
        
        lines.extend([
            "",
            f"识别到 {len(periods)} 个市场周期:",
        ])
        
        for p in periods[-10:]:  # 只显示最近10个
            lines.append(
                f"  {p.start_date.date()} ~ {p.end_date.date()}: "
                f"{p.state.value} ({p.return_pct:+.1%}, {p.duration_days}天)"
            )
        
        return "\n".join(lines)


def classify_market_from_file(
    daily_path: str = "~/.cache/quant-autoresearch/daily.parquet",
) -> pd.DataFrame:
    """从文件计算市场状态的便捷函数"""
    daily_path = Path(daily_path).expanduser()
    
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily data not found: {daily_path}")
    
    daily = pd.read_parquet(daily_path)
    classifier = MarketClassifier()
    return classifier.classify(daily)


if __name__ == "__main__":
    # 测试代码
    print("Testing MarketClassifier...")
    
    try:
        daily_path = Path("~/.cache/quant-autoresearch/daily.parquet").expanduser()
        daily = pd.read_parquet(daily_path)
        
        classifier = MarketClassifier()
        
        # 测试分类
        classified = classifier.classify(daily)
        print(f"Classified {len(classified)} trading days")
        print()
        
        # 测试状态分布
        distribution = classifier.get_state_distribution(daily)
        print("State distribution:")
        for state, ratio in distribution.items():
            print(f"  {state}: {ratio:.1%}")
        print()
        
        # 测试周期识别
        periods = classifier.get_periods(daily)
        print(f"Identified {len(periods)} market periods")
        print()
        
        # 打印摘要
        print(classifier.summary(daily))
        
    except FileNotFoundError as e:
        print(f"Test skipped: {e}")
