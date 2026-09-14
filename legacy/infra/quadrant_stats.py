#!/usr/bin/env python3
"""
四象限统计模块

计算策略预测的 TP/FP/FN/TN 以及相关指标。

核心概念：
- TP (True Positive): 预测买入且实际涨幅>=3%
- FP (False Positive): 预测买入但实际涨幅<3%  
- FN (False Negative): 未预测但实际涨幅>=3%（漏掉的机会）
- TN (True Negative): 未预测且实际涨幅<3%（正确忽略）

交易逻辑：T日收盘信号 → T+1开盘买入 → T+2开盘卖出
收益计算：(T+2 open - T+1 open) / T+1 open
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class QuadrantResult:
    """四象限统计结果"""
    # 基础计数
    tp_count: int = 0  # True Positive
    fp_count: int = 0  # False Positive
    fn_count: int = 0  # False Negative
    tn_count: int = 0  # True Negative
    
    # 详细数据
    tp_records: List[Dict] = field(default_factory=list)  # TP详细记录
    fp_records: List[Dict] = field(default_factory=list)  # FP详细记录
    fn_records: List[Dict] = field(default_factory=list)  # FN详细记录（漏掉的机会）
    
    # 派生指标
    @property
    def precision(self) -> float:
        """准确率 = TP / (TP + FP)"""
        total = self.tp_count + self.fp_count
        return self.tp_count / total if total > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """召回率 = TP / (TP + FN)"""
        total = self.tp_count + self.fn_count
        return self.tp_count / total if total > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * precision * recall / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """准确率 = (TP + TN) / Total"""
        total = self.tp_count + self.fp_count + self.fn_count + self.tn_count
        return (self.tp_count + self.tn_count) / total if total > 0 else 0.0
    
    @property
    def total_predicted(self) -> int:
        """总预测数 = TP + FP"""
        return self.tp_count + self.fp_count
    
    @property
    def total_opportunities(self) -> int:
        """总机会数（实际涨>=3%） = TP + FN"""
        return self.tp_count + self.fn_count
    
    @property
    def missed_opportunity_ratio(self) -> float:
        """漏掉机会占比 = FN / (TP + FN)"""
        total = self.tp_count + self.fn_count
        return self.fn_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'tp_count': self.tp_count,
            'fp_count': self.fp_count,
            'fn_count': self.fn_count,
            'tn_count': self.tn_count,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'total_predicted': self.total_predicted,
            'total_opportunities': self.total_opportunities,
            'missed_opportunity_ratio': self.missed_opportunity_ratio,
        }
    
    def summary(self) -> str:
        """生成摘要文本"""
        lines = [
            "=== 四象限统计 ===",
            f"TP (预测对): {self.tp_count}",
            f"FP (预测错): {self.fp_count}",
            f"FN (漏掉的): {self.fn_count}",
            f"TN (正确忽略): {self.tn_count}",
            "",
            f"准确率 (Precision): {self.precision:.1%}",
            f"召回率 (Recall): {self.recall:.1%}",
            f"F1 Score: {self.f1_score:.3f}",
            f"漏掉机会占比: {self.missed_opportunity_ratio:.1%}",
        ]
        return "\n".join(lines)


class QuadrantStats:
    """
    四象限统计计算器
    
    用法:
        stats = QuadrantStats(threshold=0.03, min_signal_score=0.15)
        result = stats.compute(daily_df, predictions_df)
        print(result.summary())
    """
    
    def __init__(
        self,
        threshold: float = 0.03,  # 涨幅阈值，默认3%
        min_signal_score: float = 0.15,  # 信号分数阈值
        near_miss_ratio: float = 0.6,  # Near-Miss FN阈值（信号分数 > min * ratio）
    ):
        self.threshold = threshold
        self.min_signal_score = min_signal_score
        self.near_miss_ratio = near_miss_ratio
    
    def compute(
        self,
        daily: pd.DataFrame,
        predictions: pd.DataFrame,
        signal_dates: Optional[List] = None,
    ) -> QuadrantResult:
        """
        计算四象限统计
        
        Args:
            daily: 日线数据，必须包含 date, code, open, close 等列
            predictions: 预测结果，包含 code, signal_date, signal（信号分数）等
            signal_dates: 限定计算的日期范围，None则使用所有预测日期
            
        Returns:
            QuadrantResult 对象
        """
        result = QuadrantResult()
        
        if len(predictions) == 0:
            return result
        
        # 确保日期格式
        daily = daily.copy()
        daily['date'] = pd.to_datetime(daily['date'])
        predictions = predictions.copy()
        predictions['signal_date'] = pd.to_datetime(predictions['signal_date'])
        
        # 获取唯一的信号日期
        if signal_dates is None:
            signal_dates = sorted(predictions['signal_date'].unique())
        else:
            signal_dates = [pd.to_datetime(d) for d in signal_dates]
        
        # 构建日期索引
        all_dates = sorted(daily['date'].unique())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        
        # 构建 open price lookup: (code, date) -> open_price
        open_lookup = {}
        for _, row in daily[['date', 'code', 'open']].iterrows():
            open_lookup[(row['code'], row['date'])] = row['open']
        
        # 构建 universe lookup: date -> set of codes
        date_codes = daily.groupby('date')['code'].apply(set).to_dict()
        
        # 构建预测lookup: (signal_date, code) -> signal_score
        pred_lookup = {}
        for _, row in predictions.iterrows():
            key = (row['signal_date'], row['code'])
            pred_lookup[key] = row.get('signal', 1.0)
        
        # 遍历每个信号日期
        for signal_date in signal_dates:
            if signal_date not in date_to_idx:
                continue
            
            idx = date_to_idx[signal_date]
            
            # 需要 T+1 和 T+2 的数据
            if idx + 2 >= len(all_dates):
                continue
            
            t1_date = all_dates[idx + 1]  # T+1 买入日
            t2_date = all_dates[idx + 2]  # T+2 卖出日
            
            # 当日可交易的股票
            tradable = date_codes.get(signal_date, set())
            if len(tradable) < 10:
                continue
            
            # 当日的预测股票
            predicted_codes = set()
            for (sd, code), score in pred_lookup.items():
                if sd == signal_date and score >= self.min_signal_score:
                    predicted_codes.add(code)
            
            # 计算所有股票的实际收益
            for code in tradable:
                t1_open = open_lookup.get((code, t1_date))
                t2_open = open_lookup.get((code, t2_date))
                
                if t1_open is None or t2_open is None:
                    continue
                if t1_open <= 0:
                    continue
                
                actual_return = (t2_open - t1_open) / t1_open
                is_opportunity = actual_return >= self.threshold
                is_predicted = code in predicted_codes
                
                signal_score = pred_lookup.get((signal_date, code), 0.0)
                
                record = {
                    'code': code,
                    'signal_date': signal_date,
                    't1_date': t1_date,
                    't2_date': t2_date,
                    't1_open': t1_open,
                    't2_open': t2_open,
                    'actual_return': actual_return,
                    'signal_score': signal_score,
                }
                
                if is_predicted and is_opportunity:
                    # TP: 预测了且涨了
                    result.tp_count += 1
                    result.tp_records.append(record)
                elif is_predicted and not is_opportunity:
                    # FP: 预测了但没涨
                    result.fp_count += 1
                    result.fp_records.append(record)
                elif not is_predicted and is_opportunity:
                    # FN: 没预测但涨了（漏掉的机会）
                    result.fn_count += 1
                    result.fn_records.append(record)
                else:
                    # TN: 没预测且没涨（正确忽略）
                    result.tn_count += 1
        
        return result
    
    def compute_from_backtest(
        self,
        daily: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> QuadrantResult:
        """
        从回测交易记录计算四象限
        
        Args:
            daily: 日线数据
            trades_df: 回测产生的交易记录，包含 code, entry_date, exit_date, pnl, signal
        
        Returns:
            QuadrantResult 对象
        """
        if len(trades_df) == 0:
            return QuadrantResult()
        
        # 从trades构建predictions格式
        predictions = []
        for _, row in trades_df.iterrows():
            # entry_date是T+1，信号日期是T（前一天）
            entry_date = pd.to_datetime(row['entry_date'])
            
            # 找信号日期（entry_date的前一个交易日）
            daily_dates = sorted(daily['date'].unique())
            daily_dates = [pd.to_datetime(d) for d in daily_dates]
            
            try:
                entry_idx = daily_dates.index(entry_date)
                if entry_idx > 0:
                    signal_date = daily_dates[entry_idx - 1]
                else:
                    continue
            except ValueError:
                continue
            
            predictions.append({
                'code': row['code'],
                'signal_date': signal_date,
                'signal': row.get('signal', 1.0),
            })
        
        if not predictions:
            return QuadrantResult()
        
        predictions_df = pd.DataFrame(predictions)
        return self.compute(daily, predictions_df)
    
    def get_near_miss_fn(self, result: QuadrantResult) -> List[Dict]:
        """
        获取Near-Miss FN（差一点就选中的漏掉机会）
        
        Near-Miss定义：信号分数 > min_signal_score * near_miss_ratio
        
        这些股票"差一点"就会被选中，分析它们可以帮助找到
        条件过严的地方。
        
        Args:
            result: QuadrantResult对象
            
        Returns:
            Near-Miss FN记录列表
        """
        near_miss_threshold = self.min_signal_score * self.near_miss_ratio
        
        near_misses = []
        for record in result.fn_records:
            if record.get('signal_score', 0) >= near_miss_threshold:
                near_misses.append(record)
        
        # 按信号分数排序（最接近选中的排前面）
        near_misses.sort(key=lambda x: x.get('signal_score', 0), reverse=True)
        return near_misses
    
    def get_fn_features(self, result: QuadrantResult, daily: pd.DataFrame) -> pd.DataFrame:
        """
        提取FN（漏掉机会）的特征
        
        Returns:
            DataFrame包含每个FN的特征
        """
        if not result.fn_records:
            return pd.DataFrame()
        
        daily = daily.copy()
        daily['date'] = pd.to_datetime(daily['date'])
        
        features = []
        for record in result.fn_records:
            code = record['code']
            signal_date = record['signal_date']
            
            # 获取该股票在信号日的数据
            stock_data = daily[(daily['code'] == code) & (daily['date'] == signal_date)]
            
            if len(stock_data) == 0:
                continue
            
            row = stock_data.iloc[0]
            
            feature = {
                'code': code,
                'signal_date': signal_date,
                'actual_return': record['actual_return'],
                'signal_score': record.get('signal_score', 0),
                
                # 基础特征
                'sector': row.get('sector', 'Unknown'),
                'close': row.get('close', 0),
                'volume': row.get('volume', 0),
                'turnover_rate': row.get('turnover_rate', 0),
                'amplitude': row.get('amplitude', 0),
                
                # 市场特征
                'mkt_ret_20d': row.get('mkt_ret_20d', 0),
                'sector_ret_20d': row.get('sector_ret_20d', 0),
                
                # 涨跌停标记
                'limit_up_flag': row.get('limit_up_flag', 0),
                'limit_down_flag': row.get('limit_down_flag', 0),
            }
            features.append(feature)
        
        return pd.DataFrame(features)


def compute_quadrant_from_files(
    daily_path: str = "~/.cache/quant-autoresearch/daily.parquet",
    trades_path: str = "~/.cache/quant-autoresearch/trades.parquet",
    threshold: float = 0.03,
) -> QuadrantResult:
    """
    从文件计算四象限统计的便捷函数
    """
    daily_path = Path(daily_path).expanduser()
    trades_path = Path(trades_path).expanduser()
    
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily data not found: {daily_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades data not found: {trades_path}")
    
    daily = pd.read_parquet(daily_path)
    trades = pd.read_parquet(trades_path)
    
    stats = QuadrantStats(threshold=threshold)
    return stats.compute_from_backtest(daily, trades)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    print("Testing QuadrantStats...")
    
    try:
        result = compute_quadrant_from_files()
        print(result.summary())
        print()
        print("Result dict:", result.to_dict())
    except FileNotFoundError as e:
        print(f"Test skipped: {e}")
        sys.exit(0)
