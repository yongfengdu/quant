"""
牛市策略 4: 量价齐升
适用场景: 牛市 (20日涨>5%)
核心逻辑: 连续放量上涨，趋势确认
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """量价齐升趋势追踪"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['ret_3d'] = g['close'].pct_change(3)
    df['ret_5d'] = g['close'].pct_change(5)
    
    # 连续上涨天数
    df['up_day'] = (df['ret'] > 0).astype(int)
    df['up_streak'] = g['up_day'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )
    
    # 成交量趋势
    df['vol_ma5'] = g['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['vol_ma20'] = g['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['vol_trend'] = df['vol_ma5'] / df['vol_ma20']  # >1 表示近期放量
    
    # 均线
    df['ma10'] = g['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 近5日有3天以上上涨
    mostly_up = (today['up_streak'] >= 3).fillna(False)
    
    # 条件2: 5日涨幅适中 (3%-12%)
    good_momentum = ((today['ret_5d'] > 0.03) & (today['ret_5d'] < 0.12)).fillna(False)
    
    # 条件3: 近期放量 (5日均量 > 20日均量)
    volume_expanding = (today['vol_trend'] > 1.1).fillna(False)
    
    # 条件4: 价格站上均线
    above_ma = ((today['close'] > today['ma10']) & (today['close'] > today['ma20'])).fillna(False)
    
    # 条件5: 今日涨幅适中 (留有空间)
    not_overextended = ((today['ret'] > 0) & (today['ret'] < 0.07)).fillna(False)
    
    # 条件6: 市值过滤
    if 'circ_cap' in today.columns:
        cap_ok = (today['circ_cap'] > 40e8).fillna(True)
    else:
        cap_ok = True
    
    mask = mostly_up & good_momentum & volume_expanding & above_ma & not_overextended & cap_ok
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 动量+放量程度
    momentum_score = (today_f['ret_5d'] - 0.03).clip(0, 0.09) / 0.09
    volume_score = (today_f['vol_trend'] - 1.1).clip(0, 0.5) / 0.5
    signal = 0.4 + 0.35 * momentum_score + 0.25 * volume_score
    
    return pd.Series(signal.values, index=today_f['code'].values)
