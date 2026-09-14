"""
牛市策略 1: 强势突破
适用场景: 牛市 (20日涨>5%)
核心逻辑: 价格突破前期高点+成交量配合
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """强势突破信号"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['ret_5d'] = g['close'].pct_change(5)
    df['ret_20d'] = g['close'].pct_change(20)
    
    # 20日最高价
    df['high_20d'] = g['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
    
    # 成交量指标
    df['vol_ma10'] = g['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['vol_ma20'] = g['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    
    # 均线
    df['ma5'] = g['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['ma10'] = g['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 突破20日新高或接近新高
    breakout = (today['close'] >= today['high_20d'] * 0.98).fillna(False)
    
    # 条件2: 放量 (成交量 > 20日均量1.5倍)
    volume_confirm = (today['vol_ratio'] > 1.5).fillna(False)
    
    # 条件3: 均线多头排列 (ma5 > ma10 > ma20)
    ma_bullish = ((today['ma5'] > today['ma10']) & (today['ma10'] > today['ma20'])).fillna(False)
    
    # 条件4: 近期涨幅合理 (5日涨幅在3%-15%之间，避免追高)
    reasonable_gain = ((today['ret_5d'] > 0.03) & (today['ret_5d'] < 0.15)).fillna(False)
    
    # 条件5: 今日涨幅适中 (不是涨停，留有空间)
    not_limit = (today['ret'] < 0.095).fillna(True)
    
    # 条件6: 市值过滤
    if 'circ_cap' in today.columns:
        cap_ok = (today['circ_cap'] > 50e8).fillna(True)
    else:
        cap_ok = True
    
    mask = breakout & volume_confirm & ma_bullish & reasonable_gain & not_limit & cap_ok
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 突破幅度 + 放量程度
    breakout_strength = ((today_f['close'] / today_f['high_20d']) - 0.98).clip(0, 0.05) / 0.05
    vol_strength = (today_f['vol_ratio'] - 1.5).clip(0, 1.5) / 1.5
    signal = 0.5 + 0.3 * breakout_strength + 0.2 * vol_strength
    
    return pd.Series(signal.values, index=today_f['code'].values)
