"""
牛市策略 2: 趋势回踩
适用场景: 牛市 (20日涨>5%)
核心逻辑: 强势股回调到均线支撑位
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """趋势回踩买入"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['ret_5d'] = g['close'].pct_change(5)
    df['ret_20d'] = g['close'].pct_change(20)
    
    # 均线
    df['ma5'] = g['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['ma10'] = g['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    
    # 距离均线
    df['dist_ma10'] = (df['close'] - df['ma10']) / df['ma10']
    df['dist_ma20'] = (df['close'] - df['ma20']) / df['ma20']
    
    # 成交量
    df['vol_ma10'] = g['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma10']
    
    # 波动率
    df['volatility'] = g['ret'].transform(lambda x: x.rolling(10, min_periods=5).std())
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 中期趋势向上 (20日涨幅 > 10%)
    uptrend = (today['ret_20d'] > 0.10).fillna(False)
    
    # 条件2: 短期回调 (5日跌幅在 -8% 到 -2% 之间)
    pullback = ((today['ret_5d'] > -0.08) & (today['ret_5d'] < -0.02)).fillna(False)
    
    # 条件3: 接近10日或20日均线支撑 (在均线附近±2%)
    near_ma = ((today['dist_ma10'].abs() < 0.03) | (today['dist_ma20'].abs() < 0.03)).fillna(False)
    
    # 条件4: 成交量萎缩 (回调缩量)
    vol_shrink = (today['vol_ratio'] < 1.0).fillna(False)
    
    # 条件5: 今日企稳 (跌幅 < 1% 或收阳)
    stabilize = (today['ret'] > -0.01).fillna(False)
    
    # 条件6: 市值过滤
    if 'circ_cap' in today.columns:
        cap_ok = (today['circ_cap'] > 30e8).fillna(True)
    else:
        cap_ok = True
    
    mask = uptrend & pullback & near_ma & vol_shrink & stabilize & cap_ok
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 趋势越强、回调越深，信号越好
    trend_strength = (today_f['ret_20d'] - 0.10).clip(0, 0.20) / 0.20
    pullback_depth = (-today_f['ret_5d'] - 0.02).clip(0, 0.06) / 0.06
    signal = 0.4 + 0.35 * trend_strength + 0.25 * pullback_depth
    
    return pd.Series(signal.values, index=today_f['code'].values)
