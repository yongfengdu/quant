"""
牛市策略 5: 龙头二波
适用场景: 牛市 (20日涨>5%)
核心逻辑: 前期强势股整理后的二次启动
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """龙头二波启动"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['ret_5d'] = g['close'].pct_change(5)
    df['ret_10d'] = g['close'].pct_change(10)
    df['ret_20d'] = g['close'].pct_change(20)
    df['ret_40d'] = g['close'].pct_change(40)
    
    # 波动率
    df['volatility'] = g['ret'].transform(lambda x: x.rolling(10, min_periods=5).std())
    
    # 成交量
    df['vol_ma10'] = g['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma10']
    
    # 40日最高价
    df['high_40d'] = g['high'].transform(lambda x: x.rolling(40, min_periods=20).max())
    df['dist_from_high'] = (df['close'] - df['high_40d']) / df['high_40d']
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 40日内有较大涨幅 (>20%，证明是强势股)
    was_strong = (today['ret_40d'] > 0.20).fillna(False)
    
    # 条件2: 近期回调整理 (距离高点回调5%-15%)
    consolidated = ((today['dist_from_high'] > -0.15) & (today['dist_from_high'] < -0.05)).fillna(False)
    
    # 条件3: 整理期波动率下降
    low_vol = (today['volatility'] < 0.03).fillna(False)
    
    # 条件4: 今日有启动迹象 (涨幅>1% 且放量)
    breakout_signal = ((today['ret'] > 0.01) & (today['vol_ratio'] > 1.3)).fillna(False)
    
    # 条件5: 不是追高 (今日涨幅<5%)
    not_chasing = (today['ret'] < 0.05).fillna(True)
    
    # 条件6: 市值过滤
    if 'circ_cap' in today.columns:
        cap_ok = (today['circ_cap'] > 50e8).fillna(True)
    else:
        cap_ok = True
    
    mask = was_strong & consolidated & low_vol & breakout_signal & not_chasing & cap_ok
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 前期涨幅+整理充分度
    prev_strength = (today_f['ret_40d'] - 0.20).clip(0, 0.30) / 0.30
    consolidation = (-today_f['dist_from_high'] - 0.05).clip(0, 0.10) / 0.10
    signal = 0.4 + 0.35 * prev_strength + 0.25 * consolidation
    
    return pd.Series(signal.values, index=today_f['code'].values)
