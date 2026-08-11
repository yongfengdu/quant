"""
熊市策略 1: 超跌反弹（简化版）
适用场景: 熊市 (20日跌>5%)
核心逻辑: 短期超跌后的技术性反弹
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """超跌反弹信号 - 简化版"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['ret_5d'] = g['close'].pct_change(5)
    df['ret_10d'] = g['close'].pct_change(10)
    
    # 成交量
    df['vol_ma10'] = g['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma10']
    
    # 收盘位置
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 5日超跌 (跌幅 5%-15%)
    oversold = ((today['ret_5d'] <= -0.05) & (today['ret_5d'] >= -0.15)).fillna(False)
    
    # 条件2: 今日企稳或反弹 (跌幅<2% 或 收在日内高位)
    stabilize = ((today['ret'] > -0.02) | (today['close_position'] > 0.6)).fillna(False)
    
    # 条件3: 成交量未暴跌 (>0.5倍均量，表明有承接)
    volume_ok = (today['vol_ratio'] > 0.5).fillna(False)
    
    # 条件4: 排除涨停板
    not_limit = (today['ret'] < 0.095).fillna(True)
    
    # 条件5: 市值过滤 (>30亿)
    if 'circ_cap' in today.columns:
        cap_ok = (today['circ_cap'] > 30e8).fillna(True)
    else:
        cap_ok = True
    
    mask = oversold & stabilize & volume_ok & not_limit & cap_ok
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 超跌越深+企稳越明显越强
    oversold_depth = (-today_f['ret_5d'] - 0.05).clip(0, 0.10) / 0.10
    stabilize_strength = today_f['close_position'].clip(0.3, 0.9)
    signal = 0.3 + 0.4 * oversold_depth + 0.3 * stabilize_strength
    
    return pd.Series(signal.values, index=today_f['code'].values)
