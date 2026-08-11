"""
熊市策略 3: 跌停板打开
适用场景: 熊市 (20日跌>5%)
核心逻辑: 跌停板打开后的超短反弹
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """跌停板打开信号"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['prev_close'] = g['close'].shift(1)
    
    # 跌停价
    df['limit_down_price'] = df['prev_close'] * 0.9
    
    # 最低价是否触及跌停
    df['touched_limit'] = (df['low'] <= df['limit_down_price'] * 1.005).fillna(False)
    
    # 收盘价是否脱离跌停
    df['escaped_limit'] = (df['close'] > df['limit_down_price'] * 1.01).fillna(False)
    
    # 成交量
    df['vol_ma10'] = g['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma10']
    
    # 收盘位置
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 日内触及跌停
    touched = today['touched_limit']
    
    # 条件2: 收盘脱离跌停
    escaped = today['escaped_limit']
    
    # 条件3: 成交量活跃 (>0.5倍均量)
    volume_up = (today['vol_ratio'] > 0.5).fillna(False)
    
    # 条件4: 收在日内高位 (>50%) - 网格搜索最优
    recover = (today['close_position'] > 0.5).fillna(False)
    
    # 条件5: 大市值过滤 (>100亿) - 关键: 大盘股跌停后修复更可靠
    if 'circ_cap' in today.columns:
        cap_ok = (today['circ_cap'] > 100e8).fillna(True)
    else:
        cap_ok = True
    
    # 条件6: 排除ST
    if 'name' in today.columns:
        not_st = ~today['name'].str.contains('ST', na=False)
    else:
        not_st = True
    
    mask = touched & escaped & volume_up & recover & cap_ok & not_st
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 收盘位置越高+放量越大越强
    position_score = (today_f['close_position'] - 0.4).clip(0, 0.5) / 0.5
    vol_score = (today_f['vol_ratio'] - 1.5).clip(0, 1.5) / 1.5
    signal = 0.4 + 0.35 * position_score + 0.25 * vol_score
    
    return pd.Series(signal.values, index=today_f['code'].values)
