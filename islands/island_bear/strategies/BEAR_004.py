"""
熊市策略 4: 超跌龙头
适用场景: 熊市 (20日跌>5%)
核心逻辑: 大市值龙头股超跌后的价值修复
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """超跌龙头信号"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['ret_5d'] = g['close'].pct_change(5)
    df['ret_20d'] = g['close'].pct_change(20)
    
    # 均线
    df['ma60'] = g['close'].transform(lambda x: x.rolling(60, min_periods=30).mean())
    df['dist_ma60'] = (df['close'] - df['ma60']) / df['ma60']
    
    # 成交量
    df['vol_ma10'] = g['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma10']
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 超大市值 (>300亿) - 网格搜索最优，龙头修复最可靠
    if 'circ_cap' in today.columns:
        cap_threshold = 300e8
        large_cap = (today['circ_cap'] >= cap_threshold).fillna(False)
    else:
        large_cap = True
    
    # 条件2: 5日超跌 (跌 > 10%)
    oversold = (today['ret_5d'] < -0.10).fillna(False)
    
    # 条件3: 远离60日均线 (低于均线 > 15%)
    below_ma = (today['dist_ma60'] < -0.15).fillna(False)
    
    # 条件4: 今日企稳 (跌幅 < 2%)
    stabilize = (today['ret'] > -0.02).fillna(False)
    
    # 条件5: 成交量未暴跌
    vol_ok = (today['vol_ratio'] > 0.5).fillna(False)
    
    # 条件6: 排除ST
    if 'name' in today.columns:
        not_st = ~today['name'].str.contains('ST', na=False)
    else:
        not_st = True
    
    mask = large_cap & oversold & below_ma & stabilize & vol_ok & not_st
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 超跌程度 + 市值
    oversold_score = (-today_f['ret_5d'] - 0.05).clip(0, 0.10) / 0.10
    cap_score = (today_f['circ_cap'] - cap_threshold) / (today_f['circ_cap'].max() - cap_threshold + 1e-10)
    cap_score = cap_score.clip(0, 1)
    signal = 0.4 + 0.4 * oversold_score + 0.2 * cap_score
    
    return pd.Series(signal.values, index=today_f['code'].values)
