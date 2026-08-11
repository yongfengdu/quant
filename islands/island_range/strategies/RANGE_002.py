"""
震荡市策略 2: 大涨不炸板
适用场景: 震荡市 (-5%~5% 20日收益)
核心逻辑: 大幅上涨(4-7%)但收在日内高位、放量，动量强劲 (grid-search: 18.5% acc, +0.72% avg)
"""
import numpy as np
import pandas as pd


def predict_next_day(df):
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)

    df['ret'] = g['close'].pct_change()
    df['vol_ma20'] = g['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    df['close_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)

    # 大涨但不涨停 (4-7%)
    big_up = ((today['ret'] > 0.04) & (today['ret'] < 0.07)).fillna(False)
    # 放量 (>1.5倍)
    vol_ok = (today['vol_ratio'] > 1.5).fillna(False)
    # 收在日内高位 (>80%)
    strong_close = (today['close_pos'] > 0.8).fillna(False)
    # 大市值
    cap_ok = (today['circ_cap'] > 100e8).fillna(True) if 'circ_cap' in today.columns else True

    mask = big_up & vol_ok & strong_close & cap_ok
    if not mask.any():
        return pd.Series(dtype=float)

    tf = today[mask].copy()
    pos_score = (tf['close_pos'] - 0.8).clip(0, 0.2) / 0.2
    vol_score = (tf['vol_ratio'] - 1.5).clip(0, 1.5) / 1.5
    signal = 0.5 + 0.3 * pos_score + 0.2 * vol_score
    return pd.Series(signal.values, index=tf['code'].values)
