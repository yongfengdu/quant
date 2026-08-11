"""
震荡市策略 1: 温和放量突破
适用场景: 震荡市 (-5%~5% 20日收益)
核心逻辑: 温和放量向上突破布林带上轨，动量延续 (grid-search: 16.3% acc, +0.41% avg)
"""
import numpy as np
import pandas as pd


def predict_next_day(df):
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)

    df['ret'] = g['close'].pct_change()
    df['vol_ma20'] = g['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    df['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['std20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=10).std())
    df['boll_pct'] = (df['close'] - (df['ma20'] - 2 * df['std20'])) / (4 * df['std20'] + 1e-10)

    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)

    # 温和放量 (1.2-2.5倍)
    vol_ok = ((today['vol_ratio'] > 1.2) & (today['vol_ratio'] < 2.5)).fillna(False)
    # 突破布林上轨附近
    breakout = (today['boll_pct'] > 0.85).fillna(False)
    # 当日上涨但不炸板 (0-7%)
    up = ((today['ret'] > 0) & (today['ret'] < 0.07)).fillna(False)
    # 大市值
    cap_ok = (today['circ_cap'] > 100e8).fillna(True) if 'circ_cap' in today.columns else True

    mask = vol_ok & breakout & up & cap_ok
    if not mask.any():
        return pd.Series(dtype=float)

    tf = today[mask].copy()
    # 信号强度: 突破程度 + 温和放量
    boll_score = (tf['boll_pct'] - 0.85).clip(0, 0.3) / 0.3
    vol_score = 1 - (tf['vol_ratio'] - 1.2).clip(0, 1.3) / 1.3
    signal = 0.4 + 0.4 * boll_score + 0.2 * vol_score
    return pd.Series(signal.values, index=tf['code'].values)
