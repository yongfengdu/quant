"""
震荡市策略 5: 放量中位突破
适用场景: 震荡市 (-5%~5% 20日收益)
核心逻辑: 布林带中上部放量启动，早于上轨突破 (grid-search: N 变体)
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

    # 明显放量 (1.5-2.5倍)
    vol_ok = ((today['vol_ratio'] > 1.5) & (today['vol_ratio'] < 2.5)).fillna(False)
    # 布林中上部 (0.6-0.9)，突破前
    boll_ok = ((today['boll_pct'] > 0.6) & (today['boll_pct'] < 0.9)).fillna(False)
    # 当日温和上涨 (1-4%)
    up = ((today['ret'] > 0.01) & (today['ret'] < 0.04)).fillna(False)
    # 大市值
    cap_ok = (today['circ_cap'] > 100e8).fillna(True) if 'circ_cap' in today.columns else True

    mask = vol_ok & boll_ok & up & cap_ok
    if not mask.any():
        return pd.Series(dtype=float)

    tf = today[mask].copy()
    boll_score = (tf['boll_pct'] - 0.6).clip(0, 0.3) / 0.3
    vol_score = (tf['vol_ratio'] - 1.5).clip(0, 1.0) / 1.0
    signal = 0.4 + 0.35 * boll_score + 0.25 * vol_score
    return pd.Series(signal.values, index=tf['code'].values)
