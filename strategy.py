"""
⚠️ 已退役 (DEPRECATED) — 此文件不再是线上策略源
=================================================
线上预测已改用岛屿路由系统 island_router.py (按市场状态路由到已验证策略池)。
本文件仅保留供以下单策略回测/验证工具使用, 不代表线上行为:
  - backtest.py / quick_backtest.py (单策略回测)
  - verify_harness.py (验证)
  - predict_june18.py (历史脚本)

⚠️ 请勿将此文件接入任何线上预测路径。
⚠️ 旧进化 pipeline.py 曾覆写此文件, 现已停用 (hermes cron paused)。
线上策略清单以 island_router.ISLAND_STRATEGIES 为唯一真相源。
"""
import pandas as pd
import numpy as np

def predict_next_day(df):
    """HH600: 超卖骨架不变+流通市值前置域过滤(硬编码) [已退役,仅回测工具用]"""
    df = df.sort_values(['code', 'date']).copy()

    # Pre-filter candidate domain BEFORE any condition computation:
    # circ_cap >= 0.5 * universe median circ_cap (hardcoded threshold constant)
    cap_threshold = 0.5 * df['circ_cap'].median()
    df = df[df['circ_cap'] >= cap_threshold].copy()
    if df.empty:
        return pd.Series(dtype=float)

    g = df.groupby('code', sort=False)

    # 20-day rolling median of turnover_rate
    df['turnover_med20'] = (
        g['turnover_rate'].shift(1).rolling(20, min_periods=20).median()
    )

    # Condition 1 helpers: turnover_rate for each of T-3, T-2, T-1 vs 20d median
    # turnover_rate at T-3, T-2, T-1 all < 0.5 * 20d median
    df['turnover_low_t3'] = g['turnover_rate'].shift(3) < g['turnover_med20'].shift(3) * 0.5
    df['turnover_low_t2'] = g['turnover_rate'].shift(2) < g['turnover_med20'].shift(2) * 0.5
    df['turnover_low_t1'] = g['turnover_rate'].shift(1) < g['turnover_med20'].shift(1) * 0.5

    # Condition 1: 3-day continuous low turnover (低波筑底)
    df['low_active_3d'] = (
        df['turnover_low_t3'].fillna(False) &
        df['turnover_low_t2'].fillna(False) &
        df['turnover_low_t1'].fillna(False)
    )

    # Condition 2: today's turnover > yesterday's * 1.5 (放量)
    df['turnover_activate'] = df['turnover_rate'] > g['turnover_rate'].shift(1) * 1.5

    # Condition 3: bullish candle + price rise (首阳)
    df['bullish_rise'] = (df['close'] > df['open']) & (df['close'] > df['prev_close'])

    # Today's data
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)

    dates = df['date'].unique()
    if len(dates) < 25:
        return pd.Series(dtype=float)

    # Apply conditions
    cond1 = today['low_active_3d'].fillna(False)
    cond2 = today['turnover_activate'].fillna(False)
    cond3 = today['bullish_rise'].fillna(False)

    mask = cond1 & cond2 & cond3

    # Quality filters
    if 'name' in today.columns:
        mask = mask & ~today['name'].str.contains('ST|退', na=False)

    if not mask.any():
        return pd.Series(dtype=float)

    today_f = today[mask].copy()

    # Scoring components

    # 1. Turnover contraction depth: how low was yesterday's turnover vs 20d median
    contraction_ratio = (g['turnover_rate'].shift(1).loc[today_f.index] / 
                         g['turnover_med20'].shift(1).loc[today_f.index])
    # Lower ratio = deeper contraction = higher score
    contraction_score = (1.0 - contraction_ratio.clip(0.1, 0.5) / 0.5)
    contraction_score = contraction_score.clip(0, 1)

    # 2. Turnover awakening intensity: today/yesterday ratio
    awaken_ratio = (today_f['turnover_rate'] / 
                    g['turnover_rate'].shift(1).loc[today_f.index])
    awaken_score = (awaken_ratio.clip(1.5, 5.0) - 1.5) / 3.5

    # 3. Market cap score: smaller is better (small cap elasticity)
    cap = today_f['circ_cap']
    cap_rank = cap.rank(pct=True)
    cap_score = 1.0 - cap_rank  # smaller cap → higher score

    raw = 0.40 * contraction_score + 0.35 * awaken_score + 0.15 * cap_score + 0.10
    raw = raw.clip(0.0, 1.0)

    rmin, rmax = raw.min(), raw.max()
    if rmax > rmin:
        signal = 0.15 + 0.85 * (raw - rmin) / (rmax - rmin)
    else:
        signal = pd.Series(0.5, index=raw.index)

    return pd.Series(signal.values, index=today_f['code'].values)
