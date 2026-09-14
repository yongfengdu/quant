import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    震荡市趋势延续策略：强板块 + 上升趋势 + 高位蓄势（允许T日小阴）。

    逻辑：只选处于上升趋势（站上向上20日线）、所属板块20日收益强于大盘、
    股价贴近20日新高的大盘股；允许T日小幅收阴（-3.5%以内）以捕捉趋势中的
    蓄势/洗盘日，同时不追高（当日涨幅>6.8%剔除）。
    信号强度 = 板块超额 + 均线斜率 + 贴近新高 + 温和量能 的加权分，映射到 0.3~1.0。
    """
    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values(['code', 'date'], kind='mergesort').reset_index(drop=True)
    g = d.groupby('code', sort=False)

    # ---------- 因子（全部按 code 分组滚动，只用历史数据） ----------
    d['ret'] = g['close'].pct_change()
    d['ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=15).mean())
    d['ma20_5ago'] = d.groupby('code', sort=False)['ma20'].shift(5)
    d['high20'] = g['high'].transform(lambda s: s.rolling(20, min_periods=15).max())
    d['vol_ma5'] = g['volume'].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d['vol_ratio'] = d['volume'] / d['vol_ma5'].replace(0, np.nan)

    # ---------- 每只股票只取最新一日 ----------
    last = d.groupby('code', sort=False).tail(1).set_index('code')

    # ---------- 股票池过滤：大盘(>100亿流通)、非ST/退市、当日非涨跌停 ----------
    name = last['name'].astype(str)
    universe = (
        (last['circ_cap'] > 100e8)
        & ~name.str.contains('ST|退', regex=True, na=False)
        & ~last['limit_up_flag'].fillna(False).astype(bool)
        & ~last['limit_down_flag'].fillna(False).astype(bool)
    )

    # ---------- 5个核心信号条件 ----------
    cond_up = last['close'] > last['ma20']                     # 1 站上20日线（上升趋势）
    cond_rising = last['ma20'] > last['ma20_5ago']             # 2 20日线向上
    cond_sector = last['sector_ret_20d'] > last['mkt_ret_20d'] # 3 板块20日强于大盘
    cond_high = last['close'] >= 0.90 * last['high20']         # 4 贴近20日新高（强度确认）
    cond_ret = last['ret'].between(-0.035, 0.068)              # 5 允许小阴蓄势、不追高

    mask = (universe & cond_up & cond_rising & cond_sector & cond_high & cond_ret).fillna(False)
    if not bool(mask.any()):
        return pd.Series(dtype=float)

    # ---------- 信号强度打分（0.3 ~ 1.0，越强越高） ----------
    sector_excess = (last['sector_ret_20d'] - last['mkt_ret_20d']).clip(0.0, 0.20) / 0.20
    slope = (last['ma20'] / last['ma20_5ago'] - 1.0).clip(0.0, 0.15) / 0.15
    near_high = (last['close'] / last['high20'] - 0.90).clip(0.0, 0.10) / 0.10
    vol_score = (last['vol_ratio'].clip(0.6, 2.4) - 0.6) / 1.8  # 温和放量加分，缩量不加分

    raw = (
        0.35 * sector_excess
        + 0.25 * slope
        + 0.25 * near_high
        + 0.15 * vol_score
    ).fillna(0.0)

    score = (0.3 + 0.7 * raw).clip(0.3, 1.0)
    return score[mask].sort_values(ascending=False)