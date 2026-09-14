import numpy as np
import pandas as pd


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    熊市大盘蓝筹放量抗跌突破策略:
    熊市环境(大盘20日跌幅>3%)下, 只选大市值(流通市值>100亿)股票中
    T日"放量(1.5-3倍) + 大涨4%-7.5%收高位(不炸板) + 显著跑赢大盘(>3%)"的标的,
    博 T+1 开盘买入到 T+2 开盘卖出的强势惯性 > 3%。
    信号强度 0.3~1.0, 由超额收益/收盘位置/量能/涨幅加权映射。
    """
    d = df.sort_values(['code', 'date']).copy()
    g = d.groupby('code', sort=False)

    # ---------- 特征计算 ----------
    # 个股日收益(用给定 prev_close, 避免重复移位)
    d['ret'] = d['close'] / d['prev_close'] - 1.0

    # 量比: 当日成交量 / 前5日均量(不含当日)
    prev_vol_ma5 = g['volume'].transform(
        lambda s: s.rolling(5, min_periods=3).mean().shift(1)
    )
    d['vol_ratio'] = d['volume'] / prev_vol_ma5.replace(0, np.nan)

    # 收盘位置: (close-low)/(high-low), 1=收在最高
    day_range = (d['high'] - d['low']).replace(0, np.nan)
    d['close_pos'] = (d['close'] - d['low']) / day_range

    # 大盘日收益与个股超额收益
    mkt_prev = g['mkt_close'].shift(1)
    d['mkt_ret'] = d['mkt_close'] / mkt_prev.replace(0, np.nan) - 1.0
    d['excess'] = d['ret'] - d['mkt_ret']

    # ---------- 只取最新一日(T日) ----------
    last = d.groupby('code', sort=False).tail(1).copy()

    # ---------- 股票池过滤: 大市值 + 排除ST/退市 ----------
    name = last['name'].fillna('').astype(str)
    universe = (
        (last['circ_cap'] > 100e8)
        & (~name.str.contains('ST', case=False, regex=False))
        & (~name.str.contains('退', regex=False))
    )

    # ---------- 信号条件(5个AND, 防过拟合) ----------
    bear_gate = last['mkt_ret_20d'] < -0.03                      # 1. 熊市环境
    strong_up = (last['ret'] >= 0.04) & (last['ret'] <= 0.075)   # 2. 大涨4%-7.5%(不追涨停)
    vol_burst = (last['vol_ratio'] >= 1.5) & (last['vol_ratio'] <= 3.0)  # 3. 放量1.5-3倍
    close_high = last['close_pos'] >= 0.7                        # 4. 收在日内高位(不炸板)
    resist = last['excess'] >= 0.03                              # 5. 显著跑赢大盘

    mask = universe.fillna(False)
    for cond in (bear_gate, strong_up, vol_burst, close_high, resist):
        mask = mask & cond.fillna(False)

    sel = last[mask].copy()
    if sel.empty:
        return pd.Series(dtype=float)

    # ---------- 信号强度打分: 0.3 ~ 1.0 ----------
    s_excess = (sel['excess'] / 0.08).clip(0, 1)
    s_pos = ((sel['close_pos'] - 0.7) / 0.3).clip(0, 1)
    s_vol = ((sel['vol_ratio'] - 1.5) / 1.5).clip(0, 1)
    s_ret = ((sel['ret'] - 0.04) / 0.035).clip(0, 1)

    score = (
        0.35 * s_excess.fillna(0)
        + 0.25 * s_pos.fillna(0)
        + 0.20 * s_vol.fillna(0)
        + 0.20 * s_ret.fillna(0)
    )
    strength = (0.3 + 0.7 * score).clip(0.3, 1.0)

    out = pd.Series(strength.values, index=sel['code'].values, dtype=float)
    out = out[~out.index.duplicated(keep='first')]
    return out.sort_values(ascending=False)