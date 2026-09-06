import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    牛市趋势动量策略: 20日新高突破 + 多头排列 + 相对强度 + 强板块
    只输出信号日(最后一天)满足条件的股票, signal_strength ∈ [0.3, 1.0]
    """
    df = df.copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    g = df.groupby("code", sort=False)

    # ---- 基础指标 ----
    df["ret"] = g["close"].pct_change()
    df["ma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["ma60"] = g["close"].transform(lambda s: s.rolling(60).mean())

    # 前20日最高收盘(不含当日), 用于判定"创20日新高突破"
    df["high_close_20"] = g["close"].transform(
        lambda s: s.rolling(20).max().shift(1)
    )

    # 个股20日收益
    df["ret_20d"] = g["close"].pct_change(20)

    # 大盘20日收益 (若无 mkt_ret_20d 则从 mkt_close 计算)
    if "mkt_ret_20d" in df.columns:
        df["mkt_ret20"] = df["mkt_ret_20d"]
    else:
        df["mkt_ret20"] = g["mkt_close"].pct_change(20)

    # 板块20日收益
    if "sector_ret_20d" in df.columns:
        df["sec_ret20"] = df["sector_ret_20d"]
    else:
        df["sec_ret20"] = g["sector_close"].pct_change(20)

    # 5日均量 / 20日均量 (放量确认, 仅用于打分不作硬条件)
    df["vol_ma5"] = g["volume"].transform(lambda s: s.rolling(5).mean())
    df["vol_ma20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
    df["vol_ratio"] = df["vol_ma5"] / df["vol_ma20"].replace(0, np.nan)

    # ---- 只取每只股票最后一天(信号日) ----
    last = df.groupby("code", sort=False).tail(1).set_index("code")

    # ---- 过滤: 名称排除 ST/退市 ----
    name_ok = ~last["name"].fillna("").str.contains("ST|退", regex=True)

    # ---- 5个硬条件 (不超过5个AND) ----
    cond_cap = last["circ_cap"] > 100e8                       # 1. 大市值过滤
    cond_breakout = last["close"] > last["high_close_20"]     # 2. 创20日新高突破
    cond_ma = (last["ma5"] > last["ma20"]) & \
              (last["ma20"] > last["ma60"])                   # 3. 多头排列
    cond_rs = last["ret_20d"] > last["mkt_ret20"]             # 4. 相对大盘正超额
    cond_sector = last["sec_ret20"] > 0                       # 5. 板块20日走强

    signal_mask = (
        name_ok & cond_cap & cond_breakout & cond_ma & cond_rs & cond_sector
    ).fillna(False)

    selected = last[signal_mask].copy()
    if selected.empty:
        return pd.Series(dtype=float)

    # ---- 信号强度打分 (0.3 ~ 1.0) ----
    def minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    # a) 20日超额收益 (相对大盘)
    excess_20d = (selected["ret_20d"] - selected["mkt_ret20"]).fillna(0)
    # b) 新高突破幅度 (收盘超出前20日高点多少)
    breakout_mag = (
        selected["close"] / selected["high_close_20"] - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    # c) 板块相对强度 (板块20日 - 大盘20日)
    sector_rs = (selected["sec_ret20"] - selected["mkt_ret20"]).fillna(0)
    # d) 放量程度
    vol_score = selected["vol_ratio"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(1.0)

    score = (
        0.35 * minmax(excess_20d)
        + 0.30 * minmax(breakout_mag)
        + 0.20 * minmax(sector_rs)
        + 0.15 * minmax(vol_score)
    )

    signal_strength = 0.3 + 0.7 * score.clip(0, 1)
    signal_strength = signal_strength.clip(0.3, 1.0)

    return signal_strength.sort_values(ascending=False)