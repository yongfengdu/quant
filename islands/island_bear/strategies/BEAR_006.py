import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    熊市大市值抗跌放量突破策略:
    T日收盘出信号 -> T+1开盘买入 -> T+2开盘卖出, 目标 T+2/T+1 开盘收益 > 3%。
    只返回高确信度信号的 code -> signal_strength(0.3~1.0)。
    """
    d = df.copy()
    d = d.sort_values(["code", "date"]).reset_index(drop=True)

    g = d.groupby("code", sort=False)

    # ---------- 基础因子 ----------
    prev_close = d["prev_close"].where(d["prev_close"] > 0, g["close"].shift(1))
    d["ret"] = d["close"] / prev_close - 1.0                      # 日收益
    d["mkt_ret"] = g["mkt_close"].pct_change()                    # 大盘日收益
    d["sector_ret"] = g["sector_close"].pct_change()              # 板块日收益

    # 5日个股/大盘/板块收益 -> 相对强度
    d["ret_5d"] = g["close"].pct_change(5)
    d["mkt_ret_5d"] = g["mkt_close"].pct_change(5)
    d["sector_ret_5d"] = g["sector_close"].pct_change(5)
    d["rs_5d"] = d["ret_5d"] - d["mkt_ret_5d"]                    # 个股相对大盘
    d["sector_rs_5d"] = d["sector_ret_5d"] - d["mkt_ret_5d"]      # 板块相对大盘

    # 量比: 当日成交量 / 过去5日均量(不含当日)
    vol_ma5 = g["volume"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    d["vol_ratio"] = d["volume"] / vol_ma5.replace(0, np.nan)

    # 收盘在日内区间的位置 (0=收最低, 1=收最高)
    span = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_pos"] = (d["close"] - d["low"]) / span

    # ---------- 取每只股票最新一天 ----------
    last = d.groupby("code", sort=False).tail(1).copy()

    # ---------- 过滤条件 (熊市大市值抗跌 + 放量突破不炸板) ----------
    name_str = last["name"].astype(str)
    cond_universe = (
        (last["circ_cap"] > 100e8)                                # 大市值过滤
        & (~name_str.str.contains("ST", case=False, na=False))    # 排除ST/退市
        & (~name_str.str.contains("退", na=False))
    )
    cond_breakout = (
        (last["ret"] >= 0.04) & (last["ret"] <= 0.075)            # 大涨但不追涨停(>7%不碰)
        & (last["close_pos"] >= 0.7)                              # 收在日内高位, 不炸板
        & (last["vol_ratio"] >= 1.5) & (last["vol_ratio"] <= 3.0) # 温和放量
        & (last["rs_5d"] > 0)                                     # 近5日跑赢大盘(抗跌)
    )
    cond = (cond_universe & cond_breakout).fillna(False)

    sel = last.loc[cond].copy()
    if sel.empty:
        return pd.Series(dtype=float)

    # ---------- 信号强度映射 0.3 ~ 1.0 ----------
    def minmax(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        lo, hi = s.min(), s.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    # 子因子: 相对强度 / 板块强度 / 收盘位置 / 量能(适度放量更好)
    rs_score = minmax(sel["rs_5d"].clip(lower=0))
    sector_score = minmax(sel["sector_rs_5d"])
    pos_score = sel["close_pos"].clip(0, 1).fillna(0.5)
    vol_score = 1.0 - (sel["vol_ratio"] - 1.5).abs() / 1.5        # 越接近1.5倍越优
    vol_score = vol_score.clip(0, 1).fillna(0.5)

    raw = 0.40 * rs_score + 0.20 * sector_score + 0.25 * pos_score + 0.15 * vol_score
    strength = 0.3 + 0.7 * minmax(raw)
    strength = strength.clip(0.3, 1.0).fillna(0.3)

    # 只保留最强的少数标的(高确信度、低频), 回测取Top5, 这里给到Top10
    strength = strength.sort_values(ascending=False).head(10)

    out = pd.Series(strength.values, index=sel.loc[strength.index, "code"].values)
    out.index.name = "code"
    return out