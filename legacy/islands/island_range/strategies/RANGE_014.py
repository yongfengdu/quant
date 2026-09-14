import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    RANGE_014: 震荡市 - 强相对强度动量 + 20日新高突破 (趋势/动量方向, 与超跌反弹正交)

    核心条件 (5个AND, 高确信度低频):
      1. 大市值过滤: circ_cap > 100e8
      2. 收盘创20日新高: close >= 前20日最高价 (平台突破)
      3. 强相对强度: 个股20日收益 - 大盘20日收益 > 5%
      4. 短期多头排列: MA5 > MA20 且 close > MA5
      5. 温和放量: volume > 1.2 * 前20日均量
    信号强度: 个股RS排名(60%) + 板块20日强度排名(40%) 映射到 0.3~1.0
    """
    work = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = work.groupby("code", sort=False)

    # ---- 因子计算 (全部基于历史数据, 用 shift 避免未来函数) ----
    work["ret_20d"] = g["close"].pct_change(20)
    work["mkt_ret_20d_calc"] = g["mkt_close"].pct_change(20)
    work["rs_20d"] = work["ret_20d"] - work["mkt_ret_20d_calc"]

    work["high_20_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(20).max())
    work["ma5"] = g["close"].transform(lambda s: s.rolling(5).mean())
    work["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    work["vol_ma20_prev"] = g["volume"].transform(lambda s: s.shift(1).rolling(20).mean())

    # 只取每只股票最新一天做信号判断
    last = work.groupby("code", sort=False).tail(1).copy()

    # ---- 过滤类条件 (不计入5个AND) ----
    cond_name = (~last["name"].astype(str).str.contains("ST|退", na=False)).fillna(False)
    cond_not_limit = (~last["limit_up_flag"].fillna(False)).fillna(False)  # 当日已涨停的不追

    # ---- 核心5个AND条件 ----
    cond_cap = (last["circ_cap"] > 100e8).fillna(False)                    # 1. 大市值
    cond_break = (last["close"] >= last["high_20_prev"] * 0.999).fillna(False)  # 2. 创20日新高
    cond_rs = (last["rs_20d"] > 0.05).fillna(False)                        # 3. 20日超额大盘>5%
    cond_ma = ((last["ma5"] > last["ma20"]) & (last["close"] > last["ma5"])).fillna(False)  # 4. 多头排列
    cond_vol = (last["volume"] > 1.2 * last["vol_ma20_prev"]).fillna(False)     # 5. 温和放量

    mask = cond_name & cond_not_limit & cond_cap & cond_break & cond_rs & cond_ma & cond_vol
    mask = mask.fillna(False)

    sel = last[mask].copy()
    if len(sel) == 0:
        return pd.Series(dtype=float)

    # ---- 信号强度: 个股RS排名 60% + 板块20日强度排名 40%, 映射到 0.3~1.0 ----
    rs_rank = sel["rs_20d"].rank(pct=True).fillna(0.5)
    sec_rank = sel["sector_ret_20d"].rank(pct=True).fillna(0.5)
    strength = 0.3 + 0.7 * (0.6 * rs_rank + 0.4 * sec_rank)
    strength = strength.clip(0.3, 1.0)

    return pd.Series(strength.values, index=sel["code"].values)