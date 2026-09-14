import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    震荡市 · 大市值趋势动量突破策略 (与池内超跌/放量策略正交)

    核心逻辑:
      震荡市中均值回归失效、动量突破有效。只选趋势中的大盘龙头股:
        1) 硬性过滤: 流通市值 > 100亿, 非 ST/退市, 非跌停, 信号日在最新交易日;
        2) 均线多头排列: MA5 > MA10 > MA20 且收盘价站上 MA20;
        3) 动量突破: 收盘价贴近 20 日收盘新高 (>= 98%);
        4) 相对强度: 个股 20 日收益跑赢大盘 (超额 > 0);
        5) 板块顺风: 所属板块 20 日涨幅分位 > 0.5 (板块前半强)。
      信号强度 = 相对强度分位 + 板块强度分位 + 新高贴近度 + 量比 复合,
      映射到 0.3~1.0, 越强越高。
    返回: pd.Series(index=code, value=signal_strength)
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    d = df.sort_values(['code', 'date']).copy()
    g = d.groupby('code')

    # ---------- 滚动指标 (按股票分组, 向量化) ----------
    def roll_mean(col, win):
        return g[col].rolling(win, min_periods=win).mean().reset_index(level=0, drop=True)

    d['ma5'] = roll_mean('close', 5)
    d['ma10'] = roll_mean('close', 10)
    d['ma20'] = roll_mean('close', 20)
    d['max_close_20'] = (
        g['close'].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    )
    # 20 日个股收益 (用 shift 实现, 兼容新版 pandas)
    d['ret_20'] = d['close'] / g['close'].shift(20) - 1.0
    d['vol_ma5'] = roll_mean('volume', 5)

    # ---------- 取每只股票的最新一行 ----------
    last = d.groupby('code').tail(1).set_index('code')
    latest_date = d['date'].max()

    close = last['close']
    excess_20 = last['ret_20'] - last['mkt_ret_20d']     # 个股 20 日超额收益
    sector_20 = last['sector_ret_20d']                   # 板块 20 日涨幅

    # 板块强度横截面分位 (rank 天然免疫并列值; > 0.5 即板块前半强)
    sector_rank_full = sector_20.rank(pct=True)

    # ---------- 入选条件 (5 组 AND) ----------
    cond_cap = (
        (last['circ_cap'] > 100e8)
        & (~last['name'].astype(str).str.contains('ST|退', na=False))
        & (~last['limit_down_flag'].fillna(False).astype(bool))
        & (last['date'] == latest_date)
        & (close > 0)
    )
    cond_trend = (
        (last['ma5'] > last['ma10'])
        & (last['ma10'] > last['ma20'])
        & (close > last['ma20'])
    )
    cond_high = close >= last['max_close_20'] * 0.98     # 贴近 20 日收盘新高
    cond_rs = excess_20 > 0                              # 跑赢大盘
    cond_sector = sector_rank_full > 0.5                 # 板块强于中位数

    mask = (cond_cap & cond_trend & cond_high & cond_rs & cond_sector).fillna(False)
    mask = mask.astype(bool)

    cand = last[mask]
    if len(cand) == 0:
        return pd.Series(dtype=float)

    # ---------- 信号强度复合 ----------
    ex = cand['ret_20'] - cand['mkt_ret_20d']
    rs_rank = ex.rank(pct=True)                                          # 相对强度分位
    sector_rank = cand['sector_ret_20d'].rank(pct=True)                  # 板块强度分位
    high_score = ((cand['close'] / cand['max_close_20'] - 0.98) / 0.02).clip(0, 1)   # 新高贴近度
    vol_ratio = cand['volume'] / cand['vol_ma5']
    vol_score = (vol_ratio.clip(0.5, 3.0) - 0.5) / 2.5                   # 温和放量得分

    composite = (
        0.40 * rs_rank.fillna(0.5)
        + 0.25 * sector_rank.fillna(0.5)
        + 0.20 * high_score.fillna(0.0)
        + 0.15 * vol_score.fillna(0.5)
    )

    strength = (0.3 + 0.7 * composite).clip(0.3, 1.0)
    return strength.sort_values(ascending=False)