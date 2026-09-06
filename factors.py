"""
原语因子库 (factors.py)
======================
信号 = 截面分数 (每股一个标量, 用于横截面排序)。
原语因子: 从 OHLCV 计算的截面因子, 全部无前视 (只用历史数据)。

约定: 每个因子函数签名为 compute(df, g) -> pd.Series, 与 df 逐行对齐。
  df 列: [code, date, open, close, high, low, volume, amount, split_factor]
  g = df.groupby('code', sort=False)
"""
import numpy as np
import pandas as pd


def _ma(g, col, n):
    return g[col].transform(lambda x: x.rolling(n, min_periods=max(2, n // 3)).mean())


def _std(g, col, n):
    return g[col].transform(lambda x: x.rolling(n, min_periods=max(2, n // 3)).std())


def _pct(g, col, n):
    return g[col].pct_change(n)


def _max(g, col, n):
    return g[col].transform(lambda x: x.rolling(n, min_periods=max(2, n // 3)).max())


def _min(g, col, n):
    return g[col].transform(lambda x: x.rolling(n, min_periods=max(2, n // 3)).min())


# 每个因子返回 (name, Series)
FACTORS = {
    # ---- 动量/反转 (带符号) ----
    "rev_1":  lambda df, g: _pct(g, "close", 1),
    "rev_3":  lambda df, g: _pct(g, "close", 3),
    "rev_5":  lambda df, g: _pct(g, "close", 5),
    "mom_10": lambda df, g: _pct(g, "close", 10),
    "mom_20": lambda df, g: _pct(g, "close", 20),
    "mom_60": lambda df, g: _pct(g, "close", 60),

    # ---- 量能 ----
    "vol_ratio_5":  lambda df, g: df["volume"] / _ma(g, "volume", 5),
    "vol_ratio_20": lambda df, g: df["volume"] / _ma(g, "volume", 20),
    "vol_chg_5":    lambda df, g: _pct(g, "volume", 5),

    # ---- 波动 ----
    "rvol_20": lambda df, g: _ret_std(g, "close", 20),
    "rvol_5":  lambda df, g: _ret_std(g, "close", 5),

    # ---- 价格位置 ----
    "dist_ma5":  lambda df, g: df["close"] / _ma(g, "close", 5) - 1,
    "dist_ma20": lambda df, g: df["close"] / _ma(g, "close", 20) - 1,
    "dist_ma60": lambda df, g: df["close"] / _ma(g, "close", 60) - 1,
    "dist_hi20": lambda df, g: df["close"] / _max(g, "high", 20) - 1,
    "dist_lo20": lambda df, g: df["close"] / _min(g, "low", 20) - 1,

    # ---- 振幅 ----
    "amplitude": lambda df, g: (df["high"] - df["low"]) / df["close"],
}


def _ret(g, col):
    return g[col].pct_change()


def _ret_std(g, col, n):
    return g[col].pct_change().transform(lambda x: x.rolling(n, min_periods=max(2, n // 3)).std())


def compute_factor(df, name):
    """计算单个因子, 返回 Series (与 df 对齐)。"""
    g = df.groupby("code", sort=False)
    return FACTORS[name](df, g)


def compute_all_factors(df, names=None):
    """计算一组因子, 以 f_<name> 列追加到 df。"""
    names = names or list(FACTORS.keys())
    out = df.copy()
    for name in names:
        out[f"f_{name}"] = compute_factor(df, name)
    return out


def add_forward_returns(df, horizons=(2, 5, 10, 20)):
    """追加前向收益列 fwd_<H> (用 split_factor 复权后的 close, 无前视方向注意: 这是未来收益, 仅供 IC 打分)。"""
    out = df.copy()
    g = out.groupby("code", sort=False)
    # 复权收盘 = close * split_factor
    adj = out["close"] * out["split_factor"]
    out["_adj_close"] = adj
    for h in horizons:
        out[f"fwd_{h}"] = g["_adj_close"].shift(-h) / adj - 1
    return out.drop(columns=["_adj_close"])


def add_market_return(df):
    """追加市场(中位数)日收益列 mkt_ret, 用于计算超额收益。"""
    out = df.copy()
    mkt = df.groupby("date")["close"].median().pct_change()
    out["mkt_ret"] = df["date"].map(mkt)
    return out
