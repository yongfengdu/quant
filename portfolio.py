"""
组合构建 (portfolio.py)
======================
把 active 信号合成截面多头组合, 月度调仓回测。

信号方向约定:
  IC = corr(factor, 前向收益)。反转因子 IC<0 (高因子→低收益)。
  合成 score = Σ sign(IC_i) * |IC_i| * zscore(factor_i)
  → score 越高 = 期望收益越高 → 买 top 档。
"""
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns


def zscore_cross_sectional(s):
    """横截面 z-score (按日)。"""
    g = s.groupby(s.index.get_level_values("date")) if hasattr(s.index, "names") and s.index.names[0] else None
    return s


def _czscore(df, col):
    """按 date 分组的横截面 z-score。"""
    return df.groupby("date")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))


def combine_signals(df, factors, weights=None):
    """合成信号 score。factors: [(name, ic)], weights 可选(默认 |ic|)。"""
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for name, ic in factors:
        col = f"f_{name}"
        w = abs(ic) if weights is None else weights.get(name, abs(ic))
        score = score + np.sign(ic) * w * _czscore(df, col)
    return score


def backtest_portfolio(df, signal_col, n_bins=5, rebalance=20, hold=20, direction="top"):
    """分档回测。每 rebalance 日调仓, 选 top/bottom 档, 持有 hold 天。
    返回 DataFrame: 每期 (rebalance_date, port_ret, market_ret, n_stocks)。"""
    dates = sorted(df["date"].unique())
    rebalance_dates = dates[::rebalance]

    # 预计算前向收益 (持有期) 与市场收益
    df = add_forward_returns(df, [hold]) if f"fwd_{hold}" not in df.columns else df
    fwd_col = f"fwd_{hold}"

    rows = []
    for d in rebalance_dates:
        sub = df[df["date"] == d].dropna(subset=[signal_col, fwd_col])
        if len(sub) < n_bins * 10:
            continue
        q = pd.qcut(sub[signal_col], n_bins, labels=False, duplicates="drop")
        if q.max() < n_bins - 1:
            continue
        if direction == "top":
            bucket = q == q.max()
        else:
            bucket = q == q.min()
        sel = sub[bucket]
        port_ret = sel[fwd_col].mean()
        mkt_ret = sub[fwd_col].mean()
        rows.append((d, port_ret, mkt_ret, len(sel), len(sub)))
    return pd.DataFrame(rows, columns=["date", "port_ret", "mkt_ret", "n_sel", "n_total"])


def metrics(result):
    """组合回测指标。result: DataFrame 含 port_ret/mkt_ret 列 (每期收益)。"""
    r = result["port_ret"].dropna()
    m = result["mkt_ret"].dropna()
    n_periods = len(r)
    # 年化 (假设每期 hold=20 交易日)
    years = n_periods * 20 / 250
    ann_port = (1 + r).prod() ** (1 / years) - 1 if years > 0 else 0
    ann_mkt = (1 + m).prod() ** (1 / years) - 1 if years > 0 else 0
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(250 / 20)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    win = (r > 0).mean()
    return {
        "n_periods": n_periods,
        "ann_port": ann_port,
        "ann_mkt": ann_mkt,
        "ann_excess": ann_port - ann_mkt,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "win_rate": win,
        "avg_ret": r.mean(),
    }
