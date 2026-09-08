"""
涨停过滤集成进 regime 门控组合 (limitup_integrate.py)
======================================================
方案:
  - 反转侧(熊/震荡): 底部档 dist_ma60, 优先选"近5日涨停"的股票(有则买, 无则 fallback 全部底部档)
  - 动量侧(牛): dist_hi20 顶部档, 不加涨停过滤(有害)
对比: 带过滤 vs 不带过滤 的 Sharpe/回撤/收益, walk-forward
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns
from portfolio import backtest_portfolio, metrics


def detect_limit_up(df):
    ret = df.groupby("code", sort=False)["close"].pct_change()
    code = df["code"]
    is_20pct = code.str.startswith("3") | code.str.startswith("688")
    thr = np.where(is_20pct, 0.195, 0.095)
    return (ret >= thr).fillna(False)


def gated_backtest(df, use_filter, n_bins=5, rebalance=20, hold=20, rl_window=5, min_filtered=3):
    dates = sorted(df["date"].unique())
    rebal = dates[::rebalance]
    rows = []
    for d in rebal:
        sub = df[df["date"] == d].copy()
        if len(sub) < n_bins * 10:
            continue
        mkt_ret = sub["mkt_ret_20d"].iloc[0]
        regime = "bull" if mkt_ret > 0.03 else "bear/range"

        if regime == "bull":
            score = sub["f_dist_hi20"]  # 动量, 买顶部
            direction = "top"
        else:
            score = -1 * sub["f_dist_ma60"]  # 反转, 买底部(超跌)
            direction = "top"

        sub = sub.assign(score=score).dropna(subset=["score", "fwd"])
        if len(sub) < n_bins * 10:
            continue
        q = pd.qcut(sub["score"], n_bins, labels=False, duplicates="drop")
        if q.max() < n_bins - 1:
            continue
        target_q = q.max() if direction == "top" else q.min()

        sel = sub[q == target_q]
        # 涨停过滤: 只在反转侧加
        if use_filter and regime != "bull":
            filtered = sel[sel[f"rl_{rl_window}"]]
            if len(filtered) >= min_filtered:
                sel = filtered
        port_ret = sel["fwd"].mean()
        rows.append((d, port_ret, sub["fwd"].mean(), regime, len(sel)))
    return pd.DataFrame(rows, columns=["date", "port_ret", "mkt_ret", "regime", "n_sel"])


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_ma60", "dist_hi20"])
    df = add_forward_returns(df, [20])  # 20日持有, 与 regime_gate 一致
    df["fwd"] = df["fwd_20"]
    df["limit_up"] = detect_limit_up(df)
    df["rl_5"] = df.groupby("code")["limit_up"].transform(
        lambda s: s.rolling(5, min_periods=1).max().fillna(0).astype(bool))
    mkt = df.groupby("date")["close"].median().pct_change(20)
    df["mkt_ret_20d"] = df["date"].map(mkt)

    print("=== regime 门控组合: 带涨停过滤 vs 不带 ===")
    for use_filter in [False, True]:
        r = gated_backtest(df, use_filter)
        m = metrics(r)
        tag = "带涨停过滤" if use_filter else "不带过滤"
        n_sel = r["n_sel"].mean()
        print(f"  {tag}: 年化超额={m['ann_excess']*100:+6.1f}%  Sharpe={m['sharpe']:.2f}  "
              f"回撤={m['max_drawdown']*100:6.1f}%  胜率={m['win_rate']*100:.0f}%  均持仓={n_sel:.1f}只")

    print("\n=== walk-forward 分时期 ===")
    for label, cond in [("2018-2022", df["date"] < "2023-01-01"),
                        ("2023-2026", df["date"] >= "2023-01-01")]:
        for use_filter in [False, True]:
            r = gated_backtest(df[cond], use_filter)
            m = metrics(r)
            tag = "带过滤" if use_filter else "不带过滤"
            print(f"  {label} {tag}: 超额={m['ann_excess']*100:+6.1f}%  Sharpe={m['sharpe']:.2f}  回撤={m['max_drawdown']*100:5.1f}%")


if __name__ == "__main__":
    main()
