"""
炸板率集成进 regime 门控组合 (brokenboard_integrate.py)
========================================================
炸板率 = 反转信号的增强开关 + 危险警报
集成方式 (仅反转侧 bear/range):
  - 危险回避: 炸板数 > 10 → 跳过反转(空仓)
  - 增强过滤: 只在炸板数 1-10 时做反转, 否则空仓
对比 baseline regime 门控。
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns
from portfolio import metrics as pf_metrics


def detect_broken(df):
    ret = df.groupby("code", sort=False)["close"].pct_change()
    ret_high = df.groupby("code", sort=False)["high"].pct_change()
    code = df["code"]
    is20 = code.str.startswith("3") | code.str.startswith("688")
    thr = np.where(is20, 0.195, 0.095)
    lu = (ret >= thr).fillna(False)
    touched = (ret_high >= thr).fillna(False)
    return (touched & ~lu).fillna(False)  # 炸板


def gated_backtest(df, mode, n_bins=5, rebalance=20, danger_thr=10):
    """mode: 'baseline' | 'avoid_danger' | 'enhance'"""
    dates = sorted(df["date"].unique())
    rebal = dates[::rebalance]
    rows = []
    n_skip = 0
    for d in rebal:
        sub = df[df["date"] == d].copy()
        if len(sub) < n_bins * 10:
            continue
        mkt_ret = sub["mkt_ret_20d"].iloc[0]
        regime = "bull" if mkt_ret > 0.03 else "bear/range"
        br_count = sub["broken"].sum()  # 当日炸板数

        if regime == "bull":
            score = sub["f_dist_hi20"]
        else:
            # 反转侧: 应用炸板过滤
            if mode == "avoid_danger" and br_count > danger_thr:
                rows.append((d, 0.0, sub["fwd_20"].mean(), regime, "skip"))
                n_skip += 1
                continue
            if mode == "enhance" and not (1 <= br_count <= danger_thr):
                rows.append((d, 0.0, sub["fwd_20"].mean(), regime, "skip"))
                n_skip += 1
                continue
            score = -1 * sub["f_dist_ma60"]

        sub = sub.assign(score=score).dropna(subset=["score", "fwd_20"])
        if len(sub) < n_bins * 10:
            continue
        q = pd.qcut(sub["score"], n_bins, labels=False, duplicates="drop")
        if q.max() < n_bins - 1:
            continue
        sel = sub[q == q.max()]
        rows.append((d, sel["fwd_20"].mean(), sub["fwd_20"].mean(), regime, "trade"))
    return pd.DataFrame(rows, columns=["date", "port_ret", "mkt_ret", "regime", "action"]), n_skip


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_hi20", "dist_ma60"])
    df = add_forward_returns(df, [20])
    df["broken"] = detect_broken(df)
    mkt = df.groupby("date")["close"].median().pct_change(20)
    df["mkt_ret_20d"] = df["date"].map(mkt)

    print("=== 炸板率集成: regime 门控组合 (全样本 2018-2026) ===\n")
    results = {}
    for mode in ["baseline", "avoid_danger", "enhance"]:
        r, n_skip = gated_backtest(df, mode)
        m = pf_metrics(r)
        results[mode] = m
        label = {"baseline": "baseline(无炸板过滤)", "avoid_danger": "危险回避(炸板>10空仓)",
                 "enhance": "增强(仅1-10炸板做反转)"}[mode]
        print(f"  {label:24s}: 超额={m['ann_excess']*100:+6.1f}%  Sharpe={m['sharpe']:.2f}  "
              f"回撤={m['max_drawdown']*100:6.1f}%  胜率={m['win_rate']*100:.0f}%  跳空{n_skip}期")

    print("\n=== walk-forward 分时期 ===\n")
    for label, cond in [("2018-2022", df["date"] < "2023-01-01"),
                        ("2023-2026", df["date"] >= "2023-01-01")]:
        for mode in ["baseline", "avoid_danger", "enhance"]:
            r, _ = gated_backtest(df[cond], mode)
            m = pf_metrics(r)
            name = {"baseline": "baseline", "avoid_danger": "危险回避", "enhance": "增强"}[mode]
            print(f"  {label} {name:8s}: 超额={m['ann_excess']*100:+6.1f}%  Sharpe={m['sharpe']:.2f}  回撤={m['max_drawdown']*100:5.1f}%")


if __name__ == "__main__":
    main()
