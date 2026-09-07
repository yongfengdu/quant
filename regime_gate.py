"""
回撤控制: regime 门控组合 (regime_gate.py)
==========================================
核心发现: 反转(熊/震荡大赚, 牛市亏) vs 动量(牛市鸡肋, 熊/震荡爆亏), 几乎互补。
方案: regime 门控
  - 熊/震荡 → 用反转 dist_ma60 (买超跌)
  - 牛市   → 用动量 dist_hi20 (买贴高) 或 空仓
walk-forward 验证 + 对比 naive 反转。
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns
from portfolio import backtest_portfolio, metrics


def regime_gated_backtest(df, bull_mode="momentum", n_bins=5, rebalance=20, hold=20):
    """regime 门控组合回测。bull_mode: 'momentum' | 'cash'。
    返回每期 (date, port_ret, mkt_ret, regime, signal)。"""
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
            if bull_mode == "cash":
                # 空仓: 超额为 0
                rows.append((d, 0.0, sub["fwd_20"].mean(), "bull", "cash"))
                continue
            score = sub["f_dist_hi20"]  # 动量: 买贴高
            signal = "momentum"
        else:
            score = -1 * sub["f_dist_ma60"]  # 反转: 买超跌
            signal = "reversal"

        sub = sub.assign(score=score).dropna(subset=["score", "fwd_20"])
        if len(sub) < n_bins * 10:
            continue
        q = pd.qcut(sub["score"], n_bins, labels=False, duplicates="drop")
        if q.max() < n_bins - 1:
            continue
        sel = sub[q == q.max()]
        rows.append((d, sel["fwd_20"].mean(), sub["fwd_20"].mean(), regime, signal))
    return pd.DataFrame(rows, columns=["date", "port_ret", "mkt_ret", "regime", "signal"])


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = compute_all_factors(df, ["dist_hi20", "dist_ma60"])
    df = add_forward_returns(df, [20])
    mkt = df.groupby("date")["close"].median().pct_change(20)
    df["mkt_ret_20d"] = df["date"].map(mkt)

    print("=== regime 门控组合 (全样本 2018-2026) ===")
    for bull_mode in ["cash", "momentum"]:
        r = regime_gated_backtest(df, bull_mode)
        m = metrics(r)
        n_bull = (r["regime"] == "bull").sum()
        print(f"  牛市策略={bull_mode:9s}: 年化超额={m['ann_excess']*100:+6.1f}%  "
              f"Sharpe={m['sharpe']:.2f}  回撤={m['max_drawdown']*100:6.1f}%  "
              f"胜率={m['win_rate']*100:.0f}%  (牛市期数 {n_bull}/{len(r)})")

    # 对比: naive 反转 (一直用 dist_ma60)
    print("\n=== 对比: naive 反转 (一直 dist_ma60) ===")
    df2 = df.copy()
    df2["score"] = -1 * df2["f_dist_ma60"]
    r = backtest_portfolio(df2, "score", n_bins=5, rebalance=20, hold=20, direction="top")
    m = metrics(r)
    print(f"  naive反转: 年化超额={m['ann_excess']*100:+6.1f}%  Sharpe={m['sharpe']:.2f}  "
          f"回撤={m['max_drawdown']*100:6.1f}%  胜率={m['win_rate']*100:.0f}%")

    # walk-forward: 分时期看门控是否稳定
    print("\n=== walk-forward: 门控组合分时期 ====")
    for label, cond in [("2018-2022", df["date"] < "2023-01-01"),
                        ("2023-2026", df["date"] >= "2023-01-01")]:
        for bull_mode in ["cash", "momentum"]:
            r = regime_gated_backtest(df[cond], bull_mode)
            m = metrics(r)
            print(f"  {label} 牛市策略={bull_mode:9s}: 超额={m['ann_excess']*100:+6.1f}%  "
                  f"Sharpe={m['sharpe']:.2f} 回撤={m['max_drawdown']*100:5.1f}%")


if __name__ == "__main__":
    main()
