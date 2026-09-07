"""
组合优化验证 (optimize_portfolio.py)
====================================
问题: 强 IC ≠ 强组合超额 (rvol_20 IC=-6.23% 但组合超额 -2.4%)
方案: 用"分档端超额"选信号 (而非 IC), walk-forward 验证
  - selection 期 (2018-2022): 算每个因子的买入档超额, 选超额>0的因子
  - validation 期 (2023-2026): 用选中的因子组合回测 (样本外)
对比: 选中组合 vs dist_ma60单信号 vs 朴素IC加权组合 vs 市场
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import FACTORS, compute_all_factors, add_forward_returns
from portfolio import backtest_portfolio, metrics
from signal_lab import compute_factor_ic

SPLIT_DATE = "2023-01-01"


def quintile_excess(df, factor_col, ic, n_bins=5, hold=20):
    """买入档(由IC方向决定)的年化超额。"""
    df = df.copy()
    df["score"] = np.sign(ic) * df[factor_col]
    res = backtest_portfolio(df, "score", n_bins=n_bins, rebalance=20, hold=hold, direction="top")
    return metrics(res)["ann_excess"]


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = compute_all_factors(df)
    df = add_forward_returns(df, [20])

    # 1. 每个因子: 全样本 IC + selection 期买入档超额
    print("=== 各因子: IC vs 买入档超额 (selection 期 2018-2022) ===")
    stats = {}
    for fname in FACTORS:
        s = compute_factor_ic(df, fname, 20)["summary"]
        ic = s["mean_ic"]
        df_sel = df[df["date"] < SPLIT_DATE].copy()
        excess = quintile_excess(df_sel, f"f_{fname}", ic)
        stats[fname] = {"ic": ic, "excess_sel": excess}
        print(f"  {fname:14s} IC={ic*100:+6.2f}%  买入档超额={excess*100:+6.2f}%")

    # 2. selection 期超额>0 的因子
    selected = [f for f, st in stats.items() if st["excess_sel"] > 0]
    dropped = [f for f, st in stats.items() if st["excess_sel"] <= 0]
    print(f"\n选中({len(selected)}): {selected}")
    print(f"剔除({len(dropped)}): {dropped}")

    # 3. validation 期回测 (样本外)
    df_val = df[df["date"] >= SPLIT_DATE].copy()
    print(f"\n=== validation 期 (2023-2026) 回测对比 ===")
    results = {}

    # (a) 选中组合 (等权, 用 IC 方向)
    def composite_score(d, factors):
        sc = pd.Series(0.0, index=d.index)
        for f in factors:
            sc = sc + np.sign(stats[f]["ic"]) * (d[f"f_{f}"] - d[f"f_{f}"].mean()) / (d[f"f_{f}"].std() + 1e-12)
        return sc

    df_val["score_sel"] = composite_score(df_val, selected)
    res = backtest_portfolio(df_val, "score_sel", n_bins=5, rebalance=20, hold=20, direction="top")
    results["选中组合"] = metrics(res)

    # (b) dist_ma60 单信号
    df_val["score_dma"] = -1 * df_val["f_dist_ma60"]
    res = backtest_portfolio(df_val, "score_dma", n_bins=5, rebalance=20, hold=20, direction="top")
    results["dist_ma60单信号"] = metrics(res)

    # (c) 朴素 IC 加权组合 (所有因子)
    df_val["score_ic"] = composite_score(df_val, list(FACTORS.keys()))
    res = backtest_portfolio(df_val, "score_ic", n_bins=5, rebalance=20, hold=20, direction="top")
    results["朴素IC加权(全因子)"] = metrics(res)

    print(f'{"方案":<20}{"年化超额":>10}{"Sharpe":>8}{"回撤":>8}{"胜率":>7}')
    for name, m in results.items():
        print(f'{name:<20}{m["ann_excess"]*100:>+9.1f}%{m["sharpe"]:>8.2f}{m["max_drawdown"]*100:>7.1f}%{m["win_rate"]*100:>6.0f}%')


if __name__ == "__main__":
    main()
