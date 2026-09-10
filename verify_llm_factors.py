"""
验证 LLM 发现的 3 个行为因子能否纳入组合
==========================================
1. 计算各因子 IC + 买入档超额
2. 与现有信号(dist_ma60/dist_hi20)的相关性(是否正交)
3. 纳入 regime 门控组合, 看能否提升 Sharpe
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns
from portfolio import metrics
from signal_registry import SignalRegistry
from signal_evolve import compute_llm_factor
from signal_lab import compute_factor_ic


def load_llm_factors():
    reg = SignalRegistry()
    return {sid: s["definition"] for sid, s in reg.data["signals"].items() if s.get("definition")}


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_ma60", "dist_hi20"])
    df = add_forward_returns(df, [20])

    # 1. 计算各 LLM 因子 + IC
    print("=== 1. LLM 因子 IC + 买入档超额 (2018-2026) ===\n")
    llm = load_llm_factors()
    factor_dfs = {}
    for sid, code in llm.items():
        df2, err = compute_llm_factor(code, df)
        if err:
            print(f"  {sid}: 运行失败 {err}")
            continue
        col = f"f_{sid}"
        df2 = df2.rename(columns={"f_llm_factor": col})
        ic = compute_factor_ic(df2, sid, 20)["summary"]
        # 买入档(正IC → 买top)
        df2["score"] = df2[col]
        res = portfolio_quintile(df2, "score", direction="top")
        m = metrics(res)
        factor_dfs[sid] = df2[col]
        print(f"  {sid}: IC={ic['mean_ic']*100:+.2f}% t={ic['t_stat']:+.1f}  "
              f"买入档超额={m['ann_excess']*100:+.2f}% Sharpe={m['sharpe']:.2f}")

    # 2. 相关性 (是否正交)
    print("\n=== 2. 因子相关性 (正交性) ===\n")
    cols = {"dist_ma60": df["f_dist_ma60"], "dist_hi20": df["f_dist_hi20"]}
    cols.update(factor_dfs)
    corr = pd.DataFrame(cols).corr()
    print(corr.round(2).to_string())

    # 3. 纳入 regime 门控组合
    print("\n=== 3. 纳入组合对比 ===\n")
    df_all = df.copy()
    for sid, col in factor_dfs.items():
        df_all[sid] = col

    def gated_composite(dfx, use_llm):
        """regime 门控 + 可选 LLM 因子等权合成。"""
        dfx = dfx.copy()
        mkt = dfx.groupby("date")["close"].median().pct_change(20)
        dfx["mkt_ret_20d"] = dfx["date"].map(mkt)
        dates = sorted(dfx["date"].unique())
        rebal = dates[::20]
        rows = []
        for d in rebal:
            sub = dfx[dfx["date"] == d].copy()
            if len(sub) < 50:
                continue
            if sub["mkt_ret_20d"].iloc[0] > 0.03:
                score = sub["f_dist_hi20"]
            else:
                score = -1 * sub["f_dist_ma60"]
            if use_llm:
                # LLM 因子等权叠加(正IC, 越高越好)
                z = lambda s: (s - s.mean()) / (s.std() + 1e-12)
                for sid in factor_dfs:
                    score = score + z(sub[sid])
            sub = sub.assign(score=score).dropna(subset=["score", "fwd_20"])
            q = pd.qcut(sub["score"], 5, labels=False, duplicates="drop")
            if q.max() < 4:
                continue
            sel = sub[q == q.max()]
            rows.append((d, sel["fwd_20"].mean(), sub["fwd_20"].mean()))
        return pd.DataFrame(rows, columns=["date", "port_ret", "mkt_ret"])

    for use_llm in [False, True]:
        r = gated_composite(df_all, use_llm)
        m = metrics(r)
        tag = "baseline(无LLM)" if not use_llm else "+3个LLM因子"
        print(f"  {tag:20s}: 超额={m['ann_excess']*100:+6.1f}%  Sharpe={m['sharpe']:.2f}  回撤={m['max_drawdown']*100:6.1f}%")


def portfolio_quintile(df, score_col, n_bins=5, rebalance=20, hold=20, direction="top"):
    from portfolio import backtest_portfolio
    return backtest_portfolio(df, score_col, n_bins=n_bins, rebalance=rebalance,
                              hold=hold, direction=direction)


if __name__ == "__main__":
    main()
