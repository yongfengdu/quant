"""
组合分仓 (portfolio_sleeves.py)
================================
验证: 把 LLM 正交因子作"卫星仓位", 与 regime 门控主仓分散化组合, 能否提升 Sharpe。
  - 主仓: regime 门控 (牛→动量dist_hi20, 熊/震荡→反转dist_ma60)
  - 卫星仓: LLM 因子 (非流动性+MAX, 正IC买top)
  - 组合: w1*主仓 + w2*卫星仓 (含风险平价)
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns
from portfolio import metrics
from signal_registry import SignalRegistry
from signal_evolve import compute_llm_factor


def quintile_returns(df, score_col, direction="top", n_bins=5, rebalance=20):
    """每期买入档的收益序列 (index=rebalance date)。"""
    dates = sorted(df["date"].unique())
    rebal = dates[::rebalance]
    out = {}
    for d in rebal:
        sub = df[df["date"] == d].copy()
        if len(sub) < n_bins * 10:
            continue
        sub = sub.dropna(subset=[score_col, "fwd_20"])
        if len(sub) < n_bins * 10:
            continue
        q = pd.qcut(sub[score_col], n_bins, labels=False, duplicates="drop")
        if q.max() < n_bins - 1:
            continue
        sel = sub[q == (q.max() if direction == "top" else q.min())]
        out[d] = sel["fwd_20"].mean()
    return pd.Series(out).sort_index()


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_ma60", "dist_hi20"])
    df = add_forward_returns(df, [20])
    mkt = df.groupby("date")["close"].median().pct_change(20)
    df["mkt_ret_20d"] = df["date"].map(mkt)
    df["regime"] = np.where(df["mkt_ret_20d"] > 0.03, "bull", "bear/range")

    # LLM 因子 (取最强的非流动性+MAX)
    reg = SignalRegistry()
    llm_sid = "LLM_1788761904"
    llm_code = reg.get(llm_sid)["definition"]
    df2, err = compute_llm_factor(llm_code, df)
    if err:
        print(f"LLM 因子运行失败: {err}")
        return
    df["llm_score"] = df2["f_llm_factor"].values

    # 主仓: regime 门控
    main_ret = {}
    for d in sorted(df["date"].unique())[::20]:
        sub = df[df["date"] == d].copy()
        if len(sub) < 50:
            continue
        reg_regime = sub["regime"].iloc[0]
        sub["score"] = sub["f_dist_hi20"] if reg_regime == "bull" else -1 * sub["f_dist_ma60"]
        sub = sub.dropna(subset=["score", "fwd_20"])
        q = pd.qcut(sub["score"], 5, labels=False, duplicates="drop")
        if q.max() < 4:
            continue
        main_ret[d] = sub[q == q.max()]["fwd_20"].mean()
    main = pd.Series(main_ret).sort_index()

    # 卫星仓: LLM 因子 (正IC买top)
    sat = quintile_returns(df, "llm_score", direction="top")

    # 对齐
    idx = main.index.intersection(sat.index)
    m = main.loc[idx]
    s = sat.loc[idx]

    print("=== 各仓独立表现 ===\n")
    def report(name, r):
        rr = pd.DataFrame({"port_ret": r.values, "mkt_ret": r.values})
        mm = metrics(rr)
        print(f"  {name}: 年化={mm['ann_port']*100:+.1f}%  Sharpe={mm['sharpe']:.2f}  "
              f"波动={r.std()*100:.2f}%  回撤={mm['max_drawdown']*100:.1f}%")

    report("主仓(regime门控)", m)
    report("卫星仓(LLM非流动性)", s)
    print(f"  两仓收益相关性: {m.corr(s):+.3f}")

    print("\n=== 分仓组合 (主仓权重 w) ===\n")
    for w in [1.0, 0.8, 0.7, 0.6, 0.5]:
        comb = w * m + (1 - w) * s
        rr = pd.DataFrame({"port_ret": comb.values, "mkt_ret": comb.values})
        mm = metrics(rr)
        print(f"  主仓{w:.0%}/卫星{1-w:.0%}: 年化={mm['ann_port']*100:+6.1f}%  Sharpe={mm['sharpe']:.3f}  回撤={mm['max_drawdown']*100:6.1f}%")

    # 风险平价: w ∝ 1/vol
    inv1, inv2 = 1 / m.std(), 1 / s.std()
    w1, w2 = inv1 / (inv1 + inv2), inv2 / (inv1 + inv2)
    comb = w1 * m + w2 * s
    rr = pd.DataFrame({"port_ret": comb.values, "mkt_ret": comb.values})
    mm = metrics(rr)
    print(f"  风险平价(主仓{w1:.0%}/卫星{w2:.0%}): 年化={mm['ann_port']*100:+6.1f}%  Sharpe={mm['sharpe']:.3f}  回撤={mm['max_drawdown']*100:6.1f}%")


if __name__ == "__main__":
    main()
