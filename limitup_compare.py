"""
涨停信号 vs 20日反转 的收益与稳定性对比
========================================
对比三个策略 (统一年化口径):
  A. 20日反转 (regime门控): 熊/震荡买超跌, 牛买动量, 20日持有
  B. 涨停打板 (买所有涨停股, 次日卖): 1日持有
  C. 超跌+近1日涨停 (短线): 1日持有, 稀疏
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns
from portfolio import metrics as pf_metrics


def detect_limit_up(df):
    ret = df.groupby("code", sort=False)["close"].pct_change()
    code = df["code"]
    is_20pct = code.str.startswith("3") | code.str.startswith("688")
    thr = np.where(is_20pct, 0.195, 0.095)
    return (ret >= thr).fillna(False)


def annualize_daily(returns, days_per_year=250):
    """日收益序列 -> 年化指标。"""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    years = n / days_per_year
    ann_ret = (1 + r).prod() ** (1 / years) - 1 if years > 0 else 0
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(days_per_year)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {
        "n_days": n, "ann_ret": ann_ret, "sharpe": sharpe,
        "max_dd": dd, "win_rate": (r > 0).mean(), "daily_std": r.std(),
    }


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_ma60", "dist_hi20"])
    g = df.groupby("code", sort=False)
    df["limit_up"] = detect_limit_up(df)
    df["fwd1"] = (g["open"].shift(-2) - g["open"].shift(-1)) / g["open"].shift(-1)  # 1日持有
    df = add_forward_returns(df, [20])  # 20日持有

    # ---- A. 20日反转 regime门控 (每20日调仓) ----
    mkt = df.groupby("date")["close"].median().pct_change(20)
    df["mkt_ret_20d"] = df["date"].map(mkt)
    df["bottom"] = False
    for d in sorted(df["date"].unique()):
        m = (df["date"] == d) & df["f_dist_ma60"].notna()
        if m.sum() < 50:
            continue
        q = pd.qcut(df.loc[m, "f_dist_ma60"], 5, labels=False, duplicates="drop")
        df.loc[m, "bottom"] = (q == 0)
    df["top_hi"] = False
    for d in sorted(df["date"].unique()):
        m = (df["date"] == d) & df["f_dist_hi20"].notna()
        if m.sum() < 50:
            continue
        q = pd.qcut(df.loc[m, "f_dist_hi20"], 5, labels=False, duplicates="drop")
        df.loc[m, "top_hi"] = (q == 4)

    dates = sorted(df["date"].unique())
    rebal = dates[::20]
    rows_a = []
    for d in rebal:
        sub = df[df["date"] == d].dropna(subset=["fwd_20"])
        if len(sub) < 50:
            continue
        if sub["mkt_ret_20d"].iloc[0] > 0.03:
            sel = sub[sub["top_hi"]]
        else:
            sel = sub[sub["bottom"]]
        if len(sel) == 0:
            continue
        rows_a.append((d, sel["fwd_20"].mean(), sub["fwd_20"].mean()))
    ra = pd.DataFrame(rows_a, columns=["date", "port_ret", "mkt_ret"])

    # ---- B. 涨停打板 (买所有涨停, 1日持有, 每日) ----
    df["fwd1"] = df["fwd1"].replace([np.inf, -np.inf], np.nan)
    daily_b = []
    for d in dates:
        sub = df[(df["date"] == d) & df["limit_up"]].dropna(subset=["fwd1"])
        if len(sub) == 0:
            daily_b.append(0.0)
        else:
            daily_b.append(sub["fwd1"].mean())
    daily_b = pd.Series(daily_b, index=dates)

    # ---- C. 超跌+近1日涨停 (1日持有) ----
    df["rl_1"] = df.groupby("code")["limit_up"].transform(
        lambda s: s.rolling(1, min_periods=1).max().fillna(0).astype(bool))
    daily_c = []
    for d in dates:
        sub = df[(df["date"] == d) & df["bottom"] & df["rl_1"]].dropna(subset=["fwd1"])
        if len(sub) == 0:
            daily_c.append(np.nan)
        else:
            daily_c.append(sub["fwd1"].mean())
    daily_c = pd.Series(daily_c, index=dates).dropna()

    print("=== 三个策略的年化对比 ===\n")
    for label, rr in [("A 20日反转(regime门控)", None), ("B 涨停打板(买所有涨停)", None)]:
        if label.startswith("A"):
            m = pf_metrics(ra)  # 用 pf 的 20日期年化
            print(f"  {label}: 年化收益={m['ann_port']*100:+6.1f}%  年化超额={m['ann_excess']*100:+6.1f}%  "
                  f"Sharpe={m['sharpe']:.2f}  回撤={m['max_drawdown']*100:6.1f}%  胜率={m['win_rate']*100:.0f}%")
        else:
            m = annualize_daily(daily_b)
            print(f"  {label}: 年化收益={m['ann_ret']*100:+6.1f}%  Sharpe={m['sharpe']:.2f}  "
                  f"回撤={m['max_dd']*100:6.1f}%  胜率={m['win_rate']*100:.0f}%  日std={m['daily_std']*100:.2f}%")

    # C 只有非空日
    m = annualize_daily(daily_c)
    print(f"  C 超跌+近1日涨停(短线): 交易天数={m['n_days']} 年化收益={m['ann_ret']*100:+6.1f}%  "
          f"Sharpe={m['sharpe']:.2f}  胜率={m['win_rate']*100:.0f}%  日std={m['daily_std']*100:.2f}%")

    # 涨停的极端风险
    print("\n=== 涨停打板的尾部风险 (B) ===")
    lu = df[df["limit_up"]].dropna(subset=["fwd1"])
    print(f"  单笔最大亏 {lu['fwd1'].min()*100:+.1f}%  最大赚 {lu['fwd1'].max()*100:+.1f}%  "
          f"亏>10%占比 {(lu['fwd1'] < -0.10).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
