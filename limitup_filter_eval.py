"""
涨停过滤的时间窗口调优 + 组合集成验证
======================================
问题: "近期涨停"过滤能提升反转信号, 但样本少(679), 需确认稳健性
验证:
  1. 时间窗口扫描 N=1/2/3/5/10
  2. walk-forward 分时期 (2018-2022 vs 2023-2026)
  3. 集成进 regime 门控组合, 对比 Sharpe/回撤
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns


def detect_limit_up(df):
    ret = df.groupby("code", sort=False)["close"].pct_change()
    code = df["code"]
    is_20pct = code.str.startswith("3") | code.str.startswith("688")
    thr = np.where(is_20pct, 0.195, 0.095)
    return (ret >= thr).fillna(False)


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_ma60", "dist_hi20"])

    g = df.groupby("code", sort=False)
    df["fwd"] = (g["open"].shift(-2) - g["open"].shift(-1)) / g["open"].shift(-1)
    df["limit_up"] = detect_limit_up(df)

    # 近N日涨停标记
    for N in [1, 2, 3, 5, 10]:
        df[f"rl_{N}"] = df.groupby("code")["limit_up"].transform(
            lambda s: s.rolling(N, min_periods=1).max().fillna(0).astype(bool))

    # 反转底部档 (每日分位)
    df["bottom"] = False
    for d in sorted(df["date"].unique()):
        m = (df["date"] == d) & df["f_dist_ma60"].notna()
        if m.sum() < 50:
            continue
        q = pd.qcut(df.loc[m, "f_dist_ma60"], 5, labels=False, duplicates="drop")
        df.loc[m, "bottom"] = (q == 0)

    valid = df.dropna(subset=["fwd"]).copy()

    def stats(mask):
        sub = valid[mask]
        if len(sub) == 0:
            return (0, np.nan, np.nan)
        return (len(sub), (sub["fwd"] > 0.03).mean() * 100, sub["fwd"].mean() * 100)

    print("=== 时间窗口扫描: 反转底部档 ∩ 近N日涨停 ===")
    print(f'{"窗口N":>6}{"样本":>8}{"3%命中率":>10}{"均收益":>10}')
    for N in [1, 2, 3, 5, 10]:
        n, hit, mean = stats(valid["bottom"] & valid[f"rl_{N}"])
        print(f"{N:>6}{n:>8}{hit:>9.1f}%{mean:>+9.2f}%")
    n, hit, mean = stats(valid["bottom"])
    print(f'{"(无过滤)":>6}{n:>8}{hit:>9.1f}%{mean:>+9.2f}%')

    # walk-forward 分时期 (用 N=2 作为代表)
    print("\n=== walk-forward: 反转底部档 ∩ 近2日涨停 分时期 ===")
    for label, cond in [("2018-2022", valid["date"] < "2023-01-01"),
                        ("2023-2026", valid["date"] >= "2023-01-01")]:
        for tag, m in [("底部档全部", cond & valid["bottom"]),
                       ("底部档∩近2日涨停", cond & valid["bottom"] & valid["rl_2"])]:
            n, hit, mean = stats(m)
            print(f"  {label} {tag}: 样本={n} 命中率={hit:.1f}% 均收益={mean:+.2f}%")

    # 动量侧: 牛市时 dist_hi20 顶部档 ∩ 涨停?
    print("\n=== 动量侧验证: dist_hi20 顶部档 ∩ 涨停 (牛市辅助) ===")
    df["top_hi20"] = False
    for d in sorted(df["date"].unique()):
        m = (df["date"] == d) & df["f_dist_hi20"].notna()
        if m.sum() < 50:
            continue
        q = pd.qcut(df.loc[m, "f_dist_hi20"], 5, labels=False, duplicates="drop")
        df.loc[m, "top_hi20"] = (q == 4)
    valid = df.dropna(subset=["fwd"]).copy()
    for tag, m in [("顶部档全部", valid["top_hi20"]),
                   ("顶部档∩近2日涨停", valid["top_hi20"] & valid["rl_2"])]:
        n, hit, mean = stats(m)
        print(f"  {tag}: 样本={n} 命中率={hit:.1f}% 均收益={mean:+.2f}%")


if __name__ == "__main__":
    main()
