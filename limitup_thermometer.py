"""
涨停温度计：市场情绪指标对反转信号的择时价值
================================================
假设: 涨停数量/炸板率反映市场情绪冷热
  - 涨停潮(情绪过热) → 反转失效(涨势延续) → 不做反转
  - 冷清(情绪冰点) → 反转有效 → 做反转
验证:
  1. 计算每日涨停数量/涨停占比/炸板率
  2. 反转信号(dist_ma60底部档)按情绪分档, 看超额是否随情绪变化
  3. 对比现有 regime 检测(mkt_ret_20d)的择时效果
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns


def detect_limits(df):
    ret = df.groupby("code", sort=False)["close"].pct_change()
    ret_high = df.groupby("code", sort=False)["high"].pct_change()
    code = df["code"]
    is_20pct = code.str.startswith("3") | code.str.startswith("688")
    thr = np.where(is_20pct, 0.195, 0.095)
    limit_up = (ret >= thr).fillna(False)
    touched = (ret_high >= thr).fillna(False)
    broken = touched & ~limit_up  # 炸板: 摸到涨停但没封住
    return limit_up, broken


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_ma60"])
    df = add_forward_returns(df, [20])

    limit_up, broken = detect_limits(df)
    df["limit_up"] = limit_up
    df["broken"] = broken

    # 每日情绪统计
    daily = df.groupby("date").agg(
        lu_count=("limit_up", "sum"),
        broken_count=("broken", "sum"),
        n=("code", "count"),
    )
    daily["lu_ratio"] = daily["lu_count"] / daily["n"]
    daily["broken_rate"] = daily["broken_count"] / (daily["lu_count"] + daily["broken_count"] + 1)

    # 反转信号每日: 底部档 fwd_20 超额
    df["bottom"] = False
    for d in sorted(df["date"].unique()):
        m = (df["date"] == d) & df["f_dist_ma60"].notna()
        if m.sum() < 50:
            continue
        q = pd.qcut(df.loc[m, "f_dist_ma60"], 5, labels=False, duplicates="drop")
        df.loc[m, "bottom"] = (q == 0)
    v = df.dropna(subset=["fwd_20"])
    sig = v.groupby("date").apply(
        lambda s: pd.Series({
            "rev_excess": s[s["bottom"]]["fwd_20"].mean() - s["fwd_20"].mean(),
            "rev_hit": (s[s["bottom"]]["fwd_20"] > 0.03).mean(),
        }), include_groups=False)
    sig = sig.join(daily[["lu_ratio", "broken_rate", "lu_count"]])

    # 市场 regime
    mkt = df.groupby("date")["close"].median().pct_change(20)
    sig["mkt_ret_20d"] = mkt

    def regime(r):
        return "bull" if r > 0.03 else ("bear" if r < -0.03 else "range")
    sig["regime"] = sig["mkt_ret_20d"].map(regime)

    print("=== 1. 涨停占比 分档 → 反转信号超额 (温度计效应) ===\n")
    sig["lu_q"] = pd.qcut(sig["lu_ratio"], 4, labels=["冷", "温", "热", "极热"], duplicates="drop")
    print(f'{"情绪档":<8}{"反转超额":>10}{"反转命中率":>11}{"平均涨停数":>10}')
    for q, grp in sig.groupby("lu_q", observed=True):
        print(f"{q:<8}{grp['rev_excess'].mean()*100:>+9.2f}%{grp['rev_hit'].mean()*100:>10.1f}%"
              f"{grp['lu_count'].mean():>10.1f}")

    print("\n=== 2. 炸板率 分档 → 反转信号超额 ===\n")
    med_br = sig["broken_rate"].median()
    sig["br_q"] = np.where(sig["broken_rate"] <= med_br, "低炸板", "高炸板")
    print(f'{"炸板档":<8}{"反转超额":>10}{"反转命中率":>11}')
    for q, grp in sig.groupby("br_q"):
        print(f"{q:<8}{grp['rev_excess'].mean()*100:>+9.2f}%{grp['rev_hit'].mean()*100:>10.1f}%")

    print("\n=== 3. 对比: regime 检测 vs 涨停温度计 择时 ===\n")
    print(f'{"择时方法":<20}{"反转超额(冷/熊震荡)":>18}{"反转超额(热/牛)":>18}')
    # regime: bear/range vs bull
    cold = sig[sig["regime"] != "bull"]["rev_excess"].mean() * 100
    hot = sig[sig["regime"] == "bull"]["rev_excess"].mean() * 100
    print(f'{"regime(mkt_ret_20d)":<20}{cold:>+17.2f}%{hot:>+17.2f}%')
    # 温度计: 冷(lu_ratio 最低档) vs 热(最高档)
    med = sig["lu_ratio"].median()
    cold2 = sig[sig["lu_ratio"] <= med]["rev_excess"].mean() * 100
    hot2 = sig[sig["lu_ratio"] > med]["rev_excess"].mean() * 100
    print(f'{"涨停温度计(lu_ratio中位)":<20}{cold2:>+17.2f}%{hot2:>+17.2f}%')

    print("\n=== 4. 温度计与 regime 的相关性 ===")
    print(f"  lu_ratio 与 mkt_ret_20d 相关: {sig['lu_ratio'].corr(sig['mkt_ret_20d']):+.2f}")


if __name__ == "__main__":
    main()
