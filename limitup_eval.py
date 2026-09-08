"""
涨停信号 vs 普通信号池 的成功率对比评估
========================================
涨停检测 (板块感知): 主板/中小板 涨>=9.5%, 创业板/科创板 涨>=19.5%
持有: T日信号 → T+1开盘买 → T+2开盘卖 (与系统一致)

对比三组:
  A. 涨停股 (limit-up)
  B. 全市场普通股 (非涨停)
  C. 反转信号 dist_ma60 底部档 (我们的最强信号)
指标: 上涨率(>0), 3%命中率, 均收益, 收益标准差(抖动)
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import compute_all_factors, add_forward_returns


def detect_limit_up(df):
    """板块感知的涨停检测。"""
    ret = df.groupby("code", sort=False)["close"].pct_change()
    code = df["code"]
    is_20pct = code.str.startswith("3") | code.str.startswith("688")
    thr = np.where(is_20pct, 0.195, 0.095)
    return ret >= thr


def main():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = compute_all_factors(df, ["dist_ma60"])

    g = df.groupby("code", sort=False)
    # 前向收益: T+1开盘 -> T+2开盘 (复权开盘)
    adj_open = df["open"] * df["split_factor"]
    df["fwd"] = (g["open"].shift(-2) - g["open"].shift(-1)) / g["open"].shift(-1)

    # 涨停标记
    df["limit_up"] = detect_limit_up(df)

    valid = df.dropna(subset=["fwd"])

    def stats(mask, label):
        sub = valid[mask]
        if len(sub) == 0:
            print(f"  {label}: 无样本")
            return
        r = sub["fwd"]
        print(f"  {label:<24s} 样本={len(sub):>7}  上涨率={ (r>0).mean()*100:5.1f}%  "
              f"3%命中率={(r>0.03).mean()*100:5.1f}%  均收益={r.mean()*100:+5.2f}%  "
              f"std={r.std()*100:5.2f}%")

    print("=== 信号成功概率对比 (T+1开盘买, T+2开盘卖) ===\n")
    print("[A] 涨停股 (limit-up)")
    stats(valid["limit_up"], "A 涨停股")

    print("\n[B] 全市场普通股 (非涨停)")
    stats(~valid["limit_up"], "B 普通股(非涨停)")

    print("\n[C] 反转信号 dist_ma60 底部档")
    for d in sorted(valid["date"].unique()):
        sub = valid[valid["date"] == d]
        if len(sub) < 50:
            continue
        q = pd.qcut(sub["f_dist_ma60"], 5, labels=False, duplicates="drop")
        valid.loc[sub.index, "bottom_quintile"] = (q == 0)
    stats(valid.get("bottom_quintile", pd.Series(False, index=valid.index)).astype(bool),
          "C 反转底部档")

    # 涨停股的次日走势细分
    print("\n=== 涨停股次日走势细分 ===\n")
    next_limit = valid["limit_up"].shift(-1).fillna(False).astype(bool)
    for lbl, m in [("首板(次日不再涨停)", valid["limit_up"] & ~next_limit),
                   ("连板(次日仍涨停)", valid["limit_up"] & next_limit)]:
        stats(m, lbl)

    # 涨停作为辅助过滤: 反转底部档 + 近期涨停 vs 无涨停
    print("\n=== 涨停作为辅助信号 (对反转信号的提升) ===\n")
    valid["recent_limit"] = valid["limit_up"] | valid["limit_up"].shift(1).fillna(False).astype(bool)
    bottom = valid.get("bottom_quintile", pd.Series(False, index=valid.index)).astype(bool)
    stats(bottom, "C 反转底部档(全部)")
    stats(bottom & valid["recent_limit"], "C1 反转底部档 ∩ 近期涨停")
    stats(bottom & ~valid["recent_limit"], "C2 反转底部档 ∩ 无涨停")
    stats(valid["recent_limit"] & ~valid["limit_up"], "D1 昨日涨停今日未涨停")

    # 涨停 vs 反转底部档 的抖动对比
    print("\n=== 抖动对比 (std 与 极端值) ===\n")
    for lbl, m in [("A 涨停股", valid["limit_up"]),
                   ("C 反转底部档", valid.get("bottom_quintile", pd.Series(False, index=valid.index)).astype(bool))]:
        r = valid[m]["fwd"]
        if len(r) == 0:
            continue
        print(f"  {lbl}: 最大单笔亏 {r.min()*100:+.1f}%  最大单笔赚 {r.max()*100:+.1f}%  "
              f" 亏>10%占比 {(r<-0.10).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
