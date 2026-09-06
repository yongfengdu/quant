"""
信号研究引擎 (signal_lab.py)
============================
IC 引擎 + 五道验证闸门。这是信号发现系统的核心。

IC = 每日横截面 Spearman 秩相关(因子, 前向收益), 对超额收益鲁棒(秩相关对市场水平平移不变)。

五闸门:
  1. 正确性: 因子有效值足够、有区分度
  2. IC 显著性: 均值 IC + t-stat, 多重比较控制
  3. 稳定性: 滚 IC 符号一致性 + 近期 IC 仍显著(内置衰减拦截)
  4. Walk-forward: 时间切分, 验证窗口显著
  5. Regime: 全 regime 显著 (简单方案=硬门槛)
"""
import numpy as np
import pandas as pd

# 阈值 (后续由元验证 T2 标定)
GATE2_T_MIN = 3.8        # 多重比较 (系统网格 ~360 信号 Bonferroni)
GATE3_CONSISTENCY = 0.6  # 符号一致性
GATE3_RECENT_T_MIN = 2.0  # 近期 IC t-stat
GATE4_TEST_T_MIN = 2.5   # walk-forward 验证窗口 t-stat
GATE5_T_MIN = 2.0        # 各 regime t-stat

HORIZONS = (2, 5, 10, 20)


def daily_rank_ic(df, factor_col, fwd_col, min_stocks=50):
    """每日横截面秩相关。返回 Series (index=date, value=IC)。"""
    ics = {}
    for d, sub in df.dropna(subset=[factor_col, fwd_col]).groupby("date"):
        if len(sub) < min_stocks:
            continue
        ic = sub[factor_col].rank().corr(sub[fwd_col].rank())
        if pd.notna(ic):
            ics[d] = ic
    return pd.Series(ics).sort_index()


def ic_summary(ic: pd.Series):
    """从每日 IC 序列计算汇总统计。"""
    if len(ic) == 0:
        return {"n": 0, "mean_ic": np.nan, "std_ic": np.nan, "t_stat": np.nan,
                "sign": 0.0, "ic_consistency": np.nan}
    ic = ic.dropna()
    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    t = mean / (std / np.sqrt(len(ic))) if std > 0 else 0.0
    sign = 1.0 if mean >= 0 else -1.0
    return {
        "n": len(ic),
        "mean_ic": mean,
        "std_ic": std,
        "t_stat": t,
        "sign": sign,
        "ic_consistency": float((ic * sign > 0).mean()),
    }


def compute_factor_ic(df, factor_name, horizon, market_ret_20d=None):
    """计算某因子在某持有期的 IC。df 需含 f_{factor} 和 fwd_{horizon} 列。"""
    from factors import compute_all_factors, add_forward_returns
    df = df.copy()
    if f"f_{factor_name}" not in df.columns:
        df = compute_all_factors(df, [factor_name])
    if f"fwd_{horizon}" not in df.columns:
        df = add_forward_returns(df, [horizon])
    ic = daily_rank_ic(df, f"f_{factor_name}", f"fwd_{horizon}")
    summary = ic_summary(ic)
    # regime 分类 (用 mkt_ret_20d)
    if market_ret_20d is None:
        mkt = df.groupby("date")["close"].median().pct_change(20)
        market_ret_20d = mkt
    reg = pd.Series(index=market_ret_20d.index, dtype=object)
    reg[market_ret_20d > 0.03] = "bull"
    reg[market_ret_20d < -0.03] = "bear"
    reg[(market_ret_20d >= -0.03) & (market_ret_20d <= 0.03)] = "range"
    regime_ic = {}
    for r in ["bull", "bear", "range"]:
        dates = reg[reg == r].index
        sub = ic[ic.index.isin(dates)]
        regime_ic[r] = ic_summary(sub) if len(sub) > 10 else {"n": 0}
    return {"ic": ic, "summary": summary, "regime_ic": regime_ic}


# ---------------- 五闸门 ----------------

def gate1_correctness(df, factor_name):
    """闸门1: 因子有效值足够且非平凡。"""
    col = f"f_{factor_name}"
    if col not in df.columns:
        return False, "因子列不存在"
    vals = df[col]
    valid_ratio = vals.notna().mean()
    if valid_ratio < 0.5:
        return False, f"有效值占比 {valid_ratio:.0%} < 50%"
    if vals.nunique() < 10:
        return False, f"区分度不足 (唯一值 {vals.nunique()})"
    return True, "ok"


def gate2_ic_significance(summary, t_min=GATE2_T_MIN):
    """闸门2: IC 显著性 + 多重比较控制。"""
    if summary["n"] < 50:
        return False, f"样本日 {summary['n']} < 50"
    if abs(summary["t_stat"]) < t_min:
        return False, f"|t|={abs(summary['t_stat']):.1f} < {t_min}"
    return True, "ok"


def gate3_stability(ic, sign, consistency=GATE3_CONSISTENCY, recent_t=GATE3_RECENT_T_MIN):
    """闸门3: 稳定性 + 近期衰减拦截。"""
    if len(ic) < 60:
        return False, "样本不足60日"
    # 符号一致性 (按20日滚窗)
    win = 20
    roll_sign = (ic * sign > 0).rolling(win, min_periods=win // 2).mean()
    frac = (roll_sign > 0.5).mean()  # 多数滚窗符号正确
    if frac < consistency:
        return False, f"符号一致性 {frac:.0%} < {consistency:.0%}"
    # 近期 IC (最后60日)
    recent = ic.iloc[-60:]
    s = ic_summary(recent)
    if abs(s["t_stat"]) < recent_t:
        return False, f"近期|t|={abs(s['t_stat']):.1f} < {recent_t} (衰减)"
    return True, "ok"


def gate4_walkforward(ic, test_t=GATE4_TEST_T_MIN, train_ratio=0.7):
    """闸门4: 时间切分 walk-forward。"""
    if len(ic) < 100:
        return False, "样本不足100日"
    split = int(len(ic) * train_ratio)
    train = ic.iloc[:split]
    test = ic.iloc[split:]
    s_test = ic_summary(test)
    if s_test["n"] < 20:
        return False, "验证窗口样本不足"
    if abs(s_test["t_stat"]) < test_t:
        return False, f"验证窗口|t|={abs(s_test['t_stat']):.1f} < {test_t}"
    return True, {"train_t": ic_summary(train)["t_stat"], "test_t": s_test["t_stat"]}


def gate5_regime(regime_ic, t_min=GATE5_T_MIN):
    """闸门5: 全 regime 显著 (简单方案=硬门槛)。"""
    for r in ["bull", "bear", "range"]:
        s = regime_ic.get(r, {"n": 0})
        if s.get("n", 0) < 10:
            return False, f"{r} regime 样本不足({s.get('n',0)})"
        if abs(s.get("t_stat", 0)) < t_min:
            return False, f"{r} regime |t|={abs(s.get('t_stat',0)):.1f} < {t_min}"
    return True, "ok"


def validate_signal(df, factor_name, horizon, t_min=GATE2_T_MIN):
    """完整验证: 五闸门。返回 report dict。"""
    from factors import compute_all_factors, add_forward_returns
    report = {"factor": factor_name, "horizon": horizon}

    # 前置: 确保因子列 + 前向收益列存在
    df = df.copy()
    if f"f_{factor_name}" not in df.columns:
        df = compute_all_factors(df, [factor_name])
    if f"fwd_{horizon}" not in df.columns:
        df = add_forward_returns(df, [horizon])

    r = compute_factor_ic(df, factor_name, horizon)
    report["ic"] = r["ic"]
    report["summary"] = r["summary"]
    report["regime_ic"] = r["regime_ic"]

    ok1, m1 = gate1_correctness(df, factor_name)
    ok2, m2 = gate2_ic_significance(r["summary"], t_min)
    ok3, m3 = gate3_stability(r["ic"], r["summary"]["sign"])
    ok4, m4 = gate4_walkforward(r["ic"])
    ok5, m5 = gate5_regime(r["regime_ic"])

    report["gates"] = {
        "1_correctness": (ok1, m1),
        "2_ic_significance": (ok2, m2),
        "3_stability": (ok3, m3),
        "4_walkforward": (ok4, m4),
        "5_regime": (ok5, m5),
    }
    report["passed"] = all([ok1, ok2, ok3, ok4, ok5])
    return report


def format_report(report):
    """格式化报告为文本。"""
    s = report["summary"]
    lines = [
        f"因子 {report['factor']} @ H={report['horizon']}天",
        f"  IC={s['mean_ic']*100:+.2f}%  t={s['t_stat']:+.1f}  n={s['n']}日  "
        f"一致性={s['ic_consistency']:.0%}",
    ]
    for g, (ok, msg) in report["gates"].items():
        icon = "✅" if ok else "❌"
        lines.append(f"  {icon} {g}: {msg}")
    lines.append(f"  总体: {'✅ 通过' if report['passed'] else '❌ 拒绝'}")
    return "\n".join(lines)
