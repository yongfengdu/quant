"""
P2 元验证 (meta_validate.py)
============================
验证"验证协议本身"是否可靠:
  1. 合成注入: 构造已知 IC 的信号 + 噪声, 测五闸门精度/召回
  2. 衰减标定: 注入已知衰减率, 测闸门3能否及时拦截衰减信号

关键问题: 样本量大(549股×597日)时, 统计显著性(gate2)是否过松?
"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import numpy as np
import pandas as pd

from factors import add_forward_returns
from signal_lab import validate_signal, format_report, compute_factor_ic


def load_df():
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily.parquet")
    df["date"] = df["date"].astype(str)
    df["split_factor"] = 1.0
    return add_forward_returns(df, [20])


def inject_signal(df, rho, seed, decay_date=None):
    """构造日 IC ≈ rho 的信号列。
    rho: 目标横截面秩相关 (0~1)
    decay_date: 给定则此日之后 rho 降为 0 (模拟衰减)
    """
    rng = np.random.default_rng(seed)
    vals = np.zeros(len(df))
    for d, sub in df.groupby("date"):
        r = sub["fwd_20"].rank(pct=True).values
        z_x = (r - 0.5) * np.sqrt(12)          # 标准化 rank
        z_u = rng.standard_normal(len(sub))    # 噪声
        rho_eff = 0.0 if (decay_date and d >= decay_date) else rho
        val = rho_eff * z_x + np.sqrt(max(1 - rho_eff**2, 0)) * z_u
        vals[sub.index] = val
    return vals


def main():
    df = load_df()
    dates = sorted(df["date"].unique())
    print("=" * 70)
    print("1. 合成注入: 不同已知 IC 的信号过闸门")
    print("=" * 70)
    results = []
    for rho in [0.0, 0.005, 0.01, 0.03, 0.05, 0.10]:
        df["f_synth"] = inject_signal(df, rho, seed=100 + int(rho * 1000))
        rep = validate_signal(df, "synth", 20)
        s = rep["summary"]
        gates = rep["gates"]
        passed = rep["passed"]
        results.append((rho, s["mean_ic"], s["t_stat"], passed))
        gate_str = "".join("✓" if gates[g][0] else "✗" for g in gates)
        print(f"  rho={rho*100:5.1f}%  实测IC={s['mean_ic']*100:+6.2f}%  "
              f"t={s['t_stat']:+6.1f}  闸门[{gate_str}]  {'通过' if passed else '拒绝'}")

    # 精度/召回 (以 rho>=3% 为"真信号", rho<3% 为"噪声")
    print("\n2. 精度/召回 (真信号 = rho>=3%)")
    tp = fp = fn = tn = 0
    for rho, _, _, passed in results:
        real = rho >= 0.03
        if real and passed: tp += 1
        if real and not passed: fn += 1
        if not real and passed: fp += 1
        if not real and not passed: tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  精度(过闸且真/过闸) = {precision:.0%}")
    print(f"  召回(过闸且真/真)   = {recall:.0%}")

    # 3. 衰减标定
    print("\n3. 衰减标定: rho=10% 信号在中途衰减")
    mid = dates[len(dates) // 2]
    df["f_decay"] = inject_signal(df, 0.10, seed=7, decay_date=mid)
    rep = validate_signal(df, "decay", 20)
    print(f"  衰减点: {mid}")
    print(format_report(rep))
    # 滚 IC 查看衰减
    ic = rep["ic"]
    print(f"  前段IC均值={ic[ic.index < mid].mean()*100:+.2f}%  后段IC均值={ic[ic.index >= mid].mean()*100:+.2f}%")


if __name__ == "__main__":
    main()
