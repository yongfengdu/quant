"""IC 引擎 + 五闸门测试 (黄金 + 负例)。"""
import numpy as np
import pandas as pd

from factors import add_forward_returns, compute_all_factors
from signal_lab import daily_rank_ic, validate_signal


def test_daily_rank_ic_perfect_signal(synthetic_bars):
    """完美单调信号 → IC 应接近 1。"""
    df = add_forward_returns(synthetic_bars, [20])
    # 构造与 fwd_20 完全同秩的信号
    df["f_perfect"] = df.groupby("date")["fwd_20"].rank(pct=True)
    ic = daily_rank_ic(df, "f_perfect", "fwd_20", min_stocks=3)
    assert ic.mean() > 0.99


def test_golden_mom_20(real_bars):
    """黄金回归: 真实数据上 mom_20 @ 20天 应显著负 IC 且通过闸门。"""
    df = add_forward_returns(real_bars, [20])
    df = compute_all_factors(df, ["mom_20"])
    rep = validate_signal(df, "mom_20", 20)
    assert rep["summary"]["mean_ic"] < -0.05, f"mom_20 IC 异常: {rep['summary']['mean_ic']:.3f}"
    assert rep["passed"], "mom_20 应通过五闸门"


def test_noise_rejected(real_bars):
    """负例: 纯噪声因子应被拒绝。"""
    rng = np.random.default_rng(42)
    df = add_forward_returns(real_bars, [20])
    df["f_noise"] = rng.random(len(df))
    rep = validate_signal(df, "noise", 20)
    assert not rep["passed"], "噪声因子不应通过"
