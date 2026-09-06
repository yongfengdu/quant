"""元验证测试 (合成注入, 快速版)。"""
import numpy as np
import pandas as pd

from factors import add_forward_returns
from signal_lab import validate_signal


def _inject(df, rho, seed):
    rng = np.random.default_rng(seed)
    vals = np.zeros(len(df))
    for d, sub in df.groupby("date"):
        r = sub["fwd_20"].rank(pct=True).values
        z_x = (r - 0.5) * np.sqrt(12)
        z_u = rng.standard_normal(len(sub))
        vals[sub.index] = rho * z_x + np.sqrt(max(1 - rho ** 2, 0)) * z_u
    return vals


def test_injection_separates_real_from_noise(real_bars):
    df = add_forward_returns(real_bars, [20])
    df["f_strong"] = _inject(df, 0.10, seed=7)
    df["f_noise"] = _inject(df, 0.0, seed=8)

    rep_strong = validate_signal(df, "strong", 20)
    rep_noise = validate_signal(df, "noise", 20)

    assert rep_strong["passed"], "rho=10% 应通过"
    assert not rep_noise["passed"], "rho=0% 应被拒"
    assert rep_strong["summary"]["mean_ic"] > rep_noise["summary"]["mean_ic"]
