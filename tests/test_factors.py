"""因子库测试 (无前视 + 正确性)。"""
import numpy as np
import pandas as pd

from factors import FACTORS, add_forward_returns, compute_all_factors


def test_mom_20_matches_pct_change(synthetic_bars):
    df = compute_all_factors(synthetic_bars, ["mom_20"])
    g = synthetic_bars.groupby("code")["close"].pct_change(20)
    assert np.allclose(df["f_mom_20"].fillna(-999), g.fillna(-999), atol=1e-9)


def test_all_factors_no_lookahead_last_day(synthetic_bars):
    """最后一个交易日的因子值不应受未来数据影响 (天然无未来)。"""
    df = compute_all_factors(synthetic_bars)
    last = df[df["date"] == df["date"].max()]
    for name in FACTORS:
        assert f"f_{name}" in df.columns
        assert last[f"f_{name}"].notna().all(), f"{name} 末日有 NaN"


def test_forward_return_correctness(synthetic_bars):
    df = add_forward_returns(synthetic_bars, [20])
    adj = synthetic_bars["close"] * synthetic_bars["split_factor"]
    g = pd.DataFrame({"code": synthetic_bars["code"], "adj": adj}).groupby("code")["adj"]
    expected = g.shift(-20) / adj - 1
    assert np.allclose(df["fwd_20"].fillna(-999), expected.fillna(-999), atol=1e-9)


def test_factor_has_variation(synthetic_bars):
    df = compute_all_factors(synthetic_bars, ["mom_20"])
    assert df["f_mom_20"].nunique() > 10
