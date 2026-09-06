"""信号生命周期测试。"""
import numpy as np
import pandas as pd

from signal_registry import Lifecycle, SignalRegistry


def _fake_report(passed=True):
    return {
        "factor": "test", "horizon": 20,
        "summary": {"n": 300, "mean_ic": -0.05, "std_ic": 0.04, "t_stat": -20.0, "sign": -1.0, "ic_consistency": 0.7},
        "regime_ic": {r: {"mean_ic": -0.05, "t_stat": -5.0, "n": 100} for r in ["bull", "bear", "range"]},
        "gates": {g: (True, "ok") for g in ["1_correctness", "2_ic_significance", "3_stability", "4_walkforward", "5_regime"]},
        "passed": passed,
    }


def _healthy_ic(n=200, sign=-1):
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.Series(sign * 0.05 + np.random.default_rng(1).normal(0, 0.03, n), index=dates)


def test_lifecycle_flow(tmp_path):
    reg = SignalRegistry(tmp_path / "reg.json")
    lc = Lifecycle(reg)

    ok, msg = lc.validate_and_promote("sig1", _fake_report(True))
    assert ok and reg.get("sig1")["status"] == "paper"

    ic = _healthy_ic()
    ok, msg = lc.promote_paper_to_active("sig1", ic)
    assert ok and reg.get("sig1")["status"] == "active"


def test_decay_retires_after_strikes(tmp_path):
    reg = SignalRegistry(tmp_path / "reg.json")
    lc = Lifecycle(reg)
    lc.validate_and_promote("sig1", _fake_report(True))
    lc.promote_paper_to_active("sig1", _healthy_ic())

    # 衰减 IC (近期 t 不显著)
    dates = pd.date_range("2024-01-01", periods=100, freq="D").strftime("%Y-%m-%d")
    ic_decay = pd.Series(np.random.default_rng(2).normal(0, 0.05, 100), index=dates)

    decayed = False
    for _ in range(3):
        d, _ = lc.check_decay("sig1", ic_decay)
        if d:
            decayed = True
            break
    assert decayed and reg.get("sig1")["status"] in ("decayed",)


def test_reject_failed_signal(tmp_path):
    reg = SignalRegistry(tmp_path / "reg.json")
    lc = Lifecycle(reg)
    ok, msg = lc.validate_and_promote("sig_bad", _fake_report(False))
    assert not ok and reg.get("sig_bad")["status"] == "retired"
