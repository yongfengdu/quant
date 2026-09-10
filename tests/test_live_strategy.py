"""线上策略测试 (regime 门控)。"""
import pandas as pd
import pytest

from live_strategy import (
    detect_regime,
    generate_recommendation,
    run_backtest,
)
from portfolio import metrics


def test_detect_regime():
    assert detect_regime(0.05) == "bull"
    assert detect_regime(-0.05) == "bear/range"
    assert detect_regime(0.01) == "bear/range"


def test_golden_backtest(real_bars):
    """黄金回归: regime 门控策略应 Sharpe ~0.84, 超额 ~+11%。"""
    r = run_backtest(real_bars)
    m = metrics(r)
    assert m["sharpe"] > 0.7, f"Sharpe 异常: {m['sharpe']:.2f}"
    assert m["ann_excess"] > 0.05, f"超额异常: {m['ann_excess']*100:.1f}%"
    assert m["n_periods"] > 50


def test_generate_recommendation(real_bars):
    rec = generate_recommendation(real_bars, top_n=20)
    assert rec["regime"] in ("bull", "bear/range")
    assert len(rec["stocks"]) == 20
    assert all("code" in s and "close" in s for s in rec["stocks"])
