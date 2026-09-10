"""pytest 共享 fixtures。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"


def make_synthetic_bars(n_codes=5, n_days=150, seed=42):
    """生成合成 OHLCV 数据 (含 split_factor)。日期用连续自然日(无周末跳过)。"""
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(1, n_codes + 1)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d")
    rows = []
    for ci, code in enumerate(codes):
        price = 10.0 + ci * 3.0
        for d in dates:
            price = price * (1 + rng.normal(0, 0.02))
            o = price * (1 + rng.normal(0, 0.006))
            close = price
            high = max(o, close) * (1 + abs(rng.normal(0, 0.008)))
            low = min(o, close) * (1 - abs(rng.normal(0, 0.008)))
            v = rng.uniform(1e5, 1e6)
            rows.append({
                "code": code, "date": d,
                "open": o, "close": close, "high": high, "low": low,
                "volume": v, "amount": close * v * 100, "split_factor": 1.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_bars():
    return make_synthetic_bars()


@pytest.fixture
def real_bars():
    """真实缓存数据 (新 daily_bars 优先, 旧 daily 兜底), 用于黄金测试。"""
    p = CACHE_DIR / "daily_bars.parquet"
    if not p.exists():
        p = CACHE_DIR / "daily.parquet"
    if not p.exists():
        pytest.skip("无缓存数据")
    df = pd.read_parquet(p)
    df["date"] = df["date"].astype(str)
    if "split_factor" not in df.columns:
        df["split_factor"] = 1.0
    return df
