"""
岛屿路由预测器 (island_router.py)
================================
根据市场状态(regime)路由到对应岛屿的已验证策略池,聚合信号输出。

只使用通过 Tiered Gate 的策略:
  - Bull  市场: BULL_002
  - Bear  市场: BEAR_001, BEAR_003, BEAR_004
  - Range 市场: RANGE_001, RANGE_002, RANGE_005

市场状态判定 (基于 mkt_ret_20d):
  > +3%  → bull
  < -3%  → bear
  否则   → range

聚合方式: 各策略输出信号后按信号强度取并集,同一股票取最高分。
"""
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
ISLANDS_DIR = PROJECT_DIR / "islands"

# 只列已通过 Gate 的策略 (2026-08 网格搜索验证)
ISLAND_STRATEGIES = {
    "bull": ["BULL_002"],
    "bear": ["BEAR_001", "BEAR_003", "BEAR_004"],
    "range": ["RANGE_001", "RANGE_002", "RANGE_005"],
}

# 缓存已加载的策略函数
_STRATEGY_CACHE = {}


def _load_strategy(island: str, name: str):
    key = f"{island}/{name}"
    if key in _STRATEGY_CACHE:
        return _STRATEGY_CACHE[key]
    path = ISLANDS_DIR / f"island_{island}" / "strategies" / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"island_{island}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "predict_next_day", None)
    _STRATEGY_CACHE[key] = fn
    return fn


def drop_incomplete_days(df: pd.DataFrame, min_ratio: float = 0.5) -> pd.DataFrame:
    """删除尾部不完整的交易日 (盘中/脏数据)。
    若某日股票数 < 历史典型股票数的 min_ratio, 视为不完整并删除。
    """
    if df.empty:
        return df
    counts = df.groupby("date")["code"].nunique().sort_index()
    if len(counts) < 5:
        return df
    # 用近期中位数作为"完整日"基准
    typical = counts.tail(20).median()
    threshold = typical * min_ratio
    # 从尾部往前删不完整日
    good_dates = counts[counts >= threshold].index
    if len(good_dates) == 0:
        return df
    latest_good = good_dates.max()
    return df[df["date"] <= latest_good].copy()


def detect_regime(df: pd.DataFrame) -> str:
    """基于 mkt_ret_20d 判定市场状态。缺失则用中位数收益回退计算。"""
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    mkt_ret = None
    if "mkt_ret_20d" in df.columns and not latest["mkt_ret_20d"].isna().all():
        val = latest["mkt_ret_20d"].iloc[0]
        # 脏数据过滤: 20日大盘收益超过±50%几乎不可能, 视为脏数据
        if pd.notna(val) and abs(val) < 0.5:
            mkt_ret = float(val)

    if mkt_ret is None:
        # 回退: 用全市场中位数 close 的20日收益
        mkt = df.groupby("date")["close"].median().sort_index()
        if len(mkt) >= 21:
            mkt_ret = float(mkt.iloc[-1] / mkt.iloc[-21] - 1)
        else:
            mkt_ret = 0.0

    if mkt_ret > 0.03:
        return "bull", mkt_ret
    elif mkt_ret < -0.03:
        return "bear", mkt_ret
    return "range", mkt_ret


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    岛屿路由主入口。与单策略 predict_next_day 签名兼容。
    返回: pd.Series(index=code, value=signal_strength)
    """
    df = df.sort_values(["code", "date"]).copy()
    # 先删除尾部不完整交易日 (盘中/脏数据), 避免污染信号和市场判断
    df = drop_incomplete_days(df)
    regime, mkt_ret = detect_regime(df)

    strategy_names = ISLAND_STRATEGIES.get(regime, [])

    # 收集各策略信号,取并集,同股取最高分
    all_signals = {}
    for name in strategy_names:
        fn = _load_strategy(regime, name)
        if fn is None:
            continue
        try:
            sig = fn(df)
        except Exception:
            continue
        if sig is None or len(sig) == 0:
            continue
        if isinstance(sig, pd.Series):
            sig = sig[sig >= 0.3]
            for code, val in sig.items():
                code = str(code)
                if code not in all_signals or val > all_signals[code]:
                    all_signals[code] = float(val)

    if not all_signals:
        return pd.Series(dtype=float)

    return pd.Series(all_signals).sort_values(ascending=False)


def predict_with_meta(df: pd.DataFrame):
    """返回 (signals, regime, mkt_ret) 供调试/日志用。"""
    df_clean = drop_incomplete_days(df.sort_values(["code", "date"]).copy())
    regime, mkt_ret = detect_regime(df_clean)
    signals = predict_next_day(df)
    return signals, regime, mkt_ret
