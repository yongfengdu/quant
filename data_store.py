"""
数据存储与校验层 (data_store.py)
================================
职责:
  1. 交易日历 (单一真相源, 从 AkShare 官方日历拉取)
  2. ingest 完整性校验 (唯一性 / OHLC 合理性 / 完整性 / 连续性)
  3. append-only 审计日志 (JSON Lines)

解决的老问题:
  - 日期对不上 → 交易日历统一对齐
  - 数据丢失 → 完整性/连续性校验
  - 完整性出错 → 唯一性 + OHLC 合理性校验
"""
import json
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"
CALENDAR_PATH = CACHE_DIR / "trading_calendar.parquet"
AUDIT_PATH = CACHE_DIR / "data_audit.jsonl"


# ---------- 交易日历 ----------

def build_trading_calendar(retries=5):
    """从 AkShare 拉官方交易日历 (含法定节假日/调休)。返回 DataFrame[date, is_trading]。"""
    import akshare as ak
    for i in range(retries):
        try:
            cal = ak.tool_trade_date_hist_sina()
            cal = cal.rename(columns={"trade_date": "date"})
            cal["date"] = pd.to_datetime(cal["date"]).dt.strftime("%Y-%m-%d")
            cal["is_trading"] = 1
            return cal[["date", "is_trading"]].sort_values("date").reset_index(drop=True)
        except Exception:
            time.sleep(4)
    raise RuntimeError("交易日历拉取失败")


def save_trading_calendar():
    """构建并保存交易日历。"""
    cal = build_trading_calendar()
    cal.to_parquet(CALENDAR_PATH, index=False)
    return cal


def load_trading_calendar():
    """加载交易日历 (不存在则先构建)。返回 set of 'YYYY-MM-DD'。"""
    if not CALENDAR_PATH.exists():
        cal = save_trading_calendar()
    else:
        cal = pd.read_parquet(CALENDAR_PATH)
    return set(cal["date"].tolist())


def next_trading_day(date, cal=None, n=1):
    """给定日期, 返回其后第 n 个交易日 (节假日感知)。date: 'YYYY-MM-DD'。"""
    cal = cal if cal is not None else load_trading_calendar()
    days = sorted(cal)
    date = str(pd.to_datetime(date).date())
    pos = 0
    for i, d in enumerate(days):
        if d > date:
            pos = i
            break
        pos = i
    idx = min(pos + n - 1, len(days) - 1)
    return days[idx] if days[idx] > date else None


# ---------- 完整性校验 ----------

def validate_bars(df, check_completeness=True):
    """校验 daily_bars。返回 (issues: list[str], metrics: dict)。"""
    issues = []
    metrics = {}

    # 1. 唯一性
    dup = int(df.duplicated(subset=["code", "date"]).sum())
    metrics["dup"] = dup
    if dup:
        issues.append(f"重复(code,date) {dup} 条")

    # 2. OHLC 合理性
    ohlc = df[["open", "close", "high", "low"]]
    bad_ohlc = int(((df["high"] < ohlc[["open", "close"]].max(axis=1)) |
                    (df["low"] > ohlc[["open", "close"]].min(axis=1)) |
                    (df["high"] < df["low"])).sum())
    metrics["bad_ohlc"] = bad_ohlc
    if bad_ohlc:
        issues.append(f"OHLC 不合理 {bad_ohlc} 条")

    bad_price = int((ohlc <= 0).any(axis=1).sum())
    metrics["bad_price"] = bad_price
    if bad_price:
        issues.append(f"非正价格 {bad_price} 条")

    # 3. split_factor 合理性 (应 >= 1 且单调不减)
    bad_factor = int((df["split_factor"] < 1).sum())
    metrics["bad_factor"] = bad_factor
    if bad_factor:
        issues.append(f"split_factor < 1 的 {bad_factor} 条")

    # 4. 完整性 (每日股票数 vs 典型数)
    counts = df.groupby("date")["code"].nunique().sort_index()
    typical = counts.median()
    metrics["n_days"] = len(counts)
    metrics["typical_count"] = int(typical)
    if check_completeness and typical > 0:
        incomplete = counts[counts < typical * 0.9]
        if len(incomplete):
            issues.append(f"不完整交易日 {len(incomplete)} 个 (股票数<90%典型, 如 {incomplete.index[0]} 仅 {incomplete.iloc[0]} 只)")

    # 5. 连续性 (对照交易日历)
    cal = load_trading_calendar()
    data_dates = set(df["date"].tolist())
    data_min, data_max = df["date"].min(), df["date"].max()
    missing = [d for d in sorted(cal) if data_min <= d <= data_max and d not in data_dates]
    metrics["missing_days"] = len(missing)
    if missing:
        issues.append(f"缺失交易日 {len(missing)} 个 (如 {missing[0]})")
    # 非交易日出现在数据里
    extra = [d for d in data_dates if d not in cal]
    metrics["non_trading_days"] = len(extra)
    if extra:
        issues.append(f"非交易日数据 {len(extra)} 个 (如 {extra[0]})")

    return issues, metrics


# ---------- 审计日志 ----------

def audit_log(event, **kwargs):
    """append-only 审计 (JSON Lines)。"""
    rec = {"ts": pd.Timestamp.now().isoformat(), "event": event}
    rec.update(kwargs)
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec
