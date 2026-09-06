"""数据存储与校验测试。"""
import pandas as pd

from data_store import next_trading_day, validate_bars


def test_next_trading_day_holiday():
    # 模拟日历: 含国庆假期
    cal = {"2026-10-01", "2026-10-08", "2026-10-09"}  # 10-01 交易日, 10-02~07 假期, 10-08 恢复
    cal = {"2026-09-30", "2026-10-01", "2026-10-08", "2026-10-09"}
    assert next_trading_day("2026-09-30", cal) == "2026-10-01"
    # 10-01 后跳过假期直接到 10-08
    assert next_trading_day("2026-10-01", cal) == "2026-10-08"


def test_validate_bars_detects_duplicates(synthetic_bars):
    df = pd.concat([synthetic_bars, synthetic_bars.iloc[:3]]).reset_index(drop=True)
    issues, m = validate_bars(df, check_completeness=False)
    assert m["dup"] == 3
    assert any("重复" in i for i in issues)


def test_validate_bars_detects_bad_ohlc(synthetic_bars):
    df = synthetic_bars.copy()
    df.loc[df.index[0], "high"] = df.loc[df.index[0], "low"] - 1  # high < low
    issues, m = validate_bars(df, check_completeness=False)
    assert m["bad_ohlc"] >= 1
    assert any("OHLC" in i for i in issues)


def test_validate_bars_clean_data(synthetic_bars):
    issues, m = validate_bars(synthetic_bars, check_completeness=False)
    assert m["dup"] == 0 and m["bad_ohlc"] == 0 and m["bad_price"] == 0
