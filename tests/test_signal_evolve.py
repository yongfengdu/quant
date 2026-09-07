"""信号进化 (LLM 假设生成) 测试。"""
import pandas as pd

from signal_evolve import (
    build_prompt,
    compute_llm_factor,
    extract_factor_code,
    validate_factor_integrity,
)


GOOD_CODE = """import pandas as pd
def factor(df):
    \"\"\"20日反转\"\"\"
    g = df.groupby('code', sort=False)
    return g['close'].pct_change(20)
"""


def test_extract_factor_code():
    resp = f"这是20日反转因子\n```python\n{GOOD_CODE}\n```"
    code = extract_factor_code(resp)
    assert code is not None and "def factor" in code


def test_extract_no_code():
    assert extract_factor_code("没有代码") is None


def test_integrity_good():
    ok, reason = validate_factor_integrity(GOOD_CODE)
    assert ok, reason


def test_integrity_stub():
    ok, reason = validate_factor_integrity("def factor(df):\n    pass  # 占位\n")
    assert not ok


def test_compute_llm_factor(synthetic_bars):
    df2, err = compute_llm_factor(GOOD_CODE, synthetic_bars)
    assert err is None
    assert "f_llm_factor" in df2.columns
    assert df2["f_llm_factor"].notna().sum() > 0


def test_build_prompt_contains_existing():
    p = build_prompt(["mom_20", "dist_ma60"], 20)
    assert "mom_20" in p and "def factor" in p
