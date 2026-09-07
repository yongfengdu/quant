"""
信号进化 (signal_evolve.py) —— LLM 假设生成 (B方案)
====================================================
职责:
  1. 让 K3 提出"横截面因子"假设 (Python 代码)
  2. 确定性验证 (五闸门 IC 验收)
  3. 通过则注册进信号生命周期

核心原则: LLM 只提假设, 不打分。打分全由 signal_lab 的确定性 IC 计算完成。
"""
import ast
import importlib.util
import re
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factors import FACTORS
from signal_lab import validate_signal, format_report

PROJECT_DIR = Path(__file__).resolve().parent
FACTOR_NAMES = list(FACTORS.keys())

# 因子接口说明 (写进 prompt)
FACTOR_INTERFACE = """def factor(df):
    \"\"\"
    df 列: code, date, open, close, high, low, volume, amount, split_factor
    返回: pd.Series, 与 df 逐行对齐 (每个 code,date 一个分数)
          分数越高 = 越看好该股未来 20 日的相对收益
    \"\"\"
    ...
"""


def build_prompt(existing_factors, horizon=20, findings=None):
    """构造因子假设 prompt。"""
    existing_str = "\n".join(f"  - {n}" for n in existing_factors) or "  (无)"
    findings_str = findings or (
        "长周期反转: 过去20-60日涨太多的股未来会跌; 反转在熊/震荡市强、牛市弱; "
        "动量(贴20日高点)在牛市中性。"
    )
    return f"""你是量化因子研究员。为 A 股设计一个"横截面因子"(信号), 直接输出可运行 Python 代码。

# 交易/预测目标
预测未来 {horizon} 日的"相对收益"排名 (哪些股票比别的股票涨得好)。

# 因子接口 (严格遵守)
{FACTOR_INTERFACE}

# 验证方式 (确定性, 你无法干预)
- 每日横截面: 计算 因子 与 未来{horizon}日收益 的秩相关 (IC)
- 通过标准: IC 显著 + 稳定 + walk-forward 样本外有效 + 全市场状态有效
- 注意: 会按买入档(分位数)实际超额再复核

# 已有因子 (不要重复这些方向)
{existing_str}

# 已发现的市场规律 (供参考, 请探索正交的新方向)
{findings_str}

# 硬要求 (违反直接失败)
1. 完整可运行代码, 无占位符/省略/"同上"/pass
2. 只用 pandas 和 numpy
3. 用 .fillna(...) 防 NaN, 避免崩溃
4. 信号有区分度 (覆盖足够多的股票, 不要只命中极少数)

# 输出格式
先用 1-2 句话说明因子逻辑, 然后输出一个 ```python 代码块, 代码块必须包含完整 import 和 factor 函数。
"""


def extract_factor_code(text):
    """从 LLM 响应提取 factor 代码块。"""
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    valid = [b for b in blocks if "def factor" in b]
    if not valid:
        return None
    code = textwrap.dedent(valid[-1])
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if non_empty:
        indent = min(len(l) - len(l.lstrip()) for l in non_empty)
        if indent > 0:
            code = "\n".join(l[indent:] if len(l) >= indent else l for l in lines)
    return code.strip()


def validate_factor_integrity(code):
    """完整性校验: 返回 (ok, reason)。"""
    if "def factor" not in code:
        return False, "缺少 factor 函数"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    for pat in [r"#\s*\.\.\.", r"pass\s*#", r"unchanged", r"your code here", r"#\s*省略"]:
        if re.search(pat, code, re.IGNORECASE):
            return False, f"含占位符: {pat}"
    # 找 factor 函数体, 检查有实质逻辑
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "factor":
            n_stmt = len([n for n in ast.walk(node) if isinstance(n, (ast.Assign, ast.Return, ast.If, ast.Call))])
            if n_stmt < 3:
                return False, "factor 函数体过于简单"
    return True, "ok"


def compute_llm_factor(code, df):
    """导入 LLM 代码并计算因子列。返回 (df_with_col, error)。"""
    tmp = PROJECT_DIR / f"_llm_factor_{int(time.time()*1000)}.py"
    tmp.write_text(code)
    try:
        spec = importlib.util.spec_from_file_location("llm_factor", tmp)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "factor", None)
        if fn is None:
            return None, "无 factor 函数"
        df = df.copy()
        df["f_llm_factor"] = fn(df)
        if not isinstance(df["f_llm_factor"], pd.Series) or len(df["f_llm_factor"]) != len(df):
            return None, "factor 返回值非对齐 Series"
        return df, None
    except Exception as e:
        return None, f"运行失败: {e}"
    finally:
        tmp.unlink(missing_ok=True)


def discover_signals(df, n=3, horizon=20, registry=None, test=False, timeout=480):
    """进化主循环: 提 n 个假设 → 验证 → 注册。返回结果列表。"""
    from kimi_watchdog import KimiWatchdog
    from signal_registry import Lifecycle, SignalRegistry

    reg = registry or SignalRegistry()
    lc = Lifecycle(reg)
    existing = list(FACTOR_NAMES) + reg.all_ids()

    wd = KimiWatchdog()
    ok, reason = wd.health_check()
    print(f"K3 服务: {'✓ 健康' if ok else '✗ ' + str(reason)}")
    if not ok and not test:
        return []

    prompt = build_prompt(existing, horizon)
    results = []
    for i in range(n):
        print(f"\n{'='*60}\n假设 {i+1}/{n}\n{'='*60}")
        if test:
            # 测试模式: 用已知因子 mock LLM
            code = "import pandas as pd\ndef factor(df):\n    g = df.groupby('code', sort=False)\n    return g['close'].pct_change(20)\n"
        else:
            resp, err = wd.call_with_retry(prompt, timeout=timeout, max_retries=2)
            if resp is None:
                print(f"  ❌ K3 失败: {err}")
                results.append({"passed": False, "error": f"K3失败:{err}"})
                continue
            code = extract_factor_code(resp)
            if not code:
                print("  ❌ 未提取到代码块")
                results.append({"passed": False, "error": "无代码块"})
                continue

        ok, reason = validate_factor_integrity(code)
        if not ok:
            print(f"  ❌ 完整性校验失败: {reason}")
            results.append({"passed": False, "error": reason})
            continue

        df2, err = compute_llm_factor(code, df)
        if err:
            print(f"  ❌ 因子运行失败: {err}")
            results.append({"passed": False, "error": err})
            continue

        report = validate_signal(df2, "llm_factor", horizon)
        print(format_report(report))
        sid = f"LLM_{int(time.time())}"
        lc.validate_and_promote(sid, report)
        results.append({"passed": report["passed"], "sid": sid, "report": report,
                        "code": code})
    return results
