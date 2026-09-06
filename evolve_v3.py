"""
路线3原型: 单Agent直接写代码 + 确定性自验证
=================================================
核心改进 (对比旧 Researcher→Engineer 双agent管线):
  1. 单个 K3 agent 同时负责"想策略"和"写代码" —— 消除翻译层断裂
  2. K3 直接输出完整可运行代码 (不是自然语言描述)
  3. 确定性回测 + Tiered Gate 验证 (LLM 不参与打分)
  4. 完整性校验: 拒绝 stub/占位符/残缺代码
  5. 通过则加入岛屿; 失败则记录简短教训 (不写长篇 LESSONS)

用法:
  python3 evolve_v3.py --island range --n 3        # 为 range 岛生成3个候选
  python3 evolve_v3.py --island bear --n 5
"""
import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path.home() / ".cache/quant-autoresearch"
KIMI_CLI = Path("/root/.kimi-code/bin/kimi")
KIMI_MODEL = os.environ.get("KIMI_CLI_MODEL", "kimi-code/k3")

REGIME_DESC = {
    "bull": "牛市 (大盘20日涨幅>3%)。动量、突破、板块龙头有效。",
    "bear": "熊市 (大盘20日跌幅>3%)。超跌反弹、跌停修复、大市值抗跌有效。",
    "range": "震荡市 (大盘20日涨跌在±3%内)。经网格搜索验证: 动量突破有效, 均值回归无效。",
}

# ---------- Tiered Gate (与全系统一致) ----------
def check_gate(trades, acc, avg):
    if avg <= 0:
        return False, "FAIL(avg<=0)"
    if acc >= 30 and trades >= 7:
        return True, "PASS(30%)"
    if acc >= 25 and trades >= 13:
        return True, "PASS(25%)"
    if acc >= 20 and trades >= 26:
        return True, "PASS(20%)"
    if acc >= 15 and trades >= 101:
        return True, "PASS(15%)"
    return False, f"FAIL(acc={acc:.1f},n={trades})"


# ---------- 数据 ----------
def load_daily():
    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    g = daily.groupby("code", sort=False)
    daily["next1_open"] = g["open"].shift(-1)
    daily["next2_open"] = g["open"].shift(-2)
    daily["target_return"] = (daily["next2_open"] - daily["next1_open"]) / daily["next1_open"]
    return daily


# ---------- 完整性校验 (堵住旧管线的 stub 漏洞) ----------
def validate_code_integrity(code: str):
    """返回 (ok, reason)。拒绝残缺/占位符代码。"""
    if "def predict_next_day" not in code:
        return False, "缺少 predict_next_day"
    # 语法检查
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    # 占位符检查 (全文,不只前50字符 —— 这是旧漏洞根源)
    placeholder_patterns = [
        r"#\s*\.\.\.\s*rest",
        r"#\s*rest of",
        r"unchanged",
        r"#\s*省略",
        r"#\s*your code here",
        r"#\s*TODO",
        r"#\s*同上",
        r"pass\s*#\s*占位",
    ]
    for pat in placeholder_patterns:
        if re.search(pat, code, re.IGNORECASE):
            return False, f"含占位符: {pat}"
    # 函数体必须有实质逻辑 (至少若干行非注释非空)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "predict_next_day":
            body_lines = len([n for n in ast.walk(node)
                              if isinstance(n, (ast.Assign, ast.Return, ast.If, ast.For, ast.Call))])
            if body_lines < 5:
                return False, f"函数体过于简单({body_lines}个语句), 疑似残缺"
    return True, "ok"


# ---------- 确定性回测 (Top5, 最近N日) ----------
def backtest_candidate(code: str, daily: pd.DataFrame, test_days=250):
    """把候选代码写入临时模块并回测。返回 metrics dict 或 (None, error)。"""
    tmp_path = PROJECT_DIR / f"_candidate_{int(time.time()*1000)}.py"
    tmp_path.write_text(code)
    try:
        spec = importlib.util.spec_from_file_location("candidate", tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predict_fn = module.predict_next_day
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return None, f"import失败: {e}"

    dates = sorted(daily["date"].unique())[-test_days:]
    date_ret = {}
    for d in dates:
        dd = daily[daily["date"] == d]
        date_ret[d] = dict(zip(dd["code"], dd["target_return"]))

    dsd = daily.sort_values("date").reset_index(drop=True)
    returns = []
    crash = None
    for d in dates:
        end = dsd["date"].searchsorted(d, side="right")
        hist = dsd.iloc[:end]
        try:
            sig = predict_fn(hist)
        except Exception as e:
            crash = f"运行崩溃@{d.date()}: {e}"
            break
        if sig is None or len(sig) == 0:
            continue
        if isinstance(sig, pd.Series):
            sig = sig[sig >= 0.3]
            if len(sig) == 0:
                continue
            top = sig.nlargest(min(5, len(sig))).index.tolist()
            rmap = date_ret[d]
            for c in top:
                r = rmap.get(str(c))
                if r is not None and pd.notna(r):
                    returns.append(r)

    tmp_path.unlink(missing_ok=True)
    if crash:
        return None, crash
    if not returns:
        return {"trades": 0, "accuracy": 0.0, "avg_return": 0.0}, None
    arr = np.array(returns)
    return {
        "trades": len(arr),
        "accuracy": float((arr > 0.03).mean() * 100),
        "avg_return": float(arr.mean() * 100),
        "win_rate": float((arr > 0).mean() * 100),
    }, None


# ---------- 已有策略摘要 (给K3看,避免重复) ----------
def summarize_existing(island: str):
    sdir = PROJECT_DIR / "islands" / f"island_{island}" / "strategies"
    lines = []
    for p in sorted(sdir.glob("*.py")):
        doc = ""
        try:
            tree = ast.parse(p.read_text())
            doc = ast.get_docstring(tree) or ""
        except Exception:
            pass
        first = doc.strip().split("\n")[0] if doc else p.stem
        lines.append(f"  - {p.stem}: {first}")
    return "\n".join(lines) if lines else "  (无)"


def get_market_feedback():
    """读取在线实盘反馈 (daily_feedback 的结论), 用于进化提示词, 关闭反馈回路。"""
    try:
        from daily_feedback import format_feedback_for_prompt
        return format_feedback_for_prompt()
    except Exception as e:
        return f"  (在线反馈获取失败: {e})"


# ---------- 构建单agent prompt ----------
def _diversity_block(island: str) -> str:
    """按岛屿差异化多样性约束。bull/range 强制趋势, bear 放宽超跌但鼓励新角度。"""
    if island in ("bull", "range"):
        return """# 策略多样性硬约束 (实证发现, 必须遵守)
- 当前策略池 12 个线上策略中 11 个是"超跌反弹/放量突破", 只有 1 个趋势类。
- 实测(300日walk-forward): 这些策略在牛市(大盘20日涨>3%)里超额命中率普遍为负, 只在震荡/下跌市有效。
- **本轮必须探索 趋势/动量 方向** (与现有策略正交), 从下列选型:
  * 创 N 日新高突破 (close 突破 20/60 日最高价)
  * 强相对强度动量 (个股20日收益全市场排名前 N%, 或 RS 相对大盘持续走强)
  * 板块轮动追涨 (强板块 + 板块内龙头, 追板块 beta)
  * 趋势跟随 (MA5>MA20>MA60 多头排列 + 平台突破)
- **禁止再产出**: 超跌反弹、低吸抄底、缩量、跌停修复 (已 11/12 饱和)。"""
    # bear 岛: 超跌反弹是熊市合理方向, 但已有 3 超跌 + 3 抗跌突破, 鼓励正交新角度
    return """# 策略多样性约束 (bear 岛)
- 超跌反弹是熊市合理方向, 但本岛已有 BEAR_001/003/004 (超跌) + BEAR_005/006/007 (抗跌突破), 共 6 个。
- **优先探索与现有正交的新角度**:
  * 恐慌极端值后的修复 (跌停潮次日 / 全市场超跌比例极端)
  * 事件驱动的错杀修复 (业绩暴雷后放量止跌)
  * 大市值避险 + 相对强度 (抗跌龙头, 弱市领涨)
- 避免与现有策略重复。"""


def build_prompt(island: str, existing: str, failed_notes: str, feedback: str = ""):
    interface = (PROJECT_DIR / "prompts" / "interface.md").read_text()
    feedback_block = ""
    if feedback.strip():
        feedback_block = f"""

# 近期在线实盘反馈 (策略正在错过这些机会, 优先补这个缺口)
{feedback}
"""
    diversity_block = _diversity_block(island)
    return f"""你是量化策略研究员兼工程师。你的任务:为【{island}市】设计一个A股T+1择时策略,并直接输出完整可运行的Python代码。

# 交易逻辑
T日收盘出信号 → T+1开盘买入 → T+2开盘卖出。目标: 预测 (T+2开盘 - T+1开盘)/T+1开盘 > 3% 的股票。

# 当前市场类型
{REGIME_DESC[island]}

# 策略接口 (必须严格遵守)
{interface}

# 已有策略 (不要重复这些方向)
{existing}

# 近期失败教训 (避免重蹈覆辙)
{failed_notes}
{feedback_block}
# 通过标准 (Tiered Gate, 必须满足其一且平均收益>0)
- 准确率≥30% 且 交易≥7笔
- 准确率≥25% 且 交易≥13笔
- 准确率≥20% 且 交易≥26笔
- 准确率≥15% 且 交易≥101笔

# 关键要求 (违反则直接失败)
1. **直接输出完整代码,不要用占位符/省略/"同上"/"unchanged"等**。所有逻辑必须完整写出。
2. 只用 pandas 和 numpy。
3. 信号强度用 0.3~1.0 之间的浮点数, 越强越高。
4. 加入大市值过滤 (circ_cap>100e8), 网格搜索证明大盘股T+1可预测性更高。
5. 避免过拟合: 不要用超过5个AND条件, 否则信号会崩溃到0。
6. 用 .fillna(False) 处理NaN, 避免运行崩溃。

{diversity_block}

# 策略设计建议 (提高通过率)
- **最容易通过的是30%档(只需7笔交易)**: 设计"高确信度、低频"策略——用较严格但不过度的条件, 每天只选最强的少数标的。宁可信号少而精, 不要多而杂。
- 趋势/动量有效因子(优先组合): 创N日新高、板块相对强度、相对强度排名、MA多头排列、放量突破平台。
- 避免的方向: 纯均值回归、低RSI超卖、缩量抄底(已被现有策略覆盖且牛市无效)。
- 信号强度要有区分度: 用因子值映射到0.3-1.0, 让最强信号排前面(回测只取Top5)。

# 输出格式
先用2-3句话说明策略核心逻辑, 然后输出一个完整的 ```python 代码块。代码块必须包含完整的 import 和 predict_next_day 函数。
"""


def extract_code(text: str):
    import textwrap
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    # 取含 predict_next_day 的最后一个块
    valid = [b for b in blocks if "def predict_next_day" in b]
    if not valid:
        return None
    code = valid[-1]
    # K3 常把代码块缩进在 bullet 下, 需去除公共前导缩进
    code = textwrap.dedent(code)
    # 若 dedent 未完全生效 (混合缩进), 按最小非空行缩进强制对齐
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if non_empty:
        indent = min(len(l) - len(l.lstrip()) for l in non_empty)
        if indent > 0:
            lines = [l[indent:] if len(l) >= indent else l for l in lines]
            code = "\n".join(lines)
    return code.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--island", required=True, choices=["bull", "bear", "range"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--test-days", type=int, default=250)
    ap.add_argument("--timeout", type=int, default=480, help="K3 单次调用超时(秒)")
    ap.add_argument("--retries", type=int, default=2, help="可恢复异常重试次数")
    ap.add_argument("--notify", action="store_true", help="运行结束推送微信汇报")
    ap.add_argument("--test-notify", action="store_true", help="汇报只打印不发送")
    args = ap.parse_args()

    from kimi_watchdog import KimiWatchdog
    import notifier

    # watchdog 带告警回调
    wd = KimiWatchdog(alert_callback=lambda msg: notifier.alert(msg, test=args.test_notify))

    # 运行前健康检查
    print("=== 运行前 Kimi/K3 健康检查 ===")
    ok, reason = wd.health_check()
    print(f"K3 服务: {'✓ 健康' if ok else '✗ 异常(' + reason + ')'}")
    if not ok:
        notifier.alert(f"进化任务启动前 K3 服务异常: {reason}, 已中止", test=args.test_notify)
        print("❌ 服务异常, 中止本次进化")
        return

    daily = load_daily()
    existing = summarize_existing(args.island)
    sdir = PROJECT_DIR / "islands" / f"island_{args.island}" / "strategies"

    feedback = get_market_feedback()
    if feedback.strip():
        print("在线市场反馈已注入提示词")
    else:
        print("(无在线市场反馈, 继续)")

    nums = [int(re.search(r"_(\d+)", p.stem).group(1))
            for p in sdir.glob(f"{args.island.upper()}_*.py")
            if re.search(r"_(\d+)", p.stem)]
    next_num = max(nums) + 1 if nums else 1

    failed_notes = "  (本次运行暂无)"
    accepted = []
    failures = []

    for i in range(args.n):
        print(f"\n{'='*60}\n候选 {i+1}/{args.n} (K3 生成中...)\n{'='*60}")
        prompt = build_prompt(args.island, existing, failed_notes, feedback)

        resp, err_type = wd.call_with_retry(prompt, timeout=args.timeout,
                                            max_retries=args.retries)
        if resp is None:
            note = f"候选{i+1}: K3调用失败({err_type})"
            print(f"  ❌ {note}")
            failures.append(note)
            # 服务级异常 (非空/超时以外) 提前中止
            if err_type in ("auth", "crash"):
                print("  ⛔ 服务级异常, 中止后续候选")
                break
            continue

        code = extract_code(resp)
        if not code:
            note = f"候选{i+1}: 未提取到代码块"
            print(f"  ❌ {note}")
            failures.append(note)
            continue

        ok, reason = validate_code_integrity(code)
        if not ok:
            note = f"候选{i+1}: 完整性校验失败({reason})"
            print(f"  ❌ {note}")
            failures.append(note)
            failed_notes = f"  - 上个候选因'{reason}'被拒, 请输出完整代码"
            continue

        print("  ✓ 完整性校验通过, 开始确定性回测...")
        metrics, err = backtest_candidate(code, daily, args.test_days)
        if err:
            note = f"候选{i+1}: 回测崩溃({err[:80]})"
            print(f"  ❌ {note}")
            failures.append(note)
            failed_notes = f"  - 上个候选运行崩溃: {err[:120]}"
            continue

        passed, gate = check_gate(metrics["trades"], metrics["accuracy"], metrics["avg_return"])
        print(f"  交易={metrics['trades']} 准确率={metrics['accuracy']:.1f}% "
              f"平均收益={metrics['avg_return']:.2f}% → {gate}")

        if passed:
            name = f"{args.island.upper()}_{next_num:03d}"
            dst = sdir / f"{name}.py"
            dst.write_text(code)
            print(f"  ✅ 通过Gate! 已保存 {name}")
            accepted.append((name, metrics, gate))
            next_num += 1
            existing = summarize_existing(args.island)
        else:
            failures.append(f"候选{i+1}: 未过Gate({gate})")
            failed_notes = (f"  - 上个候选未过Gate ({gate}): "
                            f"准确率{metrics['accuracy']:.1f}%/{metrics['trades']}笔/"
                            f"均收益{metrics['avg_return']:.2f}%. 请换正交方向或调整条件.")

    # 汇总
    print(f"\n{'='*60}\n结果: {len(accepted)}/{args.n} 通过Gate")
    for name, m, gate in accepted:
        print(f"  {name}: {gate} 准确率{m['accuracy']:.1f}% {m['trades']}笔 均收益{m['avg_return']:.2f}%")

    health = wd.get_summary()
    print(f"\nK3服务健康: {health['status']} (调用{health['total_calls']} 失败率{health['fail_rate']}%)")

    # 汇报
    if args.notify or args.test_notify:
        notifier.report_evolution(args.island, accepted, args.n, health,
                                  failures, test=args.test_notify)


if __name__ == "__main__":
    main()
