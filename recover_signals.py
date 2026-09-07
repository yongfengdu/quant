"""恢复 LLM 进化发现的信号代码 (从 kimi_sessions 提取并注册)。"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import json
import re
from pathlib import Path

from signal_evolve import extract_factor_code, validate_factor_integrity
from signal_registry import SignalRegistry

SESSIONS = [
    "kimi_sessions/v3_1788761904_0/stdout.txt",
    "kimi_sessions/v3_1788761984_0/stdout.txt",
    "kimi_sessions/v3_1788762253_0/stdout.txt",
]

reg = SignalRegistry()

# 清理旧的无代码孤儿信号
for sid in list(reg.all_ids()):
    s = reg.get(sid)
    if not s.get("definition"):
        del reg.data["signals"][sid]
        print(f"删除孤儿信号 {sid}")

# 恢复
for f in SESSIONS:
    p = Path(f)
    if not p.exists():
        continue
    text = p.read_text()
    code = extract_factor_code(text)
    if not code:
        print(f"{f}: 未提取到代码")
        continue
    ok, reason = validate_factor_integrity(code)
    if not ok:
        print(f"{f}: 完整性校验失败 {reason}")
        continue
    # 用 session 时间戳做 ID
    ts = re.search(r"v3_(\d+)_", f).group(1)
    sid = f"LLM_{ts}"
    reg.register(sid, "llm_factor", 20, definition=code)
    print(f"{f} → {sid}: 已恢复代码 ({len(code)} 字符)")

reg.save()
print(f"\n注册表现有 {len(reg.all_ids())} 个信号: {reg.all_ids()}")
