"""
岛屿状态同步 (sync_island_state.py)
=====================================
消除"三处真相源不一致"问题:
  磁盘 active .py 文件 / island_router.ISLAND_STRATEGIES / state.json

以 island_router.ISLAND_STRATEGIES (上线清单) + 磁盘实际文件 为权威,
自动重写各 island 的 state.json, 保证监控/进化看到的是真实状态。

用法: python3 sync_island_state.py
"""
import json
from pathlib import Path

from island_router import ISLAND_STRATEGIES

PROJECT_DIR = Path(__file__).resolve().parent
ISLANDS_DIR = PROJECT_DIR / "islands"


def list_active_strategies(island: str):
    """磁盘上实际存在的 active 策略 (排除 archive)。"""
    sdir = ISLANDS_DIR / f"island_{island}" / "strategies"
    if not sdir.exists():
        return []
    return sorted(p.stem for p in sdir.glob("*.py"))


def sync():
    for island in ["bull", "bear", "range"]:
        state_file = ISLANDS_DIR / f"island_{island}" / "state.json"
        active = list_active_strategies(island)
        live = ISLAND_STRATEGIES.get(island, [])  # 上线清单
        candidates = [s for s in active if s not in live]  # 候选(未上线)

        # 读旧state保留其他字段
        old = {}
        if state_file.exists():
            try:
                old = json.loads(state_file.read_text())
            except Exception:
                pass

        new_state = {
            "regime": island,
            "live_strategies": live,          # 实际路由上线的
            "candidate_strategies": candidates,  # 磁盘有但未上线(进化产出待验证)
            "all_active_files": active,       # 磁盘所有active文件
            "rounds_since_new": old.get("rounds_since_new", 0),
            "last_sync": __import__("datetime").datetime.now().isoformat(),
        }
        state_file.write_text(json.dumps(new_state, ensure_ascii=False, indent=2))
        print(f"[{island}] 上线={len(live)} 候选={len(candidates)} 磁盘={len(active)}")
        if candidates:
            print(f"    候选待验证: {candidates}")


if __name__ == "__main__":
    print("=== 同步岛屿状态 (以 ISLAND_STRATEGIES + 磁盘为准) ===")
    sync()
    print("完成。state.json 已与实际路由对齐。")
