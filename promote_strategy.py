"""
策略晋升工具 (promote_strategy.py)
====================================
把进化产出的"候选策略"晋升为"上线策略"。

流程:
  1. 确认候选文件存在于 islands/island_X/strategies/
  2. 用确定性回测 + Tiered Gate 重新验证 (双保险, 防止误上线)
  3. 通过则加入 island_router.ISLAND_STRATEGIES 上线清单
  4. 同步 state.json

用法:
  python3 promote_strategy.py --island bear --name BEAR_005          # 晋升单个
  python3 promote_strategy.py --island range --name RANGE_006 --force # 跳过回测强制晋升
  python3 promote_strategy.py --list                                  # 列出所有候选
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path.home() / ".cache/quant-autoresearch"
ROUTER_FILE = PROJECT_DIR / "island_router.py"


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


def load_daily():
    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    g = daily.groupby("code", sort=False)
    daily["next1_open"] = g["open"].shift(-1)
    daily["next2_open"] = g["open"].shift(-2)
    daily["target_return"] = (daily["next2_open"] - daily["next1_open"]) / daily["next1_open"]
    return daily


def backtest_strategy(island, name, daily, test_days=250):
    import importlib.util
    path = PROJECT_DIR / "islands" / f"island_{island}" / "strategies" / f"{name}.py"
    if not path.exists():
        return None, f"文件不存在: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        predict_fn = module.predict_next_day
    except Exception as e:
        return None, f"加载失败: {e}"

    dates = sorted(daily["date"].unique())[-test_days:]
    date_ret = {d: dict(zip(daily[daily["date"] == d]["code"],
                            daily[daily["date"] == d]["target_return"])) for d in dates}
    dsd = daily.sort_values("date").reset_index(drop=True)
    returns = []
    for d in dates:
        end = dsd["date"].searchsorted(d, side="right")
        hist = dsd.iloc[:end]
        try:
            sig = predict_fn(hist)
        except Exception as e:
            return None, f"运行崩溃@{d.date()}: {e}"
        if sig is None or len(sig) == 0:
            continue
        if isinstance(sig, pd.Series):
            sig = sig[sig >= 0.3]
            if len(sig) == 0:
                continue
            for c in sig.nlargest(min(5, len(sig))).index:
                r = date_ret[d].get(str(c))
                if r is not None and pd.notna(r):
                    returns.append(r)
    if not returns:
        return {"trades": 0, "accuracy": 0.0, "avg_return": 0.0}, None
    arr = np.array(returns)
    return {"trades": len(arr), "accuracy": float((arr > 0.03).mean() * 100),
            "avg_return": float(arr.mean() * 100)}, None


def get_island_strategies():
    """从 island_router 读当前上线清单。"""
    import importlib
    import island_router
    importlib.reload(island_router)
    return island_router.ISLAND_STRATEGIES


def list_candidates():
    live = get_island_strategies()
    print("=== 候选策略 (磁盘有但未上线) ===")
    for island in ["bull", "bear", "range"]:
        sdir = PROJECT_DIR / "islands" / f"island_{island}" / "strategies"
        active = sorted(p.stem for p in sdir.glob("*.py"))
        candidates = [s for s in active if s not in live.get(island, [])]
        print(f"\n[{island}] 上线: {live.get(island, [])}")
        if candidates:
            print(f"  候选待晋升: {candidates}")
        else:
            print("  (无候选)")


def add_to_router(island, name):
    """把策略名加入 island_router.py 的 ISLAND_STRATEGIES 字典。"""
    content = ROUTER_FILE.read_text()
    # 匹配该 island 的列表行, 例如: "bear":  ["BEAR_001", "BEAR_003", "BEAR_004"],
    pattern = rf'("{island}":\s*\[)([^\]]*)(\])'
    m = re.search(pattern, content)
    if not m:
        return False, f"未在 island_router 找到 {island} 的策略列表"
    current = m.group(2)
    if f'"{name}"' in current:
        return False, f"{name} 已在上线清单中"
    # 追加
    existing = current.rstrip().rstrip(",")
    if existing.strip():
        new_list = f'{existing}, "{name}"'
    else:
        new_list = f'"{name}"'
    new_content = content[:m.start()] + f'"{island}": [{new_list}]' + content[m.end():]
    ROUTER_FILE.write_text(new_content)
    return True, "已加入上线清单"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--island", choices=["bull", "bear", "range"])
    ap.add_argument("--name", help="策略名, 如 BEAR_005")
    ap.add_argument("--force", action="store_true", help="跳过回测强制晋升")
    ap.add_argument("--list", action="store_true", help="列出所有候选")
    ap.add_argument("--test-days", type=int, default=250)
    args = ap.parse_args()

    if args.list:
        list_candidates()
        return

    if not args.island or not args.name:
        print("需指定 --island 和 --name (或 --list 查看候选)")
        return

    # 1. 回测验证 (除非 --force)
    if not args.force:
        print(f"=== 验证 {args.name} (确定性回测+Gate) ===")
        daily = load_daily()
        metrics, err = backtest_strategy(args.island, args.name, daily, args.test_days)
        if err:
            print(f"❌ 回测失败: {err}")
            return
        passed, gate = check_gate(metrics["trades"], metrics["accuracy"], metrics["avg_return"])
        print(f"  交易={metrics['trades']} 准确率={metrics['accuracy']:.1f}% "
              f"平均收益={metrics['avg_return']:.2f}% → {gate}")
        if not passed:
            print(f"❌ 未通过 Gate, 拒绝晋升。如确需晋升用 --force")
            return
        print("  ✓ 通过 Gate")

    # 2. 加入上线清单
    ok, msg = add_to_router(args.island, args.name)
    if not ok:
        print(f"❌ {msg}")
        return
    print(f"✅ {msg}: {args.island} ← {args.name}")

    # 3. 同步 state.json
    import subprocess
    subprocess.run(["python3", "sync_island_state.py"], cwd=str(PROJECT_DIR))
    print(f"\n🎉 {args.name} 已上线! 下次预测将纳入该策略。")


if __name__ == "__main__":
    main()
