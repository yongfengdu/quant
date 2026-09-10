"""
每日流水线 (run_daily.py)
========================
主线自动化: 每天盘后自动执行
  1. 增量更新数据 (build_data.update_incremental)
  2. 信号生命周期审视 (衰减监控)
  3. 生成组合推荐 (live_strategy)

用法:
  python run_daily.py              # 完整流水线
  python run_daily.py --no-update  # 跳过数据更新
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/root/quant-autoresearch")
import pandas as pd

from data_store import audit_log, validate_bars

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"
LOG_PATH = Path("/root/quant-autoresearch") / "pipeline_daily.log"
RECOMMENDATIONS_FILE = Path("/root/quant-autoresearch") / "data" / "recommendations.csv"


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def step_update():
    log("=== Step 1: 增量更新数据 ===")
    from build_data import update_incremental
    update_incremental()
    # 完整性校验
    df = pd.read_parquet(CACHE_DIR / "daily_bars.parquet")
    issues, m = validate_bars(df, check_completeness=False)
    if issues:
        for i in issues:
            log(f"  ⚠️ {i}")
    audit_log("daily_update", n_rows=len(df), latest_date=df["date"].max())


def step_lifecycle():
    log("=== Step 2: 信号生命周期审视 ===")
    from signal_registry import SignalRegistry, Lifecycle
    reg = SignalRegistry()
    lc = Lifecycle(reg)
    retired = lc.review()  # 无新 IC 时, 用注册表里已存的
    if retired:
        for sid, detail in retired:
            log(f"  🔻 退役 {sid}: {detail}")
    active = reg.list_by_status("active")
    paper = reg.list_by_status("paper")
    log(f"  active={len(active)}, paper={len(paper)}, 总={len(reg.all_ids())}")


def step_recommend():
    log("=== Step 3: 生成组合推荐 ===")
    from live_strategy import generate_recommendation
    df = pd.read_parquet(CACHE_DIR / "daily_bars.parquet")
    rec = generate_recommendation(df)
    log(f"  信号日 {rec['date']}  市场状态 {rec['regime']}")
    # 持久化推荐 (供监控检测预测效果)
    rows = [{"signal_date": rec["date"], "regime": rec["regime"],
             "code": s["code"], "close": s["close"], "score": s["score"]}
            for s in rec["stocks"]]
    RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if RECOMMENDATIONS_FILE.exists():
        old = pd.read_csv(RECOMMENDATIONS_FILE, dtype={"code": str})
        new_df = pd.concat([old, new_df], ignore_index=True).drop_duplicates(
            subset=["signal_date", "code"], keep="last")
    new_df.to_csv(RECOMMENDATIONS_FILE, index=False)
    for i, s in enumerate(rec["stocks"][:10], 1):
        log(f"    {i}. {s['code']}  收盘 {s['close']}  分 {s['score']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args()

    log("=== 每日流水线开始 ===")
    if not args.no_update:
        step_update()
    else:
        log("跳过数据更新")
    step_lifecycle()
    step_recommend()
    log("=== 每日流水线完成 ===")


if __name__ == "__main__":
    main()
