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


def _compute_signal_ic(df, s):
    """计算某信号的每日 IC 序列 (含 fwd_20)。失败返回 None。"""
    from live_strategy import _is_code_signal
    from factors import compute_all_factors
    from signal_evolve import compute_llm_factor
    from signal_lab import daily_rank_ic

    direction = s.get("direction", 1)
    if _is_code_signal(s):
        df2, err = compute_llm_factor(s["definition"], df)
        if err:
            return None
        score = direction * df2["f_llm_factor"].reset_index(drop=True)
    else:
        df2 = compute_all_factors(df, [s["factor"]])
        score = direction * df2[f"f_{s['factor']}"]
    df2 = df2.copy()
    df2["__score"] = score
    return daily_rank_ic(df2, "__score", "fwd_20", min_stocks=50)


def step_lifecycle():
    log("=== Step 2: 信号生命周期审视 ===")
    from signal_registry import SignalRegistry, Lifecycle
    from factors import add_forward_returns
    from live_strategy import _sid

    reg = SignalRegistry()
    lc = Lifecycle(reg)

    # 计算各 paper/active 信号的近期 IC (真实数据)
    df = pd.read_parquet(CACHE_DIR / "daily_bars.parquet")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df = add_forward_returns(df, [20])

    targets = [s for s in reg.data["signals"].values() if s.get("status") in ("paper", "active")]
    ic_by_signal = {}
    for s in targets:
        ic = _compute_signal_ic(df, s)
        if ic is not None:
            ic_by_signal[_sid(s)] = ic

    # 只检查 active 信号衰减; 晋升改为人工 (supervisor promote), 避免自动晋升弱信号稀释组合
    promoted, retired = [], []
    for s in targets:
        sid = _sid(s)
        if s.get("status") != "active":
            continue
        ic = ic_by_signal.get(sid)
        decayed, detail = lc.check_decay(sid, ic)
        if decayed:
            retired.append((sid, detail))

    for sid, detail in retired:
        log(f"  🔻 退役 {sid}: {detail}")
    active = reg.list_by_status("active")
    paper = reg.list_by_status("paper")
    log(f"  active={len(active)}, paper={len(paper)}(待人工晋升), 总={len(reg.all_ids())}")


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
