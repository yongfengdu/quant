"""
线上预测健康检查 (predict_health.py)
======================================
在每日预测前后做健康检查, 异常时告警。检查项:
  1. 数据新鲜度: daily.parquet 最新完整交易日是否够新 (不落后过多)
  2. 数据完整度: 最新交易日股票数是否接近universe (防盘中脏数据)
  3. 信号产出: 岛屿路由是否能出信号 (连续空信号告警)
  4. 市场状态合理性: mkt_ret_20d 是否在合理范围 (防脏数据污染)

用法:
  python3 predict_health.py           # 检查并打印
  python3 predict_health.py --notify  # 异常时推送微信
返回码: 0=健康, 1=有警告, 2=严重异常
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path.home() / ".cache/quant-autoresearch"


def check_health():
    """返回 (level, issues, info)。level: 0健康/1警告/2严重。"""
    issues = []
    info = {}
    level = 0

    # 加载数据
    try:
        daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
        daily["date"] = pd.to_datetime(daily["date"])
    except Exception as e:
        return 2, [f"无法读取 daily.parquet: {e}"], {}

    try:
        universe = pd.read_parquet(CACHE_DIR / "universe.parquet")
        n_universe = len(universe)
    except Exception:
        n_universe = 550

    # --- 1. 数据完整度 (最新日股票数) ---
    counts = daily.groupby("date")["code"].nunique().sort_index()
    latest_date = counts.index[-1]
    latest_count = counts.iloc[-1]
    typical = counts.tail(20).median()
    info["latest_date"] = str(latest_date.date())
    info["latest_count"] = int(latest_count)
    info["typical_count"] = int(typical)

    if latest_count < typical * 0.5:
        issues.append(f"⚠️ 最新日({latest_date.date()})仅{latest_count}只股票"
                      f"(正常~{int(typical)}), 疑似盘中/脏数据")
        level = max(level, 1)
        # 找最近的完整日
        complete = counts[counts >= typical * 0.5]
        effective_date = complete.index[-1] if len(complete) else latest_date
    else:
        effective_date = latest_date
    info["effective_date"] = str(effective_date.date())

    # --- 2. 数据新鲜度 ---
    today = pd.Timestamp.now().normalize()
    days_behind = (today - effective_date).days
    info["days_behind"] = days_behind
    # 考虑周末: 落后>4天才告警
    if days_behind > 4:
        issues.append(f"🔴 数据落后 {days_behind} 天 (最新完整日 {effective_date.date()})")
        level = max(level, 2)
    elif days_behind > 2:
        issues.append(f"⚠️ 数据落后 {days_behind} 天")
        level = max(level, 1)

    # --- 3. 市场状态合理性 + 信号产出 ---
    try:
        import island_router
        df = daily[daily["code"].isin(set(universe["code"]))].copy() if n_universe else daily
        signals, regime, mkt_ret = island_router.predict_with_meta(df)
        info["regime"] = regime
        info["mkt_ret_20d"] = round(mkt_ret * 100, 1)
        info["n_signals"] = len(signals)

        if abs(mkt_ret) > 0.5:
            issues.append(f"🔴 市场20日收益异常({mkt_ret*100:.0f}%), 疑似脏数据污染")
            level = max(level, 2)

        if len(signals) == 0:
            issues.append(f"⚠️ 当前市场({regime})无信号产出")
            level = max(level, 1)
    except Exception as e:
        issues.append(f"🔴 岛屿路由执行失败: {e}")
        level = max(level, 2)

    return level, issues, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notify", action="store_true", help="异常时推送微信")
    ap.add_argument("--test-notify", action="store_true")
    args = ap.parse_args()

    level, issues, info = check_health()

    print("=== 线上预测健康检查 ===")
    print(f"最新数据日: {info.get('latest_date')} ({info.get('latest_count')}只)")
    print(f"有效交易日: {info.get('effective_date')} (落后{info.get('days_behind')}天)")
    print(f"市场状态: {info.get('regime')} ({info.get('mkt_ret_20d')}%)")
    print(f"信号数: {info.get('n_signals')}")
    print()

    status = {0: "✅ 健康", 1: "⚠️ 警告", 2: "🔴 严重异常"}[level]
    print(f"总体: {status}")
    if issues:
        print("问题:")
        for i in issues:
            print(f"  {i}")

    # 告警
    if level >= 1 and (args.notify or args.test_notify):
        import notifier
        msg = f"线上预测健康检查 {status}\n数据日:{info.get('effective_date')} 市场:{info.get('regime')} 信号:{info.get('n_signals')}\n"
        msg += "\n".join(issues)
        notifier.alert(msg, test=args.test_notify)

    sys.exit(level)


if __name__ == "__main__":
    main()
