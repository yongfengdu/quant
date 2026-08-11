#!/usr/bin/env python3
"""
每日预测跟踪系统

功能:
1. predict:  用截止到昨天的数据, 跑策略生成今日预测
2. verify:   拉取今日收盘数据, 验证昨天的预测准确率
3. summary:  输出近N天的预测准确率统计
4. lesson:   将统计结果写入 LESSONS.md, 供 pipeline 进化使用
5. adaptive: 使用自适应策略路由器生成预测 (多策略聚合)
6. monitor:  显示策略健康状态报告

Usage:
    python daily_tracker.py predict       # 生成今日预测(收盘后跑)
    python daily_tracker.py adaptive      # 自适应多策略预测(推荐)
    python daily_tracker.py verify        # 验证昨日预测(次日收盘后跑)
    python daily_tracker.py summary       # 查看近期准确率
    python daily_tracker.py lesson        # 将统计写入 LESSONS.md
    python daily_tracker.py monitor       # 策略健康状态检查
"""

import json
import os
import sys
import re
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

CACHE_DIR = Path.home() / '.cache' / 'quant-autoresearch'
PROJECT_DIR = Path("/root/quant-autoresearch")
PROXY_URL = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", ""))

# 预测记录文件
PREDICTIONS_FILE = PROJECT_DIR / "daily_predictions.csv"
ACCURACY_LOG = PROJECT_DIR / "daily_accuracy.log"


def http_get(url, timeout=20):
    req = urllib.request.Request(url)
    req.add_header("Referer", "https://stockapp.finance.qq.com/")
    req.add_header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
    if PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler({"http": PROXY_URL, "https": PROXY_URL})
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()
    resp = opener.open(req, timeout=timeout)
    raw = resp.read()
    for enc in ["gbk", "gb18030", "utf-8"]:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ====================================================================
# 选股逻辑 (与 backtest.py 一致)
# ====================================================================

def load_data():
    universe = pd.read_parquet(CACHE_DIR / "universe.parquet")
    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    from backtest import compute_derived_columns
    daily = compute_derived_columns(daily)
    return universe, daily


def find_next_trading_days(daily, from_date, n=3):
    """从 from_date 开始往后找 n 个交易日，返回日期列表（含 from_date 本身）。
    如果数据中不够，通过 API 查询后续交易日。"""
    all_dates = sorted(daily["date"].unique())
    dates_only = pd.to_datetime(all_dates)
    pos = dates_only.searchsorted(pd.to_datetime(from_date))
    if pos < len(dates_only):
        result = dates_only[pos:pos + n].tolist()
        if len(result) >= n:
            return result
    # 数据不够：跳过周末，取后续工作日作为估计的交易日
    result = [pd.to_datetime(from_date)]
    cursor = pd.to_datetime(from_date) + pd.Timedelta(days=1)
    while len(result) < n:
        if cursor.weekday() < 5:  # 周一到周五
            result.append(cursor)
        cursor += pd.Timedelta(days=1)
        if len(result) > 30:  # 安全阀
            break
    return result


def _fallback_mean_reversion(df, n=5):
    """简单均值回归：选5日跌幅最大的n只股票"""
    grp = df.sort_values("date").groupby("code")
    latest = grp.last().reset_index()
    codes = latest["code"].values
    ret5_col = latest.get("ret_5d")
    if ret5_col is None:
        latest["ret_5d"] = grp["close"].transform(lambda x: x.pct_change(5).iloc[-1] if len(x) > 5 else 0)
        ret5_col = latest["ret_5d"]
    sorted_idx = ret5_col.dropna().sort_values().index[:n]
    # 负 return → 正信号
    ticker_codes = latest.loc[sorted_idx, "code"].values
    signals = pd.Series(
        (-ret5_col.loc[sorted_idx].values).clip(0.02, 0.5),
        index=ticker_codes
    )
    return signals


def _fallback_random(df, n=3):
    """终极保底：从当日有交易的前50%市值股票中随机抽n只"""
    latest = df[df["date"] == df["date"].max()].copy()
    if "amount" in latest.columns:
        cutoff = latest["amount"].median()
        candidates = latest[latest["amount"] >= cutoff]
    else:
        candidates = latest[latest["volume"] > 0]
    codes = candidates["code"].unique().tolist()
    import random as _rd
    _rd.seed(42)
    chosen = _rd.sample(codes, min(n, len(codes)))
    return pd.Series([0.5 + i * 0.1 for i in range(len(chosen))], index=chosen)


def run_prediction_for_date(signal_date):
    """
    用截止到 signal_date 的数据, 生成对 T+1/T+2 的预测。
    与回测使用完全相同的 universe.parquet 选股池。

    交易逻辑（与 backtest.py v4 一致）：
      T 日收盘出信号 → T+1 开盘买入 → T+2 开盘卖出

    返回: list of {code, signal, entry_open, exit_open, entry_date, exit_date, ...}
    """
    # 使用岛屿路由系统 (按市场状态路由到已验证策略池)
    # 替代原损坏的单一 strategy.py (HH587 stub)
    from island_router import predict_next_day

    universe_df, daily = load_data()
    universe_codes = set(universe_df["code"].values)

    daily_cut = daily[(daily["date"] <= signal_date) &
                      (daily["code"].isin(universe_codes))].copy()
    if len(daily_cut) == 0:
        print(f"ERROR: No data on or before {signal_date.date()}")
        return []

    # 找 T+1 和 T+2
    next_dates = find_next_trading_days(daily, signal_date, 3)
    if len(next_dates) < 3:
        print(f"WARNING: Not enough future trading data after {signal_date.date()}")
        return []
    entry_date, exit_date = next_dates[1], next_dates[2]
    sector_map = dict(zip(universe_df["code"], universe_df["sector"]))

    # 构建历史数据（仅 universe 内的股票）
    hist_data = []
    for code in universe_codes & set(daily_cut["code"].unique()):
        cd = daily_cut[daily_cut["code"] == code]
        if len(cd) > 0:
            hist_data.append(cd)

    if not hist_data:
        print("ERROR: No historical data")
        return []

    all_hist = pd.concat(hist_data, ignore_index=True)

    try:
        signals = predict_next_day(all_hist)
    except Exception as e:
        print(f"ERROR: Strategy failed: {e}")
        signals = pd.Series(dtype=float)

    # 回退链：主策略 0 信号 → H175 骨架 → 简单均值回归 → 随机
    if signals is None or (isinstance(signals, pd.Series) and len(signals) == 0):
        print("  ⚠️ 主策略 0 信号，尝试 H175 骨架回退...")
        try:
            import strategy_h175_backup
            signals = strategy_h175_backup.predict_next_day(all_hist)
        except Exception:
            signals = pd.Series(dtype=float)

    if signals is None or (isinstance(signals, pd.Series) and len(signals) == 0):
        print("  ⚠️ H175 回退 0 信号，尝试简单均值回归...")
        signals = _fallback_mean_reversion(all_hist, n=5)

    if signals is None or (isinstance(signals, pd.Series) and len(signals) == 0):
        print("  ⚠️ 均值回归 0 信号，尝试随机保底...")
        signals = _fallback_random(all_hist, n=3)

    top_signals = signals.nlargest(min(3, len(signals)))

    # 读取 T+1 / T+2 开盘价（未来日期的数据还不存在，预测时跳过）
    entry_data = daily[daily["date"] == entry_date].set_index("code") if entry_date in daily["date"].values else pd.DataFrame()
    exit_data = daily[daily["date"] == exit_date].set_index("code") if exit_date in daily["date"].values else pd.DataFrame()

    latest = daily_cut[daily_cut["date"] == signal_date]
    results = []
    for code, signal in top_signals.items():
        code = str(code)
        info = latest[latest["code"] == code]
        row = info.iloc[0] if len(info) > 0 else None
        entry_open = float(entry_data.loc[code, "open"]) if code in entry_data.index else None
        exit_open = float(exit_data.loc[code, "open"]) if code in exit_data.index else None
        results.append({
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "code": code,
            "signal": round(float(signal), 6),
            "close": round(float(row["close"]), 2) if row is not None else 0.0,
            "sector": sector_map.get(str(code), "其他"),
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "entry_open": entry_open,
            "exit_open": exit_open,
            "actual_return": None,
            "hit_3pct": None,
            "any_gain": None,
        })

    return results


# ====================================================================
# 预测命令
# ====================================================================

def cmd_predict():
    """用昨天收盘数据生成今日预测"""
    print("=" * 60)
    print("🔮 生成预测")
    print("=" * 60)

    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    latest_date = daily["date"].max()
    print(f"最新数据日期: {latest_date.date()}")

    # 用最新日期作为信号日, 预测次日
    predictions = run_prediction_for_date(latest_date)

    if not predictions:
        print("策略没有发出信号")
        return

    # 追加写入
    if PREDICTIONS_FILE.exists():
        existing = pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})
        key = f"{predictions[0]['signal_date']}-{predictions[0]['entry_date']}"
        existing_keys = set()
        if len(existing) > 0:
            for _, r in existing.iterrows():
                existing_keys.add(f"{r['signal_date']}-{r['entry_date']}")
        if key in existing_keys:
            print("今日预测已存在, 跳过")
            return
        existing = pd.concat([existing, pd.DataFrame(predictions)], ignore_index=True)
        existing.to_csv(PREDICTIONS_FILE, index=False)
    else:
        pd.DataFrame(predictions).to_csv(PREDICTIONS_FILE, index=False)

    print(f"\n信号日: {latest_date.date()} → 买入日: {predictions[0]['entry_date']} → 卖出日: {predictions[0]['exit_date']}")
    print(f"推荐买入（T+1 开盘买入，T+2 开盘卖出）:\n")
    for i, p in enumerate(predictions, 1):
        print(f"  {i}. {p['code']} | 板块: {p['sector']} | "
              f"信号日收盘: {p['close']} | 入场开盘: {p['entry_open']} | 信号: {p['signal']:.4f}")

    print(f"\n已保存到: {PREDICTIONS_FILE}")


# ====================================================================
# 验证命令
# ====================================================================

def cmd_verify():
    """验证昨日预测（需 T+2 开盘价数据已到位）"""
    print("=" * 60)
    print("✅ 验证预测（T+2 开盘卖出验证）")
    print("=" * 60)

    if not PREDICTIONS_FILE.exists():
        print("没有预测记录")
        return

    predictions = pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})
    for col in ["hit_3pct", "any_gain"]:
        if col in predictions.columns:
            predictions[col] = predictions[col].astype("object")
    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    latest_date = daily["date"].max()
    print(f"最新数据日期: {latest_date.date()}")

    # 找 exit_date == latest_date 的预测（T+2 已到，可验证）
    exit_str = latest_date.strftime("%Y-%m-%d")
    to_verify = predictions[(predictions["exit_date"] == exit_str) &
                            (predictions["actual_return"].isna())].copy()

    if len(to_verify) == 0:
        print(f"没有待验证的预测（exit_date={exit_str}）")
        pending = predictions[predictions["actual_return"].isna()]
        if len(pending) > 0:
            print(f"\n待验证的预测:")
            for _, p in pending.head(10).iterrows():
                print(f"  {p['signal_date']} → 入场:{p['entry_date']} 出场:{p['exit_date']}: {p['code']}")
            if len(pending) > 10:
                print(f"  ... +{len(pending)-10} 条")
        return

    # 计算实际收益: (exit_open - entry_open) / entry_open
    exit_data = daily[daily["date"] == latest_date].set_index("code")
    
    # 获取 entry_date 的开盘价数据（修复：从 daily 获取而非 CSV）
    entry_date_str = to_verify["entry_date"].iloc[0] if len(to_verify) > 0 else None
    entry_data = pd.DataFrame()
    if entry_date_str:
        entry_date = pd.to_datetime(entry_date_str)
        entry_data = daily[daily["date"] == entry_date].set_index("code")
    
    results = []
    for _, p in to_verify.iterrows():
        code = str(p["code"])
        if code not in exit_data.index:
            continue
        actual_exit_open = exit_data.loc[code, "open"]
        
        # 优先从 daily 获取 entry_open，回退到 CSV 中的值
        if code in entry_data.index:
            entry_open = float(entry_data.loc[code, "open"])
        else:
            entry_open = p["entry_open"]
        
        actual_return = (actual_exit_open - entry_open) / entry_open if entry_open and entry_open > 0 else 0
        hit = actual_return > 0.03
        p["entry_open"] = round(float(entry_open), 2) if entry_open else None
        p["exit_open"] = round(float(actual_exit_open), 2)
        p["actual_return"] = round(float(actual_return), 6)
        p["hit_3pct"] = hit
        p["any_gain"] = actual_return > 0
        results.append(p)

    # 更新 CSV
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    for _, row in results_df.iterrows():
        mask = (predictions["signal_date"] == row["signal_date"]) & \
               (predictions["code"] == row["code"])
        predictions.loc[mask, "entry_open"] = row["entry_open"]
        predictions.loc[mask, "exit_open"] = row["exit_open"]
        predictions.loc[mask, "actual_return"] = row["actual_return"]
        predictions.loc[mask, "hit_3pct"] = row["hit_3pct"]
        predictions.loc[mask, "any_gain"] = row["any_gain"]

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    # 输出结果
    hits = sum(1 for r in results if r["hit_3pct"])
    gains = sum(1 for r in results if r["any_gain"])
    total = len(results)

    print(f"\n验证 {latest_date.date()} (卖出日):")
    print(f"  预测 {total} 只, 涨幅>3%: {hits} 只, 上涨: {gains} 只")
    print()

    for r in results:
        status = "✅+3%" if r["hit_3pct"] else ("✅涨" if r["any_gain"] else "❌跌")
        print(f"  {r['code']} | 板块: {r['sector']} | "
              f"入场:{r['entry_open']} → 出场:{r['exit_open']} | "
              f"{r['actual_return']*100:+.2f}% | {status} | "
              f"信号: {r['signal']:.4f}")

    # 追加到准确率日志
    with open(ACCURACY_LOG, "a") as f:
        f.write(f"{latest_date.strftime('%Y-%m-%d')}\t{total}\t{hits}\t{gains}\n")

    # 更新策略监控器 (如果有来源信息)
    try:
        from infra.strategy_pool import StrategyPool, StrategyMonitor
        pool = StrategyPool()
        pool.load_verified_strategies()
        monitor = StrategyMonitor(pool)
        
        for r in results:
            sources = r.get('sources', '')
            if sources:
                # 解析来源策略
                source_list = sources.split(',')
                for sid in source_list:
                    sid = sid.strip()
                    if sid:
                        monitor.record_outcome(
                            strategy_id=sid,
                            date=r['signal_date'],
                            code=r['code'],
                            actual_return=r['actual_return']
                        )
        
        # 检查是否有告警
        report = monitor.health_check()
        problem_count = report['status_summary']['warning'] + report['status_summary']['suspended']
        if problem_count > 0:
            print(f"\n⚠️ 策略健康警告: {problem_count}个策略需要关注")
            print("  运行 'python daily_tracker.py monitor' 查看详情")
    except Exception as e:
        # 监控器失败不影响主流程
        pass

    print(f"\n已更新: {PREDICTIONS_FILE}")
    print(f"准确率日志: {ACCURACY_LOG}")


# ====================================================================
# 统计命令
# ====================================================================

def cmd_summary(n_days=30):
    """输出近 N 天的预测准确率"""
    print("=" * 60)
    print("📊 预测准确率统计")
    print("=" * 60)

    if not PREDICTIONS_FILE.exists():
        print("没有预测记录")
        return

    predictions = pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})

    # 需要有实际结果的
    verified = predictions[predictions["actual_return"].notna()].copy()

    if len(verified) == 0:
        print("没有已验证的预测")
        print(f"\n总预测数: {len(predictions)}")
        pending = predictions[predictions["actual_return"].isna()]
        if len(pending) > 0:
            print(f"待验证: {len(pending)}")
        return

    # 按卖出日分组统计
    verified["exit_date"] = pd.to_datetime(verified["exit_date"])

    if n_days:
        cutoff = verified["exit_date"].max() - timedelta(days=n_days)
        verified = verified[verified["exit_date"] >= cutoff]

    daily_stats = verified.groupby("exit_date").agg(
        total=("code", "count"),
        hits_3pct=("hit_3pct", "sum"),
        any_gain=("any_gain", "sum"),
        avg_return=("actual_return", "mean"),
    ).reset_index()

    total_all = len(verified)
    hits_all = verified["hit_3pct"].sum()
    gains_all = verified["any_gain"].sum()

    print(f"\n已验证: {total_all} 条预测")
    print(f"  涨幅>3%: {hits_all} ({hits_all/total_all*100:.1f}%)")
    print(f"  上涨: {gains_all} ({gains_all/total_all*100:.1f}%)")
    print(f"  平均收益: {verified['actual_return'].mean()*100:+.2f}%")
    print(f"  中位数收益: {verified['actual_return'].median()*100:+.2f}%")

    print(f"\n每日统计:")
    print(f"{'日期':<12} {'预测':>4} {'>3%':>4} {'上涨':>4} {'>3%率':>6} {'上涨率':>6} {'均收益':>7}")
    print("-" * 50)
    for _, row in daily_stats.iterrows():
        d = row["exit_date"].strftime("%Y-%m-%d")
        t = int(row["total"])
        h = int(row["hits_3pct"])
        g = int(row["any_gain"])
        print(f"{d:<12} {t:>4} {h:>4} {g:>4} "
              f"{h/t*100:>5.1f}% {g/t*100:>5.1f}% "
              f"{row['avg_return']*100:+6.2f}%")

    # 板块统计
    sector_stats = verified.groupby("sector").agg(
        total=("code", "count"),
        hits_3pct=("hit_3pct", "sum"),
        any_gain=("any_gain", "sum"),
        avg_return=("actual_return", "mean"),
    ).reset_index()
    sector_stats = sector_stats.sort_values("avg_return", ascending=False)

    print(f"\n板块统计:")
    print(f"{'板块':<10} {'预测':>4} {'>3%':>4} {'上涨':>4} {'均收益':>7}")
    print("-" * 35)
    for _, row in sector_stats.iterrows():
        t = int(row["total"])
        h = int(row["hits_3pct"])
        g = int(row["any_gain"])
        print(f"{row['sector']:<10} {t:>4} {h:>4} {g:>4} "
              f"{row['avg_return']*100:+6.2f}%")


# ====================================================================
# Lesson 命令
# ====================================================================

def cmd_lesson():
    """将预测准确率统计写入 LESSONS.md"""
    if not PREDICTIONS_FILE.exists():
        print("没有预测记录")
        return

    predictions = pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})
    verified = predictions[predictions["actual_return"].notna()]

    if len(verified) < 5:
        print(f"验证样本不足 ({len(verified)}), 至少需要5条")
        return

    lessions_file = PROJECT_DIR / "LESSONS.md"

    # 统计
    total = len(verified)
    hits = verified["hit_3pct"].sum()
    gains = verified["any_gain"].sum()
    avg_ret = verified["actual_return"].mean()

    # 板块表现
    sector_stats = verified.groupby("sector").agg(
        hits=("hit_3pct", "sum"),
        total=("code", "count"),
        avg_return=("actual_return", "mean"),
    ).reset_index()
    sector_stats["rate"] = sector_stats["hits"] / sector_stats["total"]
    sector_stats = sector_stats.sort_values("avg_return", ascending=False)

    best_sector = sector_stats.iloc[0] if len(sector_stats) > 0 else None
    worst_sector = sector_stats.iloc[-1] if len(sector_stats) > 0 else None

    lesson_text = f"""
---
## 预测准确率追踪 (更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')})
- 总预测: {total} 条 | 涨幅>3%: {hits} ({hits/total*100:.1f}%) | 上涨: {gains} ({gains/total*100:.1f}%) | 均收益: {avg_ret*100:+.2f}%
- 最佳板块: {best_sector['sector'] if best_sector else 'N/A'} ({best_sector['avg_return']*100:+.2f}%, {int(best_sector['hits'])}/{int(best_sector['total'])})
- 最差板块: {worst_sector['sector'] if worst_sector else 'N/A'} ({worst_sector['avg_return']*100:+.2f}%, {int(worst_sector['hits'])}/{int(worst_sector['total'])})
- 信号质量评估: {"信号质量可接受" if avg_ret > 0 else "信号质量偏低，需要改进策略"}

"""

    lessions_file.write_text(lessions_file.read_text() + lesson_text)
    print(f"已写入 LESSONS.md")
    print(lesson_text.strip())


# ====================================================================
# 自适应路由器命令
# ====================================================================

def cmd_adaptive(mode='weighted'):
    """
    使用岛屿路由系统生成预测 (已切换,替代旧 AdaptiveRouter)

    特点:
    1. 按市场状态(bull/bear/range)路由到对应岛屿
    2. 只使用通过 Tiered Gate 的已验证策略
    3. 多策略信号取并集,同股取最高分
    """
    print("=" * 60)
    print("🏝️  岛屿路由预测系统")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_DIR))
    import island_router

    # 加载数据
    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    # 加载universe过滤
    universe_df = pd.read_parquet(CACHE_DIR / "universe.parquet")
    universe_codes = set(universe_df["code"].values)
    daily_filtered = daily[daily["code"].isin(universe_codes)].copy()

    # 删除尾部不完整交易日 (盘中/脏数据), 保证信号日与数据一致
    daily_filtered = island_router.drop_incomplete_days(
        daily_filtered.sort_values(["code", "date"]))
    latest_date = daily_filtered["date"].max()
    print(f"最新数据日期: {latest_date.date()}")

    # 岛屿路由信号
    signals, regime, mkt_ret = island_router.predict_with_meta(daily_filtered)

    print(f"\n市场状态: {regime}  (20日涨跌: {mkt_ret*100:+.1f}%)")
    print(f"激活岛屿: island_{regime}")
    print(f"参与策略: {island_router.ISLAND_STRATEGIES.get(regime, [])}")
    print(f"信号数: {len(signals)}")

    # 转成 final 格式 (top 3)
    final = []
    if signals is not None and len(signals) > 0:
        for code, s in signals.head(3).items():
            final.append({'code': str(code), 'weight': float(s), 'sources': [regime]})

    if not final:
        print("\n⚠️ 岛屿策略无信号 (当前市场状态下无符合条件标的)")
        return
    
    # 找 T+1 和 T+2
    next_dates = find_next_trading_days(daily, latest_date, 3)
    entry_date, exit_date = next_dates[1], next_dates[2]
    sector_map = dict(zip(universe_df["code"], universe_df["sector"]))
    
    # 构建预测结果
    predictions = []
    print(f"\n信号日: {latest_date.date()} → 买入日: {entry_date.date()} → 卖出日: {exit_date.date()}")
    print(f"推荐买入（按聚合权重排序）:\n")
    
    entry_data = daily[daily["date"] == entry_date].set_index("code")
    
    for i, sig in enumerate(final[:3], 1):
        code = str(sig['code'])
        latest_row = daily_filtered[(daily_filtered["date"] == latest_date) & 
                                     (daily_filtered["code"] == code)]
        close = float(latest_row["close"].iloc[0]) if len(latest_row) > 0 else 0.0
        entry_open = float(entry_data.loc[code, "open"]) if code in entry_data.index else None
        
        sources = sig.get('sources', [])[:3]
        n_sources = sig.get('n_sources', len(sources))
        
        print(f"  {i}. {code} | 板块: {sector_map.get(code, '其他')} | "
              f"收盘: {close:.2f} | 聚合权重: {sig['weight']:.2f} | "
              f"来源({n_sources}): {', '.join(sources)}")
        
        predictions.append({
            "signal_date": latest_date.strftime("%Y-%m-%d"),
            "code": code,
            "signal": round(float(sig['weight']), 6),
            "close": round(close, 2),
            "sector": sector_map.get(code, "其他"),
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "entry_open": entry_open,
            "exit_open": None,
            "actual_return": None,
            "hit_3pct": None,
            "any_gain": None,
            "sources": ','.join(sources),
            "n_sources": n_sources,
        })
    
    # 保存预测
    if predictions:
        if PREDICTIONS_FILE.exists():
            existing = pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})
            key = f"{predictions[0]['signal_date']}-{predictions[0]['entry_date']}"
            existing_keys = set()
            if len(existing) > 0:
                for _, r in existing.iterrows():
                    existing_keys.add(f"{r['signal_date']}-{r['entry_date']}")
            if key in existing_keys:
                print("\n今日预测已存在, 跳过保存")
                return
            existing = pd.concat([existing, pd.DataFrame(predictions)], ignore_index=True)
            existing.to_csv(PREDICTIONS_FILE, index=False)
        else:
            pd.DataFrame(predictions).to_csv(PREDICTIONS_FILE, index=False)
        
        print(f"\n已保存到: {PREDICTIONS_FILE}")


# ====================================================================
# 策略监控命令
# ====================================================================

def cmd_monitor():
    """显示策略健康状态报告"""
    print("=" * 60)
    print("策略健康监控")
    print("=" * 60)
    
    try:
        from infra.strategy_pool import StrategyPool, StrategyMonitor
        
        pool = StrategyPool()
        n = pool.load_verified_strategies()
        print(f"\n策略池: {n}个策略")
        
        monitor = StrategyMonitor(pool)
        report = monitor.health_check()
        
        print(monitor.format_report(report))
        
        # 详细信息
        print("\n" + "=" * 60)
        print("策略详情")
        print("=" * 60)
        
        for s in report['strategies']:
            status_icon = {"active": "✓", "warning": "⚠", "suspended": "✗"}.get(s['status'], "?")
            acc_str = f"{s['recent_accuracy']:.1f}%" if s['recent_accuracy'] is not None else "N/A"
            baseline_str = f"{s['baseline_accuracy']:.1f}%"
            
            print(f"\n{status_icon} {s['id']}:")
            print(f"  状态: {s['status']}")
            print(f"  基线准确率: {baseline_str}")
            print(f"  近期准确率: {acc_str}")
            print(f"  记录数: {s['n_outcomes']}")
            print(f"  连续失败: {s['consecutive_fails']}")
            print(f"  CUSUM: {s['cusum_neg']:.2f}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


# ====================================================================
# 考评报告命令 (report)
# ====================================================================

def cmd_report(period='all'):
    """
    生成系统考评报告
    
    包含:
    1. 核心KPI - 累计收益、胜率、夏普比率
    2. 风控指标 - 最大回撤、连续亏损、单日最大亏损
    3. 趋势分析 - 周度表现、近期 vs 历史
    4. 板块分析 - 各板块胜率和收益
    5. 决策建议 - 是否该继续/暂停
    
    Args:
        period: 'all', 'week', 'month'
    """
    print("=" * 70)
    print("📊 量化系统考评报告")
    print("=" * 70)
    
    if not PREDICTIONS_FILE.exists():
        print("没有预测记录")
        return
    
    predictions = pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})
    
    # 过滤已验证且有效的记录
    verified = predictions[
        predictions["actual_return"].notna() & 
        predictions["entry_open"].notna()
    ].copy()
    
    if len(verified) == 0:
        print("没有已验证的预测")
        return
    
    verified["exit_date"] = pd.to_datetime(verified["exit_date"])
    verified["signal_date"] = pd.to_datetime(verified["signal_date"])
    verified = verified.sort_values("exit_date")
    
    # 按周期过滤
    if period == 'week':
        cutoff = verified["exit_date"].max() - timedelta(days=7)
        verified = verified[verified["exit_date"] >= cutoff]
        period_name = "近一周"
    elif period == 'month':
        cutoff = verified["exit_date"].max() - timedelta(days=30)
        verified = verified[verified["exit_date"] >= cutoff]
        period_name = "近一月"
    else:
        period_name = "全部"
    
    if len(verified) == 0:
        print(f"{period_name}没有已验证的预测")
        return
    
    print(f"\n统计周期: {period_name}")
    print(f"数据范围: {verified['exit_date'].min().date()} ~ {verified['exit_date'].max().date()}")
    print(f"验证样本: {len(verified)} 条")
    
    # ===== 1. 核心KPI =====
    print("\n" + "=" * 70)
    print("一、核心KPI")
    print("=" * 70)
    
    hits_3pct = (verified["hit_3pct"] == True).sum()
    wins = (verified["any_gain"] == True).sum()
    avg_ret = verified["actual_return"].mean()
    median_ret = verified["actual_return"].median()
    
    # 累计收益（按日等权：同一天多只股票分散投资）
    daily_ret = verified.groupby("exit_date")["actual_return"].mean()
    cumulative_ret = (1 + daily_ret).prod() - 1
    n_trading_days = len(daily_ret)
    
    # 夏普比率（按日计算，年化）
    if daily_ret.std() > 0:
        # 假设一年约250个交易日
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(250)
    else:
        sharpe = 0
    
    # 盈亏比
    wins_data = verified[verified["actual_return"] > 0]["actual_return"]
    losses_data = verified[verified["actual_return"] < 0]["actual_return"]
    avg_win = wins_data.mean() if len(wins_data) > 0 else 0
    avg_loss = abs(losses_data.mean()) if len(losses_data) > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    baseline_accuracy = 0.1014  # 随机选股 3% 命中率
    
    print(f"\n┌{'─'*30}┬{'─'*20}┬{'─'*15}┐")
    print(f"│ {'指标':<28} │ {'数值':>18} │ {'评价':>13} │")
    print(f"├{'─'*30}┼{'─'*20}┼{'─'*15}┤")
    
    # 3%命中率
    hit_rate = hits_3pct / len(verified)
    hit_eval = "✅ 优秀" if hit_rate > 0.2 else ("✅ 达标" if hit_rate > baseline_accuracy else "❌ 低于基线")
    print(f"│ {'3%命中率':<26} │ {hit_rate*100:>16.1f}% │ {hit_eval:>11} │")
    
    # 胜率
    win_rate = wins / len(verified)
    win_eval = "✅ 优秀" if win_rate > 0.6 else ("✅ 达标" if win_rate > 0.5 else "⚠️ 偏低")
    print(f"│ {'胜率(上涨)':<26} │ {win_rate*100:>16.1f}% │ {win_eval:>11} │")
    
    # 平均收益
    ret_eval = "✅ 盈利" if avg_ret > 0 else "❌ 亏损"
    print(f"│ {'平均收益':<26} │ {avg_ret*100:>+15.2f}% │ {ret_eval:>11} │")
    
    # 累计收益
    cum_eval = "✅ 盈利" if cumulative_ret > 0 else "❌ 亏损"
    print(f"│ {'累计收益':<26} │ {cumulative_ret*100:>+15.2f}% │ {cum_eval:>11} │")
    
    # 夏普比率
    sharpe_eval = "✅ 优秀" if sharpe > 1.5 else ("✅ 达标" if sharpe > 0.5 else "⚠️ 偏低")
    print(f"│ {'夏普比率(年化)':<26} │ {sharpe:>18.2f} │ {sharpe_eval:>11} │")
    
    # 盈亏比
    pf_eval = "✅ 优秀" if profit_factor > 1.5 else ("✅ 达标" if profit_factor > 1 else "⚠️ 偏低")
    pf_str = f"{profit_factor:.2f}" if profit_factor < 100 else "∞"
    print(f"│ {'盈亏比':<26} │ {pf_str:>18} │ {pf_eval:>11} │")
    
    print(f"└{'─'*30}┴{'─'*20}┴{'─'*15}┘")
    
    # ===== 2. 风控指标 =====
    print("\n" + "=" * 70)
    print("二、风控指标")
    print("=" * 70)
    
    # 最大回撤（按日计算）
    daily_ret_sorted = daily_ret.sort_index()
    cumulative_values = (1 + daily_ret_sorted).cumprod()
    peak = cumulative_values.expanding().max()
    drawdown = (cumulative_values - peak) / peak
    max_drawdown = drawdown.min()
    
    # 连续亏损/盈利（按日计算）
    max_consecutive_loss = 0
    max_consecutive_win = 0
    current_loss = 0
    current_win = 0
    for ret in daily_ret_sorted:
        if ret < 0:
            current_loss += 1
            current_win = 0
            max_consecutive_loss = max(max_consecutive_loss, current_loss)
        else:
            current_win += 1
            current_loss = 0
            max_consecutive_win = max(max_consecutive_win, current_win)
    
    # 单笔最大
    max_single_loss = verified["actual_return"].min()
    max_single_win = verified["actual_return"].max()
    
    print(f"\n{'指标':<25} {'数值':>15} {'阈值':>15} {'状态':>10}")
    print("-" * 70)
    
    # 最大回撤
    dd_status = "✅ 正常" if max_drawdown > -0.10 else ("⚠️ 警告" if max_drawdown > -0.20 else "❌ 危险")
    print(f"{'最大回撤':<23} {max_drawdown*100:>+14.2f}% {'<-10%':>15} {dd_status:>10}")
    
    # 连续亏损
    loss_status = "✅ 正常" if max_consecutive_loss <= 5 else ("⚠️ 警告" if max_consecutive_loss <= 7 else "❌ 危险")
    print(f"{'最大连续亏损':<23} {max_consecutive_loss:>15}次 {'≤5次':>15} {loss_status:>10}")
    
    # 单笔最大亏损
    single_status = "✅ 正常" if max_single_loss > -0.05 else ("⚠️ 警告" if max_single_loss > -0.08 else "❌ 危险")
    print(f"{'单笔最大亏损':<23} {max_single_loss*100:>+14.2f}% {'>-5%':>15} {single_status:>10}")
    
    # 单笔最大盈利
    print(f"{'单笔最大盈利':<23} {max_single_win*100:>+14.2f}%")
    
    # 最大连续盈利
    print(f"{'最大连续盈利':<23} {max_consecutive_win:>15}次")
    
    # ===== 3. 趋势分析 =====
    print("\n" + "=" * 70)
    print("三、趋势分析（周度）")
    print("=" * 70)
    
    verified["week"] = verified["exit_date"].dt.strftime("%Y-W%W")
    weekly = verified.groupby("week").agg(
        n=("actual_return", "count"),
        wins=("any_gain", "sum"),
        hits=("hit_3pct", "sum"),
        total_ret=("actual_return", "sum"),
        avg_ret=("actual_return", "mean"),
    ).reset_index()
    
    print(f"\n{'周':<10} {'样本':>6} {'胜率':>8} {'3%命中':>8} {'周收益':>10} {'均收益':>10}")
    print("-" * 60)
    for _, row in weekly.iterrows():
        wr = row["wins"] / row["n"] * 100 if row["n"] > 0 else 0
        hr = row["hits"] / row["n"] * 100 if row["n"] > 0 else 0
        trend = "📈" if row["total_ret"] > 0 else "📉"
        print(f"{row['week']:<10} {int(row['n']):>6} {wr:>7.1f}% {hr:>7.1f}% "
              f"{row['total_ret']*100:>+9.2f}% {row['avg_ret']*100:>+9.2f}% {trend}")
    
    # 近期 vs 历史对比
    if len(verified) >= 10:
        recent = verified.tail(10)
        historical = verified.head(len(verified) - 10)
        
        recent_wr = (recent["any_gain"] == True).mean()
        hist_wr = (historical["any_gain"] == True).mean() if len(historical) > 0 else 0
        recent_ret = recent["actual_return"].mean()
        hist_ret = historical["actual_return"].mean() if len(historical) > 0 else 0
        
        print(f"\n近期(最近10条) vs 历史对比:")
        trend_wr = "↑" if recent_wr > hist_wr else ("↓" if recent_wr < hist_wr else "→")
        trend_ret = "↑" if recent_ret > hist_ret else ("↓" if recent_ret < hist_ret else "→")
        print(f"  胜率: {recent_wr*100:.1f}% vs {hist_wr*100:.1f}% {trend_wr}")
        print(f"  均收益: {recent_ret*100:+.2f}% vs {hist_ret*100:+.2f}% {trend_ret}")
    
    # ===== 4. 板块分析 =====
    print("\n" + "=" * 70)
    print("四、板块分析")
    print("=" * 70)
    
    sector_stats = verified.groupby("sector").agg(
        n=("actual_return", "count"),
        wins=("any_gain", "sum"),
        hits=("hit_3pct", "sum"),
        avg_ret=("actual_return", "mean"),
        total_ret=("actual_return", "sum"),
    ).reset_index()
    sector_stats["win_rate"] = sector_stats["wins"] / sector_stats["n"]
    sector_stats = sector_stats.sort_values("avg_ret", ascending=False)
    
    print(f"\n{'板块':<12} {'样本':>6} {'胜率':>8} {'3%命中':>8} {'均收益':>10} {'累计收益':>10}")
    print("-" * 65)
    for _, row in sector_stats.iterrows():
        wr = row["win_rate"] * 100
        hr = row["hits"] / row["n"] * 100 if row["n"] > 0 else 0
        grade = "⭐" if row["avg_ret"] > 0.01 else ("" if row["avg_ret"] > 0 else "⚠️")
        print(f"{row['sector']:<10} {int(row['n']):>6} {wr:>7.1f}% {hr:>7.1f}% "
              f"{row['avg_ret']*100:>+9.2f}% {row['total_ret']*100:>+9.2f}% {grade}")
    
    # ===== 5. 决策建议 =====
    print("\n" + "=" * 70)
    print("五、决策建议")
    print("=" * 70)
    
    # 评分系统
    score = 0
    reasons = []
    
    # 核心指标评分
    if hit_rate > baseline_accuracy:
        score += 2
        reasons.append(f"✅ 3%命中率({hit_rate*100:.1f}%)超过基线({baseline_accuracy*100:.1f}%)")
    else:
        score -= 2
        reasons.append(f"❌ 3%命中率({hit_rate*100:.1f}%)低于基线({baseline_accuracy*100:.1f}%)")
    
    if win_rate > 0.5:
        score += 1
        reasons.append(f"✅ 胜率{win_rate*100:.1f}%超过50%")
    else:
        score -= 1
        reasons.append(f"⚠️ 胜率{win_rate*100:.1f}%低于50%")
    
    if avg_ret > 0:
        score += 2
        reasons.append(f"✅ 平均收益为正({avg_ret*100:+.2f}%)")
    else:
        score -= 2
        reasons.append(f"❌ 平均收益为负({avg_ret*100:+.2f}%)")
    
    if cumulative_ret > 0:
        score += 1
        reasons.append(f"✅ 累计收益为正({cumulative_ret*100:+.2f}%)")
    else:
        score -= 1
        reasons.append(f"❌ 累计收益为负({cumulative_ret*100:+.2f}%)")
    
    # 风控指标评分
    if max_drawdown > -0.10:
        score += 1
        reasons.append(f"✅ 最大回撤可控({max_drawdown*100:.1f}%)")
    else:
        score -= 1
        reasons.append(f"⚠️ 最大回撤较大({max_drawdown*100:.1f}%)")
    
    if max_consecutive_loss <= 5:
        score += 1
    else:
        score -= 1
        reasons.append(f"⚠️ 连续亏损次数较多({max_consecutive_loss}次)")
    
    print(f"\n综合评分: {score}/8")
    print()
    for r in reasons:
        print(f"  {r}")
    
    # 最终建议
    print()
    if score >= 5:
        print("📗 建议: 系统表现良好，可继续运行")
    elif score >= 2:
        print("📙 建议: 系统表现一般，建议观察并优化")
    elif score >= 0:
        print("📙 建议: 系统表现较弱，建议减小仓位或暂停")
    else:
        print("📕 建议: 系统表现不佳，建议暂停并检查策略")
    
    # 样本量警告
    if len(verified) < 30:
        print(f"\n⚠️ 注意: 当前样本量({len(verified)})较少，统计结果可能不稳定，建议积累更多数据后再做决策")


# ====================================================================
# Main
# ====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python daily_tracker.py [predict|adaptive|verify|summary|lesson|monitor|report]")
        return

    cmd = sys.argv[1]
    if cmd == "predict":
        cmd_predict()
    elif cmd == "adaptive":
        mode = sys.argv[2] if len(sys.argv) > 2 else 'weighted'
        cmd_adaptive(mode)
    elif cmd == "verify":
        cmd_verify()
    elif cmd == "summary":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        cmd_summary(n)
    elif cmd == "lesson":
        cmd_lesson()
    elif cmd == "monitor":
        cmd_monitor()
    elif cmd == "report":
        period = sys.argv[2] if len(sys.argv) > 2 else 'all'
        cmd_report(period)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
