#!/usr/bin/env python3
"""
量化系统周报生成器

生成精简的微信友好格式报告，通过 hermes 发送到微信。

Usage:
    python weekly_report.py              # 生成并发送周报
    python weekly_report.py --test       # 仅生成不发送
    python weekly_report.py --period=month  # 生成月报
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).parent
PREDICTIONS_FILE = PROJECT_DIR / "daily_predictions.csv"
HERMES_BIN = "/root/.local/bin/hermes"
WEIXIN_TARGET = "weixin:o9cq801nD9gYMZnU-b7HdGqc8upc@im.wechat"


def load_verified_data(period='week'):
    """加载已验证的预测数据"""
    if not PREDICTIONS_FILE.exists():
        return None
    
    predictions = pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})
    
    # 过滤已验证且有效的记录
    verified = predictions[
        predictions["actual_return"].notna() & 
        predictions["entry_open"].notna()
    ].copy()
    
    if len(verified) == 0:
        return None
    
    verified["exit_date"] = pd.to_datetime(verified["exit_date"])
    verified = verified.sort_values("exit_date")
    
    # 按周期过滤
    if period == 'week':
        cutoff = verified["exit_date"].max() - timedelta(days=7)
        verified = verified[verified["exit_date"] >= cutoff]
    elif period == 'month':
        cutoff = verified["exit_date"].max() - timedelta(days=30)
        verified = verified[verified["exit_date"] >= cutoff]
    
    return verified


def generate_report(period='week'):
    """生成报告文本"""
    verified = load_verified_data(period)
    
    period_name = {"week": "周报", "month": "月报", "all": "总报"}[period]
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    lines = []
    lines.append(f"[ 量化系统{period_name} ]")
    lines.append(f"{date_str}")
    lines.append("")
    
    if verified is None or len(verified) == 0:
        lines.append("暂无已验证的预测数据")
        return "\n".join(lines)
    
    # 基础统计
    n = len(verified)
    date_range = f"{verified['exit_date'].min().strftime('%m/%d')}-{verified['exit_date'].max().strftime('%m/%d')}"
    
    hits_3pct = (verified["hit_3pct"] == True).sum()
    wins = (verified["any_gain"] == True).sum()
    avg_ret = verified["actual_return"].mean()
    
    # 按日等权计算累计收益
    daily_ret = verified.groupby("exit_date")["actual_return"].mean()
    cumulative_ret = (1 + daily_ret).prod() - 1
    n_days = len(daily_ret)
    
    # 夏普比率
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(250)
    else:
        sharpe = 0
    
    # 最大回撤
    daily_ret_sorted = daily_ret.sort_index()
    cumulative_values = (1 + daily_ret_sorted).cumprod()
    peak = cumulative_values.expanding().max()
    drawdown = (cumulative_values - peak) / peak
    max_drawdown = drawdown.min()
    
    # 连续亏损
    max_consecutive_loss = 0
    current_loss = 0
    for ret in daily_ret_sorted:
        if ret < 0:
            current_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, current_loss)
        else:
            current_loss = 0
    
    # 盈亏比
    wins_data = verified[verified["actual_return"] > 0]["actual_return"]
    losses_data = verified[verified["actual_return"] < 0]["actual_return"]
    avg_win = wins_data.mean() if len(wins_data) > 0 else 0
    avg_loss = abs(losses_data.mean()) if len(losses_data) > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # 核心指标
    lines.append(f"-- 核心指标 --")
    lines.append(f"周期: {date_range} ({n_days}个交易日)")
    lines.append(f"样本: {n}条预测")
    lines.append("")
    
    hit_rate = hits_3pct / n * 100
    win_rate = wins / n * 100
    baseline = 10.14
    
    hit_emoji = "V" if hit_rate > baseline else "X"
    win_emoji = "V" if win_rate > 50 else "X"
    ret_emoji = "V" if avg_ret > 0 else "X"
    
    lines.append(f"{hit_emoji} 3%命中: {hit_rate:.1f}% (基线{baseline:.1f}%)")
    lines.append(f"{win_emoji} 胜率: {win_rate:.1f}%")
    lines.append(f"{ret_emoji} 平均收益: {avg_ret*100:+.2f}%")
    lines.append(f"  累计收益: {cumulative_ret*100:+.2f}%")
    lines.append(f"  夏普比率: {sharpe:.2f}")
    lines.append(f"  盈亏比: {profit_factor:.2f}" if profit_factor < 100 else f"  盈亏比: INF")
    lines.append("")
    
    # 风控指标
    lines.append(f"-- 风控指标 --")
    dd_emoji = "V" if max_drawdown > -0.10 else "!"
    loss_emoji = "V" if max_consecutive_loss <= 5 else "!"
    lines.append(f"{dd_emoji} 最大回撤: {max_drawdown*100:.2f}%")
    lines.append(f"{loss_emoji} 最大连亏: {max_consecutive_loss}天")
    lines.append("")
    
    # 板块表现（前3好 + 前2差）
    sector_stats = verified.groupby("sector").agg(
        n=("actual_return", "count"),
        avg_ret=("actual_return", "mean"),
    ).reset_index()
    sector_stats = sector_stats[sector_stats["n"] >= 2]  # 至少2条
    sector_stats = sector_stats.sort_values("avg_ret", ascending=False)
    
    if len(sector_stats) > 0:
        lines.append(f"-- 板块表现 --")
        # 最好的
        for _, row in sector_stats.head(3).iterrows():
            emoji = "^" if row["avg_ret"] > 0 else "v"
            lines.append(f"{emoji} {row['sector']}: {row['avg_ret']*100:+.2f}% ({int(row['n'])})")
        # 最差的
        if len(sector_stats) > 3:
            lines.append("...")
            for _, row in sector_stats.tail(2).iterrows():
                emoji = "^" if row["avg_ret"] > 0 else "v"
                lines.append(f"{emoji} {row['sector']}: {row['avg_ret']*100:+.2f}% ({int(row['n'])})")
        lines.append("")
    
    # 综合评分
    score = 0
    if hit_rate > baseline:
        score += 2
    if win_rate > 50:
        score += 1
    if avg_ret > 0:
        score += 2
    if cumulative_ret > 0:
        score += 1
    if max_drawdown > -0.10:
        score += 1
    if max_consecutive_loss <= 5:
        score += 1
    
    lines.append(f"-- 综合评分 --")
    lines.append(f"得分: {score}/8")
    
    if score >= 5:
        lines.append("建议: 继续运行")
    elif score >= 2:
        lines.append("建议: 观察优化")
    else:
        lines.append("建议: 考虑暂停")
    
    # 样本量警告
    if n < 30:
        lines.append("")
        lines.append(f"! 样本量({n})较少，统计不稳定")
    
    return "\n".join(lines), score


def send_to_weixin(message, test=False):
    """通过 hermes 发送到微信"""
    if test:
        print("=" * 50)
        print("[TEST MODE - 不发送]")
        print("=" * 50)
        print(message)
        print("=" * 50)
        return True
    
    try:
        result = subprocess.run(
            [HERMES_BIN, "send", "--to", WEIXIN_TARGET, message],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"[OK] 已发送到微信")
            return True
        else:
            print(f"[ERROR] 发送失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] 发送异常: {e}")
        return False


def run_diagnosis(test=False):
    """评分低时运行诊断分析"""
    try:
        from auto_improve import cmd_suggest
        print("[INFO] 评分较低，运行诊断分析...")
        cmd_suggest(days=14, test=test)
    except Exception as e:
        print(f"[WARN] 诊断分析失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='量化系统周报生成器')
    parser.add_argument('--test', action='store_true', help='测试模式，不发送')
    parser.add_argument('--period', choices=['week', 'month', 'all'], default='week', 
                        help='报告周期')
    parser.add_argument('--no-diagnosis', action='store_true', help='禁用自动诊断')
    args = parser.parse_args()
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 生成{args.period}报告...")
    
    report, score = generate_report(args.period)
    
    success = send_to_weixin(report, test=args.test)
    
    # 评分 < 4 时自动运行诊断
    if score < 4 and not args.no_diagnosis:
        print(f"[INFO] 评分 {score}/8 较低，触发诊断分析")
        run_diagnosis(test=args.test)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
