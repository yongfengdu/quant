#!/usr/bin/env python3
"""
每日回测反馈系统

分析当天实际涨幅超过3%但策略没有命中的股票，生成反馈用于策略改进。

流程:
1. 获取当天所有股票的实际涨幅 (T+1开盘买 → T+2开盘卖)
2. 找出涨幅>3%的"黄金机会"
3. 对比策略预测，分析哪些机会被错过
4. 分析错过原因（特征分析）
5. 写入 LESSONS.md 供 pipeline 进化使用

Usage:
    python daily_feedback.py              # 分析昨天的错过机会
    python daily_feedback.py --date=2026-07-25  # 分析指定日期
    python daily_feedback.py --test       # 测试模式
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).parent
CACHE_DIR = Path.home() / '.cache' / 'quant-autoresearch'
PREDICTIONS_FILE = PROJECT_DIR / "daily_predictions.csv"
FEEDBACK_LOG = PROJECT_DIR / "data" / "daily_feedback.json"
LESSONS_FILE = PROJECT_DIR / "LESSONS.md"

# 确保目录存在
(PROJECT_DIR / "data").mkdir(exist_ok=True)


def load_data():
    """加载数据"""
    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    
    universe = pd.read_parquet(CACHE_DIR / "universe.parquet")
    universe_codes = set(universe["code"].values)
    
    return daily, universe_codes


def load_predictions():
    """加载预测记录"""
    if not PREDICTIONS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_FILE, dtype={"code": str})


def get_trading_days(daily, around_date, n_before=3, n_after=3):
    """获取某日期前后的交易日"""
    all_dates = sorted(daily["date"].unique())
    target = pd.to_datetime(around_date)
    
    try:
        idx = list(all_dates).index(target)
    except ValueError:
        # 找最近的交易日
        for i, d in enumerate(all_dates):
            if d >= target:
                idx = i
                break
        else:
            idx = len(all_dates) - 1
    
    start = max(0, idx - n_before)
    end = min(len(all_dates), idx + n_after + 1)
    
    return all_dates[start:end]


def calculate_actual_returns(daily, signal_date, universe_codes):
    """
    计算从 signal_date 出发的实际收益
    交易逻辑: signal_date 收盘信号 → T+1 开盘买 → T+2 开盘卖
    """
    all_dates = sorted(daily["date"].unique())
    signal_dt = pd.to_datetime(signal_date)
    
    # 找 T+1 和 T+2
    try:
        idx = list(all_dates).index(signal_dt)
    except ValueError:
        return None
    
    if idx + 2 >= len(all_dates):
        return None  # 数据不足
    
    entry_date = all_dates[idx + 1]
    exit_date = all_dates[idx + 2]
    
    # 获取开盘价
    entry_data = daily[daily["date"] == entry_date].set_index("code")["open"]
    exit_data = daily[daily["date"] == exit_date].set_index("code")["open"]
    
    # 计算收益
    results = []
    for code in universe_codes:
        if code in entry_data.index and code in exit_data.index:
            entry_price = entry_data[code]
            exit_price = exit_data[code]
            ret = (exit_price - entry_price) / entry_price
            
            # 获取 signal_date 的特征
            signal_data = daily[(daily["date"] == signal_dt) & (daily["code"] == code)]
            if len(signal_data) > 0:
                row = signal_data.iloc[0]
                results.append({
                    "code": code,
                    "signal_date": str(signal_dt.date()),
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(exit_date.date()),
                    "entry_open": float(entry_price),
                    "exit_open": float(exit_price),
                    "actual_return": float(ret),
                    "hit_3pct": ret > 0.03,
                    # 特征
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                    "amount": float(row.get("amount", 0)),
                    "ret_1d": float(row.get("ret_1d", 0)) if "ret_1d" in row else None,
                    "ret_5d": float(row.get("ret_5d", 0)) if "ret_5d" in row else None,
                    "ret_20d": float(row.get("ret_20d", 0)) if "ret_20d" in row else None,
                    "vol_ratio": float(row.get("vol_ratio", 0)) if "vol_ratio" in row else None,
                })
    
    return pd.DataFrame(results)


def analyze_missed_opportunities(actual_returns, predictions, signal_date):
    """
    分析错过的机会
    
    返回:
    - golden: 涨幅>3%的黄金机会
    - predicted: 策略预测的股票
    - hit: 预测且命中
    - missed: 黄金机会但没预测
    """
    # 黄金机会：实际涨幅 > 3%
    golden = actual_returns[actual_returns["hit_3pct"] == True].copy()
    
    # 策略预测了哪些
    signal_date_str = str(pd.to_datetime(signal_date).date())
    predicted_codes = set()
    if len(predictions) > 0:
        pred_on_date = predictions[predictions["signal_date"] == signal_date_str]
        predicted_codes = set(pred_on_date["code"].values)
    
    golden["was_predicted"] = golden["code"].isin(predicted_codes)
    
    hit = golden[golden["was_predicted"] == True]
    missed = golden[golden["was_predicted"] == False]
    
    return {
        "signal_date": signal_date_str,
        "total_stocks": len(actual_returns),
        "golden_opportunities": len(golden),
        "predicted_count": len(predicted_codes),
        "hit_count": len(hit),
        "missed_count": len(missed),
        "hit_rate": len(hit) / len(golden) if len(golden) > 0 else 0,
        "golden_codes": golden["code"].tolist(),
        "predicted_codes": list(predicted_codes),
        "hit_codes": hit["code"].tolist(),
        "missed_codes": missed["code"].tolist(),
        "missed_details": missed.to_dict("records") if len(missed) > 0 else [],
        "golden_details": golden.to_dict("records"),
    }


def extract_missed_patterns(missed_details, daily, signal_date):
    """
    分析错过机会的共性特征，用于策略改进
    
    深入分析为什么策略没选中这些股票
    """
    if not missed_details:
        return None
    
    df = pd.DataFrame(missed_details)
    signal_dt = pd.to_datetime(signal_date)
    
    patterns = {
        "count": len(df),
        "avg_return": df["actual_return"].mean(),
        "max_return": df["actual_return"].max(),
        "codes": df["code"].tolist(),
    }
    
    insights = []
    strategy_gaps = []
    
    # 深入分析每只错过的股票
    for _, row in df.iterrows():
        code = row["code"]
        hist = daily[(daily["code"] == code) & (daily["date"] <= signal_dt)].copy()
        
        if len(hist) < 20:
            continue
        
        hist = hist.sort_values("date")
        hist["ma20"] = hist["close"].shift(1).rolling(20, min_periods=20).mean()
        hist["low_N"] = hist["low"].shift(1).rolling(20, min_periods=20).min()
        hist["vol_ma20"] = hist["volume"].shift(1).rolling(20, min_periods=20).mean()
        hist["yang"] = (hist["close"] > hist["open"]).astype(int)
        hist["ret_5d"] = hist["close"].pct_change(5)
        
        today = hist[hist["date"] == signal_dt]
        if len(today) == 0:
            continue
        
        today = today.iloc[0]
        
        # 检查策略条件
        cond1 = today["close"] < today["ma20"]  # 下跌趋势
        cond2 = today["low"] < today["low_N"]   # 创新低
        cond3_yang = today["yang"] == 1         # 收阳
        cond3_vol = today["volume"] > today["vol_ma20"] * 1.5  # 放量
        
        gap_info = {
            "code": code,
            "sector": row.get("sector", "未知"),
            "actual_return": row["actual_return"],
            "cond1_downtrend": cond1,
            "cond2_new_low": cond2,
            "cond3_yang": cond3_yang,
            "cond3_volume": cond3_vol,
            "close_vs_ma20": (today["close"] / today["ma20"] - 1) if today["ma20"] > 0 else 0,
            "ret_5d": today.get("ret_5d", 0),
        }
        strategy_gaps.append(gap_info)
        
        # 分类失败原因
        if not cond2:
            insights.append(f"{code}({row.get('sector','?')}): 未创新低，可能是趋势延续型上涨")
        elif not cond3_yang:
            insights.append(f"{code}({row.get('sector','?')}): 未收阳，但次日仍大涨")
        elif not cond3_vol:
            insights.append(f"{code}({row.get('sector','?')}): 量能不足1.5x，但仍上涨")
    
    patterns["strategy_gaps"] = strategy_gaps
    
    # 汇总分析
    if strategy_gaps:
        gap_df = pd.DataFrame(strategy_gaps)
        
        # 统计失败原因
        n_no_newlow = sum(1 for g in strategy_gaps if not g["cond2_new_low"])
        n_no_yang = sum(1 for g in strategy_gaps if g["cond2_new_low"] and not g["cond3_yang"])
        n_no_vol = sum(1 for g in strategy_gaps if g["cond2_new_low"] and g["cond3_yang"] and not g["cond3_volume"])
        
        if n_no_newlow > len(strategy_gaps) * 0.5:
            insights.append(f"多数错过机会({n_no_newlow}/{len(strategy_gaps)})非超跌反弹，可能需要趋势策略")
        
        if n_no_yang > 0:
            insights.append(f"{n_no_yang}只股票当日未收阳但次日大涨，可能次日跳空高开")
        
        # 板块分析
        if "sector" in gap_df.columns:
            sector_counts = gap_df["sector"].value_counts()
            top_sector = sector_counts.index[0] if len(sector_counts) > 0 else None
            if top_sector and sector_counts[top_sector] >= 2:
                insights.append(f"板块{top_sector}多次被错过，可能需要板块轮动策略")
    
    patterns["insights"] = insights
    
    return patterns


def format_feedback_for_lessons(analysis, patterns):
    """格式化反馈写入 LESSONS.md"""
    lines = []
    lines.append(f"\n### 每日反馈 {analysis['signal_date']}")
    lines.append(f"- 黄金机会: {analysis['golden_opportunities']}只 (涨幅>3%)")
    lines.append(f"- 策略预测: {analysis['predicted_count']}只")
    lines.append(f"- 命中: {analysis['hit_count']}只, 错过: {analysis['missed_count']}只")
    
    if analysis['missed_count'] > 0:
        lines.append(f"- 错过的股票: {', '.join(analysis['missed_codes'][:5])}")
        
        if patterns and patterns.get("insights"):
            lines.append("- 错过原因分析:")
            for insight in patterns["insights"]:
                lines.append(f"  - {insight}")
        
        # 给出改进建议
        lines.append("- 改进方向:")
        if patterns:
            if patterns.get("ret_5d_mean", 0) < -0.05:
                lines.append("  - 当前超跌阈值可能太严格，考虑放宽")
            if patterns.get("vol_ratio_mean", 0) > 1.5:
                lines.append("  - 放量信号权重可以提高")
            if patterns.get("ret_1d_mean", 0) > 0.02:
                lines.append("  - 可能需要增加趋势跟随策略")
    else:
        lines.append("- 没有错过的黄金机会！策略表现良好")
    
    return "\n".join(lines)


def save_feedback(analysis, patterns):
    """保存反馈日志"""
    # 加载历史
    history = []
    if FEEDBACK_LOG.exists():
        try:
            with open(FEEDBACK_LOG) as f:
                history = json.load(f)
        except:
            history = []
    
    # 添加新记录
    record = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "patterns": patterns,
    }
    history.append(record)
    
    # 只保留最近30天
    history = history[-30:]
    
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)


def append_to_lessons(content):
    """追加到 LESSONS.md"""
    with open(LESSONS_FILE, "a") as f:
        f.write(content)
        f.write("\n")


def cmd_feedback(signal_date=None, test=False, write_lessons=True):
    """运行反馈分析"""
    print("=" * 60)
    print("📊 每日回测反馈分析")
    print("=" * 60)
    
    # 加载数据
    daily, universe_codes = load_data()
    predictions = load_predictions()
    
    # 默认分析前一个交易日
    if signal_date is None:
        all_dates = sorted(daily["date"].unique())
        # 找最近一个有完整 T+2 数据的日期
        latest = all_dates[-1]
        if len(all_dates) >= 3:
            signal_date = all_dates[-3]  # T-2 作为 signal_date，这样 T 就是 exit_date
        else:
            print("数据不足")
            return None
    else:
        signal_date = pd.to_datetime(signal_date)
    
    print(f"分析日期: {signal_date.date()}")
    
    # 计算实际收益
    actual_returns = calculate_actual_returns(daily, signal_date, universe_codes)
    
    if actual_returns is None or len(actual_returns) == 0:
        print("无法计算收益（数据不足）")
        return None
    
    print(f"股票数量: {len(actual_returns)}")
    
    # 分析错过的机会
    analysis = analyze_missed_opportunities(actual_returns, predictions, signal_date)
    
    print(f"\n黄金机会 (涨幅>3%): {analysis['golden_opportunities']}只")
    print(f"策略预测: {analysis['predicted_count']}只")
    print(f"命中: {analysis['hit_count']}只")
    print(f"错过: {analysis['missed_count']}只")
    
    if analysis['golden_opportunities'] > 0:
        print(f"命中率: {analysis['hit_rate']*100:.1f}%")
    
    # 分析错过的模式
    patterns = extract_missed_patterns(analysis["missed_details"], daily, signal_date)
    
    if patterns and patterns.get("insights"):
        print(f"\n错过原因分析:")
        for insight in patterns["insights"]:
            print(f"  - {insight}")
    
    if analysis['missed_count'] > 0:
        print(f"\n错过的股票 (前5):")
        for detail in analysis["missed_details"][:5]:
            print(f"  {detail['code']}: +{detail['actual_return']*100:.2f}%")
    
    # 保存
    if not test:
        save_feedback(analysis, patterns)
        print(f"\n反馈已保存到: {FEEDBACK_LOG}")
        
        if write_lessons and (analysis['missed_count'] > 0 or analysis['hit_count'] > 0):
            lesson_content = format_feedback_for_lessons(analysis, patterns)
            append_to_lessons(lesson_content)
            print(f"已追加到: {LESSONS_FILE}")
    
    return analysis, patterns


def cmd_summary(days=7):
    """汇总最近N天的反馈"""
    if not FEEDBACK_LOG.exists():
        print("没有反馈记录")
        return
    
    with open(FEEDBACK_LOG) as f:
        history = json.load(f)
    
    if not history:
        print("没有反馈记录")
        return
    
    print(f"最近 {len(history)} 天反馈汇总:")
    print("-" * 60)
    
    total_golden = 0
    total_hit = 0
    total_missed = 0
    all_insights = []
    
    for record in history[-days:]:
        analysis = record["analysis"]
        date = analysis["signal_date"]
        golden = analysis["golden_opportunities"]
        hit = analysis["hit_count"]
        missed = analysis["missed_count"]
        
        total_golden += golden
        total_hit += hit
        total_missed += missed
        
        hit_rate = hit / golden * 100 if golden > 0 else 0
        print(f"{date}: 黄金{golden}只, 命中{hit}, 错过{missed} ({hit_rate:.0f}%)")
        
        if record.get("patterns") and record["patterns"].get("insights"):
            all_insights.extend(record["patterns"]["insights"])
    
    print("-" * 60)
    overall_rate = total_hit / total_golden * 100 if total_golden > 0 else 0
    print(f"总计: 黄金{total_golden}只, 命中{total_hit}, 错过{total_missed} ({overall_rate:.0f}%)")
    
    if all_insights:
        print(f"\n常见错过原因:")
        # 统计频率（去掉股票代码，只保留模式）
        from collections import Counter
        patterns = []
        for insight in all_insights:
            # 提取通用模式
            if "未创新低" in insight:
                patterns.append("未创新低(趋势延续)")
            elif "未收阳" in insight:
                patterns.append("未收阳但次日大涨")
            elif "量能不足" in insight:
                patterns.append("量能不足但仍上涨")
            elif "趋势策略" in insight:
                patterns.append("需要趋势策略")
            elif "板块轮动" in insight:
                patterns.append("需要板块轮动策略")
            else:
                patterns.append(insight[:30])
        
        pattern_counts = Counter(patterns)
        for pattern, count in pattern_counts.most_common(5):
            print(f"  ({count}次) {pattern}")
    
    # 返回是否需要触发进化
    return {
        "total_golden": total_golden,
        "total_missed": total_missed,
        "hit_rate": overall_rate,
        "top_patterns": pattern_counts.most_common(3) if all_insights else []
    }


def write_feedback_to_lessons():
    """将积累的反馈汇总写入 LESSONS.md"""
    if not FEEDBACK_LOG.exists():
        print("没有反馈记录")
        return
    
    with open(FEEDBACK_LOG) as f:
        history = json.load(f)
    
    if len(history) < 3:
        print(f"反馈记录不足 ({len(history)}/3)，等待更多数据")
        return
    
    # 汇总分析
    from collections import Counter
    all_insights = []
    total_missed = 0
    
    for record in history:
        analysis = record["analysis"]
        total_missed += analysis["missed_count"]
        if record.get("patterns") and record["patterns"].get("insights"):
            all_insights.extend(record["patterns"]["insights"])
    
    if total_missed == 0:
        print("没有错过的机会，策略表现完美！")
        return
    
    # 生成汇总
    lines = []
    lines.append(f"\n## 每日反馈汇总 ({datetime.now().strftime('%Y-%m-%d')})")
    lines.append(f"- 分析周期: {len(history)}天")
    lines.append(f"- 错过的黄金机会: {total_missed}只")
    
    if all_insights:
        lines.append("- 主要错过原因:")
        patterns = []
        for insight in all_insights:
            if "未创新低" in insight:
                patterns.append("趋势延续型上涨（非超跌反弹）")
            elif "未收阳" in insight:
                patterns.append("当日未收阳但次日跳空")
            elif "量能不足" in insight:
                patterns.append("量能不足但价格上涨")
            elif "趋势策略" in insight:
                patterns.append("需要增加趋势跟随策略")
            elif "板块轮动" in insight:
                patterns.append("需要增加板块轮动策略")
        
        pattern_counts = Counter(patterns)
        for pattern, count in pattern_counts.most_common(3):
            lines.append(f"  - {pattern} ({count}次)")
    
    lines.append("- 改进建议:")
    if any("趋势" in p for p in patterns):
        lines.append("  - 增加趋势跟随策略，捕捉非超跌的上涨")
    if any("板块" in p for p in patterns):
        lines.append("  - 增加板块轮动策略，跟随热点板块")
    if any("收阳" in p for p in patterns):
        lines.append("  - 放宽收阳条件或增加次日跳空预判")
    
    content = "\n".join(lines)
    append_to_lessons(content)
    print(f"已写入 LESSONS.md:\n{content}")
    
    # 清空已处理的反馈
    with open(FEEDBACK_LOG, "w") as f:
        json.dump([], f)


def main():
    parser = argparse.ArgumentParser(description="每日回测反馈系统")
    parser.add_argument("command", nargs="?", default="feedback", 
                        choices=["feedback", "summary", "flush"],
                        help="命令: feedback/summary/flush")
    parser.add_argument("--date", help="分析日期 (YYYY-MM-DD)")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--no-lessons", action="store_true", help="不写入LESSONS.md")
    parser.add_argument("--days", type=int, default=7, help="汇总天数")
    
    args = parser.parse_args()
    
    if args.command == "feedback":
        cmd_feedback(args.date, args.test, not args.no_lessons)
    elif args.command == "summary":
        cmd_summary(args.days)
    elif args.command == "flush":
        write_feedback_to_lessons()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

def format_feedback_for_prompt():
    """返回一段简短的在线市场反馈文本，用于写入分析师提示词。"""
    try:
        daily, universe_codes = load_data()
        predictions = load_predictions()
        if daily is None or predictions is None:
            return ""
        all_dates = sorted(daily["date"].unique())
        if len(all_dates) < 3:
            return ""
        signal_date = all_dates[-3]
        actual_returns = calculate_actual_returns(daily, signal_date, universe_codes)
        if actual_returns is None or len(actual_returns) == 0:
            return ""
        analysis = analyze_missed_opportunities(actual_returns, predictions, signal_date)
        patterns = extract_missed_patterns(analysis["missed_details"], daily, signal_date)
        
        lines = []
        lines.append(f"📊 在线市场反馈（信号日: {signal_date.date()}）")
        lines.append(f"   黄金机会: {analysis['golden_opportunities']}只 | 命中: {analysis['hit_count']}只 | 错过: {analysis['missed_count']}只")
        if analysis['golden_opportunities'] > 0:
            lines.append(f"   命中率: {analysis['hit_rate']*100:.1f}%")
        if patterns and patterns.get("insights"):
            lines.append("   模式洞察:")
            insights = patterns["insights"]
            actionable_kw = ("多数错过", "趋势策略", "板块轮动", "跳空", "收阳")
            actionable = [p for p in insights if any(k in p for k in actionable_kw)]
            picked = (actionable + [p for p in insights if p not in actionable])[:3]
            for p in picked:
                lines.append(f"     - {p}")
        if analysis['missed_count'] > 0 and analysis['missed_details']:
            top_missed = sorted(analysis['missed_details'], key=lambda x: x.get('actual_return', 0), reverse=True)[:3]
            missed_str = ", ".join(f"{d['code']}(+{d['actual_return']*100:.1f}%)" for d in top_missed)
            lines.append(f"   最大错过: {missed_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"在线反馈获取失败: {e}"
