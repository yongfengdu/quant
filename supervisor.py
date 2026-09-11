#!/usr/bin/env python3
"""
supervisor.py — 编排器 (上帝服务)
==================================
统一入口, 管理 pipeline 调度 + 信号生命周期 + 状态查询。

子命令:
  init              初始化: 注册内置核心信号(反转/动量)到注册表
  daily [--no-update]  每日: 数据更新 → 生命周期审视 → 预测(读active信号)
  evolve [n]        进化: LLM 找 n 个新信号假设
  monitor [--alert] 监控: 5大类健康检测
  status            状态: 系统全景 (信号/数据/预测/健康, CLI dashboard)
  promote <sid>     晋升: paper → active (人工确认后)
  retire <sid>      退役: active → retired

设计原则: 编排器只做"调度 + 状态管理", 具体逻辑留在各模块。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/root/quant-autoresearch")
import pandas as pd

from signal_registry import SignalRegistry, Lifecycle

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"

# 内置核心信号 (regime 门控, 回测 Sharpe 0.84)
CORE_SIGNALS = [
    {"sid": "dist_ma60", "factor": "dist_ma60", "horizon": 20,
     "direction": -1, "regime": "bear/range", "desc": "反转: 买超跌"},
    {"sid": "dist_hi20", "factor": "dist_hi20", "horizon": 20,
     "direction": 1, "regime": "bull", "desc": "动量: 买贴高"},
]


def cmd_init():
    """注册内置核心信号 (幂等: 已存在则跳过)。"""
    reg = SignalRegistry()
    for s in CORE_SIGNALS:
        if reg.get(s["sid"]) is None:
            reg.register(s["sid"], s["factor"], s["horizon"],
                         type_="builtin", direction=s["direction"],
                         regime=s["regime"], status="active")
            print(f"  注册 {s['sid']}: {s['desc']} (regime={s['regime']}, active)")
        else:
            print(f"  已存在 {s['sid']}, 跳过")
    reg.save()
    print(f"注册表共 {len(reg.all_ids())} 个信号")


def cmd_daily(no_update=False):
    """每日: 数据更新 → 生命周期审视 → 预测。"""
    from run_daily import step_update, step_lifecycle, step_recommend
    if not no_update:
        step_update()
    step_lifecycle()
    step_recommend()


def cmd_evolve(n=3):
    """进化: LLM 找 n 个新信号。"""
    from run_evolve import discover_signals
    df = pd.read_parquet(CACHE_DIR / "daily_bars.parquet")
    discover_signals(df, n=n, horizon=20, test=False)


def cmd_monitor(alert=False):
    from monitor import SystemMonitor
    mon = SystemMonitor()
    mon.run_all()
    print(mon.report())
    critical = sum(1 for i in mon.issues if i[0] == "CRITICAL")
    warning = sum(1 for i in mon.issues if i[0] == "WARNING")
    if alert and (critical or warning):
        try:
            import notifier
            notifier.alert(mon.report())
        except Exception:
            pass
    return 2 if critical else (1 if warning else 0)


def cmd_status():
    """CLI dashboard: 系统全景。"""
    print("=" * 60)
    print("系统状态 (supervisor status)")
    print("=" * 60)

    # 1. 信号注册表
    print("\n[信号注册表]")
    reg = SignalRegistry()
    signals = reg.data.get("signals", {})
    if not signals:
        print("  (空, 请先 supervisor init)")
    for sid, s in signals.items():
        ic = s.get("rolling_ic")
        ic_str = f"{ic*100:+.2f}%" if ic is not None else "-"
        print(f"  {s['status']:<9} {sid:<20} regime={s.get('regime','all'):<10} "
              f"dir={s.get('direction',1):+d} 滚IC={ic_str}")

    # 2. 数据状态
    print("\n[数据状态]")
    try:
        df = pd.read_parquet(CACHE_DIR / "daily_bars.parquet", columns=["date", "code"])
        latest = df["date"].max()
        lag = len(pd.bdate_range(pd.Timestamp(latest), pd.Timestamp.now().normalize())) - 1
        print(f"  最新日期 {latest} (滞后 {lag} 工作日), {df['code'].nunique()} 只股票")
    except Exception as e:
        print(f"  读取失败: {e}")

    # 3. 推荐状态
    print("\n[预测状态]")
    rec_file = Path("/root/quant-autoresearch") / "data" / "recommendations.csv"
    if rec_file.exists():
        rec = pd.read_csv(rec_file, dtype={"code": str})
        print(f"  推荐记录 {len(rec)} 条, 最新信号日 {rec['signal_date'].max()}")
    else:
        print("  尚无推荐记录")

    # 4. 预测效果 (回测 vs 实盘)
    eff_file = Path("/root/quant-autoresearch") / "data" / "prediction_effectiveness.csv"
    if eff_file.exists():
        eff = pd.read_csv(eff_file)
        last = eff.iloc[-1]
        print(f"  已验证 {last['n_cum']} 条, 累计3%命中 {last['cum_hit3']*100:.1f}%, "
              f"上涨 {last['cum_win']*100:.1f}% (回测基线 11%/50%)")
    else:
        print("  尚无验证数据 (需 T+20)")

    # 5. 健康
    print("\n[健康]")
    import json
    health = Path("/root/quant-autoresearch") / "kimi_health.json"
    if health.exists():
        h = json.loads(health.read_text())
        print(f"  K3: {h.get('status')} (调用{h.get('total_calls')} 失败率"
              f"{h.get('total_failures',0)/max(h.get('total_calls',1),1)*100:.0f}%)")


def cmd_promote(sid):
    reg = SignalRegistry()
    s = reg.get(sid)
    if s is None:
        print(f"信号 {sid} 不存在")
        return
    if s["status"] not in ("paper", "proposed"):
        print(f"信号 {sid} 状态 {s['status']}, 不可晋升")
        return
    reg.set_status(sid, "active")
    print(f"✅ {sid} → active (已上线)")


def cmd_retire(sid):
    reg = SignalRegistry()
    s = reg.get(sid)
    if s is None:
        print(f"信号 {sid} 不存在")
        return
    reg.set_status(sid, "retired")
    print(f"✅ {sid} → retired (已退役)")


def main():
    ap = argparse.ArgumentParser(description="量化系统编排器")
    ap.add_argument("command", choices=["init", "daily", "evolve", "monitor",
                                        "status", "promote", "retire"])
    ap.add_argument("arg", nargs="?", help="promote/retire 的信号id, 或 evolve 的假设数")
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--alert", action="store_true")
    args = ap.parse_args()

    if args.command == "init":
        cmd_init()
    elif args.command == "daily":
        cmd_daily(args.no_update)
    elif args.command == "evolve":
        cmd_evolve(int(args.arg) if args.arg else 3)
    elif args.command == "monitor":
        sys.exit(cmd_monitor(args.alert))
    elif args.command == "status":
        cmd_status()
    elif args.command == "promote":
        if not args.arg:
            print("用法: supervisor.py promote <sid>")
            return
        cmd_promote(args.arg)
    elif args.command == "retire":
        if not args.arg:
            print("用法: supervisor.py retire <sid>")
            return
        cmd_retire(args.arg)


if __name__ == "__main__":
    main()
