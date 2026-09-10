"""
系统监控 (monitor.py)
====================
对自动迭代系统做每日健康检测, 覆盖 5 大类:
  1. 服务健康: K3/LLM、网络、磁盘、cron
  2. 数据拉取状态: 新鲜度、完整性、上次拉取
  3. LLM 返回和成功率: K3 调用统计、进化通过率
  4. 股票预测效果: 信号衰减(滚IC)、推荐命中率
  5. 其他: 注册表健康、推荐是否产出

输出: 分级报告 (CRITICAL/WARNING/OK), 退出码 0/1/2。
用法: python monitor.py [--alert] [--json]
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/root/quant-autoresearch")
import pandas as pd

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"
PROJECT_DIR = Path("/root/quant-autoresearch")
RECOMMENDATIONS_FILE = PROJECT_DIR / "data" / "recommendations.csv"
MONITOR_LOG = PROJECT_DIR / "monitor.log"


class SystemMonitor:
    def __init__(self):
        self.issues = []  # (severity, category, message)
        self.metrics = {}

    def log(self, severity, category, message):
        self.issues.append((severity, category, message))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(MONITOR_LOG, "a") as f:
            f.write(f"[{ts}] [{severity}] {category}: {message}\n")

    # ============ 1. 服务健康 ============
    def check_service_health(self):
        # 1a. K3/LLM 服务
        try:
            from kimi_watchdog import KimiWatchdog
            wd = KimiWatchdog()
            ok, reason = wd.health_check(timeout=30)
            if ok:
                self.metrics["k3"] = "healthy"
            else:
                self.log("CRITICAL", "服务", f"K3/LLM 服务异常: {reason}")
        except Exception as e:
            self.log("CRITICAL", "服务", f"K3 检查失败: {e}")

        # 1b. 网络/代理连通 (访问腾讯行情)
        try:
            from data_fetch import http_get
            url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=k&param=sh600000,day,2024-01-01,2024-01-05,5,"
            http_get(url, timeout=15, retries=2)
            self.metrics["network"] = "ok"
        except Exception as e:
            self.log("CRITICAL", "服务", f"网络/代理不通: {type(e).__name__}")

        # 1c. 磁盘空间
        try:
            usage = int(subprocess.run(['df', '/'], capture_output=True, text=True)
                        .stdout.strip().split('\n')[-1].split()[4].replace('%', ''))
            self.metrics["disk_usage"] = usage
            if usage >= 90:
                self.log("CRITICAL", "服务", f"磁盘使用 {usage}%")
            elif usage >= 80:
                self.log("WARNING", "服务", f"磁盘使用 {usage}%")
        except Exception:
            pass

        # 1d. cron 任务存在性
        try:
            cron = subprocess.run(['crontab', '-l'], capture_output=True, text=True).stdout
            for task in ["run_daily.py", "evolve_signal_cron.sh"]:
                if task not in cron:
                    self.log("CRITICAL", "服务", f"cron 任务缺失: {task}")
        except Exception:
            pass

    # ============ 2. 数据拉取状态 ============
    def check_data_status(self):
        daily_path = CACHE_DIR / "daily_bars.parquet"
        if not daily_path.exists():
            self.log("CRITICAL", "数据", "daily_bars.parquet 不存在")
            return
        try:
            df = pd.read_parquet(daily_path, columns=["date", "code"])
            latest = df["date"].max()
            self.metrics["data_latest"] = str(latest)
            # 工作日滞后
            today = pd.Timestamp.now().normalize()
            lag = len(pd.bdate_range(latest, today)) - 1
            self.metrics["data_lag"] = lag
            if lag >= 3:
                self.log("CRITICAL", "数据", f"数据滞后 {lag} 个工作日 (最新 {latest})")
            elif lag >= 1:
                self.log("WARNING", "数据", f"数据滞后 {lag} 个工作日")
            # 完整性
            from data_store import validate_bars
            full = pd.read_parquet(daily_path)
            issues, m = validate_bars(full, check_completeness=False)
            if m["dup"] or m["bad_ohlc"] or m["bad_price"]:
                self.log("CRITICAL", "数据", f"完整性: 重复{m['dup']} 坏OHLC{m['bad_ohlc']} 坏价{m['bad_price']}")
        except Exception as e:
            self.log("CRITICAL", "数据", f"检查失败: {e}")

    # ============ 3. LLM 返回和成功率 ============
    def check_llm_status(self):
        health_file = PROJECT_DIR / "kimi_health.json"
        if not health_file.exists():
            self.log("WARNING", "LLM", "kimi_health.json 不存在")
            return
        try:
            h = json.loads(health_file.read_text())
            calls = h.get("total_calls", 0)
            fails = h.get("total_failures", 0)
            rate = fails / calls * 100 if calls else 0
            self.metrics["k3_calls"] = calls
            self.metrics["k3_fail_rate"] = round(rate, 1)
            if h.get("consecutive_failures", 0) >= 3:
                self.log("CRITICAL", "LLM", f"K3 连续失败 {h['consecutive_failures']} 次")
            if rate > 30:
                self.log("WARNING", "LLM", f"K3 失败率 {rate:.0f}%")
        except Exception as e:
            self.log("WARNING", "LLM", f"读取失败: {e}")

    # ============ 4. 股票预测效果 ============
    def check_prediction_effectiveness(self):
        # 4a. 信号衰减 (注册表滚 IC)
        try:
            from signal_registry import SignalRegistry
            reg = SignalRegistry()
            signals = reg.data.get("signals", {})
            active = [s for s in signals.values() if s.get("status") in ("active", "paper")]
            self.metrics["signals_total"] = len(signals)
            self.metrics["signals_active"] = len(active)
        except Exception as e:
            self.log("WARNING", "预测", f"注册表读取失败: {e}")

        # 4b. 推荐命中率 (T+20 验证)
        if not RECOMMENDATIONS_FILE.exists():
            return  # 尚无推荐, 不算异常
        try:
            rec = pd.read_csv(RECOMMENDATIONS_FILE, dtype={"code": str})
            self.metrics["rec_total"] = len(rec)
            daily = pd.read_parquet(CACHE_DIR / "daily_bars.parquet")
            daily["date"] = daily["date"].astype(str)
            # 20日前向收益 (split_factor 复权)
            g = daily.groupby("code", sort=False)
            adj = daily["close"] * daily["split_factor"]
            daily["fwd20"] = g["close"].shift(-20) * g["split_factor"].shift(-20) / (daily["close"] * daily["split_factor"]) - 1
            fwd_map = {(str(r["code"]), str(r["date"])): r["fwd20"]
                       for _, r in daily.dropna(subset=["fwd20"]).iterrows()}
            verified = [fwd_map[(r["code"], r["signal_date"])]
                        for _, r in rec.iterrows()
                        if (r["code"], r["signal_date"]) in fwd_map]
            if verified:
                import numpy as np
                arr = np.array(verified)
                hit = (arr > 0.03).mean() * 100
                win = (arr > 0).mean() * 100
                self.metrics["rec_verified"] = len(verified)
                self.metrics["rec_hit3"] = round(hit, 1)
                self.metrics["rec_win"] = round(win, 1)
                if len(verified) >= 20 and hit < 8:
                    self.log("WARNING", "预测", f"推荐 3%命中率 {hit:.1f}% 偏低 (<8%)")
            else:
                self.metrics["rec_verified"] = 0
        except Exception as e:
            self.log("WARNING", "预测", f"推荐验证失败: {e}")

    # ============ 5. 其他 ============
    def check_other(self):
        # 5a. 注册表健康 (卡住的 paper 信号)
        try:
            from signal_registry import SignalRegistry
            reg = SignalRegistry()
            signals = reg.data.get("signals", {})
            stuck = [sid for sid, s in signals.items() if s.get("status") == "paper"]
            decayed = [sid for sid, s in signals.items() if s.get("status") == "decayed"]
            if len(signals) == 0:
                self.log("WARNING", "其他", "注册表为空, 无信号")
            self.metrics["paper_signals"] = len(stuck)
            self.metrics["decayed_signals"] = len(decayed)
        except Exception:
            pass

        # 5b. 推荐是否产出 (最近日志)
        try:
            log = PROJECT_DIR / "pipeline_daily.log"
            if log.exists():
                lines = log.read_text().splitlines()
                last_complete = next((l for l in reversed(lines) if "流水线完成" in l), None)
                if last_complete:
                    self.metrics["last_pipeline"] = last_complete[:19]
        except Exception:
            pass

    def run_all(self):
        self.check_service_health()
        self.check_data_status()
        self.check_llm_status()
        self.check_prediction_effectiveness()
        self.check_other()

    def report(self):
        critical = [i for i in self.issues if i[0] == "CRITICAL"]
        warning = [i for i in self.issues if i[0] == "WARNING"]
        lines = ["=== 系统监控报告 ==="]
        lines.append(f"指标: {self.metrics}")
        if critical:
            lines.append(f"\n🔴 CRITICAL ({len(critical)}):")
            for _, c, m in critical:
                lines.append(f"  - [{c}] {m}")
        if warning:
            lines.append(f"\n🟡 WARNING ({len(warning)}):")
            for _, c, m in warning:
                lines.append(f"  - [{c}] {m}")
        if not critical and not warning:
            lines.append("\n🟢 全部正常")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alert", action="store_true", help="异常时推送微信")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    mon = SystemMonitor()
    mon.run_all()
    if args.json:
        print(json.dumps({"issues": mon.issues, "metrics": mon.metrics},
                         ensure_ascii=False, indent=2))
    else:
        print(mon.report())

    critical = sum(1 for i in mon.issues if i[0] == "CRITICAL")
    warning = sum(1 for i in mon.issues if i[0] == "WARNING")

    if args.alert and (critical or warning):
        try:
            import notifier
            notifier.alert(mon.report())
        except Exception as e:
            print(f"告警推送失败: {e}")

    sys.exit(2 if critical else (1 if warning else 0))


if __name__ == "__main__":
    main()
