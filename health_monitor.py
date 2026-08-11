#!/usr/bin/env python3
"""
量化系统健康监控与自动修复

功能:
1. 检查数据是否最新
2. 检查策略池是否正常加载
3. 检查磁盘空间
4. 检查 cron 任务执行状态
5. 自动修复可修复的问题
6. 生成健康报告

设计原则:
- 每个检查项独立，失败不影响其他检查
- 可修复的问题自动修复
- 不可修复的问题生成告警
- 输出简洁报告供 hermes 发送
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd

# 配置
PROJECT_DIR = Path("/root/quant-autoresearch")
CACHE_DIR = Path.home() / ".cache/quant-autoresearch"
ISLANDS_DIR = PROJECT_DIR / "islands"
HEALTH_LOG = PROJECT_DIR / "health_monitor.log"

# 阈值
DATA_LAG_WARN_DAYS = 1  # 数据滞后1天告警
DATA_LAG_CRIT_DAYS = 3  # 数据滞后3天严重
DISK_WARN_PERCENT = 80
DISK_CRIT_PERCENT = 90
MIN_STRATEGIES = 5  # 最少策略数

class HealthChecker:
    def __init__(self):
        self.issues = []  # (severity, category, message)
        self.fixes = []   # 已修复的问题
        self.metrics = {} # 监控指标
    
    def log(self, msg: str):
        """写入日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(HEALTH_LOG, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")
    
    def add_issue(self, severity: str, category: str, message: str):
        """添加问题"""
        self.issues.append((severity, category, message))
        self.log(f"[{severity}] {category}: {message}")
    
    def add_fix(self, message: str):
        """记录修复"""
        self.fixes.append(message)
        self.log(f"[FIX] {message}")

    # ========== 检查项 ==========
    
    def check_data_freshness(self) -> bool:
        """检查数据是否最新"""
        try:
            daily_file = CACHE_DIR / "daily.parquet"
            if not daily_file.exists():
                self.add_issue("CRITICAL", "DATA", "daily.parquet 不存在")
                return False
            
            daily = pd.read_parquet(daily_file)
            daily['date'] = pd.to_datetime(daily['date'])
            latest_date = daily['date'].max()
            
            # 计算工作日滞后
            today = pd.Timestamp.now().normalize()
            lag_days = 0
            check_date = latest_date
            while check_date < today:
                check_date += timedelta(days=1)
                if check_date.weekday() < 5:  # 周一到周五
                    lag_days += 1
            
            self.metrics['data_latest_date'] = str(latest_date.date())
            self.metrics['data_lag_days'] = lag_days
            
            if lag_days >= DATA_LAG_CRIT_DAYS:
                self.add_issue("CRITICAL", "DATA", f"数据滞后 {lag_days} 个工作日 (最新: {latest_date.date()})")
                return False
            elif lag_days >= DATA_LAG_WARN_DAYS:
                self.add_issue("WARNING", "DATA", f"数据滞后 {lag_days} 个工作日")
                return True
            
            return True
        except Exception as e:
            self.add_issue("CRITICAL", "DATA", f"检查数据失败: {e}")
            return False
    
    def check_strategy_pool(self) -> bool:
        """检查策略池"""
        try:
            # 检查岛屿策略
            total_strategies = 0
            island_status = {}
            
            for regime in ['bear', 'bull', 'range']:
                island_dir = ISLANDS_DIR / f"island_{regime}"
                state_file = island_dir / "state.json"
                
                if state_file.exists():
                    state = json.loads(state_file.read_text())
                    count = len(state.get('strategies', []))
                    island_status[regime] = count
                    total_strategies += count
                else:
                    island_status[regime] = 0
                    if regime == 'range':
                        self.add_issue("WARNING", "STRATEGY", f"{regime} 岛屿未初始化")
            
            self.metrics['strategies_bear'] = island_status.get('bear', 0)
            self.metrics['strategies_bull'] = island_status.get('bull', 0)
            self.metrics['strategies_range'] = island_status.get('range', 0)
            self.metrics['strategies_total'] = total_strategies
            
            if total_strategies < MIN_STRATEGIES:
                self.add_issue("WARNING", "STRATEGY", f"策略总数 {total_strategies} < {MIN_STRATEGIES}")
            
            # 检查 verified_strategies 目录
            verified_dir = PROJECT_DIR / "verified_strategies"
            if not verified_dir.exists():
                self.add_issue("CRITICAL", "STRATEGY", "verified_strategies 目录不存在")
                return False
            
            # 检查是否有策略文件
            strategy_files = list(verified_dir.glob("*.py"))
            if len(strategy_files) == 0:
                self.add_issue("WARNING", "STRATEGY", "verified_strategies 目录为空")
            
            return True
        except Exception as e:
            self.add_issue("CRITICAL", "STRATEGY", f"检查策略池失败: {e}")
            return False
    
    def check_disk_space(self) -> bool:
        """检查磁盘空间"""
        try:
            result = subprocess.run(['df', '/'], capture_output=True, text=True)
            line = result.stdout.strip().split('\n')[-1]
            usage = int(line.split()[4].replace('%', ''))
            
            self.metrics['disk_usage_percent'] = usage
            
            if usage >= DISK_CRIT_PERCENT:
                self.add_issue("CRITICAL", "DISK", f"磁盘使用 {usage}% >= {DISK_CRIT_PERCENT}%")
                return False
            elif usage >= DISK_WARN_PERCENT:
                self.add_issue("WARNING", "DISK", f"磁盘使用 {usage}%")
            
            return True
        except Exception as e:
            self.add_issue("WARNING", "DISK", f"检查磁盘失败: {e}")
            return True
    
    def check_cron_status(self) -> bool:
        """检查 cron 任务状态"""
        try:
            result = subprocess.run(
                ['/root/.local/bin/hermes', 'cron', 'list'],
                capture_output=True, text=True, timeout=30
            )
            output = result.stdout
            
            # 检查关键任务是否存在
            if '量化策略每日预测' not in output:
                self.add_issue("WARNING", "CRON", "量化策略每日预测 任务未找到")
            
            if '岛屿进化' not in output:
                self.add_issue("WARNING", "CRON", "岛屿进化 任务未找到")
            
            # 检查是否有失败记录
            if 'Delivery failed' in output:
                self.add_issue("WARNING", "CRON", "存在消息发送失败")
            
            return True
        except Exception as e:
            self.add_issue("WARNING", "CRON", f"检查 cron 失败: {e}")
            return True
    
    def check_recent_predictions(self) -> bool:
        """检查最近预测质量"""
        try:
            pred_file = PROJECT_DIR / "daily_predictions.csv"
            if not pred_file.exists():
                self.add_issue("WARNING", "PREDICTION", "无预测记录文件")
                return True
            
            df = pd.read_csv(pred_file)
            if len(df) == 0:
                return True
            
            df['signal_date'] = pd.to_datetime(df['signal_date'])
            
            # 最近7天
            week_ago = pd.Timestamp.now() - timedelta(days=7)
            recent = df[df['signal_date'] >= week_ago]
            
            if len(recent) == 0:
                self.add_issue("WARNING", "PREDICTION", "最近7天无预测")
                return True
            
            # 计算指标
            hit_rate = recent['hit_3pct'].mean() * 100 if 'hit_3pct' in recent.columns else 0
            avg_return = recent['actual_return'].mean() * 100 if 'actual_return' in recent.columns else 0
            
            self.metrics['prediction_7d_count'] = len(recent)
            self.metrics['prediction_7d_hit_rate'] = round(hit_rate, 1)
            self.metrics['prediction_7d_avg_return'] = round(avg_return, 2)
            
            if hit_rate < 10:
                self.add_issue("WARNING", "PREDICTION", f"7天3%命中率 {hit_rate:.1f}% < 10%")
            
            return True
        except Exception as e:
            self.add_issue("WARNING", "PREDICTION", f"检查预测失败: {e}")
            return True

    # ========== 自动修复 ==========
    
    def fix_verified_strategies(self) -> bool:
        """修复 verified_strategies 目录"""
        try:
            verified_dir = PROJECT_DIR / "verified_strategies"
            verified_dir.mkdir(exist_ok=True)
            
            # 从岛屿复制策略
            copied = 0
            for regime in ['bear', 'bull', 'range']:
                island_strategies = ISLANDS_DIR / f"island_{regime}" / "strategies"
                if not island_strategies.exists():
                    continue
                
                for py_file in island_strategies.glob("*.py"):
                    target = verified_dir / py_file.name
                    if not target.exists():
                        # 复制文件而不是符号链接（更稳定）
                        import shutil
                        shutil.copy2(py_file, target)
                        copied += 1
            
            if copied > 0:
                self.add_fix(f"已复制 {copied} 个策略到 verified_strategies")
            
            return True
        except Exception as e:
            self.log(f"修复 verified_strategies 失败: {e}")
            return False
    
    def fix_range_island(self) -> bool:
        """初始化 range 岛屿"""
        try:
            range_dir = ISLANDS_DIR / "island_range"
            range_dir.mkdir(parents=True, exist_ok=True)
            
            state_file = range_dir / "state.json"
            if not state_file.exists():
                state = {
                    "regime": "range",
                    "strategies": [],
                    "rounds_since_new": 0,
                    "live_cumulative_return": 0.0,
                    "last_migration": "",
                    "frozen_paths": {}
                }
                state_file.write_text(json.dumps(state, indent=2))
                
                # 创建策略目录
                (range_dir / "strategies").mkdir(exist_ok=True)
                
                # 创建空的 LESSONS.md
                (range_dir / "LESSONS.md").write_text("# Range 市场策略教训\n\n")
                
                self.add_fix("已初始化 range 岛屿")
            
            return True
        except Exception as e:
            self.log(f"初始化 range 岛屿失败: {e}")
            return False
    
    def trigger_data_update(self) -> bool:
        """触发数据更新（后台执行）"""
        try:
            # 创建一个标记文件，让 cron 任务知道需要强制更新
            flag_file = PROJECT_DIR / ".need_data_update"
            flag_file.write_text(datetime.now().isoformat())
            self.add_fix("已标记需要数据更新")
            return True
        except Exception as e:
            self.log(f"触发数据更新失败: {e}")
            return False

    # ========== 主流程 ==========
    
    def run_checks(self) -> Dict:
        """运行所有检查"""
        self.log("=== 开始健康检查 ===")
        
        # 运行检查
        self.check_disk_space()
        self.check_data_freshness()
        self.check_strategy_pool()
        self.check_cron_status()
        self.check_recent_predictions()
        
        return {
            'issues': self.issues,
            'metrics': self.metrics,
        }
    
    def run_fixes(self) -> List[str]:
        """运行自动修复"""
        self.log("=== 开始自动修复 ===")
        
        # 根据问题类型修复
        for severity, category, message in self.issues:
            if category == "STRATEGY" and "verified_strategies" in message:
                self.fix_verified_strategies()
            elif category == "STRATEGY" and "range" in message and "未初始化" in message:
                self.fix_range_island()
            elif category == "DATA" and "滞后" in message:
                self.trigger_data_update()
        
        return self.fixes
    
    def generate_report(self) -> str:
        """生成健康报告"""
        lines = ["📊 量化系统健康报告", ""]
        
        # 状态概览
        critical = sum(1 for s, _, _ in self.issues if s == "CRITICAL")
        warning = sum(1 for s, _, _ in self.issues if s == "WARNING")
        
        if critical > 0:
            lines.append(f"🔴 状态: 严重问题 {critical} 个")
        elif warning > 0:
            lines.append(f"🟡 状态: 警告 {warning} 个")
        else:
            lines.append("🟢 状态: 健康")
        
        lines.append("")
        
        # 关键指标
        lines.append("📈 关键指标:")
        if 'data_latest_date' in self.metrics:
            lag = self.metrics.get('data_lag_days', 0)
            lag_str = f" (滞后{lag}天)" if lag > 0 else ""
            lines.append(f"  数据: {self.metrics['data_latest_date']}{lag_str}")
        if 'strategies_total' in self.metrics:
            lines.append(f"  策略: bear={self.metrics['strategies_bear']} bull={self.metrics['strategies_bull']} range={self.metrics['strategies_range']}")
        if 'disk_usage_percent' in self.metrics:
            lines.append(f"  磁盘: {self.metrics['disk_usage_percent']}%")
        if 'prediction_7d_hit_rate' in self.metrics:
            lines.append(f"  7天命中率: {self.metrics['prediction_7d_hit_rate']}%")
        
        # 问题列表
        if self.issues:
            lines.append("")
            lines.append("⚠️ 问题:")
            for severity, category, message in self.issues[:5]:  # 最多显示5个
                icon = "🔴" if severity == "CRITICAL" else "🟡"
                lines.append(f"  {icon} {message}")
        
        # 修复记录
        if self.fixes:
            lines.append("")
            lines.append("🔧 已修复:")
            for fix in self.fixes[:3]:
                lines.append(f"  ✅ {fix}")
        
        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='量化系统健康监控')
    parser.add_argument('--check-only', action='store_true', help='只检查不修复')
    parser.add_argument('--fix-only', action='store_true', help='只修复已知问题')
    parser.add_argument('--json', action='store_true', help='输出JSON格式')
    args = parser.parse_args()
    
    checker = HealthChecker()
    
    if not args.fix_only:
        checker.run_checks()
    
    if not args.check_only:
        checker.run_fixes()
    
    if args.json:
        result = {
            'timestamp': datetime.now().isoformat(),
            'issues': [(s, c, m) for s, c, m in checker.issues],
            'fixes': checker.fixes,
            'metrics': checker.metrics,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(checker.generate_report())


if __name__ == "__main__":
    main()
