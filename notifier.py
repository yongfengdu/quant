"""
通知汇报模块 (notifier.py)
============================
统一的告警/汇报出口, 通过 hermes 发送到微信。
供 evolve_v3 / watchdog / cron 使用。
"""
import subprocess
from datetime import datetime

HERMES_BIN = "/root/.local/bin/hermes"
WEIXIN_TARGET = "weixin:o9cq801nD9gYMZnU-b7HdGqc8upc@im.wechat"


def send(message, test=False):
    """发送消息到微信。test=True 只打印不发送。"""
    if test:
        print("=" * 50)
        print("[TEST 通知 - 不发送]")
        print(message)
        print("=" * 50)
        return True
    try:
        r = subprocess.run(
            [HERMES_BIN, "send", "--to", WEIXIN_TARGET, message],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print("[通知] ✓ 已发送到微信")
            return True
        print(f"[通知] ✗ 发送失败: {r.stderr[:200]}")
        return False
    except Exception as e:
        print(f"[通知] ✗ 异常: {e}")
        return False


def alert(msg, test=False):
    """告警消息 (带🚨前缀和时间戳)。"""
    ts = datetime.now().strftime("%m-%d %H:%M")
    return send(f"🚨 [量化系统告警 {ts}]\n{msg}", test=test)


def report_evolution(island, accepted, total, health_summary, failures, test=False):
    """进化运行汇报。"""
    ts = datetime.now().strftime("%m-%d %H:%M")
    lines = [f"🧬 [策略进化汇报 {ts}]", f"岛屿: {island} | 通过: {len(accepted)}/{total}"]

    if accepted:
        lines.append("\n✅ 新增策略:")
        for name, m, gate in accepted:
            lines.append(f"  {name}: {gate} "
                         f"准确率{m['accuracy']:.1f}% {m['trades']}笔 均涨{m['avg_return']:.2f}%")
    else:
        lines.append("\n本轮无策略通过Gate")

    if failures:
        lines.append(f"\n⚠️ 失败 {len(failures)} 次:")
        for f in failures[:5]:
            lines.append(f"  {f}")

    # Kimi 健康
    hs = health_summary
    status_icon = "✓" if hs["status"] == "healthy" else "✗"
    lines.append(f"\n🤖 K3服务: {status_icon} {hs['status']} "
                 f"(调用{hs['total_calls']} 失败率{hs['fail_rate']}%)")
    if hs["consecutive_failures"] > 0:
        lines.append(f"  ⚠️ 连续失败{hs['consecutive_failures']}次, "
                     f"最近: {hs['last_error_type']}")

    return send("\n".join(lines), test=test)


if __name__ == "__main__":
    import sys
    test = "--test" in sys.argv
    send("📢 notifier.py 测试消息", test=test)
