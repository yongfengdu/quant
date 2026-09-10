"""系统监控测试。"""
from monitor import SystemMonitor


def test_monitor_report_structure():
    mon = SystemMonitor()
    mon.log("CRITICAL", "测试", "测试问题")
    mon.log("WARNING", "测试", "测试警告")
    report = mon.report()
    assert "CRITICAL" in report
    assert "测试问题" in report
    assert "WARNING" in report


def test_monitor_all_green():
    mon = SystemMonitor()
    report = mon.report()
    assert "全部正常" in report


def test_run_all_no_crash():
    mon = SystemMonitor()
    mon.run_all()  # 不应抛异常
    assert isinstance(mon.issues, list)
    assert isinstance(mon.metrics, dict)
