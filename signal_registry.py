"""
信号生命周期 (signal_registry.py)
=================================
信号状态机 + 衰减监控 + 自动退役。

状态流转:
  proposed(提出) → validated(过5闸) → paper(纸面跟踪) → active(实盘)
                                                              ↓
                                                        decayed(滚IC跌破阈值)
                                                              ↓
                                                        retired(退役→触发补位)

存储: JSON (信号注册表), 含验证报告 + IC 历史。
衰减判定: 近期(60日)滚 IC 的 |t-stat| < 阈值 连续 N 期 → decayed。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path("/root/quant-autoresearch")
REGISTRY_PATH = PROJECT_DIR / "data" / "signal_registry.json"

# 衰减判定参数
DECAY_WINDOW = 60        # 滚 IC 窗口 (交易日)
DECAY_T_THRESH = 1.5     # 近期 |t-stat| 低于此值视为衰减
DECAY_N_PERIODS = 3      # 连续 N 次判定衰减才退役 (防抖)

STATUS_FLOW = ["proposed", "validated", "paper", "active", "decayed", "retired"]


class SignalRegistry:
    """信号注册表 (JSON 持久化)。"""

    def __init__(self, path=REGISTRY_PATH):
        self.path = Path(path)
        self.data = {"signals": {}, "meta": {"version": 1}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                pass

    def _sig(self, sid):
        return self.data["signals"].setdefault(sid, {
            "id": sid, "status": "proposed", "factor": None, "horizon": None,
            "definition": None, "created_at": pd.Timestamp.now().isoformat(),
            "ic_history": [], "rolling_ic": None, "decay_strikes": 0,
            "validation": None, "last_updated": pd.Timestamp.now().isoformat(),
        })

    def register(self, sid, factor, horizon, definition=None, status="proposed"):
        s = self._sig(sid)
        s["factor"] = factor
        s["horizon"] = horizon
        s["definition"] = definition
        s["status"] = status
        self.save()
        return s

    def set_status(self, sid, status):
        s = self._sig(sid)
        assert status in STATUS_FLOW, f"非法状态 {status}"
        s["status"] = status
        s["last_updated"] = pd.Timestamp.now().isoformat()
        self.save()
        return s

    def set_validation(self, sid, report):
        """存五闸门验证报告 (精简, 去掉 ic 序列)。"""
        s = self._sig(sid)
        compact = {
            "factor": report["factor"], "horizon": report["horizon"],
            "summary": {k: v for k, v in report["summary"].items()},
            "gates": {k: {"ok": v[0], "msg": str(v[1])} for k, v in report["gates"].items()},
            "passed": report["passed"],
            "regime_ic": {k: ({"mean_ic": v.get("mean_ic"), "t_stat": v.get("t_stat"), "n": v.get("n")})
                          for k, v in report["regime_ic"].items()},
        }
        s["validation"] = compact
        self.save()
        return s

    def set_ic_history(self, sid, ic_series, window=DECAY_WINDOW):
        """存每日 IC 序列 (list of [date, ic]) 并更新滚 IC。"""
        s = self._sig(sid)
        hist = [[str(idx), float(v)] for idx, v in ic_series.dropna().items()]
        s["ic_history"] = hist
        s["rolling_ic"] = float(ic_series.dropna().iloc[-window:].mean()) if len(ic_series.dropna()) else None
        s["last_updated"] = pd.Timestamp.now().isoformat()
        self.save()
        return s

    def get(self, sid):
        return self.data["signals"].get(sid)

    def list_by_status(self, status):
        return [sid for sid, s in self.data["signals"].items() if s["status"] == status]

    def all_ids(self):
        return list(self.data["signals"].keys())

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))


def rolling_ic(ic_series, window=DECAY_WINDOW):
    """滚 IC (trailing mean)。返回 Series。"""
    return ic_series.rolling(window, min_periods=max(10, window // 3)).mean()


def recent_t_stat(ic_series, window=DECAY_WINDOW):
    """近期 IC 的 t-stat。"""
    recent = ic_series.dropna().iloc[-window:]
    if len(recent) < 10:
        return np.nan
    m, s = recent.mean(), recent.std(ddof=1)
    return m / (s / np.sqrt(len(recent))) if s > 0 else np.nan


class Lifecycle:
    """生命周期管理: 晋升 + 衰减监控 + 退役。"""

    def __init__(self, registry=None):
        self.reg = registry or SignalRegistry()

    def validate_and_promote(self, sid, report):
        """五闸门验证 → 通过则 proposed→validated→paper。"""
        self.reg.set_validation(sid, report)
        if report["passed"]:
            self.reg.set_status(sid, "paper")
            return True, "通过闸门, 进入纸面跟踪"
        self.reg.set_status(sid, "retired")
        return False, "未通过闸门, 退役"

    def check_decay(self, sid, ic_series=None):
        """检查 active/paper 信号是否衰减。返回 (decayed, detail)。"""
        s = self.reg.get(sid)
        if s is None:
            return False, "不存在"
        if s["status"] not in ("paper", "active"):
            return False, s["status"]
        if ic_series is not None:
            self.reg.set_ic_history(sid, ic_series)
            s = self.reg.get(sid)
        ic = pd.Series([v for _, v in s["ic_history"]])
        t = recent_t_stat(ic)
        if np.isnan(t) or abs(t) < DECAY_T_THRESH:
            s["decay_strikes"] = s.get("decay_strikes", 0) + 1
            detail = f"近期|t|={abs(t):.1f} < {DECAY_T_THRESH} (第{s['decay_strikes']}次)"
            if s["decay_strikes"] >= DECAY_N_PERIODS:
                self.reg.set_status(sid, "decayed")
                return True, detail + " → 退役"
            self.reg.save()
            return False, detail
        s["decay_strikes"] = 0
        self.reg.save()
        return False, f"健康 (近期|t|={abs(t):.1f})"

    def promote_paper_to_active(self, sid, ic_series=None):
        """纸面跟踪期间 IC 持续显著 → 转 active。"""
        s = self.reg.get(sid)
        if s["status"] != "paper":
            return False, s["status"]
        if ic_series is not None:
            self.reg.set_ic_history(sid, ic_series)
        t = recent_t_stat(pd.Series([v for _, v in self.reg.get(sid)["ic_history"]]))
        if not np.isnan(t) and abs(t) >= DECAY_T_THRESH * 2:
            self.reg.set_status(sid, "active")
            return True, f"转正 (近期|t|={abs(t):.1f})"
        return False, f"纸面中 (近期|t|={abs(t):.1f})"

    def review(self, ic_by_signal=None):
        """周期审视: 对每个 paper/active 信号做衰减检查。返回退役列表。"""
        retired = []
        for sid in self.reg.all_ids():
            s = self.reg.get(sid)
            if s["status"] not in ("paper", "active"):
                continue
            ic = ic_by_signal.get(sid) if ic_by_signal else None
            decayed, detail = self.check_decay(sid, ic)
            if decayed:
                retired.append((sid, detail))
        return retired
