"""
Kimi 服务看门狗 (kimi_watchdog.py)
=====================================
职责:
  1. 健康检查: 调用前探测 Kimi/K3 服务是否可用
  2. 异常检测: 分类识别失败类型 (超时/认证/配额/网络/空响应/崩溃)
  3. 自恢复: 针对可恢复异常自动重试 (指数退避); OAuth 过期尝试刷新
  4. 状态记录: 写 health 状态文件, 供汇报机制读取
  5. 告警: 连续失败超阈值时触发告警回调

用法:
    from kimi_watchdog import KimiWatchdog
    wd = KimiWatchdog()
    ok, reason = wd.health_check()
    result, err_type = wd.call_with_retry(prompt, timeout=480, max_retries=2)
"""
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
KIMI_CLI = Path("/root/.kimi-code/bin/kimi")
KIMI_MODEL = os.environ.get("KIMI_CLI_MODEL", "kimi-code/k3")
HEALTH_FILE = PROJECT_DIR / "kimi_health.json"
OAUTH_FILE = Path("/root/.kimi-code/oauth/kimi-code")

# 异常分类
ERR_TIMEOUT = "timeout"
ERR_AUTH = "auth"          # OAuth/认证失败
ERR_QUOTA = "quota"        # 429 配额
ERR_NETWORK = "network"    # 代理/连接
ERR_EMPTY = "empty"        # 空响应
ERR_CRASH = "crash"        # CLI 崩溃/非0返回
ERR_NONE = "ok"

RECOVERABLE = {ERR_TIMEOUT, ERR_NETWORK, ERR_EMPTY, ERR_QUOTA}


class KimiWatchdog:
    def __init__(self, alert_callback=None):
        self.alert_callback = alert_callback
        self.state = self._load_state()

    def _load_state(self):
        if HEALTH_FILE.exists():
            try:
                return json.loads(HEALTH_FILE.read_text())
            except Exception:
                pass
        return {
            "last_check": None, "status": "unknown",
            "consecutive_failures": 0, "total_calls": 0,
            "total_failures": 0, "last_error_type": None,
            "last_error_msg": None, "error_history": [],
        }

    def _save_state(self):
        try:
            HEALTH_FILE.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _build_env(self):
        env = os.environ.copy()
        env_file = PROJECT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
        return env

    def _classify_error(self, returncode, output, timed_out):
        """根据返回码和输出内容分类异常。
        注意: 只在'成功但可疑'或'明确失败'时分类, 避免误判正常输出。
        """
        if timed_out:
            return ERR_TIMEOUT
        low = (output or "").lower()

        # 先剔除 session id 行 (含随机hex, 会误匹配 429 等数字)
        low_clean = "\n".join(
            l for l in low.split("\n")
            if "session_" not in l and "resume this session" not in l
        )

        # 只有在返回码非0 或 明确错误措辞时才判定服务异常
        has_error_context = returncode != 0 or any(
            kw in low_clean for kw in ["error", "failed", "错误", "失败", "exception"]
        )

        if has_error_context:
            if any(k in low_clean for k in ["401", "unauthorized", "oauth",
                                            "token expired", "authentication failed"]):
                return ERR_AUTH
            if any(k in low_clean for k in ["429", "rate limit", "quota exceeded",
                                            "too many requests", "配额"]):
                return ERR_QUOTA
            if any(k in low_clean for k in ["connection refused", "proxy error",
                                            "econnrefused", "dns", "unreachable",
                                            "network error", "代理错误"]):
                return ERR_NETWORK
            if returncode != 0:
                return ERR_CRASH

        if not output or len(output.strip()) < 10:
            return ERR_EMPTY
        return ERR_NONE

    def health_check(self, timeout=60):
        """轻量探活: 发一个极简请求确认服务可用。返回 (ok, reason)。"""
        workdir = PROJECT_DIR / "kimi_sessions" / f"health_{int(time.time())}"
        workdir.mkdir(parents=True, exist_ok=True)
        out_f = workdir / "health.txt"
        cmd = [str(KIMI_CLI), "-m", KIMI_MODEL, "--prompt", "reply OK"]
        timed_out = False
        rc = -1
        try:
            with open(out_f, "w") as f:
                r = subprocess.run(cmd, env=self._build_env(), timeout=timeout,
                                   cwd=str(workdir), stdout=f, stderr=subprocess.STDOUT)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        output = out_f.read_text() if out_f.exists() else ""
        err_type = self._classify_error(rc, output, timed_out)

        ok = err_type == ERR_NONE
        self.state["last_check"] = datetime.now().isoformat()
        self.state["status"] = "healthy" if ok else f"unhealthy({err_type})"
        self._save_state()
        return ok, err_type

    def _try_recover_auth(self):
        """OAuth 过期时尝试刷新 (kimi 有自动刷新, 这里触发一次探活)。"""
        # kimi CLI 通常自动刷新 oauth; 记录事件即可
        print("  🔧 [watchdog] 检测到认证异常, OAuth 可能过期, 尝试探活刷新...")
        ok, _ = self.health_check(timeout=60)
        return ok

    def call_with_retry(self, prompt, timeout=480, max_retries=2, base_backoff=15):
        """
        带重试的 K3 调用。返回 (output_or_None, err_type)。
        - 可恢复异常 (超时/网络/空/配额) 自动重试, 指数退避
        - 认证异常尝试刷新一次
        - 崩溃异常不重试 (代码/逻辑问题, 重试无用)
        """
        self.state["total_calls"] += 1
        last_err = ERR_NONE

        for attempt in range(max_retries + 1):
            workdir = PROJECT_DIR / "kimi_sessions" / f"v3_{int(time.time())}_{attempt}"
            workdir.mkdir(parents=True, exist_ok=True)
            out_f = workdir / "stdout.txt"
            cmd = [str(KIMI_CLI), "-m", KIMI_MODEL, "--prompt", prompt]
            timed_out = False
            rc = -1
            t0 = time.time()
            try:
                with open(out_f, "w") as f:
                    r = subprocess.run(cmd, env=self._build_env(), timeout=timeout,
                                       cwd=str(workdir), stdout=f, stderr=subprocess.STDOUT)
                rc = r.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            elapsed = time.time() - t0
            output = out_f.read_text() if out_f.exists() else ""
            err_type = self._classify_error(rc, output, timed_out)

            if err_type == ERR_NONE:
                self.state["consecutive_failures"] = 0
                self.state["last_error_type"] = None
                self._save_state()
                print(f"  ✓ [watchdog] K3 成功 (尝试{attempt+1}, 耗时{elapsed:.0f}s)")
                return output, ERR_NONE

            # 失败处理
            last_err = err_type
            self._record_failure(err_type, f"attempt{attempt+1} elapsed{elapsed:.0f}s")
            print(f"  ⚠️ [watchdog] K3 失败: {err_type} (尝试{attempt+1}/{max_retries+1}, 耗时{elapsed:.0f}s)")

            if err_type == ERR_AUTH:
                if self._try_recover_auth():
                    print("  🔧 [watchdog] 认证恢复, 重试...")
                    continue
                else:
                    break  # 认证无法恢复, 停止
            if err_type == ERR_CRASH:
                break  # 崩溃不重试
            if err_type in RECOVERABLE and attempt < max_retries:
                backoff = base_backoff * (2 ** attempt)
                print(f"  ⏳ [watchdog] {backoff}s 后重试...")
                time.sleep(backoff)
                continue
            break

        self._maybe_alert()
        return None, last_err

    def _record_failure(self, err_type, msg):
        self.state["total_failures"] += 1
        self.state["consecutive_failures"] += 1
        self.state["last_error_type"] = err_type
        self.state["last_error_msg"] = msg
        hist = self.state.get("error_history", [])
        hist.append({"time": datetime.now().isoformat(), "type": err_type, "msg": msg})
        self.state["error_history"] = hist[-50:]  # 保留最近50条
        self._save_state()

    def _maybe_alert(self, threshold=3):
        """连续失败超阈值时告警。"""
        if self.state["consecutive_failures"] >= threshold:
            msg = (f"🚨 Kimi/K3 服务连续失败 {self.state['consecutive_failures']} 次! "
                   f"最近错误: {self.state['last_error_type']} ({self.state['last_error_msg']})")
            print(f"\n{msg}\n")
            if self.alert_callback:
                try:
                    self.alert_callback(msg)
                except Exception as e:
                    print(f"  告警回调失败: {e}")

    def get_summary(self):
        """返回健康摘要 dict, 供汇报机制使用。"""
        s = self.state
        fail_rate = (s["total_failures"] / s["total_calls"] * 100) if s["total_calls"] else 0
        return {
            "status": s["status"],
            "last_check": s["last_check"],
            "total_calls": s["total_calls"],
            "total_failures": s["total_failures"],
            "fail_rate": round(fail_rate, 1),
            "consecutive_failures": s["consecutive_failures"],
            "last_error_type": s["last_error_type"],
            "recent_errors": s.get("error_history", [])[-5:],
        }


if __name__ == "__main__":
    # 独立探活模式
    wd = KimiWatchdog()
    print("=== Kimi/K3 服务健康检查 ===")
    ok, reason = wd.health_check()
    print(f"状态: {'✓ 健康' if ok else '✗ 异常'} ({reason})")
    print(f"\n健康摘要:")
    print(json.dumps(wd.get_summary(), ensure_ascii=False, indent=2))
