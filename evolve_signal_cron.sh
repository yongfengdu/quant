#!/bin/bash
# ============================================================
# 信号进化 cron 包装脚本 (每周自动找新信号)
# ============================================================
# 调度: 每周六 02:00 (避开交易日盘后, K3 调用成本低)
# 职责: 让 LLM 提因子假设 → 五闸门验证 → 注册进信号生命周期
# 产出: data/signal_registry.json 里的新信号 (paper 状态, 待衰减监控转正)
# ============================================================
cd /root/quant-autoresearch || exit 1

echo "=== 信号进化 $(date '+%Y-%m-%d %H:%M') ==="
/root/quant-autoresearch/.venv/bin/python -u /root/quant-autoresearch/run_evolve.py 3 2>&1
echo "=== 进化完成 $(date '+%H:%M') ==="
