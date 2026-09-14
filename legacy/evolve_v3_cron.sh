#!/bin/bash
# ============================================================
# evolve_v3 夜间进化 cron 包装脚本
# ============================================================
# 调度: 每工作日凌晨 02:00
# 职责: 用 K3 单agent进化生成候选策略, 带 watchdog 监控 + 微信汇报
#
# 重要解耦原则:
#   进化产出的策略只写入 islands/*/strategies/ (候选)
#   不自动加入 island_router.ISLAND_STRATEGIES (上线清单)
#   需人工/独立验证后才上线 → 进化出问题也污染不到线上预测
# ============================================================

cd /root/quant-autoresearch || exit 1

export PYTHONPATH=/root/quant-autoresearch

# 轮流进化三个岛屿 (每天一个, 按星期几分配)
DOW=$(date +%u)   # 1=周一 ... 5=周五
case $DOW in
    1|4) ISLAND=range ;;   # 周一/周四: 震荡
    2|5) ISLAND=bear ;;    # 周二/周五: 熊市
    3)   ISLAND=bull ;;    # 周三: 牛市
    *)   ISLAND=range ;;
esac

echo "=== evolve_v3 夜间进化 $(date '+%Y-%m-%d %H:%M') ==="
echo "目标岛屿: $ISLAND"

# 运行进化 (3候选, 480s超时, 带微信汇报)
# watchdog 会在 K3 服务异常时告警; 完整性校验+Gate 保证产出质量
# --notify: evolve_v3 内部通过 notifier 直接推送汇报到微信
/usr/bin/python3 evolve_v3.py \
    --island "$ISLAND" \
    --n 3 \
    --timeout 480 \
    --retries 2 \
    --test-days 250 \
    --notify 2>&1

# 同步 state.json (让新候选策略反映到岛屿状态)
/usr/bin/python3 sync_island_state.py 2>&1

echo "=== 进化完成 $(date '+%H:%M') ==="
echo ""
echo "提示: 新策略已写入 islands/island_${ISLAND}/strategies/ (候选状态)"
echo "      需人工验证后手动加入 island_router.ISLAND_STRATEGIES 才会上线"
