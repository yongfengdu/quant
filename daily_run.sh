#!/bin/bash
#===============================================================================
# 量化系统每日执行脚本
# 
# 执行顺序: 更新数据 → 验证预测 → 生成新预测
# 运行时间: 每日15:30后 (A股收盘后)
# 
# Usage:
#   ./daily_run.sh           # 正常执行
#   ./daily_run.sh --force   # 强制更新数据
#===============================================================================

set -e

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/cron_daily.log"
CACHE_DIR="$HOME/.cache/quant-autoresearch"
PYTHON="/usr/bin/python3"

# 代理设置 (如果需要)
# export HTTPS_PROXY=http://proxy-dmz.intel.com:912
# export HTTP_PROXY=http://proxy-dmz.intel.com:912

cd "$SCRIPT_DIR"

#-------------------------------------------------------------------------------
# 日志函数 (只写文件，不输出到 stdout 避免重复)
#-------------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log_section() {
    echo "" >> "$LOG_FILE"
    echo "========================================" >> "$LOG_FILE"
    log "$1"
    echo "========================================" >> "$LOG_FILE"
}

#-------------------------------------------------------------------------------
# 主流程
#-------------------------------------------------------------------------------
log_section "🚀 量化系统每日任务开始"

# 检查Python环境
if [ ! -f "$PYTHON" ]; then
    PYTHON=$(which python3)
fi
log "Python: $PYTHON"

#-------------------------------------------------------------------------------
# Step 1: 更新数据
#-------------------------------------------------------------------------------
log_section "📥 Step 1: 更新数据"

# 检查是否需要强制更新数据 (health_monitor 写入的标记)
FORCE_UPDATE=""
if [ -f "$SCRIPT_DIR/.need_data_update" ]; then
    log "发现数据更新标记，执行强制更新"
    FORCE_UPDATE="--force"
    rm -f "$SCRIPT_DIR/.need_data_update"
fi

if [ "$1" = "--force" ]; then
    FORCE_UPDATE="--force"
    log "强制更新模式"
fi

if [ -f "$CACHE_DIR/daily.parquet" ]; then
    LATEST_DATE=$($PYTHON -c "
import pandas as pd
d = pd.read_parquet('$CACHE_DIR/daily.parquet')
print(d['date'].max().strftime('%Y-%m-%d'))
" 2>/dev/null || echo "unknown")
    log "当前数据最新日期: $LATEST_DATE"
else
    log "数据文件不存在，需要初始化"
    FORCE_UPDATE="--force"
fi

# 执行数据更新 (增加超时到30分钟，并发优化后通常5-10分钟完成)
log "开始更新数据..."
if timeout 1800 $PYTHON update_data.py $FORCE_UPDATE >> "$LOG_FILE" 2>&1; then
    log "✅ 数据更新完成"
else
    log "⚠️ 数据更新失败或超时"
fi

# 验证更新后的数据日期
if [ -f "$CACHE_DIR/daily.parquet" ]; then
    NEW_DATE=$($PYTHON -c "
import pandas as pd
d = pd.read_parquet('$CACHE_DIR/daily.parquet')
print(d['date'].max().strftime('%Y-%m-%d'))
" 2>/dev/null || echo "unknown")
    log "更新后数据日期: $NEW_DATE"
fi

#-------------------------------------------------------------------------------
# Step 2: 验证预测
#-------------------------------------------------------------------------------
log_section "🔍 Step 2: 验证预测"

log "验证2天前的预测结果..."
if $PYTHON daily_tracker.py verify >> "$LOG_FILE" 2>&1; then
    log "✅ 验证完成"
else
    log "⏭️ 无可验证的预测或验证失败"
fi

#-------------------------------------------------------------------------------
# Step 3: 生成预测
#-------------------------------------------------------------------------------
log_section "🔮 Step 3: 生成预测"

# 预测前健康检查 (数据新鲜度/完整度/市场状态/信号产出), 严重异常时告警
log "运行线上预测健康检查..."
HEALTH_OUTPUT=$($PYTHON predict_health.py --notify 2>&1)
HEALTH_CODE=$?
echo "$HEALTH_OUTPUT" >> "$LOG_FILE"
if [ $HEALTH_CODE -eq 2 ]; then
    log "🔴 健康检查严重异常, 已告警。仍尝试预测但结果可能不可靠"
elif [ $HEALTH_CODE -eq 1 ]; then
    log "⚠️ 健康检查有警告 (详见日志)"
else
    log "✅ 健康检查通过"
fi

log "运行自适应策略路由器 (岛屿聚合)..."
PREDICTION_RESULT=$($PYTHON daily_tracker.py adaptive 2>&1)
echo "$PREDICTION_RESULT" >> "$LOG_FILE"

# 检查是否有预测结果
if echo "$PREDICTION_RESULT" | grep -q "已保存到"; then
    log "✅ 预测生成成功"
    # 提取推荐的股票
    echo "$PREDICTION_RESULT" | grep -E "^\s+[0-9]+\." | head -5
elif echo "$PREDICTION_RESULT" | grep -q "无推荐\|无信号"; then
    log "📭 今日无交易信号 (策略未触发)"
else
    log "⚠️ adaptive失败，尝试单策略..."
    if $PYTHON daily_tracker.py predict >> "$LOG_FILE" 2>&1; then
        log "✅ 单策略预测完成"
    else
        log "⚠️ 单策略也无信号"
    fi
fi

#-------------------------------------------------------------------------------
# Step 4: 策略健康检查 (可选)
#-------------------------------------------------------------------------------
log_section "📊 Step 4: 策略健康检查"

HEALTH_RESULT=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
from infra.strategy_pool import StrategyPool, StrategyMonitor
pool = StrategyPool()
pool.load_verified_strategies()
monitor = StrategyMonitor(pool)
report = monitor.health_check()
print(f\"策略状态: 活跃={report['status_summary']['active']}, 警告={report['status_summary']['warning']}, 暂停={report['status_summary']['suspended']}\")
if report['status_summary']['warning'] + report['status_summary']['suspended'] > 0:
    print('⚠️ 有策略需要关注!')
    for s in report['strategies']:
        if s['status'] != 'active':
            print(f\"  - {s['id']}: {s['status']}\")
" 2>&1)
echo "$HEALTH_RESULT" >> "$LOG_FILE"
log "$HEALTH_RESULT"

#-------------------------------------------------------------------------------
# Step 5: 每日回测反馈 (分析错过的机会)
#-------------------------------------------------------------------------------
log_section "📝 Step 5: 每日回测反馈"

log "分析昨日错过的黄金机会..."
FEEDBACK_RESULT=$($PYTHON daily_feedback.py feedback --no-lessons 2>&1)
echo "$FEEDBACK_RESULT" >> "$LOG_FILE"

# 提取关键信息
if echo "$FEEDBACK_RESULT" | grep -q "错过:"; then
    MISSED=$(echo "$FEEDBACK_RESULT" | grep "错过:" | head -1)
    log "$MISSED"
    
    # 如果有错过的机会，显示原因
    if echo "$FEEDBACK_RESULT" | grep -q "错过原因分析"; then
        echo "$FEEDBACK_RESULT" | grep -A5 "错过原因分析:" | head -6 >> "$LOG_FILE"
    fi
else
    log "📭 无黄金机会或无法分析"
fi

#-------------------------------------------------------------------------------
# 完成
#-------------------------------------------------------------------------------
log_section "✅ 每日任务完成"
log "日志文件: $LOG_FILE"
log "查看预测: cat daily_predictions.csv | tail -10"
log "查看统计: python daily_tracker.py summary"

# 显示最近的日志
echo ""
echo "=== 最近日志 ==="
tail -20 "$LOG_FILE"
