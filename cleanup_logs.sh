#!/bin/bash
#===============================================================================
# 日志清理脚本 - 防止磁盘爆满
# 
# 功能:
# 1. 清理超过指定大小的日志文件
# 2. 清理超过指定天数的临时文件
# 3. 清理 git/cache 中的大文件
# 4. 发送告警（如果磁盘使用超过阈值）
#
# 运行时间: 每天凌晨 1:00
#===============================================================================

set -e

# 配置
LOG_DIR="/root/quant-autoresearch"
CACHE_DIR="/root/.cache"
TMP_DIR="/tmp"
MAX_LOG_SIZE_MB=10          # 日志文件最大 10MB
MAX_LOG_LINES=2000          # 保留最新 2000 行
MAX_CACHE_DAYS=7            # 缓存文件保留 7 天
DISK_WARN_PERCENT=80        # 磁盘使用超过 80% 告警
DISK_CRIT_PERCENT=90        # 磁盘使用超过 90% 强制清理

# 日志
CLEANUP_LOG="/root/quant-autoresearch/cleanup.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 开始清理 ===" >> "$CLEANUP_LOG"

#-------------------------------------------------------------------------------
# 函数：获取磁盘使用百分比
#-------------------------------------------------------------------------------
get_disk_usage() {
    df / | tail -1 | awk '{print $5}' | tr -d '%'
}

#-------------------------------------------------------------------------------
# 函数：清理大日志文件
#-------------------------------------------------------------------------------
cleanup_large_logs() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检查大日志文件..." >> "$CLEANUP_LOG"
    
    # 查找超过阈值的日志文件
    find "$LOG_DIR" -name "*.log" -type f -size +${MAX_LOG_SIZE_MB}M 2>/dev/null | while read logfile; do
        size=$(du -h "$logfile" | cut -f1)
        echo "  发现大日志: $logfile ($size)" >> "$CLEANUP_LOG"
        
        # 保留最新 N 行
        tail -n $MAX_LOG_LINES "$logfile" > "${logfile}.tmp"
        mv "${logfile}.tmp" "$logfile"
        
        new_size=$(du -h "$logfile" | cut -f1)
        echo "  已截断: $logfile ($size -> $new_size)" >> "$CLEANUP_LOG"
    done
    
    # 特别检查 cron_daily.log（历史问题文件）
    if [ -f "$LOG_DIR/cron_daily.log" ]; then
        size_bytes=$(stat -c%s "$LOG_DIR/cron_daily.log" 2>/dev/null || echo 0)
        if [ "$size_bytes" -gt 10485760 ]; then  # 10MB
            tail -n $MAX_LOG_LINES "$LOG_DIR/cron_daily.log" > "$LOG_DIR/cron_daily.log.tmp"
            mv "$LOG_DIR/cron_daily.log.tmp" "$LOG_DIR/cron_daily.log"
            echo "  强制截断 cron_daily.log" >> "$CLEANUP_LOG"
        fi
    fi
}

#-------------------------------------------------------------------------------
# 函数：清理临时文件
#-------------------------------------------------------------------------------
cleanup_temp_files() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 清理临时文件..." >> "$CLEANUP_LOG"
    
    # 清理 /tmp 中超过 1 天的 Python 临时文件
    deleted=$(find /tmp -name "tmp*.py" -mtime +1 -delete -print 2>/dev/null | wc -l)
    echo "  删除 /tmp/*.py: $deleted 个" >> "$CLEANUP_LOG"
    
    # 清理 /tmp 中超过 7 天的所有文件
    deleted=$(find /tmp -type f -mtime +7 -delete -print 2>/dev/null | wc -l)
    echo "  删除 /tmp 旧文件: $deleted 个" >> "$CLEANUP_LOG"
    
    # 清理 kimi_sessions 中超过 7 天的文件
    if [ -d "$LOG_DIR/kimi_sessions" ]; then
        deleted=$(find "$LOG_DIR/kimi_sessions" -type f -mtime +$MAX_CACHE_DAYS -delete -print 2>/dev/null | wc -l)
        echo "  删除 kimi_sessions: $deleted 个" >> "$CLEANUP_LOG"
    fi
    
    # 清理 rounds 中超过 30 天的文件（保留更久用于分析）
    if [ -d "$LOG_DIR/rounds" ]; then
        deleted=$(find "$LOG_DIR/rounds" -type f -mtime +30 -delete -print 2>/dev/null | wc -l)
        echo "  删除 rounds 旧文件: $deleted 个" >> "$CLEANUP_LOG"
    fi
}

#-------------------------------------------------------------------------------
# 函数：清理 Python 缓存
#-------------------------------------------------------------------------------
cleanup_python_cache() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 清理 Python 缓存..." >> "$CLEANUP_LOG"
    
    # 清理 __pycache__
    deleted=$(find "$LOG_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; echo $?)
    
    # 清理 .pyc 文件
    deleted=$(find "$LOG_DIR" -name "*.pyc" -delete -print 2>/dev/null | wc -l)
    echo "  删除 .pyc: $deleted 个" >> "$CLEANUP_LOG"
}

#-------------------------------------------------------------------------------
# 函数：紧急清理（磁盘严重不足时）
#-------------------------------------------------------------------------------
emergency_cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 紧急清理模式!" >> "$CLEANUP_LOG"
    
    # 清空所有日志文件（保留 100 行）
    find "$LOG_DIR" -name "*.log" -type f 2>/dev/null | while read logfile; do
        tail -n 100 "$logfile" > "${logfile}.tmp" 2>/dev/null
        mv "${logfile}.tmp" "$logfile" 2>/dev/null
    done
    
    # 清理 pip 缓存
    rm -rf /root/.cache/pip/* 2>/dev/null
    
    # 清理旧的 hermes 会话
    find /root/.hermes -name "*.log" -mtime +3 -delete 2>/dev/null
    
    # 清理 git gc
    cd "$LOG_DIR" && git gc --aggressive --prune=now 2>/dev/null || true
    
    echo "  紧急清理完成" >> "$CLEANUP_LOG"
}

#-------------------------------------------------------------------------------
# 主流程
#-------------------------------------------------------------------------------

# 获取当前磁盘使用率
DISK_USAGE=$(get_disk_usage)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 磁盘使用率: ${DISK_USAGE}%" >> "$CLEANUP_LOG"

# 根据磁盘使用率决定清理策略
if [ "$DISK_USAGE" -ge "$DISK_CRIT_PERCENT" ]; then
    echo "⚠️ 磁盘使用率 ${DISK_USAGE}% >= ${DISK_CRIT_PERCENT}%，执行紧急清理" >> "$CLEANUP_LOG"
    emergency_cleanup
    cleanup_large_logs
    cleanup_temp_files
    cleanup_python_cache
elif [ "$DISK_USAGE" -ge "$DISK_WARN_PERCENT" ]; then
    echo "⚠️ 磁盘使用率 ${DISK_USAGE}% >= ${DISK_WARN_PERCENT}%，执行常规清理" >> "$CLEANUP_LOG"
    cleanup_large_logs
    cleanup_temp_files
else
    # 正常清理
    cleanup_large_logs
    cleanup_temp_files
fi

# 清理后再次检查
NEW_DISK_USAGE=$(get_disk_usage)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 清理后磁盘使用率: ${NEW_DISK_USAGE}%" >> "$CLEANUP_LOG"

# 保持清理日志自身不要太大
if [ -f "$CLEANUP_LOG" ]; then
    size_bytes=$(stat -c%s "$CLEANUP_LOG" 2>/dev/null || echo 0)
    if [ "$size_bytes" -gt 1048576 ]; then  # 1MB
        tail -n 500 "$CLEANUP_LOG" > "${CLEANUP_LOG}.tmp"
        mv "${CLEANUP_LOG}.tmp" "$CLEANUP_LOG"
    fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 清理完成 ===" >> "$CLEANUP_LOG"
echo "" >> "$CLEANUP_LOG"

# 输出摘要（供 cron 邮件或 hermes 发送）
echo "磁盘清理完成: ${DISK_USAGE}% -> ${NEW_DISK_USAGE}%"
