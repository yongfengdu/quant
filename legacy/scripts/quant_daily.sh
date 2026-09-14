#!/bin/bash
#===============================================================================
# 量化策略每日汇报脚本 (hermes cron 专用)
# 
# 功能:
# 1. 验证昨日预测结果
# 2. 生成今日预测信号
# 3. 分析错过的黄金机会
# 4. 输出简洁报告到 stdout (hermes 会发送到微信)
#
# 运行时间: 每日 16:30 (收盘后)
#===============================================================================

cd /root/quant-autoresearch
PYTHON=/usr/bin/python3

# 静默执行数据更新和验证
$PYTHON daily_tracker.py verify > /dev/null 2>&1

# 获取预测结果
PREDICTION=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
from daily_tracker import load_data, find_next_trading_days
from infra.strategy_pool import StrategyPool, AdaptiveRouter
import pandas as pd
from pathlib import Path

CACHE_DIR = Path.home() / '.cache/quant-autoresearch'

try:
    daily = pd.read_parquet(CACHE_DIR / 'daily.parquet')
    daily['date'] = pd.to_datetime(daily['date'])
    latest_date = daily['date'].max()
    
    universe = pd.read_parquet(CACHE_DIR / 'universe.parquet')
    universe_codes = set(universe['code'].values)
    daily_filtered = daily[daily['code'].isin(universe_codes)].copy()
    
    pool = StrategyPool()
    n = pool.load_verified_strategies()
    router = AdaptiveRouter(pool)
    result = router.get_signals(daily_filtered, date=latest_date, mode='weighted')
    
    ms = result['market_state']
    signals = result['final_signals']
    
    print(f'[ 量化策略日报 ]')
    print(f'{latest_date.date()}')
    print()
    print(f'市场: {ms[\"state_name\"]}')
    print(f'20日: {ms[\"ret_20d\"]:+.1f}% | 5日: {ms[\"ret_5d\"]:+.1f}%')
    print()
    
    if signals:
        sector_map = dict(zip(universe['code'], universe['sector']))
        next_dates = []
        all_dates = sorted(daily['date'].unique())
        idx = list(all_dates).index(latest_date)
        if idx + 2 < len(all_dates):
            entry = all_dates[idx + 1]
            exit_d = all_dates[idx + 2]
            print(f'买入: {entry.date()} 开盘')
            print(f'卖出: {exit_d.date()} 开盘')
            print()
        
        print('-- 推荐 --')
        for i, sig in enumerate(signals[:3], 1):
            code = sig['code']
            sector = sector_map.get(code, '?')
            print(f'{i}. {code} ({sector})')
    else:
        print('今日无信号')
        print('(策略条件未触发)')
        
except Exception as e:
    print(f'执行出错: {e}')
" 2>&1)

echo "$PREDICTION"

# 获取反馈分析
FEEDBACK=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
from daily_feedback import load_data, calculate_actual_returns, analyze_missed_opportunities
import pandas as pd
from pathlib import Path

CACHE_DIR = Path.home() / '.cache/quant-autoresearch'
PREDICTIONS_FILE = Path('/root/quant-autoresearch/daily_predictions.csv')

try:
    daily, universe_codes = load_data()
    all_dates = sorted(daily['date'].unique())
    
    if len(all_dates) >= 3:
        signal_date = all_dates[-3]
        actual = calculate_actual_returns(daily, signal_date, universe_codes)
        
        if actual is not None and len(actual) > 0:
            preds = pd.read_csv(PREDICTIONS_FILE, dtype={'code': str}) if PREDICTIONS_FILE.exists() else pd.DataFrame()
            analysis = analyze_missed_opportunities(actual, preds, signal_date)
            
            golden = analysis['golden_opportunities']
            hit = analysis['hit_count']
            missed = analysis['missed_count']
            
            if golden > 0:
                print()
                print('-- 昨日复盘 --')
                print(f'黄金机会: {golden}只')
                print(f'命中: {hit} | 错过: {missed}')
                
                if missed > 0 and analysis['missed_codes']:
                    codes = ', '.join(analysis['missed_codes'][:3])
                    print(f'错过: {codes}')
except Exception as e:
    pass
" 2>&1)

echo "$FEEDBACK"

# 底部说明
echo ""
echo "---"
echo "详情: python daily_tracker.py report"
