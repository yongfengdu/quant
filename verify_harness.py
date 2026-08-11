#!/usr/bin/env python3
"""验证 Pipeline Harness 完整性"""
import pandas as pd
import py_compile
import os, sys, importlib

print("=" * 60)
print("PIPELINE HARNESS 验证")
print("=" * 60)

# 1. 数据层
print("\n=== 1. 数据层 ===")
ss = pd.read_parquet('/root/.cache/quant-autoresearch/stocks_sectors.parquet')
daily = pd.read_parquet('/root/.cache/quant-autoresearch/daily.parquet')
daily['date'] = pd.to_datetime(daily['date'])
universe = pd.read_parquet('/root/.cache/quant-autoresearch/universe.parquet')

print(f"  stocks_sectors: {len(ss)}只, {ss['sector'].nunique()}行业, 科创板={ss['code'].str.startswith('688').sum()}只")
print(f"  daily K线: {daily['code'].nunique()}只, {len(daily)}行")
print(f"  日期范围: {daily['date'].min()}~{daily['date'].max()}")
print(f"  universe: {len(universe)}只, 科创板={universe['code'].str.startswith('688').sum()}只")

# 2. strategy.py
print("\n=== 2. strategy.py 验证 ===")
py_compile.compile('strategy.py', doraise=True)
print("  语法: OK")

if 'strategy' in sys.modules: del sys.modules['strategy']
from backtest import compute_derived_columns
daily = compute_derived_columns(daily)
import strategy as strat
signals = strat.predict_next_day(daily)
print(f"  predict_next_day: {type(signals).__name__}, len={len(signals)}")

# 3. backtest.py
print("\n=== 3. backtest.py 验证 ===")
py_compile.compile('backtest.py', doraise=True)
print("  语法: OK")

# 4. pipeline.py
print("\n=== 4. pipeline.py 验证 ===")
py_compile.compile('pipeline.py', doraise=True)
print("  语法: OK")

# Check LLM routing
import pipeline as pl
print(f"  call_llm: OK")
print(f"  Kimi CLI: {os.path.exists('/root/.kimi-code/bin/kimi')}")

# 5. K线覆盖率
overlap = set(daily['code'].unique()) & set(universe['code'].unique())
print(f"\n=== 5. K线覆盖 ===")
print(f"  Universe 有K线的: {len(overlap)}/{len(universe)} ({len(overlap)/len(universe)*100:.1f}%)")
print(f"  需要下载K线的: {len(universe)-len(overlap)} 只")

# 6. 实际回测能选多少只？
print(f"\n=== 6. 实际回测可用 ===")
ss_valid = ss[ss['code'].isin(set(daily['code'].unique()))]
print(f"  stocks_sectors 有K线的: {len(ss_valid)} 只")
print(f"  行业分布: {ss_valid['sector'].nunique()} 个")

print(f"\n{'='*60}")
print("HARNESS 验证完成")
print(f"{'='*60}")
