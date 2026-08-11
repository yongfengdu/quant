"""Fast batch backtest for all island strategies with Tiered Gate."""
import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import sys

cache = Path.home() / '.cache/quant-autoresearch'
daily = pd.read_parquet(cache / 'daily.parquet')
daily['date'] = pd.to_datetime(daily['date'])
daily = daily.sort_values(['code', 'date']).reset_index(drop=True)

g = daily.groupby('code', sort=False)
daily['next1_open'] = g['open'].shift(-1)
daily['next2_open'] = g['open'].shift(-2)
daily['target_return'] = (daily['next2_open'] - daily['next1_open']) / daily['next1_open']

dates = sorted(daily['date'].unique())
test_dates = dates[-120:]

# Pre-index: for each test date, the historical slice end position
# Group daily by date for fast target lookup
date_return_map = {}
for date in test_dates:
    day_data = daily[daily['date'] == date][['code', 'target_return']]
    date_return_map[date] = dict(zip(day_data['code'], day_data['target_return']))

# Pre-slice histories once (share across strategies via date boundaries)
# Build cumulative index by date for slicing
daily_sorted_by_date = daily.sort_values('date').reset_index(drop=True)
date_end_idx = {}
for date in test_dates:
    # position where date <= given date ends
    date_end_idx[date] = daily_sorted_by_date['date'].searchsorted(date, side='right')


def backtest(path):
    spec = importlib.util.spec_from_file_location('strategy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict_fn = module.predict_next_day
    all_trades = []
    for date in test_dates:
        end = date_end_idx[date]
        hist = daily_sorted_by_date.iloc[:end]
        try:
            signals = predict_fn(hist)
            if signals is None or len(signals) == 0:
                continue
            if isinstance(signals, pd.Series):
                signals = signals[signals >= 0.3]
            if len(signals) == 0:
                continue
            top_codes = signals.nlargest(min(5, len(signals))).index.tolist()
            ret_map = date_return_map[date]
            for code in top_codes:
                r = ret_map.get(code)
                if r is not None and pd.notna(r):
                    all_trades.append(r)
        except Exception:
            continue
    if not all_trades:
        return None
    arr = np.array(all_trades)
    return {'trades': len(arr), 'accuracy': (arr > 0.03).mean() * 100,
            'avg_return': arr.mean() * 100, 'win_rate': (arr > 0).mean() * 100}


def check_gate(trades, acc, avg):
    if avg <= 0:
        return 'FAIL(neg)'
    if acc >= 30 and trades >= 7:
        return 'PASS(30%)'
    if acc >= 25 and trades >= 13:
        return 'PASS(25%)'
    if acc >= 20 and trades >= 26:
        return 'PASS(20%)'
    if acc >= 15 and trades >= 101:
        return 'PASS(15%)'
    return 'FAIL'


islands = sys.argv[1:] if len(sys.argv) > 1 else ['bull', 'range', 'bear']
for island in islands:
    print(f'=== {island.upper()} ISLAND ===')
    print(f"{'Strategy':<14} {'Trades':>7} {'Accuracy':>9} {'AvgRet':>8} {'Gate':>12}")
    print('-' * 54)
    sdir = Path(f'islands/island_{island}/strategies')
    for path in sorted(sdir.glob('*.py')):
        r = backtest(str(path))
        if r:
            gate = check_gate(r['trades'], r['accuracy'], r['avg_return'])
            print(f"{path.stem:<14} {r['trades']:>7} {r['accuracy']:>8.1f}% {r['avg_return']:>7.2f}% {gate:>12}")
        else:
            print(f'{path.stem:<14} {"no trades":>40}')
    print(flush=True)
