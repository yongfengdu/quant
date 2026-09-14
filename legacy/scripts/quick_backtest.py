#!/usr/bin/env python3
"""快速回测验证 harness — 最近1年数据"""
import pandas as pd, numpy as np, time, sys, importlib
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"

stocks_sectors = pd.read_parquet(CACHE_DIR / "stocks_sectors.parquet")
daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
daily["date"] = pd.to_datetime(daily["date"])
from backtest import compute_derived_columns
daily = compute_derived_columns(daily)

cutoff = daily["date"].max() - pd.Timedelta(days=365)
daily = daily[daily["date"] >= cutoff].copy()

dates = sorted(daily["date"].unique())
print(f"回测: {len(dates)} 天, {daily['code'].nunique()} 只股票")

daily_dict = {}
for code in daily["code"].unique():
    daily_dict[code] = daily[daily["code"] == code].sort_values("date")

if "strategy" in sys.modules:
    del sys.modules["strategy"]
from strategy import predict_next_day

start_time = time.time()
positions = {}
daily_pnl = []
trades = []
capital = 1_000_000
portfolio_value = 1_000_000
peak_value = 1_000_000

TOP_N = 8
BUY_N = 3
COMMISSION = 0.001
SLIPPAGE = 0.001

for i, date in enumerate(dates):
    if i + 1 >= len(dates):
        break
    next_date = dates[i + 1]

    tradable = set(daily[daily["date"] == date]["code"].unique())
    latest_cap = stocks_sectors[stocks_sectors["code"].isin(tradable)][["code", "sector", "circ_cap"]]
    universe = latest_cap.groupby("sector").head(TOP_N)

    if len(universe) == 0:
        daily_pnl.append({"date": date, "value": portfolio_value})
        continue

    hist = []
    for code in universe["code"]:
        if code in daily_dict:
            d = daily_dict[code][daily_dict[code]["date"] <= date]
            if len(d) > 0:
                hist.append(d)
    if not hist:
        daily_pnl.append({"date": date, "value": portfolio_value})
        continue
    all_hist = pd.concat(hist, ignore_index=True)

    signals = predict_next_day(all_hist)

    for code in list(positions.keys()):
        pos = positions[code]
        if pos["entry_date"] <= date:
            nxt = daily[(daily["code"] == code) & (daily["date"] == next_date)]
            if len(nxt) > 0:
                sell = nxt["close"].values[0] * (1 - COMMISSION - SLIPPAGE)
                capital += sell * pos["quantity"]
                pnl = (sell - pos["entry_price"]) / pos["entry_price"]
                trades.append({"code": code, "pnl": pnl})
                positions.pop(code)

    if len(signals) > 0:
        top = signals.nlargest(min(BUY_N, len(signals)))
        for code, sig in top.items():
            today = daily[(daily["code"] == code) & (daily["date"] == date)]
            if len(today) > 0:
                buy = today["close"].values[0] * (1 + COMMISSION + SLIPPAGE)
                alloc = capital / max(len(top), 1)
                qty = int(alloc / buy / 100) * 100
                if qty > 0 and buy * qty <= capital:
                    capital -= buy * qty
                    positions[code] = {"entry_price": buy, "quantity": qty, "entry_date": date + pd.Timedelta(days=1), "signal": sig}

    pv = 0
    for code, pos in positions.items():
        lat = daily[(daily["code"] == code) & (daily["date"] <= date)]
        if len(lat) > 0:
            pv += lat["close"].values[-1] * pos["quantity"]
    portfolio_value = capital + pv
    daily_pnl.append({"date": date, "value": portfolio_value, "positions": len(positions)})
    peak_value = max(peak_value, portfolio_value)

    if (i + 1) % 100 == 0:
        print(f"  Day {i+1}/{len(dates)} ({date.date()}) - PV: {portfolio_value/10000:.1f}w, Pos: {len(positions)}")

elapsed = time.time() - start_time
result = pd.DataFrame(daily_pnl)
result["daily_return"] = result["value"].pct_change().fillna(0)

total_return = portfolio_value / 1_000_000 - 1
years = len(result) / 252
annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
excess = result["daily_return"] - 0.02 / 252
sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
cummax = result["value"].cummax()
drawdown = (result["value"] - cummax) / cummax
max_dd = drawdown.min()
win_rate = (result["daily_return"] > 0).sum() / len(result)

print(f"\n{'='*50}")
print(f"HARNESS 验证 — Mini Backtest")
print(f"{'='*50}")
print(f"  val_sharpe:      {sharpe:.4f}")
print(f"  annual_return:   {annual_return:.4f}")
print(f"  total_return:    {total_return:.4f}")
print(f"  max_drawdown:    {max_dd:.4f}")
print(f"  win_rate_daily:  {win_rate:.4f}")
print(f"  total_trades:    {len(trades)}")
print(f"  trading_days:    {len(result)}")
print(f"  training_seconds: {elapsed:.1f}")
print(f"\n✅ Harness 验证通过 — backtest 能正常执行并返回指标")
