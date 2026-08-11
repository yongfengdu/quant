#!/usr/bin/env python3
"""
回测引擎 v4 — 使用精选 Universe
交易逻辑：T日收盘出信号 → T+1日开盘买入 → T+2日开盘卖出
"""

import os, sys, time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

CACHE_DIR = Path("~/.cache/quant-autoresearch").expanduser()

# ========== 固定参数 ==========
BACKTEST_START = "20210101"
BACKTEST_END = "20251231"  # 截止到2025年底，排除2026年失效期
BUY_SIGNALS = 3  # 买最强的3个信号
MIN_SIGNAL_SCORE = 0.15  # 信号门槛
COMMISSION_RATE = 0.001
SLIPPAGE = 0.001
INITIAL_CAPITAL = 1_000_000


def compute_derived_columns(daily):
    """
    可重入：保留索引列不变，覆盖已有的衍生列。
    """
    drop_cols = [c for c in daily.columns if c.startswith(("mkt_", "sector_"))
                 or c in ("prev_close", "turnover_rate", "amplitude",
                          "limit_up_flag", "limit_down_flag",
                          "mkt_ret_20d", "sector_ret_20d")]
    daily = daily.drop(columns=[c for c in drop_cols if c in daily.columns])

    # ===== 构建市场指数 & 板块指数 =====
    mkt_idx = daily.groupby("date")[["open", "close", "high", "low", "volume", "amount"]].median()
    mkt_idx.columns = [f"mkt_{c}" for c in mkt_idx.columns]
    daily = daily.merge(mkt_idx, on="date", how="left")

    sector_idx = daily.groupby(["date", "sector"])[["open", "close", "high", "low", "volume", "amount"]].median()
    sector_idx.columns = [f"sector_{c}" for c in sector_idx.columns]
    daily = daily.merge(sector_idx, on=["date", "sector"], how="left")

    # ===== 预计算常用指标 =====
    daily = daily.sort_values(["code", "date"])
    daily["prev_close"] = daily.groupby("code")["close"].shift(1)
    daily["turnover_rate"] = daily["volume"] * daily["close"] / daily["circ_cap"]
    daily["amplitude"] = (daily["high"] - daily["low"]) / daily["prev_close"]
    daily["limit_up_flag"] = (daily["close"] >= daily["prev_close"] * 1.098).astype(int)
    daily["limit_down_flag"] = (daily["close"] <= daily["prev_close"] * 0.902).astype(int)
    mkt_ret = daily.groupby("date")["mkt_close"].first().pct_change(20)
    daily["mkt_ret_20d"] = daily["date"].map(mkt_ret)
    sector_ret = daily.groupby(["date", "sector"])["sector_close"].first().reset_index()
    sector_ret["sector_ret_20d"] = sector_ret.groupby("sector")["sector_close"].transform(lambda x: x.pct_change(20))
    daily = daily.merge(sector_ret[["date", "sector", "sector_ret_20d"]], on=["date", "sector"], how="left")

    return daily


def load_data():
    """加载精选 universe + K线数据，构建快速查找索引"""
    universe_df = pd.read_parquet(CACHE_DIR / "universe.parquet")
    universe_codes = set(universe_df["code"].values)

    daily = pd.read_parquet(CACHE_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[(daily["date"] >= pd.to_datetime(BACKTEST_START)) &
                  (daily["date"] <= pd.to_datetime(BACKTEST_END))]
    daily = daily[daily["code"].isin(universe_codes)]

    daily = compute_derived_columns(daily)

    # 构建 O(1) 查询：code -> {date -> price}
    # 分别构建 open_lookup（买入价/卖出价用）和 close_lookup（策略计算用）
    open_lookup = {}
    close_lookup = {}
    daily_dict = {}
    for code in daily["code"].unique():
        cd = daily[daily["code"] == code].set_index("date").sort_index()
        open_lookup[code] = cd["open"]
        close_lookup[code] = cd["close"]
        daily_dict[code] = cd.reset_index()

    return universe_df, daily, open_lookup, close_lookup, daily_dict


def run_backtest():
    print("=" * 60)
    print("Quant Autoresearch: Backtest Engine v4")
    print("交易逻辑: T日收盘信号 → T+1开盘买入 → T+2开盘卖出")
    print("=" * 60)

    universe_df, daily, open_lookup, close_lookup, daily_dict = load_data()
    if len(daily) == 0:
        print("ERROR: No daily data.")
        return None

    dates = sorted(daily["date"].unique())
    print(f"Min signal threshold: {MIN_SIGNAL_SCORE} (signals below this are discarded)")
    print(f"\nDate range: {dates[0].date()} to {dates[-1].date()}")
    print(f"Total trading days: {len(dates)}")
    print(f"Stocks in universe: {daily['code'].nunique()}")

    date_codes = daily.groupby("date")["code"].apply(set).to_dict()

    from strategy import predict_next_day

    # ===== 信号预检：快速检查策略是否能产生信号 =====
    print("\n预检信号...")
    sample_dates = dates[40::50][:5]  # 每50天采样一次，最多5个点
    precheck_signals = 0
    for sample_date in sample_dates:
        cutoff = dates[max(0, dates.index(sample_date) - 120)]
        sample_hist = daily[(daily["date"] >= cutoff) & (daily["date"] <= sample_date)]
        try:
            signals = predict_next_day(sample_hist)
            if len(signals) > 0:
                signals = signals[signals >= MIN_SIGNAL_SCORE]
                precheck_signals += len(signals)
        except Exception as e:
            print(f"  预检失败 ({sample_date.date()}): {e}")
    
    if precheck_signals == 0:
        print(f"  ⚠️ 预检警告: 采样 {len(sample_dates)} 个日期均无信号，策略可能有问题")
    else:
        print(f"  ✓ 预检通过: 采样 {len(sample_dates)} 个日期发现 {precheck_signals} 个信号")

    print("\nRunning backtest...")
    positions = {}       # code -> {entry_price, quantity, entry_date, signal}
    pending_buys = {}    # date -> [(code, signal)]
    trades = []
    daily_pnl = []
    capital = INITIAL_CAPITAL
    portfolio_value = INITIAL_CAPITAL
    peak_value = INITIAL_CAPITAL
    start_time = time.time()
    total_signals_gen = 0

    for i, date in enumerate(dates):
        prev_date = dates[i - 1] if i >= 1 else None

        # 当日有交易的 universe 股票
        tradable = date_codes.get(date, set())
        if not tradable or len(tradable) < 10:
            continue

        # ====== 卖出：T+2 开盘卖出 ======
        # 交易逻辑：T+1 开盘买入 → T+2 开盘卖出
        # positions 的 entry_date 记录的是买入日（T+1）
        # 因此到 date = T+2 时（prev_date = T+1），卖出入场日期为 prev_date 的仓位
        if prev_date:
            for code in list(positions.keys()):
                pos = positions[code]
                if pos["entry_date"] == prev_date:
                    try:
                        if code in open_lookup and date in open_lookup[code].index:
                            sell_price = open_lookup[code].loc[date]
                            sell_price_net = sell_price * (1 - COMMISSION_RATE - SLIPPAGE)
                        else:
                            continue
                    except Exception:
                        continue

                    value = sell_price_net * pos["quantity"]
                    capital += value
                    pnl = (sell_price_net - pos["entry_price"]) / pos["entry_price"]
                    trades.append({
                        "code": code,
                        "entry_date": pos["entry_date"].date(),
                        "exit_date": date.date(),
                        "entry_price": pos["entry_price"],
                        "exit_price": sell_price_net,
                        "pnl": pnl,
                        "signal": pos.get("signal", 0),
                    })
                    positions.pop(code)

        # ====== 买入：执行 T-1 收盘生成的信号，在 T 开盘买入 ======
        if date in pending_buys:
            buy_list = pending_buys.pop(date)
            buy_list.sort(key=lambda x: x[1], reverse=True)
            buy_list = buy_list[:BUY_SIGNALS]

            allocation = capital / max(len(buy_list), 1)
            for code, signal in buy_list:
                try:
                    if code in open_lookup and date in open_lookup[code].index:
                        buy_price = open_lookup[code].loc[date]
                        buy_price_gross = buy_price * (1 + COMMISSION_RATE + SLIPPAGE)
                    else:
                        continue
                except Exception:
                    continue

                quantity = int(allocation / buy_price_gross / 100) * 100
                if quantity <= 0:
                    continue
                cost = buy_price_gross * quantity
                if cost > capital:
                    continue

                capital -= cost
                positions[code] = {
                    "entry_price": buy_price_gross,
                    "quantity": quantity,
                    "entry_date": date,
                    "signal": signal,
                }

        # ====== 生成信号：用 T 日收盘数据预测 T+1，计划 T+1 开盘买入、T+2 开盘卖出 ======
        if i < len(dates) - 2:  # 需要 T, T+1, T+2 都存在
            next_date = dates[i + 1]

            cutoff = dates[max(0, i - 120)]
            all_hist = daily[(daily["date"] >= cutoff) & (daily["date"] <= date)]

            try:
                signals = predict_next_day(all_hist)
            except Exception as e:
                print(f"\nERROR: strategy failed on {date.date()}: {e}")
                return None

            if len(signals) > 0:
                signals = signals[signals >= MIN_SIGNAL_SCORE]
            if len(signals) > 0:
                top = signals.nlargest(min(BUY_SIGNALS, len(signals)))
                pending_buys.setdefault(next_date, []).extend(
                    (code, sig) for code, sig in top.items()
                )
                total_signals_gen += len(top)

        # ====== 组合价值（用于回撤计算） ======
        positions_value = 0
        for code, pos in positions.items():
            try:
                if code in open_lookup and date in open_lookup[code].index:
                    positions_value += open_lookup[code].loc[date] * pos["quantity"]
            except Exception:
                pass

        portfolio_value = capital + positions_value
        daily_pnl.append({
            "date": date,
            "value": portfolio_value,
            "positions": len(positions),
        })
        peak_value = max(peak_value, portfolio_value)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Day {i+1}/{len(dates)} ({date.date()}) - "
                  f"Capital: {capital/10000:.1f}w, "
                  f"Positions: {len(positions)}, "
                  f"Elapsed: {elapsed:.1f}s")

    # ====== 强制平仓：用最后一个交易日收盘价 ======
    for code, pos in list(positions.items()):
        try:
            if code in close_lookup:
                final_price = close_lookup[code].iloc[-1] * (1 - COMMISSION_RATE - SLIPPAGE)
                capital += final_price * pos["quantity"]
        except Exception:
            pass
    positions.clear()
    final_value = capital

    # ========== 计算指标 ==========
    result = pd.DataFrame(daily_pnl)
    if len(result) == 0:
        print("ERROR: No trading days processed.")
        return None

    result["daily_return"] = result["value"].pct_change().fillna(0)

    total_return = final_value / INITIAL_CAPITAL - 1
    trading_days = len(result)
    years = trading_days / 252
    annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    cumulative = result["value"].cummax()
    drawdown = (result["value"] - cumulative) / cumulative
    max_drawdown = drawdown.min()

    daily_rf = 0.02 / 252
    excess = result["daily_return"] - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        trade_win_rate = (trades_df["pnl"] > 0).sum() / max(len(trades_df), 1)
        avg_trade_pnl = trades_df["pnl"].mean()
    else:
        trade_win_rate = 0.0
        avg_trade_pnl = 0.0

    # ========== 预测准确率（核心指标）==========
    predictions = []
    for t in trades:
        actual_pnl = t["pnl"]
        predictions.append({
            "code": t["code"],
            "signal_date": t["entry_date"],
            "actual_return": actual_pnl,
            "hit_3pct": actual_pnl > 0.03,
            "signal": t.get("signal", 0),
        })
    pred_df = pd.DataFrame(predictions)
    total_signals = len(pred_df)
    hits_3pct = pred_df["hit_3pct"].sum() if total_signals > 0 else 0
    accuracy_3pct = hits_3pct / total_signals * 100 if total_signals > 0 else 0
    avg_return = pred_df["actual_return"].mean() * 100 if total_signals > 0 else 0
    median_return = pred_df["actual_return"].median() * 100 if total_signals > 0 else 0

    # 也计算等权持仓的收益（按信号日期分组，每日等权买入 top N）
    if len(pred_df) > 0:
        pred_df["signal_date_dt"] = pd.to_datetime(pred_df["signal_date"])
        daily_groups = pred_df.groupby("signal_date_dt")
        daily_equal = daily_groups["actual_return"].mean()
        avg_equal_return = daily_equal.mean() * 100
        win_rate_equal = (daily_equal > 0).mean() * 100
    else:
        avg_equal_return = 0
        win_rate_equal = 0

    elapsed = time.time() - start_time

    # ========== 输出结果 ==========
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"🎯 预测准确率 (买入后开盘到卖出开盘 >3%): {accuracy_3pct:.1f}% ({hits_3pct}/{total_signals})")
    print(f"   信号平均收益: {avg_return:.2f}%")
    print(f"   信号中位收益: {median_return:.2f}%")
    print(f"   等权日均收益: {avg_equal_return:.2f}%")
    print(f"   等权日胜率:   {win_rate_equal:.1f}%")
    print(f"   年化收益:     {annual_return:.4f}")
    print(f"   最大回撤:     {max_drawdown:.4f}")
    print(f"   总交易次数:   {len(trades_df)}")
    print(f"   交易胜率:     {trade_win_rate:.2%}")
    # 机器可读行（供 pipeline 解析）
    print(f"accuracy_3pct:      {accuracy_3pct:.4f}")
    print(f"hits_3pct:          {hits_3pct:.4f}")
    print(f"avg_signal_return:  {avg_return:.4f}")
    print(f"median_signal_return: {median_return:.4f}")
    print(f"avg_equal_return:   {avg_equal_return:.4f}")
    print(f"win_rate_equal:     {win_rate_equal:.4f}")
    print(f"val_sharpe:         {sharpe:.4f}")
    print(f"annual_return:      {annual_return:.4f}")
    print(f"total_return:       {total_return:.4f}")
    print(f"max_drawdown:       {max_drawdown:.4f}")
    print(f"trade_win_rate:     {trade_win_rate:.4f}")
    print(f"avg_trade_pnl:      {avg_trade_pnl:.4f}")
    print(f"total_trades:       {len(trades_df)}")
    print(f"trading_days:       {trading_days}")
    print(f"training_seconds:   {elapsed:.1f}")

    trades_df.to_parquet(CACHE_DIR / "trades.parquet", index=False)
    pred_df.to_parquet(CACHE_DIR / "predictions.parquet", index=False)

    return True


if __name__ == "__main__":
    success = run_backtest()
    sys.exit(0 if success else 1)
