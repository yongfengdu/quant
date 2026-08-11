#!/usr/bin/env python3
"""
增量数据更新脚本 — 只拉取每个股票的最新N天K线，合并到已有数据中。

Usage:
    python update_data.py          # 增量更新所有股票
    python update_data.py --force  # 强制全量重拉（用于数据校验）
"""

import os
import sys
import time
import json
import re
import urllib.request
import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

CACHE_DIR = Path.home() / '.cache' / 'quant-autoresearch'
PROXY_URL = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", ""))

# 低调参数 (增量更新时可以更快，因为请求量小)
MIN_DELAY = 0.5      # 增量更新时降低延时
MAX_DELAY = 1.5
JITTER = 0.3
CONCURRENT_WORKERS = 3  # 并发数（降低以避免被限速）


def http_get(url, timeout=20):
    req = urllib.request.Request(url)
    req.add_header("Referer", "https://stockapp.finance.qq.com/")
    req.add_header("User-Agent", random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]))
    if PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler({"http": PROXY_URL, "https": PROXY_URL})
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()
    resp = opener.open(req, timeout=timeout)
    raw = resp.read()
    for enc in ["gbk", "gb18030", "utf-8"]:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def fetch_kline_incremental(code, existing_dates):
    """
    只拉取比已有数据更新的K线。
    existing_dates: 该股票已有的日期集合 (pd.Timestamp)
    """
    prefix = "sh" if code.startswith("6") else "sz"
    tcode = f"{prefix}{code}"

    if existing_dates:
        # 从已有最新日期往后拉
        latest_date = max(existing_dates)
        start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        # 全新股票，拉最近640天
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=960)).strftime("%Y-%m-%d")

    r = str(random.random())
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
           f"_var=kline_dayqfq&param={tcode},day,{start_date},{end_date},640,qfq&r={r}")

    try:
        text = http_get(url, timeout=15)
        match = re.search(r'kline_dayqfq=(.+)', text)
        if not match:
            return []
        data = json.loads(match.group(1))
        if data.get("code") != 0:
            return []
        rows = data.get("data", {}).get(tcode, {}).get("qfqday", [])
        if not rows:
            rows = data.get("data", {}).get(tcode, {}).get("day", [])
        return rows
    except Exception as e:
        print(f"  ❌ {code} 增量获取失败: {e}")
        return []


def update_data():
    cache = CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    force = "--force" in sys.argv

    print("=" * 60)
    print("🔄 增量数据更新")
    print("=" * 60)

    # Load existing data
    stocks_path = cache / "stocks.parquet"
    daily_path = cache / "daily.parquet"

    # 优先使用 universe.parquet，回退到 stocks.parquet
    universe_path = cache / "universe.parquet"
    stocks = pd.DataFrame()
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        codes = universe["code"].tolist()
        print(f"股票数: {len(codes)} (universe)")
        if stocks_path.exists():
            stocks = pd.read_parquet(stocks_path)
    elif stocks_path.exists():
        stocks = pd.read_parquet(stocks_path)
        codes = stocks["code"].tolist()
        print(f"股票数: {len(codes)} (stocks)")
    else:
        print("❌ universe.parquet 或 stocks.parquet 不存在")
        return

    new_rows_all = []
    updated_count = 0

    # Load existing daily data
    if daily_path.exists() and not force:
        existing_daily = pd.read_parquet(daily_path)
        existing_daily["date"] = pd.to_datetime(existing_daily["date"])
        latest_date = existing_daily["date"].max()
        print(f"已有数据最新日期: {latest_date.date()}")
        
        # Build per-stock date sets FIRST
        existing_dates_map = {}
        latest_by_stock = {}
        for code in codes:
            mask = existing_daily["code"] == code
            if mask.any():
                dates = existing_daily.loc[mask, "date"]
                existing_dates_map[code] = set(dates.tolist())
                latest_by_stock[code] = dates.max()
        
        # 检查有多少股票需要更新（最新日期 < 全局最新日期）
        need_update = [c for c in codes if c in latest_by_stock and latest_by_stock[c] < latest_date]
        missing_data = [c for c in codes if c not in existing_dates_map]
        
        if need_update:
            print(f"⚠ {len(need_update)} 只股票数据滞后，需要更新")
        if missing_data:
            print(f"⚠ {len(missing_data)} 只股票缺少数据")
        
        # 智能跳过：只有当所有股票都是最新时才跳过
        today = pd.Timestamp.now().normalize()
        weekday = today.weekday()
        
        # 计算上一个交易日
        if weekday == 0:  # 周一
            last_trading_day = today - timedelta(days=3)
        elif weekday == 6:  # 周日
            last_trading_day = today - timedelta(days=2)
        else:
            last_trading_day = today - timedelta(days=1)
        
        if latest_date >= last_trading_day and not need_update and not missing_data:
            print(f"✅ 数据已是最新 ({latest_date.date()})，采样检查...")
            # 只检查 5 只股票确认
            sample_codes = random.sample(codes, min(5, len(codes)))
            sample_ok = True
            for code in sample_codes:
                rows = fetch_kline_incremental(code, set())
                if rows:
                    newest = pd.to_datetime(rows[-1][0])
                    if newest > latest_date:
                        sample_ok = False
                        print(f"  ⚠ {code} 有新数据 ({newest.date()})")
                        break
                time.sleep(0.2)
            
            if sample_ok:
                print(f"✅ 采样检查通过，无需更新")
                return
            print(f"  需要更新，继续...")
    else:
        existing_dates_map = {}
        existing_daily = None
        print("没有已有数据或 --force 模式，全量拉取")

    start_time = time.time()
    
    # 使用线程池并发获取数据
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    new_rows_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress = {"done": 0, "updated": 0}
    
    def fetch_one(code):
        """获取单个股票的增量数据"""
        existing_dates = existing_dates_map.get(code, set())
        rows = fetch_kline_incremental(code, existing_dates)
        
        if rows:
            clean = [row[:6] for row in rows]
            new_df = pd.DataFrame(clean, columns=["date", "open", "close", "high", "low", "volume"])
            new_df["code"] = code
            new_df["date"] = pd.to_datetime(new_df["date"])
            for c in ["open", "close", "high", "low", "volume"]:
                new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
            
            if existing_dates:
                new_df = new_df[~new_df["date"].isin(existing_dates)]
            
            if len(new_df) > 0:
                new_df["amount"] = new_df["close"] * new_df["volume"] * 100
                return code, new_df
        
        return code, None
    
    # 并发执行
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {executor.submit(fetch_one, code): code for code in codes}
        
        for future in as_completed(futures):
            code, new_df = future.result()
            
            with progress_lock:
                progress["done"] += 1
                
                if new_df is not None and len(new_df) > 0:
                    new_rows_all.append(new_df)
                    progress["updated"] += 1
                    earliest_new = new_df["date"].min().date()
                    latest_new = new_df["date"].max().date()
                    print(f"  ✅ {code}: +{len(new_df)} 行 ({earliest_new} to {latest_new})")
                
                if progress["done"] % 50 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / progress["done"] * (len(codes) - progress["done"])
                    print(f"  进度: {progress['done']}/{len(codes)}, 更新 {progress['updated']} 只, ETA: {eta/60:.1f}分钟")
    
    updated_count = progress["updated"]

    # Merge with existing
    if new_rows_all:
        new_combined = pd.concat(new_rows_all, ignore_index=True)
        if daily_path.exists() and not force and existing_daily is not None:
            all_data = pd.concat([existing_daily, new_combined], ignore_index=True)
        else:
            all_data = new_combined

        # Deduplicate (date + code)
        all_data = all_data.sort_values(["code", "date"]).drop_duplicates(
            subset=["code", "date"], keep="last"
        ).reset_index(drop=True)

        # 填充 sector/name/circ_cap（新拉取的行缺少这些列）
        ss_path = cache / "stocks_sectors.parquet"
        if ss_path.exists():
            ss = pd.read_parquet(ss_path)
            lookup = ss[["code", "sector", "name", "circ_cap", "total_cap"]].drop_duplicates("code")
            all_data = all_data.merge(lookup, on="code", how="left", suffixes=("", "_ss"))
            for col in ["sector", "name", "circ_cap", "total_cap"]:
                ss_col = f"{col}_ss"
                if ss_col in all_data.columns:
                    all_data[col] = all_data[col].fillna(all_data[ss_col])
                    all_data.drop(columns=[ss_col], inplace=True)
            print(f"   补全 sector/name/circ_cap (from stocks_sectors)")
        
        # 备用: 从universe.parquet填充circ_cap (更可靠的数据源)
        universe_path = cache / "universe.parquet"
        if universe_path.exists():
            universe = pd.read_parquet(universe_path)
            cap_cols = ["circ_cap", "total_cap"]
            for col in cap_cols:
                if col in universe.columns:
                    cap_map = dict(zip(universe["code"], universe[col]))
                    missing_mask = (all_data[col].isna()) | (all_data[col] == 0)
                    if missing_mask.any():
                        all_data.loc[missing_mask, col] = all_data.loc[missing_mask, "code"].map(cap_map)
            circ_valid = (all_data["circ_cap"] > 0).mean() * 100
            print(f"   补全 circ_cap (from universe): {circ_valid:.1f}% 有效")

        # 计算派生指标并存入 parquet
        from backtest import compute_derived_columns
        all_data = compute_derived_columns(all_data)

        all_data.to_parquet(daily_path, index=False)

        # 验证写入
        verify = pd.read_parquet(daily_path)
        expected_cols = {"code", "date", "open", "close", "high", "low", "volume", "amount",
                         "sector", "name", "circ_cap", "total_cap",
                         "mkt_open", "mkt_close", "mkt_high", "mkt_low", "mkt_volume", "mkt_amount",
                         "sector_open", "sector_close", "sector_high", "sector_low", "sector_volume", "sector_amount",
                         "prev_close", "turnover_rate", "amplitude",
                         "limit_up_flag", "limit_down_flag", "mkt_ret_20d", "sector_ret_20d"}
        actual = set(verify.columns)
        missing_cols = expected_cols - actual
        if missing_cols:
            print(f"   ⚠️ 缺少列: {missing_cols}")
        else:
            print(f"   ✅ 全部 {len(expected_cols)} 列写入完成")

        dates = sorted(all_data["date"].unique())
        print(f"\n✅ 更新完成!")
        print(f"   新增: {updated_count} 只股票的数据")
        print(f"   新增行数: {len(new_combined)}")
        print(f"   总行数: {len(all_data)}")
        print(f"   最新日期: {dates[-1].date()}")
    else:
        print(f"\n✅ 无新数据需要更新（已有数据已是最新）")

    # Also update stock list + market cap
    if len(stocks) > 0:
        print(f"\n更新股票市值信息...")
        for i, code in enumerate(codes):
            prefix = "sh" if code.startswith("6") else "sz"
            url = f"https://qt.gtimg.cn/q={prefix}{code}"
            try:
                text = http_get(url, timeout=10)
                m = re.search(r'v_\w+="([^"]+)"', text)
                if m:
                    fields = m.group(1).split("~")
                    if len(fields) >= 50:
                        name = fields[1]
                        current = fields[3]
                        try:
                            total_share = float(fields[45]) if fields[45] else 0
                            price = float(current) if current and current != "--" else 0
                            circ_cap = total_share * price * 10000
                            # Update in stocks df
                            mask = stocks["code"] == code
                            if mask.any():
                                stocks.loc[mask, "name"] = name
                                stocks.loc[mask, "circ_cap"] = round(circ_cap, 2)
                                stocks.loc[mask, "total_cap"] = round(circ_cap, 2)
                        except (ValueError, IndexError):
                            pass
            except Exception:
                pass

            time.sleep(0.2)  # minimal pause

        stocks.to_parquet(stocks_path, index=False)
    else:
        print(f"  跳过: stocks.parquet 不存在")

    # Rebuild stocks_sectors
    if len(stocks) > 0:
        sectors_path = cache / "sectors.parquet"
        if sectors_path.exists():
            sectors = pd.read_parquet(sectors_path)
            stocks_sectors = stocks.merge(sectors, on="code", how="inner")
            stocks_sectors.to_parquet(cache / "stocks_sectors.parquet", index=False)

    elapsed = time.time() - start_time
    print(f"   耗时: {elapsed/60:.1f} 分钟")
    print("=" * 60)


if __name__ == "__main__":
    update_data()
