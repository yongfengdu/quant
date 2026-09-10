"""
数据采集层 (data_fetch.py)
==========================
职责:
  1. 拉取不复权原始 OHLCV (raw)
  2. 从分红送配事件构建"送转因子" (split_factor), 用于复权
  3. 带重试 + 双源兜底 (腾讯为主, AkShare 为辅)

复权方案 (point-in-time 正确):
  - 存不复权 raw 价 + split_factor (仅处理送股/转增, 即份额变化)
  - adj_close(day) = raw_close(day) * split_factor(day)
  - split_factor(day) = Π_{除权除息日 <= day} (1 + (送股+转增)/10)
  - 返回: adj_ret = adj_close[T2]/adj_close[T1] - 1
  - 说明: 现金分红(~1-2%/次, 对20日收益影响~0.1%)暂不调整, 作为后续 total-return 增强

为什么不用腾讯 hfq/raw: 实测 hfq/raw 逐日漂移(非阶梯函数), 含现金分红再投资成分,
  不是干净的 point-in-time 因子。
"""
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROXY = "http://proxy.ims.intel.com:911"
CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"

OUT_COLS = ["date", "open", "close", "high", "low", "volume", "amount", "split_factor"]


def http_get(url, timeout=20, retries=5, delay=3):
    """带重试的 HTTP GET (走代理)。"""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Referer", "https://stockapp.finance.qq.com/")
            req.add_header("User-Agent", "Mozilla/5.0")
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
            raw = opener.open(req, timeout=timeout).read()
            for enc in ("gbk", "gb18030", "utf-8"):
                try:
                    return raw.decode(enc)
                except (UnicodeDecodeError, LookupError):
                    continue
            return raw.decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(delay * (i + 1))
    raise RuntimeError(f"http_get 重试 {retries} 次仍失败: {last}")


def _tencent_prefix(code):
    return "sh" if str(code).startswith("6") else "sz"


def fetch_tencent_kline(code, start="2018-01-01", end="2030-01-01"):
    """腾讯不复权日线。返回 DataFrame[date,open,close,high,low,volume]。"""
    p = _tencent_prefix(code)
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
           f"_var=k&param={p}{code},day,{start},{end},640,")
    text = http_get(url)
    d = json.loads(re.search(r"k=(.+)", text).group(1))
    node = d.get("data", {}).get(f"{p}{code}", {})
    rows = node.get("day") or []
    out = []
    for r in rows:
        out.append({"date": r[0], "open": float(r[1]), "close": float(r[2]),
                    "high": float(r[3]), "low": float(r[4]), "volume": float(r[5])})
    return pd.DataFrame(out)


def fetch_akshare_kline(code, start="20180101", end="20300101"):
    """AkShare 兜底 (不复权)。返回统一格式。"""
    import akshare as ak
    df = ak.stock_zh_a_hist(symbol=str(code), period="daily",
                            start_date=start, end_date=end, adjust="")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "成交量": "volume"})
    df["date"] = df["date"].astype(str)
    return df[["date", "open", "close", "high", "low", "volume"]]


def fetch_raw_kline(code, start="2018-01-01", end="2030-01-01"):
    """双源拉不复权: 腾讯优先, 失败转 AkShare。"""
    try:
        return fetch_tencent_kline(code, start, end)
    except Exception:
        return fetch_akshare_kline(code, start.replace("-", ""), end.replace("-", ""))


SPLIT_EVENTS_PATH = CACHE_DIR / "split_events.parquet"


def load_split_events_cache():
    """加载分红送转事件缓存 (code -> [(ex_date, factor)])。"""
    if not SPLIT_EVENTS_PATH.exists():
        return {}
    df = pd.read_parquet(SPLIT_EVENTS_PATH)
    cache = {}
    for code, grp in df.groupby("code"):
        events = [(ex, f) for ex, f in zip(grp["ex_date"], grp["factor"]) if ex != ""]
        cache[str(code)] = events
    return cache


def save_split_events_cache(cache):
    """保存分红送转事件缓存 (无送转的用哨兵行 ex_date='' 持久化, 避免重复拉取)。"""
    rows = []
    for code, events in cache.items():
        if not events:
            rows.append({"code": code, "ex_date": "", "factor": 1.0})
        for ex, factor in events:
            rows.append({"code": code, "ex_date": ex, "factor": factor})
    pd.DataFrame(rows).to_parquet(SPLIT_EVENTS_PATH, index=False)


def fetch_split_events(code, retries=5):
    """从 AkShare 拉分红送配事件, 返回 [(除权除息日, 送转比例)] 升序。"""
    import akshare as ak
    for i in range(retries):
        try:
            dv = ak.stock_history_dividend_detail(symbol=str(code))
            break
        except Exception:
            time.sleep(4)
    else:
        return []
    dv = dv[dv["进度"] == "实施"].copy()
    events = []
    for _, r in dv.iterrows():
        ex = str(pd.to_datetime(r["除权除息日"]).date())
        s = float(r["送股"]) if pd.notna(r["送股"]) else 0.0
        z = float(r["转增"]) if pd.notna(r["转增"]) else 0.0
        ratio = (s + z) / 10.0
        if ratio > 0:
            events.append((ex, 1.0 + ratio))
    return sorted(events)


def get_split_events(code, cache=None):
    """获取分红送转事件 (优先用缓存, 未命中才拉取)。"""
    cache = cache if cache is not None else load_split_events_cache()
    code = str(code)
    if code in cache:
        return cache[code]
    events = fetch_split_events(code)
    cache[code] = events
    return events


def fetch_raw_history(code, start="2018-01-01", end="2030-01-01", chunk_years=2):
    """分段拉取不复权历史 (腾讯单次上限 640 行 ≈ 2.5 年)。"""
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur < end_ts:
        nxt = min(cur + pd.DateOffset(years=chunk_years), end_ts)
        df = fetch_raw_kline(code, cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"))
        if df is not None and len(df) > 0:
            chunks.append(df)
        cur = nxt + pd.Timedelta(days=1)
    if not chunks:
        return None
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_raw_with_factor(code, start="2018-01-01", end="2030-01-01", split_cache=None):
    """拉不复权 + 构建送转因子。返回 DataFrame(OUT_COLS)。"""
    raw = fetch_raw_history(code, start, end)
    if raw is None or len(raw) == 0:
        return None
    events = get_split_events(code, split_cache)
    raw = raw.sort_values("date").reset_index(drop=True)

    # split_factor: 阶梯函数, 在除权日跳变
    factor = 1.0
    factor_by_date = {}
    ev_idx = 0
    for d in raw["date"]:
        while ev_idx < len(events) and events[ev_idx][0] <= d:
            factor *= events[ev_idx][1]
            ev_idx += 1
        factor_by_date[d] = factor

    raw["split_factor"] = raw["date"].map(factor_by_date)
    raw["amount"] = raw["close"] * raw["volume"] * 100  # A股 1手=100股
    return raw[OUT_COLS]


def load_universe():
    """加载股票池 (code 列表)。"""
    uni_path = CACHE_DIR / "universe.parquet"
    if not uni_path.exists():
        raise FileNotFoundError(f"universe.parquet 不存在: {uni_path}")
    uni = pd.read_parquet(uni_path)
    return sorted(set(str(c) for c in uni["code"].tolist()))
