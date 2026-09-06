"""
数据构建 (build_data.py)
========================
拉取全量历史数据 (不复权 raw + 送转因子) 存入 parquet。

用法:
  python build_data.py --n 20          # 小样本验证 (前20只)
  python build_data.py --codes 000001,600519   # 指定代码
  python build_data.py                  # 全量 (universe 全部)
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from data_fetch import fetch_raw_with_factor, load_universe

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"
OUT_PATH = CACHE_DIR / "daily_bars.parquet"
START, END = "2018-01-01", "2026-12-31"


def pull(codes, out_path=OUT_PATH, start=START, end=END, resume=True):
    existing = {}
    if resume and out_path.exists():
        existing = {c for c in pd.read_parquet(out_path, columns=["code"])["code"].unique()}
    frames = []
    ok = fail = skip = 0
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        if code in existing:
            skip += 1
            continue
        try:
            df = fetch_raw_with_factor(code, start, end)
            if df is None or len(df) < 100:
                fail += 1
                print(f"[{i}/{len(codes)}] {code}: 数据不足, 跳过")
                continue
            df["code"] = code
            frames.append(df)
            ok += 1
            print(f"[{i}/{len(codes)}] {code}: {len(df)} 行 "
                  f"(split_factor 范围 {df['split_factor'].min():.2f}~{df['split_factor'].max():.2f})")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(codes)}] {code}: 失败 {type(e).__name__}")
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        if out_path.exists() and resume:
            old = pd.read_parquet(out_path)
            all_df = pd.concat([old, all_df], ignore_index=True).drop_duplicates(
                subset=["code", "date"], keep="last")
        all_df.sort_values(["code", "date"]).to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\n完成: 成功 {ok}, 失败 {fail}, 跳过(已存在) {skip}, 耗时 {elapsed/60:.1f} 分钟")
    if out_path.exists():
        d = pd.read_parquet(out_path)
        print(f"总数据: {len(d)} 行, {d['code'].nunique()} 只股票, "
              f"日期 {d['date'].min()} ~ {d['date'].max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="只拉前 N 只 (小样本验证)")
    ap.add_argument("--codes", help="逗号分隔的代码")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",")]
    else:
        codes = load_universe()
        if args.n:
            codes = codes[:args.n]
    print(f"拉取 {len(codes)} 只股票, {args.start} ~ {args.end}")
    pull(codes, start=args.start, end=args.end)
