"""运行一轮 LLM 信号进化 (真实 K3 调用)。"""
import sys
sys.path.insert(0, "/root/quant-autoresearch")
import pandas as pd

from signal_evolve import discover_signals

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    df = pd.read_parquet("/root/.cache/quant-autoresearch/daily_bars.parquet")
    print(f"数据: {len(df)} 行, {df['code'].nunique()} 只")
    results = discover_signals(df, n=n, horizon=20, test=False)
    passed = [r for r in results if r.get("passed")]
    print(f"\n=== 本轮进化: {len(passed)}/{n} 通过 ===")
    for r in passed:
        s = r["report"]["summary"]
        print(f"  {r['sid']}: IC={s['mean_ic']*100:+.2f}% t={s['t_stat']:+.1f}")
