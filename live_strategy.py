"""
线上策略 (live_strategy.py)
==========================
已验证的 regime 门控截面策略 (主线固化):

  市场状态判定 (mkt_ret_20d):
    - 牛市 (> +3%): 用动量 dist_hi20, 买"贴20日高点"的股票
    - 熊/震荡 (其余): 用反转 dist_ma60, 买"跌到60日均线下方"的超跌股

  交易节奏: 每 20 交易日调仓一次, 持有 20 交易日
  回测 (2018-2026): 年化超额 +11.0%, Sharpe 0.84, 回撤 -29.1%

用法:
  python live_strategy.py --backtest          # 回测验证
  python live_strategy.py --recommend         # 生成当前推荐
"""
import argparse
import sys
sys.path.insert(0, "/root/quant-autoresearch")
from pathlib import Path

import pandas as pd

from factors import compute_all_factors, add_forward_returns
from portfolio import metrics

CACHE_DIR = Path.home() / ".cache" / "quant-autoresearch"

# 已验证的策略配置
BULL_THRESHOLD = 0.03
REBALANCE_DAYS = 20
N_BINS = 5


def detect_regime(mkt_ret_20d):
    """市场状态: bull 或 bear/range。"""
    if pd.isna(mkt_ret_20d):
        return "bear/range"
    return "bull" if mkt_ret_20d > BULL_THRESHOLD else "bear/range"


# 默认核心信号 (注册表为空时回退)
DEFAULT_SIGNALS = {
    "bull": [{"sid": "dist_hi20", "factor": "dist_hi20", "direction": 1,
              "type": "builtin", "regime": "bull"}],
    "bear/range": [{"sid": "dist_ma60", "factor": "dist_ma60", "direction": -1,
                    "type": "builtin", "regime": "bear/range"}],
}


def _sid(s):
    """信号 id (兼容注册表 'id' 键与默认 'sid' 键)。"""
    return s.get("sid") or s.get("id")


def load_active_signals():
    """读注册表 active 信号 (空则回退默认核心信号)。返回全部 active 列表。"""
    from signal_registry import SignalRegistry
    active = SignalRegistry().active_signals()
    if active:
        return active
    return [s for lst in DEFAULT_SIGNALS.values() for s in lst]


def signals_for_regime(signals, regime):
    """过滤出某 regime 适用的信号 (regime='all' 的也算)。"""
    return [s for s in signals if s.get("regime", "all") in ("all", regime)]


def _is_code_signal(s):
    """代码型信号 (LLM 生成的, 有 definition 或 type='code')。"""
    return s.get("type") == "code" or s.get("definition") is not None


def add_strategy_columns(df, signals=None):
    """追加策略所需列: 各 active 信号的分数列 + 前向收益 + 市场状态。"""
    from signal_evolve import compute_llm_factor
    df = df.copy()
    signals = signals if signals is not None else load_active_signals()
    df = add_forward_returns(df, [REBALANCE_DAYS])
    mkt = df.groupby("date")["close"].median().pct_change(REBALANCE_DAYS)
    df["mkt_ret_20d"] = df["date"].map(mkt)

    builtin_factors = sorted({s["factor"] for s in signals if not _is_code_signal(s)})
    if builtin_factors:
        df = compute_all_factors(df, builtin_factors)

    for s in signals:
        col = f"score_{_sid(s)}"
        if _is_code_signal(s):
            df2, err = compute_llm_factor(s["definition"], df)
            if err:
                continue
            df[col] = s.get("direction", 1) * df2["f_llm_factor"].reset_index(drop=True)
        else:
            df[col] = s.get("direction", 1) * df[f"f_{s['factor']}"]
    return df


def compute_combined_score(sub, regime, signals):
    """合成买入分数: 当前 regime 适用信号的 z-score 等权求和。"""
    scores = []
    for s in signals_for_regime(signals, regime):
        col = f"score_{_sid(s)}"
        if col not in sub.columns:
            continue
        sc = sub[col]
        if sc.isna().all():
            continue
        z = (sc - sc.mean()) / (sc.std() + 1e-12)
        scores.append(z)
    return sum(scores) if scores else None


def run_backtest(df):
    """回测 regime 门控策略, 返回每期 (date, port_ret, mkt_ret, regime)。"""
    signals = load_active_signals()
    df = add_strategy_columns(df, signals)
    dates = sorted(df["date"].unique())
    rebal = dates[::REBALANCE_DAYS]
    rows = []
    for d in rebal:
        sub = df[df["date"] == d].copy()
        if len(sub) < N_BINS * 10:
            continue
        regime = detect_regime(sub["mkt_ret_20d"].iloc[0])
        score = compute_combined_score(sub, regime, signals)
        if score is None:
            continue
        sub["score"] = score
        sub = sub.dropna(subset=["score", "fwd_20"])
        if len(sub) < N_BINS * 10:
            continue
        q = pd.qcut(sub["score"], N_BINS, labels=False, duplicates="drop")
        if q.max() < N_BINS - 1:
            continue
        sel = sub[q == q.max()]
        rows.append((d, sel["fwd_20"].mean(), sub["fwd_20"].mean(), regime))
    return pd.DataFrame(rows, columns=["date", "port_ret", "mkt_ret", "regime"])


def generate_recommendation(df, date=None, top_n=20):
    """生成当前推荐: 给定数据(截至某日), 返回该买的股票列表。"""
    signals = load_active_signals()
    df = add_strategy_columns(df, signals)
    if date is None:
        date = df["date"].max()
    sub = df[df["date"] == date].copy()
    regime = detect_regime(sub["mkt_ret_20d"].iloc[0])
    score = compute_combined_score(sub, regime, signals)
    if score is None:
        return {"date": str(date), "regime": regime, "stocks": []}
    sub["score"] = score
    sub = sub.dropna(subset=["score"]).sort_values("score", ascending=False)
    top = sub.head(top_n)
    return {
        "date": str(date),
        "regime": regime,
        "stocks": [
            {"code": str(r["code"]), "close": round(float(r["close"]), 2),
             "score": round(float(r["score"]), 4)}
            for _, r in top.iterrows()
        ],
    }


def cmd_backtest():
    df = pd.read_parquet(CACHE_DIR / "daily_bars.parquet")
    r = run_backtest(df)
    m = metrics(r)
    print("=== regime 门控策略回测 (2018-2026) ===")
    print(f"  年化收益: {m['ann_port']*100:+.1f}%")
    print(f"  年化超额: {m['ann_excess']*100:+.1f}%")
    print(f"  Sharpe: {m['sharpe']:.2f}")
    print(f"  最大回撤: {m['max_drawdown']*100:.1f}%")
    print(f"  胜率: {m['win_rate']*100:.0f}%")
    print(f"  期数: {m['n_periods']}")
    print(f"\n  各 regime 期数: {r['regime'].value_counts().to_dict()}")


def cmd_recommend():
    df = pd.read_parquet(CACHE_DIR / "daily_bars.parquet")
    rec = generate_recommendation(df)
    print(f"=== 当前推荐 (信号日 {rec['date']}, 市场状态 {rec['regime']}) ===")
    for i, s in enumerate(rec["stocks"][:10], 1):
        print(f"  {i}. {s['code']}  收盘 {s['close']}  分 {s['score']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--recommend", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        cmd_backtest()
    else:
        cmd_recommend()
