import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    BEAR_010: 熊市趋势延续 + 板块轮动共振策略 (大市值避险领涨)

    逻辑: 熊市中资金抱团强者, 抗跌且趋势完好的大市值股在弱市中容易延续上涨。
    条件 (核心5条):
      1. 大市值: circ_cap > 100e8 (避险属性, T+1可预测性高)
      2. 趋势完好: close >= 0.97 * 20日最高价 且 close > MA20 (未创新低/趋势延续)
      3. 相对强度: 个股5日收益 > 大盘5日收益 (弱市领涨)
      4. 板块轮动: 板块20日收益 > 大盘20日收益 (板块相对强弱共振)
      5. 温和放量: 当日量 >= 5日均量 (资金确认), 且非一字涨停(避免追高风险)
    信号强度: 由相对强度、贴近新高程度、板块强度三因子加权映射到 0.3~1.0
    """
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    g = df.groupby('code', sort=False)

    # ---- 逐股滚动指标 ----
    df['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df['high20'] = g['high'].transform(lambda x: x.rolling(20, min_periods=10).max())
    df['vol_ma5'] = g['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df['ret5'] = g['close'].pct_change(5)
    df['mkt_ret5'] = g['mkt_close'].pct_change(5)

    # ---- 只取每只股票最新一日 (T日收盘) ----
    last = df.groupby('code', sort=False).tail(1).copy()

    # ---- 基础过滤 ----
    cap_ok = last['circ_cap'] > 100e8
    name_ok = ~last['name'].fillna('').str.contains('ST|退', regex=True)

    # ---- 核心条件 (5条 AND) ----
    trend_ok = (last['close'] >= 0.97 * last['high20']) & (last['close'] > last['ma20'])
    rs_ok = (last['ret5'] - last['mkt_ret5']) > 0.0
    sector_ok = (last['sector_ret_20d'] - last['mkt_ret_20d']) > 0.0
    vol_ok = (last['volume'] >= last['vol_ma5']) & (last['limit_up_flag'].fillna(0) == 0)

    cond = (cap_ok & name_ok & trend_ok & rs_ok & sector_ok & vol_ok).fillna(False)
    sel = last[cond].copy()

    if len(sel) == 0:
        return pd.Series(dtype=float)

    # ---- 信号强度打分: 三因子加权, 映射到 0.3~1.0 ----
    eps = 1e-12
    # 因子1: 5日超额收益 (0~1)
    rs_score = ((sel['ret5'] - sel['mkt_ret5']) / 0.08).clip(0, 1)
    # 因子2: 贴近20日新高程度 (0~1)
    pos_score = ((sel['close'] / (sel['high20'] + eps)) - 0.97) / 0.03
    pos_score = pos_score.clip(0, 1)
    # 因子3: 板块超额强度 (0~1)
    sec_score = ((sel['sector_ret_20d'] - sel['mkt_ret_20d']) / 0.05).clip(0, 1)

    raw = 0.45 * rs_score + 0.30 * pos_score + 0.25 * sec_score
    strength = (0.3 + 0.7 * raw).clip(0.3, 1.0).fillna(0.3)

    # ---- 高确信度、低频: 只保留打分最高的前10只 ----
    result = strength.sort_values(ascending=False).head(10)
    result.index = sel.loc[result.index, 'code']
    result = result.groupby(level=0).first()

    return result.astype(float)