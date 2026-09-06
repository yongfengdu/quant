import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    RANGE_TREND: 震荡市趋势动量策略 —— 强板块龙头新高突破

    核心思路(与超跌反弹类策略正交):
      1) 大市值过滤: circ_cap > 100e8 (大盘股 T+1 可预测性更高)
      2) 新高突破: 收盘价站上/贴近前 60 日最高价 (平台突破, 惯性延续)
      3) 多头排列: MA5 > MA20 > MA60 (趋势完好, 非一日游)
      4) 板块轮动: 所属板块 20 日收益强于大盘 (sector_ret_20d > mkt_ret_20d)
      5) 温和放量: 当日量/5日均量 ∈ [1.2, 4.0] (放量确认但非衰竭性天量)

    信号强度 = 个股20日动量、板块强度、量能的截面百分位加权, 映射到 0.3~1.0。
    返回: pd.Series(index=code, value=signal_strength)
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)

    # ---------- 按个股时序计算趋势/动量指标 ----------
    df['_ma5'] = g['close'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df['_ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df['_ma60'] = g['close'].transform(lambda s: s.rolling(60, min_periods=60).mean())
    # 前 60 日最高价 (不含当日, 至少20日数据), 用于判断平台突破
    df['_prev_high60'] = g['high'].transform(
        lambda s: s.rolling(60, min_periods=20).max().shift(1)
    )
    # 前 5 日均量 (不含当日), 计算量比
    df['_prev_vol_ma5'] = g['volume'].transform(
        lambda s: s.rolling(5, min_periods=3).mean().shift(1)
    )
    # 个股 20 日动量
    df['_ret20'] = g['close'].transform(lambda s: s.pct_change(20))

    # ---------- 只取每只股票最新一天 (T 日) ----------
    last = df.groupby('code', sort=False).tail(1).copy()

    # 卫生过滤: 剔除 ST/退市股、剔除 T 日已涨停的 (避免追一字板, 也与"大涨不炸板"策略区分)
    name = last['name'].fillna('').astype(str)
    name_ok = ~name.str.contains('ST|退', regex=True)
    not_limit_up = last['limit_up_flag'].fillna(0).astype(float) == 0
    last = last[name_ok & not_limit_up]
    if len(last) == 0:
        return pd.Series(dtype=float)

    # ---------- 5 个核心 AND 条件 ----------
    # 1) 大市值
    cond_cap = last['circ_cap'] > 100e8
    # 2) 60 日新高突破 (允许 0.5% 容差, 贴近前高也算, 防止信号过少)
    cond_breakout = last['close'] >= last['_prev_high60'] * 0.995
    # 3) 均线多头排列
    cond_trend = (last['_ma5'] > last['_ma20']) & (last['_ma20'] > last['_ma60'])
    # 4) 板块相对强度: 板块 20 日收益跑赢大盘
    cond_sector = last['sector_ret_20d'] > last['mkt_ret_20d']
    # 5) 温和放量: 量比在 [1.2, 4.0]
    vol_ratio = last['volume'] / last['_prev_vol_ma5'].replace(0, np.nan)
    cond_vol = (vol_ratio >= 1.2) & (vol_ratio <= 4.0)

    mask = (cond_cap & cond_breakout & cond_trend & cond_sector & cond_vol).fillna(False)
    sel = last[mask].copy()
    if len(sel) == 0:
        return pd.Series(dtype=float)

    # ---------- 信号强度: 截面百分位加权, 映射到 0.3~1.0 ----------
    sel['_vr'] = vol_ratio.reindex(sel.index).clip(upper=3.0)
    mom_rank = sel['_ret20'].rank(pct=True)                      # 个股动量
    sec_rank = sel['sector_ret_20d'].rank(pct=True)              # 板块强度
    vol_rank = sel['_vr'].rank(pct=True)                         # 量能确认

    score = (0.45 * mom_rank + 0.35 * sec_rank + 0.20 * vol_rank).fillna(0.5)
    strength = (0.3 + 0.7 * score).clip(0.3, 1.0)

    result = pd.Series(strength.values, index=sel['code'].values, dtype=float)
    result = result[~result.index.duplicated(keep='first')]
    return result