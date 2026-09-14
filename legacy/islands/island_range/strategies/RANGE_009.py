import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    RANGE_TREND: 震荡市趋势突破策略
    逻辑: T日收盘出现 创20日新高 + MA5>MA20>MA60 多头排列 + 放量 + 相对大盘强势
    的大盘股, T+1开盘买入, T+2开盘卖出。
    返回: pd.Series(index=code, value=signal_strength in [0.3, 1.0])
    """
    df = df.copy()
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # ---------- 排除 ST / 退市等风险标的 ----------
    name = df['name'].astype(str)
    bad_name = name.str.contains('ST', case=False, na=False) | name.str.contains('退', na=False)
    df = df[~bad_name].copy()
    if df.empty:
        return pd.Series(dtype=float)

    g = df.groupby('code', sort=False)

    # ---------- 因子计算 (按股票分组, 时序滚动) ----------
    # 均线
    ma5 = g['close'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    ma20 = g['close'].transform(lambda x: x.rolling(20, min_periods=20).mean())
    ma60 = g['close'].transform(lambda x: x.rolling(60, min_periods=60).mean())

    # 前20日最高价(不含当日), 用于判断当日是否创新高
    prev_high_20 = g['high'].transform(lambda x: x.shift(1).rolling(20, min_periods=20).max())

    # 前20日均量(不含当日), 用于放量判断
    prev_vol_ma20 = g['volume'].transform(lambda x: x.shift(1).rolling(20, min_periods=20).mean())

    # 个股20日收益 与 大盘20日收益 -> 相对强度
    ret20 = g['close'].pct_change(20)
    mkt_ret20 = df['mkt_ret_20d'].astype(float)
    rs20 = ret20 - mkt_ret20

    # 突破幅度: 收盘价超出前20日高点的比例
    breakout_pct = df['close'] / prev_high_20 - 1.0

    df['_ma5'] = ma5
    df['_ma20'] = ma20
    df['_ma60'] = ma60
    df['_prev_high_20'] = prev_high_20
    df['_prev_vol_ma20'] = prev_vol_ma20
    df['_rs20'] = rs20
    df['_breakout_pct'] = breakout_pct

    # ---------- 只取每只股票最新一行 (T日信号) ----------
    last = df.groupby('code', sort=False).tail(1).copy()

    # ---------- 信号条件 (5个AND, 防过拟合) ----------
    cond_large_cap = last['circ_cap'].astype(float) > 100e8                      # 1 大市值
    cond_breakout = last['close'] > last['_prev_high_20']                        # 2 创20日新高
    cond_ma_bull = (last['_ma5'] > last['_ma20']) & (last['_ma20'] > last['_ma60'])  # 3 多头排列
    cond_volume = last['volume'] > 1.5 * last['_prev_vol_ma20']                  # 4 放量确认
    cond_rs = last['_rs20'] > 0                                                  # 5 跑赢大盘

    signal_mask = (
        cond_large_cap.fillna(False)
        & cond_breakout.fillna(False)
        & cond_ma_bull.fillna(False)
        & cond_volume.fillna(False)
        & cond_rs.fillna(False)
    )

    picked = last[signal_mask].copy()
    if picked.empty:
        return pd.Series(dtype=float)

    # ---------- 信号强度: 相对强度排名 + 突破幅度, 映射到 0.3~1.0 ----------
    rs_rank = picked['_rs20'].rank(pct=True)          # 0~1, 越大越强
    bo_rank = picked['_breakout_pct'].rank(pct=True)  # 0~1, 越大越强
    score = 0.6 * rs_rank + 0.4 * bo_rank
    strength = 0.3 + 0.7 * score
    strength = strength.clip(0.3, 1.0)

    result = pd.Series(strength.values, index=picked['code'].values)
    result = result[~result.index.duplicated(keep='last')]
    result = result.sort_values(ascending=False)
    return result