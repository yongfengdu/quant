import numpy as np
import pandas as pd


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    RANGE_TREND: 震荡市趋势动量策略 —— 60日新高 + 均线多头排列 + 相对强度

    逻辑:
      1) 大市值过滤 (circ_cap > 100e8), 剔除 ST/退市
      2) 收盘价创 60 日新高 (平台/趋势突破)
      3) MA5 > MA20 > MA60 多头排列 (趋势确认)
      4) 个股 20 日绝对动量 > 0 (顺势, 不抄底)
      5) 剔除当日涨停/跌停 (无法成交或追高风险)
    信号强度: 相对大盘20日超额收益、突破幅度、量比 三因子截面排名合成,
              映射到 0.3 ~ 1.0, 最强信号排最前。
    """
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    g = df.groupby('code', sort=False)

    # ---------- 滚动特征 (按股票分组) ----------
    df['ma5'] = g['close'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    df['ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df['ma60'] = g['close'].transform(lambda s: s.rolling(60, min_periods=60).mean())

    # 前 60 日最高价 (不含当日), 用于判断"当日收盘创新高"
    df['prev_high_60'] = g['high'].transform(
        lambda s: s.shift(1).rolling(60, min_periods=20).max()
    )

    # 个股 20 日收益 与 相对大盘超额收益
    df['ret_20'] = g['close'].pct_change(20)
    mkt_ret_20 = df['mkt_ret_20d'] if 'mkt_ret_20d' in df.columns else 0.0
    df['rs_20'] = df['ret_20'] - mkt_ret_20

    # 量比: 当日成交量 / 20 日均量
    df['vol_ma20'] = g['volume'].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean()
    )
    df['vol_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)

    # 突破幅度: 收盘相对前 60 日高点超出多少
    df['breakout_gap'] = df['close'] / df['prev_high_60'] - 1.0

    # ---------- 只取每只股票最新一根 K 线 (T 日) ----------
    last = df.groupby('code', sort=False).tail(1).copy()

    # ---------- 过滤条件 (不超过 5 个 AND) ----------
    cond_cap = last['circ_cap'] > 100e8
    cond_name = ~last['name'].astype(str).str.contains('ST|退', na=False)
    cond_breakout = last['close'] > last['prev_high_60']
    cond_ma_bull = (last['ma5'] > last['ma20']) & (last['ma20'] > last['ma60'])
    cond_momentum = last['ret_20'] > 0
    cond_not_limit = (~last['limit_up_flag'].fillna(False).astype(bool)) & \
                     (~last['limit_down_flag'].fillna(False).astype(bool))

    mask = (
        cond_cap.fillna(False)
        & cond_name.fillna(False)
        & cond_breakout.fillna(False)
        & cond_ma_bull.fillna(False)
        & cond_momentum.fillna(False)
        & cond_not_limit.fillna(False)
    )

    cand = last.loc[mask].copy()
    if cand.empty:
        return pd.Series(dtype=float)

    # ---------- 信号强度: 三因子截面排名合成, 映射到 0.3 ~ 1.0 ----------
    n = len(cand)
    if n == 1:
        strength = pd.Series(0.8, index=cand['code'].values)
    else:
        rank_rs = cand['rs_20'].rank(pct=True)                       # 相对大盘强度
        rank_bo = cand['breakout_gap'].rank(pct=True)                # 突破幅度
        rank_vol = cand['vol_ratio'].clip(upper=3.0).rank(pct=True)  # 温和放量加分
        composite = 0.45 * rank_rs + 0.35 * rank_bo + 0.20 * rank_vol
        strength = 0.3 + 0.7 * composite.clip(0.0, 1.0)
        strength.index = cand['code'].values

    strength = strength.astype(float).clip(0.3, 1.0)
    return strength.sort_values(ascending=False)