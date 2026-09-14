import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    BULL 相对强度放量突破:
    T日收盘: 大市值 + 温和放量大涨(3%~7%, 不追涨停) + 收盘接近日内高位
    + 相对大盘有显著超额收益  -> T+1开盘买入, T+2开盘卖出。
    返回: pd.Series(index=code, value=signal_strength 0.3~1.0)
    """
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code')

    # 个股日收益 & 大盘日收益
    prev_close = g['close'].shift(1)
    df['ret'] = df['close'] / prev_close - 1.0
    df['mkt_ret'] = g['mkt_close'].pct_change()

    # 放量倍数: 当日量 / 前5日均量(不含当日)
    df['vol_ma5'] = g['volume'].transform(lambda s: s.rolling(5, min_periods=1).mean().shift(1))
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    # 收盘在日内区间中的位置 (0~1, 越接近1越强)
    day_range = (df['high'] - df['low']).replace(0, np.nan)
    df['close_pos'] = (df['close'] - df['low']) / day_range

    # 相对大盘超额收益
    df['excess'] = df['ret'] - df['mkt_ret']

    # 只取每只股票最新一根K线
    last = df.groupby('code').tail(1).set_index('code')

    # ST/退市 卫生过滤
    name_ok = ~last['name'].astype(str).str.contains('ST|退', na=False)

    # 5个核心AND条件
    cond = (
        (last['circ_cap'] > 100e8) &                       # 1 大市值
        (last['ret'] >= 0.03) & (last['ret'] <= 0.07) &    # 2 大涨但不追涨停
        (last['vol_ratio'] >= 1.5) & (last['vol_ratio'] <= 3.5) &  # 3 温和放量
        (last['close_pos'] >= 0.7) &                       # 4 收在日内高位(不炸板)
        (last['excess'] >= 0.02)                           # 5 显著跑赢大盘
    ) & name_ok
    cond = cond.fillna(False)

    # 信号强度: 因子映射到 0.3~1.0, 有区分度
    excess_score = (last['excess'] / 0.06).clip(0, 1)
    vol_score = ((last['vol_ratio'] - 1.5) / 2.0).clip(0, 1)
    pos_score = last['close_pos'].clip(0, 1)
    composite = (0.5 * excess_score + 0.3 * vol_score + 0.2 * pos_score).fillna(0.0)
    strength = (0.3 + 0.7 * composite).clip(0.3, 1.0)

    return strength.loc[cond].sort_values(ascending=False)