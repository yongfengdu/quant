import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    BEAR 策略: 大市值放量抗跌强势延续
    熊市资金抱团大盘蓝筹。筛选: 大市值(circ_cap>100亿) + 当日涨3%~7% +
    收在日内高位(不炸板) + 温和放量(1.5~3.5倍) + 非ST/退市,
    再用板块相对强度与20日新高加分, 输出0.3~1.0的信号强度。
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)

    # ---- 个股量价特征 ----
    df['ret'] = g['close'].pct_change()
    prev_vol_ma5 = g['volume'].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    df['vol_ratio'] = df['volume'] / prev_vol_ma5.replace(0, np.nan)
    hl_span = (df['high'] - df['low']).replace(0, np.nan)
    df['close_pos'] = ((df['close'] - df['low']) / hl_span).fillna(0.5)
    prev_high20 = g['high'].transform(lambda s: s.shift(1).rolling(20, min_periods=10).max())
    df['is_new_high'] = (df['close'] >= prev_high20).fillna(False)

    # ---- 板块相对强度: 近5日板块指数收益 - 大盘收益 ----
    df['mkt_ret5'] = g['mkt_close'].pct_change(5)
    df['sec_ret5'] = g['sector_close'].pct_change(5)
    df['rs_sector'] = df['sec_ret5'] - df['mkt_ret5']

    # ---- 取每只股票最新一行 ----
    last = df.groupby('code', sort=False).tail(1).set_index('code')

    # ---- 硬条件 (共5个AND) ----
    not_bad = ~last['name'].astype(str).str.contains('ST|退', regex=True, na=False)
    big_cap = (last['circ_cap'] > 100e8).fillna(False)
    strong_close = ((last['ret'] >= 0.03) & (last['ret'] <= 0.07)).fillna(False)
    near_high = (last['close_pos'] >= 0.80).fillna(False)
    vol_ok = ((last['vol_ratio'] >= 1.5) & (last['vol_ratio'] <= 3.5)).fillna(False)

    mask = big_cap & not_bad & strong_close & near_high & vol_ok

    if int(mask.sum()) == 0:
        return pd.Series(dtype=float)

    cand = last.loc[mask]

    # ---- 信号强度打分 (区分度: 板块RS + 新高 + 量价质量) ----
    score = pd.Series(0.0, index=cand.index)
    score += np.clip((cand['ret'] - 0.03) / 0.04, 0, 1) * 0.25          # 涨幅质量(4~7%越高越好)
    score += np.clip((cand['vol_ratio'] - 1.5) / 2.0, 0, 1) * 0.20       # 放量程度
    score += np.clip((cand['close_pos'] - 0.80) / 0.20, 0, 1) * 0.15     # 收盘位置
    score += np.clip(cand['rs_sector'] / 0.05, 0, 1).fillna(0.0) * 0.25  # 板块相对强度
    score += np.where(cand['is_new_high'].fillna(False), 0.15, 0.0)      # 20日新高加分

    strength = (0.3 + 0.7 * score).clip(0.3, 1.0)
    return strength.sort_values(ascending=False)