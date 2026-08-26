import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    熊市强势抗跌延续策略 (与 BEAR_001/003/004 方向不同):
    熊市(大盘20日跌幅>3%)中, 选择T日温和放量大涨、收在日内高位、
    且显著跑赢大盘的大市值股票, 赌其强势延续到 T+1开盘 -> T+2开盘。
    信号强度 0.3~1.0, 由相对强度/收盘位置/量能加权映射。
    """
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code')

    # ---------------- 基础因子 ----------------
    df['prev_close'] = df['prev_close'].replace(0, np.nan)
    ret = df['close'] / df['prev_close'] - 1.0
    df['ret'] = ret.fillna(g['close'].pct_change())
    df['mkt_ret'] = g['mkt_close'].pct_change()
    df['sector_ret'] = g['sector_close'].pct_change()
    vol_ma5 = g['volume'].transform(lambda s: s.rolling(5, min_periods=3).mean().shift(1))
    df['vol_ratio'] = df['volume'] / vol_ma5.replace(0, np.nan)

    # ---------------- 取每只股票最新一日(T日) ----------------
    last = df.groupby('code').tail(1).copy()
    last = last.set_index('code')

    # ---------------- 选股条件 (恰好5个AND条件 + 卫生过滤) ----------------
    cond = (
        (last['circ_cap'] > 100e8) &                                # 1 大市值, 熊市抗跌
        (last['ret'] >= 0.04) & (last['ret'] <= 0.075) &            # 2 大涨4%~7.5%, 不追涨停
        (last['vol_ratio'] >= 1.5) & (last['vol_ratio'] <= 3.0) &   # 3 温和放量突破
        (last['close'] >= last['high'] * 0.985) &                   # 4 收在最高位附近, 不炸板
        ((last['ret'] - last['mkt_ret']) >= 0.03)                   # 5 显著跑赢大盘(弱市强势)
    )
    # 卫生过滤: 剔除ST/退市、停牌(不计入5个alpha条件)
    cond = cond & (~last['name'].astype(str).str.contains('ST|退', na=False))
    cond = cond & (last['volume'] > 0)
    cond = cond.fillna(False)

    sel = last[cond]
    if sel.empty:
        return pd.Series(dtype=float)

    # ---------------- 信号强度: 映射到 0.3 ~ 1.0 ----------------
    rs_mkt = (sel['ret'] - sel['mkt_ret']).clip(0.0, 0.12) / 0.12      # 对大盘相对强度
    rs_sec = (sel['ret'] - sel['sector_ret']).clip(0.0, 0.10) / 0.10   # 对板块相对强度
    day_rng = (sel['high'] - sel['low']).replace(0, np.nan)
    close_pos = ((sel['close'] - sel['low']) / day_rng).fillna(1.0).clip(0.0, 1.0)  # 收盘位置
    vol_score = ((sel['vol_ratio'] - 1.5) / 1.5).clip(0.0, 1.0)        # 量能得分

    score = (0.35 * rs_mkt.fillna(0.0)
             + 0.25 * rs_sec.fillna(0.0)
             + 0.25 * close_pos
             + 0.15 * vol_score.fillna(0.0))

    if len(sel) > 1 and score.max() > score.min():
        strength = 0.3 + 0.7 * (score - score.min()) / (score.max() - score.min())
    else:
        strength = pd.Series(0.65, index=sel.index)

    strength = strength.clip(0.3, 1.0)
    return strength