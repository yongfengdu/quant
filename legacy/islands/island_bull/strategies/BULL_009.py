import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    BULL 趋势动量策略: 多头排列 + 20日新高突破 + 相对强度领先 + 强势板块。
    仅选流通市值>100亿的大盘股, 输出次日(T+1买->T+2卖)涨幅有望>3%的标的。
    """
    cols = ['code', 'date', 'open', 'close', 'high', 'low', 'volume',
            'sector', 'name', 'circ_cap',
            'mkt_close', 'mkt_ret_20d', 'sector_ret_20d']
    d = df[cols].copy()
    d = d.sort_values(['code', 'date']).reset_index(drop=True)

    g = d.groupby('code', sort=False)

    # ---- 个股时序指标 ----
    d['ret20'] = g['close'].pct_change(20)                       # 个股20日收益
    d['ma5'] = g['close'].transform(lambda s: s.rolling(5).mean())
    d['ma10'] = g['close'].transform(lambda s: s.rolling(10).mean())
    d['ma20'] = g['close'].transform(lambda s: s.rolling(20).mean())
    d['high20_prev'] = g['high'].transform(lambda s: s.rolling(20).max().shift(1))  # 前20日最高(不含当日)

    # ---- 相对强度: 个股20日收益 - 大盘20日收益 ----
    d['rs20'] = d['ret20'] - d['mkt_ret_20d']

    # ---- 取最新交易日截面 ----
    last_date = d['date'].max()
    snap = d[d['date'] == last_date].copy()

    # 截面排名(0~1): 相对强度排名 / 板块强度排名
    snap['rs_rank'] = snap['rs20'].rank(pct=True)
    snap['sec_rank'] = snap['sector_ret_20d'].rank(pct=True)

    # ---- 选股条件 (5个核心AND条件) ----
    cond_cap = snap['circ_cap'] > 100e8                                   # 1) 大市值
    cond_trend = (snap['ma5'] > snap['ma10']) & (snap['ma10'] > snap['ma20'])  # 2) 多头排列
    cond_break = snap['close'] >= 0.985 * snap['high20_prev']             # 3) 触及/突破20日新高
    cond_rs = snap['rs_rank'] >= 0.85                                     # 4) 相对强度全市场前15%
    cond_sector = snap['sector_ret_20d'] > snap['mkt_ret_20d']            # 5) 板块强于大盘

    # 卫生过滤: 剔除ST/退市
    cond_name = ~snap['name'].astype(str).str.contains('ST|退', na=False)

    mask = (cond_cap & cond_trend & cond_break & cond_rs & cond_sector & cond_name).fillna(False)

    # ---- 信号强度: 0.3~1.0, 因子值映射, 越强越高 ----
    prox = ((snap['close'] / snap['high20_prev'] - 0.97) / 0.06).clip(0, 1)  # 新高接近/突破程度
    score = (0.45 * snap['rs_rank'] + 0.30 * snap['sec_rank'] + 0.25 * prox).clip(0, 1)
    snap['strength'] = (0.3 + 0.7 * score).clip(0.3, 1.0)

    sel = snap[mask]
    out = pd.Series(sel['strength'].values, index=sel['code'].values)
    out = out[~out.index.duplicated(keep='first')]
    return out.sort_values(ascending=False)