import numpy as np
import pandas as pd


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    BEAR 板块轮动 + 板块内趋势领涨延续策略 (与超跌/跌停/抗跌类策略正交)

    核心逻辑:
      熊市中资金抱团取暖, 当日轮动走强的板块里、趋势完好且温和放量的
      大市值领涨股, 次日存在趋势延续惯性 (T+1开盘买入, T+2开盘卖出)。

    筛选 (5个AND条件):
      1) 名称过滤: 剔除 ST / 退市;
      2) 大市值: circ_cap > 100e8 (大盘股 T+1 可预测性更高);
      3) 板块轮动: 所属板块5日收益处于全市场前 25% (资金避风港);
      4) 趋势延续: 收盘价站上 MA20 且 MA20 近3日走平/向上 且 5日收益>0
         (不创新低的趋势延续型, 非超跌反弹);
      5) 温和放量: 当日成交量 > 1.2倍 20日均量 (有增量资金确认)。

    信号强度: 板块强度分位 + 个股相对板块超额收益分位 + 乖离动量分位,
      映射到 0.3~1.0, 每日最多保留最强的 10 只 (少而精)。
    """
    empty = pd.Series(dtype=float)
    if df is None or len(df) == 0:
        return empty

    d = df.copy()
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values(['code', 'date'], kind='mergesort').reset_index(drop=True)

    # ---- 逐股票滚动特征 (向量化) ----
    g = d.groupby('code', sort=False)
    d['ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d['vol_ma20'] = g['volume'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d['ret5'] = g['close'].pct_change(5)
    d['sector_ret5'] = g['sector_close'].pct_change(5)
    d['ma20_lag'] = d.groupby('code', sort=False)['ma20'].shift(3)

    # ---- 只保留每只股票最新一行 (T日信号) ----
    last = d.groupby('code', sort=False).tail(1).copy()
    if len(last) == 0:
        return empty

    # ---- 板块轮动强度: 按板块5日收益取全市场前25% (至少2个) ----
    sec_strength = (
        last.groupby('sector')['sector_ret5']
        .first()
        .dropna()
        .sort_values(ascending=False)
    )
    if len(sec_strength) == 0:
        return empty
    n_top = max(2, int(np.ceil(len(sec_strength) * 0.25)))
    top_sectors = set(sec_strength.head(n_top).index)

    # ---- 5个筛选条件 ----
    cond_name = ~last['name'].astype(str).str.contains('ST|退', regex=True, na=False)
    cond_cap = last['circ_cap'] > 100e8
    cond_sector = last['sector'].isin(top_sectors)
    cond_trend = (
        (last['close'] > last['ma20'])
        & (last['ma20'] >= last['ma20_lag'])
        & (last['ret5'] > 0)
    )
    cond_vol = last['volume'] > 1.2 * last['vol_ma20']

    conds = pd.concat(
        [cond_name, cond_cap, cond_sector, cond_trend, cond_vol], axis=1
    )
    mask = conds.fillna(False).all(axis=1)

    sel = last[mask].copy()
    if len(sel) == 0:
        return empty

    # ---- 信号强度打分 (0.3~1.0, 越强越高) ----
    sel['rs'] = sel['ret5'] - sel['sector_ret5']            # 相对板块超额收益
    sel['sector_pct'] = sel['sector_ret5'].rank(pct=True)   # 板块强度分位
    sel['rs_pct'] = sel['rs'].rank(pct=True)                # 相对强度分位
    sel['mom_pct'] = (sel['close'] / sel['ma20'] - 1.0).rank(pct=True)  # 乖离动量分位

    score = 0.4 * sel['sector_pct'] + 0.4 * sel['rs_pct'] + 0.2 * sel['mom_pct']
    sel['strength'] = (0.3 + 0.7 * score).clip(0.3, 1.0)

    # ---- 每日最多保留最强的10只 ----
    sel = sel.sort_values('strength', ascending=False).head(10)
    result = pd.Series(sel['strength'].values, index=sel['code'].values, dtype=float)
    result = result[~result.index.duplicated(keep='first')].dropna()
    return result