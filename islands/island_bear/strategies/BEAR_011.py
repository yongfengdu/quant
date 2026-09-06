import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    BEAR: 大市值趋势延续 + 板块相对强度领涨策略.

    熊市资金向大市值强势股抱团。选择:
      1) 流通市值 > 100亿, 非ST/退市
      2) 未创新低且贴近20日新高 (趋势延续, 而非超跌反弹)
      3) 个股10日收益强于大盘 (相对强度为正)
      4) 所属板块10日收益强于大盘 (板块轮动中的领涨板块)
      5) 可交易性: 未涨停/未跌停, 单日涨幅未过热, 有量能配合
    信号强度 = 相对强度/板块强度/贴高度/量比的截面排名加权, 映射到 0.3~1.0
    """
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)

    # ---- 因子计算 (每只股票内部按时间滚动) ----
    df['ret_1'] = g['close'].pct_change()
    df['ret_10'] = g['close'].pct_change(10)
    df['mkt_ret_10'] = g['mkt_close'].pct_change(10)
    df['sec_ret_10'] = g['sector_close'].pct_change(10)

    # 相对强度: 个股 / 板块 相对大盘
    df['rs_10'] = df['ret_10'] - df['mkt_ret_10']
    df['sec_rs_10'] = df['sec_ret_10'] - df['mkt_ret_10']

    # 贴近20日新高的程度 (趋势延续而非创新低)
    df['max_20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=10).max())
    df['high_prox'] = df['close'] / df['max_20'].replace(0, np.nan)

    # 量比: 当日成交量 / 5日均量
    df['vol_ma5'] = g['volume'].transform(lambda s: s.rolling(5, min_periods=2).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma5'].replace(0, np.nan)

    # ---- 取每只股票最新一行 ----
    last = g.tail(1).set_index('code')

    name = last['name'].astype(str)
    limit_up = last['limit_up_flag'].fillna(False).astype(bool)
    limit_down = last['limit_down_flag'].fillna(False).astype(bool)

    # ---- 选股条件 (5条主条件) ----
    # 1) 大市值 + 非ST/退市
    cap_ok = (last['circ_cap'] > 100e8) & ~name.str.contains('ST|退', na=False)
    # 2) 趋势延续: 收盘价不低于20日最高价的95% (未创新低, 贴近新高)
    trend_ok = last['high_prox'] >= 0.95
    # 3) 个股相对强度为正 (10日跑赢大盘)
    rs_ok = last['rs_10'] > 0
    # 4) 板块相对强度为正 (领涨板块); 板块数据缺失时要求个股强度更高作为补偿
    sec_rs = last['sec_rs_10']
    sec_ok = (sec_rs > 0) | (sec_rs.isna() & (last['rs_10'] > 0.03))
    # 5) 可交易且未过热: 非涨停非跌停, 单日涨跌在(-5%, +6.5%), 有基本量能
    tradable = (
        (~limit_up) & (~limit_down)
        & last['ret_1'].between(-0.05, 0.065)
        & (last['vol_ratio'] > 0.8)
    )

    mask = (cap_ok & trend_ok & rs_ok & sec_ok & tradable).fillna(False)
    sel = last[mask]
    if sel.empty:
        return pd.Series(dtype=float)

    # ---- 信号强度: 四因子截面排名加权, 映射到 0.3~1.0 ----
    rs_rank = sel['rs_10'].rank(pct=True)                       # 个股相对强度
    sec_rank = sel['sec_rs_10'].fillna(0.0).rank(pct=True)      # 板块相对强度
    prox_rank = sel['high_prox'].rank(pct=True)                 # 贴近新高程度
    vol_rank = sel['vol_ratio'].clip(0.5, 3.0).rank(pct=True)   # 量比(截断防爆量)

    score = 0.40 * rs_rank + 0.25 * sec_rank + 0.20 * prox_rank + 0.15 * vol_rank
    signal = (0.3 + 0.7 * score).clip(0.3, 1.0)

    return signal