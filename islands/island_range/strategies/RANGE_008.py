import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    RANGE_008: 震荡市趋势动量策略 —— 强板块龙头 + 20日新高突破 + 相对强度
    思路: 震荡市中资金抱团少数强势板块, 板块内创20日新高且跑赢大盘的大市值
    龙头股具有动量延续性, T+1开盘买入、T+2开盘卖出捕捉趋势惯性。
    """
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    g = df.groupby('code')

    # ---- 因子计算 (全部按个股时间序列, 无未来函数) ----
    # 前20日(不含今日)最高收盘价 -> 今日是否创20日新高
    prev_high20 = g['close'].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).max())
    # 个股20日动量
    ret20 = g['close'].pct_change(20)
    # 20日均量(截至昨日) -> 量比
    vol_ma20 = g['volume'].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    vol_ratio = df['volume'] / vol_ma20.replace(0, np.nan)

    df['_prev_high20'] = prev_high20
    df['_ret20'] = ret20
    df['_vol_ratio'] = vol_ratio

    # ---- 取最新截面 ----
    last = df.groupby('code').tail(1).set_index('code')

    # 板块强度: 最新日 sector_ret_20d 在全板块中的分位排名
    sector_strength = last.groupby('sector')['sector_ret_20d'].first()
    sector_rank = sector_strength.rank(pct=True)
    last['_sector_rank'] = last['sector'].map(sector_rank)

    # ---- 选股条件 (5个AND, 高确信度低频) ----
    c_cap = last['circ_cap'] > 100e8                                    # 1 大市值
    c_name = ~last['name'].fillna('').str.contains('ST|退')              # 2 非ST/退市
    c_break = last['close'] > last['_prev_high20']                      # 3 创20日新高
    c_sector = (last['_sector_rank'] >= 0.65) & (last['sector_ret_20d'] > 0)  # 4 强板块
    c_rs = (last['_ret20'] > last['mkt_ret_20d']) & (last['_ret20'] > 0)      # 5 强于大盘

    cond = (c_cap & c_name & c_break & c_sector & c_rs).fillna(False)
    sel = last[cond]
    if sel.empty:
        return pd.Series(dtype=float)

    # ---- 信号强度: 0.3 ~ 1.0, 越强越高 ----
    mom_q = sel['_ret20'].rank(pct=True)                                # 动量分位(候选池内)
    brk = ((sel['close'] / sel['_prev_high20'] - 1) / 0.08).clip(0, 1)  # 突破幅度 0~8%->0~1
    sec_q = sel['_sector_rank'].clip(0, 1)                              # 板块强度分位
    vol_q = ((sel['_vol_ratio'] - 1) / 2).clip(0, 1)                    # 量比 1~3 -> 0~1

    score = 0.35 * mom_q + 0.25 * brk + 0.25 * sec_q + 0.15 * vol_q
    strength = (0.3 + 0.7 * score).clip(0.3, 1.0).fillna(0.3)
    strength.index.name = 'code'
    return strength