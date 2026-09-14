import numpy as np
import pandas as pd


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    RANGE_TREND_011: 震荡市 趋势动量 + 创20日新高突破 + 强板块龙头

    交易流程: T日收盘出信号 -> T+1开盘买入 -> T+2开盘卖出
    目标: 预测 (T+2开盘 - T+1开盘)/T+1开盘 > 3% 的股票

    核心逻辑:
      震荡市中资金抱团少数主线, 已走出多头排列的大盘趋势股在创阶段新高后
      惯性强、抛压轻(无套牢盘), 次日继续冲高的概率大。
      要求 趋势/新高/动量/相对强度/板块强度 五者共振, 宁缺毋滥。
      与池中超跌反弹/放量抄底类策略正交。
    """
    d = df.sort_values(['code', 'date']).copy()
    g = d.groupby('code', sort=False)

    # ---------- 因子计算 (按个股时序) ----------
    d['ma5'] = g['close'].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d['ma20'] = g['close'].transform(lambda s: s.rolling(20, min_periods=10).mean())
    d['ma60'] = g['close'].transform(lambda s: s.rolling(60, min_periods=20).mean())
    # 前20日最高价(不含当日), 用于判定"真突破"
    d['high20_prev'] = g['high'].transform(
        lambda s: s.rolling(20, min_periods=10).max().shift(1)
    )
    # 个股20日动量
    d['ret20'] = d['close'] / g['close'].shift(20) - 1.0
    # 放量配合度: 当日量 / 5日均量
    d['vol_ma5'] = g['volume'].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d['vol_ratio'] = d['volume'] / d['vol_ma5']

    # ---------- 取每只股票最新一日 (T日) ----------
    last = g.tail(1).copy()

    # 板块强度: 用给定的 sector_ret_20d 在板块间做截面排名 (0~1)
    sec_mean = last.groupby('sector')['sector_ret_20d'].mean()
    sec_rank = sec_mean.rank(pct=True)
    last['sector_rank'] = last['sector'].map(sec_rank)

    name = last['name'].fillna('').astype(str)
    limit_up = last['limit_up_flag'].fillna(False).astype(bool)

    # ---------- 基础过滤 (不计入5个AND条件) ----------
    cond_cap = last['circ_cap'] > 100e8                  # 大盘股过滤(必须)
    cond_name = ~name.str.contains('ST|退', regex=True)  # 排除ST/退市
    cond_live = ~limit_up                                # 当日未涨停(T+1开盘可买)

    # ---------- 5个核心AND条件 ----------
    # 1. 多头排列趋势: MA5 > MA20 > MA60 且价格站在MA20上
    cond_trend = (
        (last['ma5'] > last['ma20'])
        & (last['ma20'] > last['ma60'])
        & (last['close'] > last['ma20'])
    )
    # 2. 创20日新高突破: 今日收盘 > 前20日最高价
    cond_break = last['close'] > last['high20_prev']
    # 3. 动量适中: 20日涨幅 8%~45% (有动量但未过热/未加速赶顶)
    cond_mom = (last['ret20'] > 0.08) & (last['ret20'] < 0.45)
    # 4. 相对大盘有超额: 个股20日收益 - 大盘20日收益 > 3%
    cond_rs = (last['ret20'] - last['mkt_ret_20d']) > 0.03
    # 5. 强板块: 所属板块强度排名全市场前40%
    cond_sec = last['sector_rank'] >= 0.60

    mask = (
        cond_cap & cond_name & cond_live
        & cond_trend & cond_break & cond_mom & cond_rs & cond_sec
    )
    mask = mask.fillna(False)

    sel = last[mask].copy()
    if sel.empty:
        return pd.Series(dtype=float)

    # ---------- 信号强度: 多因子加权, 映射到 0.3 ~ 1.0 ----------
    # 突破力度: 收盘高出前20日最高价的比例, 截断于6%
    brk = ((sel['close'] / sel['high20_prev'] - 1.0)
           .clip(0.0, 0.06) / 0.06).fillna(0.0)
    # 相对大盘超额强度, 截断于25%
    rs = ((sel['ret20'] - sel['mkt_ret_20d'])
          .clip(0.0, 0.25) / 0.25).fillna(0.0)
    # 板块强度排名 (0~1)
    secr = sel['sector_rank'].clip(0.0, 1.0).fillna(0.0)
    # 趋势斜率: MA20/MA60 - 1, 截断于15%
    slope = ((sel['ma20'] / sel['ma60'] - 1.0)
             .clip(0.0, 0.15) / 0.15).fillna(0.0)
    # 放量配合: vol_ratio-1, 截断于2倍
    vr = ((sel['vol_ratio'] - 1.0).clip(0.0, 2.0) / 2.0).fillna(0.0)

    raw = 0.30 * brk + 0.25 * rs + 0.20 * secr + 0.15 * slope + 0.10 * vr
    strength = (0.30 + 0.70 * raw).clip(0.30, 1.00)

    out = pd.Series(strength.values, index=sel['code'].values)
    out = out.sort_values(ascending=False)
    return out