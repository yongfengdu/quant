import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    BULL_TREND_RS: 牛市趋势动量 + 相对强度策略

    核心逻辑: 牛市中强者恒强。只在大盘处于牛市区间时, 选取
      1) 大市值 (circ_cap > 100亿)、非ST/非退市;
      2) 趋势多头: close > MA20 > MA60;
      3) 贴近 60 日新高 (close >= 0.94 * 60日最高价) 的平台突破型强势股;
      4) 20日动量排名全市场前 15% (强相对强度);
      5) 所属板块 20 日收益为正 (板块 beta 走强);
      并用 5 日涨幅 < 25% 排除短期过热。信号日不要求收阳——强势股
    横盘/小阴蓄势次日照样大涨(补线上错过的缺口)。
    信号强度由 动量排名/超额收益/新高贴近度/板块强度/量能 综合打分映射到 0.3~1.0。
    """
    d = df.sort_values(['code', 'date']).copy()
    g = d.groupby('code', sort=False)

    # ---------- 逐股票时序特征 ----------
    d['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=20).mean())
    d['ma60'] = g['close'].transform(lambda x: x.rolling(60, min_periods=60).mean())
    d['high60'] = g['high'].transform(lambda x: x.rolling(60, min_periods=20).max())
    d['ret20'] = g['close'].transform(lambda x: x.pct_change(20))
    d['ret5'] = g['close'].transform(lambda x: x.pct_change(5))
    d['vma20'] = g['volume'].transform(lambda x: x.rolling(20, min_periods=20).mean())

    # ---------- 取每只股票最新一根K线 (T日截面) ----------
    last = d.groupby('code', sort=False).tail(1).set_index('code')
    if len(last) == 0:
        return pd.Series(dtype=float)

    # ---------- 硬过滤 (5个核心AND条件 + 卫生过滤) ----------
    name = last['name'].astype(str)
    hygiene = (~name.str.contains('ST', case=False, na=False)) & \
              (~name.str.contains('退', na=False)) & \
              (last['circ_cap'] > 100e8) & \
              (last['volume'] > 0)

    c1_trend = (last['close'] > last['ma20']) & (last['ma20'] > last['ma60'])
    c2_near_high = last['close'] >= 0.94 * last['high60']
    c3_momo_top = last['ret20'] >= last['ret20'].quantile(0.85)
    c4_sector = last['sector_ret_20d'] > 0
    c5_not_hot = last['ret5'] < 0.25

    mask = (hygiene & c1_trend & c2_near_high & c3_momo_top
            & c4_sector & c5_not_hot).fillna(False)

    cand = last[mask]
    if len(cand) == 0:
        return pd.Series(dtype=float)

    # ---------- 信号强度: 截面分位加权打分 ----------
    rs_mkt = cand['ret20'] - cand['mkt_ret_20d']                 # 相对大盘超额
    high_prox = cand['close'] / cand['high60']                   # 新高贴近度
    vol_ratio = (cand['volume'] / cand['vma20']).clip(0, 3)      # 量比(温和放量加分)

    score = (0.30 * cand['ret20'].rank(pct=True)
             + 0.25 * rs_mkt.rank(pct=True)
             + 0.20 * high_prox.rank(pct=True)
             + 0.15 * cand['sector_ret_20d'].rank(pct=True)
             + 0.10 * vol_ratio.rank(pct=True))

    sig = (0.3 + 0.7 * score).clip(0.3, 1.0)
    sig = sig.fillna(0.3)
    return sig.sort_values(ascending=False)