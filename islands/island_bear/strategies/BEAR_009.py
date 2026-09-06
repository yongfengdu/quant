"""
BEAR_009: 熊市抗跌相对强度趋势延续策略 (大市值避险领涨)

核心逻辑 (与 BEAR_001/003/004 超跌反弹、005-008 抗跌突破正交):
  熊市中资金抱团少数抗跌领涨的大市值龙头, 这类股票呈现趋势延续特征
  (对应实盘反馈: 错过机会多为"非超跌反弹"的趋势延续型上涨, 如688253未创新低)。
  选股条件:
    1) 大市值: circ_cap > 100e8 (熊市大盘股T+1可预测性更高, 且抗跌属性强)
    2) 趋势向上: close > MA20, 且收盘价贴近20日新高 (趋势延续, 非抄底)
    3) 相对强度: 个股10日收益显著跑赢大盘 (熊市避险资金抱团的直接证据)
    4) 放量确认: 成交量 > 1.2倍5日均量 (突破有资金支持, 非无量虚涨)
  信号强度: 由相对强度 / 贴近新高程度 / 量比 / 板块相对强度合成, 映射到 0.3~1.0
"""

import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    参数: df - 全量股票历史数据 (每只股票按时间排序, 最新在最后)
    返回: pd.Series(index=code, value=signal_strength), 只包含预期次日涨幅>3%的股票
    """
    # ---------- 0. 数据准备 ----------
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    g = df.groupby('code', sort=False)

    # ---------- 1. 因子计算 (全部按 code 分组滚动, 只用T日及之前数据) ----------
    # 个股与大盘10日动量 -> 相对强度
    df['ret_10'] = g['close'].pct_change(10)
    df['mkt_ret_10'] = g['mkt_close'].pct_change(10)
    df['rs_10'] = df['ret_10'] - df['mkt_ret_10']            # 相对大盘超额收益

    # 板块10日相对强度 (板块 vs 大盘, 用于强度加权)
    df['sec_ret_10'] = g['sector_close'].pct_change(10)
    df['sec_rs_10'] = df['sec_ret_10'] - df['mkt_ret_10']

    # 趋势: MA20 与 20日最高收盘价
    df['ma20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df['high_20'] = g['close'].transform(lambda x: x.rolling(20, min_periods=20).max())

    # 量能: 5日均量与量比
    df['vol_ma5'] = g['volume'].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    # ---------- 2. 只取每只股票最新一天 (T日) 的截面 ----------
    last = df.groupby('code', sort=False).tail(1).copy()

    # 名称过滤: 排除 ST / 退市股 (name可能为NaN)
    name = last['name'].fillna('').astype(str)
    name_ok = ~(name.str.contains('ST', case=False) | name.str.contains('退'))

    # ---------- 3. 选股条件 (4个核心AND条件 + 市值/名称过滤, 避免过拟合) ----------
    cond_cap = last['circ_cap'] > 100e8                      # 1) 大市值 > 100亿
    cond_trend = (last['close'] > last['ma20']) & \
                 (last['close'] >= 0.95 * last['high_20'])   # 2) MA20上方且贴近20日新高
    cond_rs = last['rs_10'] > 0.05                           # 3) 10日跑赢大盘5%以上
    cond_vol = last['vol_ratio'] > 1.2                       # 4) 放量确认

    mask = (cond_cap & cond_trend & cond_rs & cond_vol & name_ok).fillna(False)
    sel = last[mask].copy()
    if sel.empty:
        return pd.Series(dtype=float)

    # ---------- 4. 信号强度合成: 因子加权, 映射到 0.3~1.0 ----------
    # (a) 个股相对强度: 5%~30% 线性映射到 0~1
    rs_score = ((sel['rs_10'].clip(0.05, 0.30) - 0.05) / 0.25).fillna(0.0)
    # (b) 贴近新高程度: 0.95~1.00 映射到 0~1
    pos_score = (((sel['close'] / sel['high_20']).clip(0.95, 1.0) - 0.95) / 0.05).fillna(0.0)
    # (c) 量比: 1.2~3.0 映射到 0~1
    vol_score = ((sel['vol_ratio'].clip(1.2, 3.0) - 1.2) / 1.8).fillna(0.0)
    # (d) 板块相对强度微调: 板块也强则加分 (弱市领涨板块抱团效应)
    sec_score = ((sel['sec_rs_10'].clip(0.0, 0.15)) / 0.15).fillna(0.0)

    score = 0.45 * rs_score + 0.25 * pos_score + 0.20 * vol_score + 0.10 * sec_score
    strength = (0.3 + 0.7 * score).clip(0.3, 1.0)

    # ---------- 5. 输出: index=code, value=强度, 按强度降序 ----------
    out = pd.Series(strength.values, index=sel['code'].values, dtype=float)
    out = out.groupby(level=0).max()          # 防御性去重
    out = out.sort_values(ascending=False)
    return out.round(4)