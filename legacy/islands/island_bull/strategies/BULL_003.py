"""
牛市策略 3: 板块轮动追涨
适用场景: 牛市 (20日涨>5%)
核心逻辑: 追踪领涨板块中的滞涨龙头
"""
import numpy as np
import pandas as pd

def predict_next_day(df):
    """板块轮动追涨"""
    df = df.sort_values(['code', 'date']).copy()
    g = df.groupby('code', sort=False)
    
    # 计算指标
    df['ret'] = g['close'].pct_change()
    df['ret_5d'] = g['close'].pct_change(5)
    df['ret_10d'] = g['close'].pct_change(10)
    
    # 板块收益
    if 'sector_close' in df.columns:
        sg = df.groupby('sector', sort=False)
        df['sector_ret_5d'] = sg['sector_close'].transform(lambda x: x.pct_change(5))
        df['sector_ret_10d'] = sg['sector_close'].transform(lambda x: x.pct_change(10))
    else:
        df['sector_ret_5d'] = 0
        df['sector_ret_10d'] = 0
    
    # 成交量
    df['vol_ma10'] = g['volume'].transform(lambda x: x.rolling(10, min_periods=5).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ma10']
    
    today = df[df['date'] == df['date'].max()].copy()
    if today.empty:
        return pd.Series(dtype=float)
    
    # 条件1: 板块5日涨幅领先 (>5%)
    sector_leading = (today['sector_ret_5d'] > 0.05).fillna(False)
    
    # 条件2: 个股跑输板块 (滞涨，有补涨空间)
    stock_lagging = (today['ret_5d'] < today['sector_ret_5d']).fillna(False)
    
    # 条件3: 个股10日涨幅不是太弱 (>-3%，基本面没问题)
    not_too_weak = (today['ret_10d'] > -0.03).fillna(False)
    
    # 条件4: 今日有启动迹象 (涨幅 > 1% 或放量)
    starting = ((today['ret'] > 0.01) | (today['vol_ratio'] > 1.3)).fillna(False)
    
    # 条件5: 排除涨停板
    not_limit = (today['ret'] < 0.095).fillna(True)
    
    # 条件6: 市值过滤 (中大盘)
    if 'circ_cap' in today.columns:
        cap_ok = ((today['circ_cap'] > 50e8) & (today['circ_cap'] < 2000e8)).fillna(True)
    else:
        cap_ok = True
    
    mask = sector_leading & stock_lagging & not_too_weak & starting & not_limit & cap_ok
    
    if not mask.any():
        return pd.Series(dtype=float)
    
    today_f = today[mask].copy()
    
    # 信号强度: 板块越强、滞涨越明显越好
    sector_strength = (today_f['sector_ret_5d'] - 0.05).clip(0, 0.10) / 0.10
    lag_amount = (today_f['sector_ret_5d'] - today_f['ret_5d']).clip(0, 0.10) / 0.10
    signal = 0.4 + 0.35 * sector_strength + 0.25 * lag_amount
    
    return pd.Series(signal.values, index=today_f['code'].values)
