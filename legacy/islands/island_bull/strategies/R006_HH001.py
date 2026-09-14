import pandas as pd
import numpy as np

def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    强势板块动量突破策略 - BULL市场
    1. 计算板块相对强度排名，选前3板块
    2. 在候选板块中选突破前高+放量+动量确认的个股
    3. 综合打分，保留信号强度>0.7的股票
    """
    # 按股票和日期排序，确保时间顺序
    df = df.sort_values(['code', 'date']).copy()
    
    # 按股票分组
    g = df.groupby('code', sort=False)
    
    # ========== 计算个股指标 ==========
    # 过去10日个股收益
    df['ret_10d'] = g['close'].pct_change(10)
    # 过去5日个股收益
    df['ret_5d'] = g['close'].pct_change(5)
    # 过去20日最高价
    df['high_20d'] = g['high'].transform(lambda x: x.rolling(20, min_periods=20).max())
    # 过去20日均量
    df['vol_ma20'] = g['volume'].transform(lambda x: x.rolling(20, min_periods=20).mean())
    # 成交量比率（当日成交量/20日均量）
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    
    # ========== 计算板块相对强度 ==========
    # 个股过去10日收益 - 对应板块收益 = 个股相对板块的超额收益
    # sector_close是板块指数收盘价，计算板块收益
    df['sector_ret_10d'] = df.groupby('code')['sector_close'].transform(lambda x: x.pct_change(10))
    # 个股相对板块超额收益
    df['excess_ret_vs_sector'] = df['ret_10d'] - df['sector_ret_10d']
    
    # 获取最新日期
    latest_date = df['date'].max()
    today = df[df['date'] == latest_date].copy()
    
    if today.empty:
        return pd.Series(dtype=float)
    
    # ========== 板块强度排名（基于最新日期的超额收益中位数） ==========
    sector_strength = today.groupby('sector')['excess_ret_vs_sector'].median().reset_index()
    sector_strength.columns = ['sector', 'sector_strength']
    sector_strength = sector_strength.sort_values('sector_strength', ascending=False)
    
    # 取前3个强势板块
    top3_sectors = sector_strength.head(3)['sector'].tolist()
    
    # 筛选属于强势板块的股票
    today = today[today['sector'].isin(top3_sectors)].copy()
    
    if today.empty:
        return pd.Series(dtype=float)
    
    # ========== 个股条件筛选 ==========
    # 条件1: 过去5日涨幅 > 8%（动量确认）
    cond1 = today['ret_5d'] > 0.08
    
    # 条件2: 当日收盘价 >= 过去20日最高价 * 0.98（接近前高突破）
    cond2 = today['close'] >= today['high_20d'] * 0.98
    
    # 条件3: 成交量 > 过去20日均量的1.2倍（放量确认）
    cond3 = today['vol_ratio'] > 1.2
    
    # 条件4: 排除流通市值 < 30亿 或 > 1000亿
    cond4 = (today['circ_cap'] >= 30e8) & (today['circ_cap'] <= 1000e8)
    
    # 条件5: 排除ST/退市股（名称包含ST、*ST、退市等）
    cond5 = ~today['name'].str.contains('ST|退市|\\*ST', na=False)
    
    # 综合条件
    mask = cond1 & cond2 & cond3 & cond4 & cond5
    candidates = today[mask].copy()
    
    if candidates.empty:
        return pd.Series(dtype=float)
    
    # ========== 信号强度计算 ==========
    # 1. 板块强度排名归一化（排名越靠前值越大）
    # 将板块强度映射到0-1之间
    sector_strength_dict = sector_strength.set_index('sector')['sector_strength'].to_dict()
    candidates['sector_strength_val'] = candidates['sector'].map(sector_strength_dict)
    
    # 板块强度归一化（min-max归一化）
    s_min = candidates['sector_strength_val'].min()
    s_max = candidates['sector_strength_val'].max()
    if s_max > s_min:
        candidates['sector_norm'] = (candidates['sector_strength_val'] - s_min) / (s_max - s_min)
    else:
        candidates['sector_norm'] = 1.0
    
    # 2. 个股5日收益归一化
    r_min = candidates['ret_5d'].min()
    r_max = candidates['ret_5d'].max()
    if r_max > r_min:
        candidates['ret_norm'] = (candidates['ret_5d'] - r_min) / (r_max - r_min)
    else:
        candidates['ret_norm'] = 1.0
    
    # 3. 成交量比率归一化
    v_min = candidates['vol_ratio'].min()
    v_max = candidates['vol_ratio'].max()
    if v_max > v_min:
        candidates['vol_norm'] = (candidates['vol_ratio'] - v_min) / (v_max - v_min)
    else:
        candidates['vol_norm'] = 1.0
    
    # 综合信号强度 = 板块强度*0.4 + 个股收益*0.3 + 成交量比率*0.3
    candidates['signal_strength'] = (
        candidates['sector_norm'] * 0.4 +
        candidates['ret_norm'] * 0.3 +
        candidates['vol_norm'] * 0.3
    )
    
    # 只保留信号强度 > 0.7 的股票
    final = candidates[candidates['signal_strength'] > 0.7].copy()
    
    if final.empty:
        return pd.Series(dtype=float)
    
    # 返回 Series(code -> signal_strength)
    return pd.Series(final['signal_strength'].values, index=final['code'].values)