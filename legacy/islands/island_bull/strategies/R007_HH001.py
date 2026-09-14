import pandas as pd
import numpy as np

def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    动量龙头追击策略 - BULL市场
    筛选条件：
    1. 最近5日收益率排名前10%
    2. 个股超额收益（相对大盘）>0 且大于板块平均超额收益
    3. 板块强度（sector_ret_20d）排名前5
    4. 流通市值30亿-500亿中盘股
    5. 排除涨停股票
    """
    # 按股票代码和日期排序，确保时间顺序
    df = df.sort_values(['code', 'date']).copy()
    
    # 按股票分组
    g = df.groupby('code', sort=False)
    
    # ========== 计算指标 ==========
    # 1. 个股5日收益率
    df['ret_5d'] = g['close'].pct_change(5)
    
    # 2. 个股相对大盘的超额收益（个股收益 - 大盘收益）
    # 大盘20日收益率已提供：mkt_ret_20d
    # 计算个股20日收益率用于超额收益比较
    df['ret_20d'] = g['close'].pct_change(20)
    df['excess_ret'] = df['ret_20d'] - df['mkt_ret_20d']
    
    # 3. 计算板块内平均超额收益（用于条件2比较）
    # 按日期和板块分组，计算板块内个股超额收益的均值
    df['sector_excess_mean'] = df.groupby(['date', 'sector'])['excess_ret'].transform('mean')
    
    # 4. 涨停标志（已提供）
    
    # ========== 获取最新日期数据 ==========
    latest_date = df['date'].max()
    today = df[df['date'] == latest_date].copy()
    
    if today.empty:
        return pd.Series(dtype=float)
    
    # ========== 条件筛选 ==========
    
    # 条件1: 5日收益率排名前10%
    # 按日期分组计算排名（虽然只有最新一天，但保持通用性）
    today['ret_5d_rank'] = today.groupby('date')['ret_5d'].rank(ascending=False, pct=True)
    cond1 = today['ret_5d_rank'] <= 0.10  # 前10%
    
    # 条件2: 超额收益为正且大于板块平均超额收益
    cond2 = (today['excess_ret'] > 0) & (today['excess_ret'] > today['sector_excess_mean'])
    
    # 条件3: 板块强度排名前5
    # 使用sector_ret_20d（板块20日收益率）进行板块排名
    sector_strength = today.groupby('sector')['sector_ret_20d'].first().sort_values(ascending=False)
    top5_sectors = sector_strength.head(5).index.tolist()
    cond3 = today['sector'].isin(top5_sectors)
    
    # 条件4: 流通市值30亿-500亿
    cond4 = (today['circ_cap'] >= 30e8) & (today['circ_cap'] <= 500e8)
    
    # 条件5: 排除涨停股票
    cond5 = ~today['limit_up_flag']
    
    # 综合所有条件
    final_mask = cond1 & cond2 & cond3 & cond4 & cond5
    
    # ========== 生成信号 ==========
    selected = today[final_mask].copy()
    
    if selected.empty:
        return pd.Series(dtype=float)
    
    # 信号强度：使用5日收益率 * 超额收益（越大表示动量越强且独立于大盘）
    signal_strength = selected['ret_5d'] * selected['excess_ret']
    
    # 返回Series，索引为股票代码，值为信号强度
    return pd.Series(
        signal_strength.values,
        index=selected['code'].values
    )