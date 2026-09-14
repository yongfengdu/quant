import pandas as pd
import numpy as np


def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    RANGE_013: 趋势动量 —— 20日新高突破 + 相对强度 + 板块共振打分

    核心逻辑(与现有超跌/低吸池正交, 纯趋势动量方向):
      震荡市中, 收盘价突破前20日最高价(创20日新高)、20日相对大盘有显著超额收益、
      温和放量的大盘股, T+1开盘买入 -> T+2开盘卖出 的动量延续概率最高。
      板块相对强度不做硬过滤(避免信号崩溃), 只用于信号强度打分, 实现板块共振优先。
      不要求收阳线(实盘反馈显示多只错过的大牛股信号日并未收阳)。
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    g = df.groupby('code')

    # ---------- 个股时序指标 ----------
    df['ret_20d'] = g['close'].pct_change(20)
    # 前20日最高价(不含当日), 用于判定"创20日新高"
    df['prev_high_20'] = g['high'].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).max()
    )
    # 前20日均量(不含当日), 用于判定"温和放量"
    df['vol_ma20'] = g['volume'].transform(
        lambda s: s.shift(1).rolling(20, min_periods=15).mean()
    )

    # ---------- 大盘/板块 20日收益(优先用给定列, 缺失则自算) ----------
    if 'mkt_ret_20d' in df.columns and df['mkt_ret_20d'].notna().any():
        df['mkt20'] = df['mkt_ret_20d']
    else:
        df['mkt20'] = g['mkt_close'].pct_change(20)
    if 'sector_ret_20d' in df.columns and df['sector_ret_20d'].notna().any():
        df['sec20'] = df['sector_ret_20d']
    else:
        df['sec20'] = g['sector_close'].pct_change(20)

    df['rs_stock'] = df['ret_20d'] - df['mkt20']    # 个股相对大盘强度
    df['rs_sector'] = df['sec20'] - df['mkt20']     # 板块相对大盘强度

    # ---------- 每只股票取最新一行 ----------
    last = df.groupby('code').tail(1).set_index('code')

    # ---------- 硬条件(5个AND以内) ----------
    cond_cap = last['circ_cap'] > 100e8                       # 1) 大市值过滤
    cond_break = last['close'] > last['prev_high_20']         # 2) 收盘创20日新高
    cond_rs = last['rs_stock'] > 0.05                         # 3) 20日超额收益>5%
    cond_vol = (last['volume'] > 1.2 * last['vol_ma20']) & \
               (last['volume'] < 5.0 * last['vol_ma20'])      # 4) 温和放量(防爆量)
    cond_buyable = (last['limit_up_flag'].fillna(0) != 1) & \
                   ~last['name'].astype(str).str.contains('ST|退', na=False)  # 5) 非涨停非ST

    cond = (cond_cap & cond_break & cond_rs & cond_vol & cond_buyable).fillna(False)

    # ---------- 信号强度: 个股RS与板块RS截面分位加权, 映射到0.3~1.0 ----------
    rs_stock_rank = last['rs_stock'].rank(pct=True).clip(0, 1).fillna(0.5)
    rs_sector_rank = last['rs_sector'].rank(pct=True).clip(0, 1).fillna(0.5)
    score = 0.7 * rs_stock_rank + 0.3 * rs_sector_rank
    strength = (0.3 + 0.7 * score).clip(0.3, 1.0)

    sig = strength[cond].sort_values(ascending=False)
    sig.index.name = 'code'
    return sig.astype(float)