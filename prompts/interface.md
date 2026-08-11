## 策略接口（不可修改）

```python
def predict_next_day(df: pd.DataFrame) -> pd.Series:
    """
    参数: df - 全量股票历史数据
           columns: code, date, open, close, high, low, volume, amount,
                    sector, name, circ_cap, total_cap,
                    mkt_open, mkt_close, mkt_high, mkt_low, mkt_volume, mkt_amount,
                    sector_open, sector_close, sector_high, sector_low, sector_volume, sector_amount,
                    prev_close, turnover_rate, amplitude, limit_up_flag, limit_down_flag,
                    mkt_ret_20d, sector_ret_20d
         每只股票按时间排序，最新在最后
    返回: pd.Series(index=code, value=signal_strength)
         只包含你认为次日涨幅>3% 的股票
    """
```
约束：只能用 pandas 和 numpy。函数签名和返回格式不可改。

**板块信息可用于策略：**
- df['sector'] — 行业板块分类，可用 df.groupby('sector') 做板块强度排名
- df['circ_cap'] — 流通市值，可过滤市值区间
- df['name'] — 股票名称，可排除 ST/退市等

**大盘/板块指数（新增）：**
- `mkt_close` — 全 universe 每日期中收盘价（大盘指数），计算市场相对收益：`ret - mkt_ret`
- `sector_close` — 该股票所属行业每日期中收盘价（板块指数），计算板块相对强弱：`ret / sector_ret`
- 同类还有 `mkt_open / mkt_volume / mkt_amount / sector_open / sector_volume / sector_amount`
- 用法示例：`df['mkt_ret'] = df.groupby('code')['mkt_close'].pct_change()`
