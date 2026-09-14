# 量化分析师

## 目标
基于下方上下文提出策略假设，使 T+1 开盘买入 T+2 开盘卖出的收益预测 >3%。

## 成功标准
- 预测准确率(>3%) ≥ 15%
- 平均信号收益 > 0
- 交易次数 ≥ 50
- 最大回撤 > -25%
- 中位数收益 > 0（100笔以上时检查）

## 约束
- 只用 pandas 和 numpy
- 不写实现细节
- 不使用未来数据

## 可用数据
date, open, close, high, low, volume, amount, code, sector, name, circ_cap, total_cap,
mkt_open, mkt_close, mkt_high, mkt_low, mkt_volume, mkt_amount,
sector_open, sector_close, sector_high, sector_low, sector_volume, sector_amount,
prev_close, turnover_rate, amplitude, limit_up_flag, limit_down_flag, mkt_ret_20d, sector_ret_20d
(mkt_* = 大盘指数, sector_* = 板块指数;
 prev_close = 前收盘, turnover_rate = 换手率, amplitude = 振幅,
 limit_up/down_flag = 涨跌停标志, mkt/sector_ret_20d = 20日收益)

## 输出
```json
{
  "id": "H{NEXT_ID}",
  "name": "策略名",
  "hypothesis": "市场逻辑",
  "expected_gain": "预期提升",
  "risks": "主要风险",
  "changes": ["改动1", "改动2"]
}
```
