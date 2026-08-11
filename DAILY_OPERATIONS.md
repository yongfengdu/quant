# 量化系统每日操作指南

## 概述

本系统用于A股短线交易信号预测：
- **交易逻辑**: T日收盘后生成信号 → T+1日开盘买入 → T+2日开盘卖出
- **目标收益**: >3%
- **股票池**: 46只A股

## 每日执行流程

### 时间线

```
T日 (信号日)              T+1日 (买入日)           T+2日 (卖出日)
     |                         |                        |
  15:00 收盘                开盘买入                  开盘卖出
     |                      (持有中)                     |
  15:30-16:00                                       15:30-16:00
  运行脚本                                          运行脚本
     |                                                  |
     v                                                  v
  1. 更新数据                                      验证T日预测的
  2. 验证(T-2)日预测                               实际收益
  3. 生成T+1日买入信号
```

### 执行顺序 (每日15:30后)

| 步骤 | 命令 | 作用 | 耗时 |
|------|------|------|------|
| 1 | `python update_data.py` | 拉取最新K线数据 | 2-5分钟 |
| 2 | `python daily_tracker.py verify` | 验证2天前预测的实际收益 | <10秒 |
| 3 | `python daily_tracker.py adaptive` | 生成明日买入信号(13策略聚合) | 3-5秒 |

### 一键执行

```bash
cd /root/quant-autoresearch
./daily_run.sh
```

## 命令详解

### 1. 数据更新 (update_data.py)

```bash
python update_data.py          # 增量更新(推荐)
python update_data.py --force  # 强制全量更新
```

- 数据源: 腾讯财经API
- 存储位置: `~/.cache/quant-autoresearch/daily.parquet`
- 包含: OHLCV、板块指数、大盘指数

### 2. 验证预测 (daily_tracker.py verify)

```bash
python daily_tracker.py verify
```

- 验证的是**2天前**的预测 (因为T日预测→T+1买→T+2卖)
- 自动计算实际收益率
- 更新策略监控器状态
- 结果保存到: `daily_predictions.csv`

### 3. 生成预测 (daily_tracker.py adaptive)

```bash
python daily_tracker.py adaptive   # 推荐: 13策略聚合
python daily_tracker.py predict    # 备选: 单策略(R187)
```

- 读取最新数据
- 运行所有验证过的策略
- 根据市场状态激活对应策略类别
- 输出: 明日开盘买入的股票代码

### 4. 其他命令

```bash
python daily_tracker.py summary      # 查看近30天准确率统计
python daily_tracker.py summary 90   # 查看近90天
python daily_tracker.py monitor      # 策略健康状态检查
python daily_tracker.py report       # 系统考评报告(全部数据)
python daily_tracker.py report week  # 近一周考评报告
python daily_tracker.py report month # 近一月考评报告
python daily_tracker.py lesson       # 将统计写入LESSONS.md
```

## 定时任务配置

### 系统级 Cron (crontab)

| 任务 | 时间 | 脚本 |
|------|------|------|
| 每日执行 | 周一-周五 15:35 | `daily_run.sh` |
| 周报 | 周六 9:00 | `weekly_report.py` |
| 月报 | 每月1日 9:00 | `weekly_report.py --period=month` |

### Hermes Cron (微信推送)

| 任务 | 时间 | 脚本 |
|------|------|------|
| 量化策略日报 | 周一-周五 16:30 | `quant_daily.sh` |

日报内容包括：
- 市场状态（恐慌/恐惧/中性/贪婪/狂热）
- 今日预测信号（如有）
- 昨日复盘（黄金机会命中/错过）

### 查看/管理 Hermes 任务

```bash
hermes cron list              # 查看所有任务
hermes cron run <job_id>      # 立即执行某任务
hermes cron pause <job_id>    # 暂停
hermes cron resume <job_id>   # 恢复
```

## 自动汇报与诊断机制

系统通过 hermes 自动发送报告到微信，并在表现不佳时自动诊断：

| 报告类型 | 时间 | 触发条件 |
|----------|------|----------|
| **周报** | 每周六 9:00 | 定时 |
| **月报** | 每月1日 9:00 | 定时 |
| **诊断报告** | 随周报 | 评分 < 4 时自动触发 |

### 自动诊断流程

当周报评分低于 4 分时，系统自动：
1. 分析失败案例共性（板块、信号强度、星期）
2. 分析策略表现（哪些策略拖后腿）
3. 分析集中度（是否过度集中某只股票）
4. 生成改进建议并发送到微信

### 手动命令

```bash
# 周报/月报
python weekly_report.py --period=week
python weekly_report.py --period=month

# 手动诊断分析
python auto_improve.py analyze              # 分析问题
python auto_improve.py suggest              # 生成并发送建议
python auto_improve.py apply --index=1 --confirm  # 应用第1条建议

# 测试模式
python weekly_report.py --test
python auto_improve.py suggest --test
```

### 改进建议类型

| 建议类型 | 说明 | 自动执行 |
|----------|------|----------|
| `exclude_sectors` | 排除表现差的板块 | 需确认 |
| `suspend_strategies` | 暂停表现差的策略 | 需确认 |
| `raise_signal_threshold` | 提高信号阈值 | 需确认 |
| `add_diversity_filter` | 增加分散度过滤 | 需确认 |

### 报告示例

```
[ 量化系统周报 ]
2026-07-24

-- 核心指标 --
周期: 07/16-07/23 (6个交易日)
样本: 18条预测

V 3%命中: 22.2% (基线10.1%)
V 胜率: 61.1%
V 平均收益: +0.99%
  累计收益: +5.96%
  夏普比率: 7.45
  盈亏比: 1.42

-- 风控指标 --
V 最大回撤: -1.36%
V 最大连亏: 2天

-- 板块表现 --
^ 白酒Ⅱ: +2.06% (5)
...

-- 综合评分 --
得分: 8/8
建议: 继续运行
```

### 符号说明
- `V` = 达标
- `X` = 不达标
- `!` = 警告
- `^` = 正收益
- `v` = 负收益

## 考评系统

使用 `report` 命令生成系统考评报告：

```bash
python daily_tracker.py report       # 全量数据
python daily_tracker.py report week  # 近一周
python daily_tracker.py report month # 近一月
```

报告包含：

| 维度 | 指标 | 说明 |
|------|------|------|
| **核心KPI** | 3%命中率 | 目标>10.14%(基线) |
| | 胜率 | 目标>50% |
| | 平均收益 | 目标>0 |
| | 累计收益 | 按日等权计算 |
| | 夏普比率 | 年化，目标>1 |
| | 盈亏比 | 目标>1 |
| **风控指标** | 最大回撤 | 警戒线-10% |
| | 连续亏损 | 警戒线5次 |
| | 单笔最大亏损 | 警戒线-5% |
| **趋势分析** | 周度表现 | 观察趋势变化 |
| | 近期vs历史 | 判断是否恶化 |
| **板块分析** | 各板块胜率/收益 | 识别优势/劣势板块 |
| **决策建议** | 综合评分(0-8) | ≥5继续，<2暂停 |

**建议观察频率**：
- 周报：每周一早上查看 `report week`
- 月报：每月初查看 `report month`

## 策略迭代

当需要改进策略时：

```bash
python run_round.py    # 运行一轮LLM策略进化
```

流程:
1. LLM分析师读取当前策略和LESSONS.md
2. 提出改进假设
3. LLM工程师实现代码
4. 自动回测评估
5. 通过Gate则保存到 `rounds/` 目录

新策略会自动被 `adaptive` 命令加载使用。

## 文件说明

| 文件 | 说明 |
|------|------|
| `daily_predictions.csv` | 所有预测记录和验证结果 |
| `daily_accuracy.log` | 每日准确率日志 |
| `cron_daily.log` | 自动任务执行日志 |
| `~/.cache/quant-autoresearch/daily.parquet` | K线数据 |
| `data/strategy_pool_state.json` | 策略权重状态 |
| `data/strategy_monitor_state.json` | 策略健康监控状态 |

## 策略池状态

当前系统包含 **13个已验证策略**:

| 类别 | 数量 | 说明 |
|------|------|------|
| reversal | 1 | 通用反转 |
| reversal_panic | 2 | 恐慌反转(跌停/暴跌后) |
| reversal_volume | 10 | 量能反转(缩量后放量) |

最佳策略: **R187** (准确率32.1%, 28笔交易)

## 市场状态门控

系统会根据市场状态自动选择策略:

| 市场状态 | 条件 | 激活策略 |
|----------|------|----------|
| 恐慌(panic) | 20日跌>10% & 5日跌>5% | 所有反转类 |
| 恐惧(fear) | 20日跌>5% | 所有反转类 + 防守 |
| 中性(neutral) | 其他 | 全部策略 |
| 贪婪(greed) | 20日涨>5% | 量能反转 |
| 狂热(euphoria) | 20日涨>10% & 5日涨>5% | 量能反转 |

## 故障排查

### 数据更新失败
```bash
# 检查网络
curl -I https://web.ifzq.gtimg.cn

# 查看日志
tail -50 cron_daily.log

# 手动重试
python update_data.py --force
```

### 无预测信号
这是正常的！策略是保守的，不是每天都有信号。

```bash
# 查看策略健康状态
python daily_tracker.py monitor

# 查看最近有信号的日期
grep "已保存" cron_daily.log | tail -10
```

### 策略被暂停
```bash
# 查看哪些策略被暂停
python daily_tracker.py monitor

# 如果需要重置某个策略
python -c "
from infra.strategy_pool import StrategyPool, StrategyMonitor
pool = StrategyPool()
pool.load_verified_strategies()
monitor = StrategyMonitor(pool)
monitor.reset_strategy('R187')  # 替换为需要重置的策略ID
"
```
