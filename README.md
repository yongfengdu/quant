# Quant Autoresearch — A股量化策略自动研究系统

一个针对 A 股的**自动化量化策略研究与在线预测系统**。核心是"人给方向 + LLM 写策略 + 确定性验证"的进化闭环, 配合按市场状态路由的多策略在线预测。

> **交易逻辑**: T日收盘出信号 → T+1开盘买入 → T+2开盘卖出, 目标预测 (T+2开盘 − T+1开盘)/T+1开盘 **> 3%** 的股票。

---

## 核心设计理念

传统"LLM 自然语言描述策略 → 工程师翻译成代码"的双 agent 管线存在**翻译层断裂**问题(504轮实验中 41% 直接崩溃, 零采纳)。本系统改为:

1. **单 Agent 直接写代码** — K3 同时负责想策略和写完整可运行代码, 消除翻译层
2. **确定性验证** — LLM 不参与打分, 所有策略必须通过统一 **Tiered Gate**(基于统计显著性)
3. **进化与线上解耦** — 进化产出的策略先进"候选", 人工验证后才上线, 进化出问题也污染不到线上
4. **服务监控** — Watchdog 监控 LLM 服务异常并告警, 数据脏了有兜底

---

## 系统架构

```
                  数据层 (共享)
          ~/.cache/quant-autoresearch/daily.parquet
                  ▲                    ▲
     ┌────────────┴───────┐  ┌─────────┴──────────┐
     │ Pipeline A: 进化    │  │ Pipeline B: 在线预测 │
     │ (夜间 02:00)        │  │ (盘后 15:35)        │
     │ evolve_v3.py       │  │ daily_run.sh       │
     └─────────┬──────────┘  └────────────────────┘
               │ 写候选策略               ▲ 读(路由)
               ▼                         │
        islands/*/strategies/  ◀── island_router (按市场状态路由)
               │                         │
        ISLAND_STRATEGIES (Gate通过的上线清单 = 唯一真相源)
```

### Pipeline A — 自动进化 (`evolve_v3.py`)
K3 单 agent 生成完整策略代码 → 完整性校验(堵 stub 漏洞) → 确定性回测 → Tiered Gate → 通过则写入岛屿候选。带 `kimi_watchdog` 服务监控 + `notifier` 微信汇报。

### Pipeline B — 在线预测 (`daily_run.sh`)
数据更新 → 健康检查(数据新鲜度/完整度/市场状态) → `island_router` 按市场状态(牛/熊/震荡)路由到对应岛屿的已验证策略池 → 多策略信号聚合 → 输出推荐。

---

## 策略岛屿

策略按市场状态分三个岛屿, 物理隔离, 统一接口 `predict_next_day(df) -> pd.Series`:

| 岛屿 | 触发条件 | 已上线策略 |
|------|----------|-----------|
| **bull** (牛市) | 大盘20日涨 >3% | BULL_002 (趋势回踩) |
| **bear** (熊市) | 大盘20日跌 >3% | BEAR_001/003/004 (超跌反弹/跌停修复/超跌龙头) |
| **range** (震荡) | ±3% 内 | RANGE_001/002/005 (放量突破/大涨不炸板/中位突破) |

**市场状态判定** (`island_router.detect_regime`): 基于 `mkt_ret_20d`, 内置脏数据过滤 + 不完整交易日自动剔除。

---

## Tiered Gate (统一门禁)

基于统计显著性设计, 准确率越高所需交易样本越少。基准: 随机选股准确率 ~10%。

```python
def check_gate(trades, acc, avg):
    if avg <= 0:                    return FAIL   # 必须正期望
    if acc >= 30 and trades >= 7:   return PASS(30%)
    if acc >= 25 and trades >= 13:  return PASS(25%)
    if acc >= 20 and trades >= 26:  return PASS(20%)
    if acc >= 15 and trades >= 101: return PASS(15%)
    return FAIL
```

---

## 关键组件

| 文件 | 职责 |
|------|------|
| `island_router.py` | 市场状态路由 + 信号聚合(在线核心) |
| `evolve_v3.py` | K3 单agent进化(推荐) |
| `kimi_watchdog.py` | K3 服务监控(健康检查/异常分类/重试/告警) |
| `notifier.py` | 微信告警/汇报 |
| `predict_health.py` | 在线预测健康检查 |
| `promote_strategy.py` | 候选→上线晋升(带 Gate 双重验证) |
| `sync_island_state.py` | 岛屿状态同步(消除真相源不一致) |
| `daily_tracker.py` | 在线预测/验证/汇报主控 |
| `daily_run.sh` | 在线预测调度脚本 |
| `update_data.py` | 数据增量更新 |
| `batch_backtest.py` | 岛屿策略批量 Gate 评估 |

详见 [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md)。

---

## 使用

```bash
# 数据更新
python3 update_data.py

# 在线预测 (按市场状态路由)
python3 daily_tracker.py adaptive

# 预测健康检查
python3 predict_health.py

# 运行一轮进化 (生成候选策略)
python3 evolve_v3.py --island range --n 3 --notify

# 批量回测评估岛屿策略
python3 batch_backtest.py bull bear range

# 列出候选策略
python3 promote_strategy.py --list

# 晋升候选策略上线 (先跑 Gate 验证)
python3 promote_strategy.py --island bear --name BEAR_006

# K3 服务健康检查
python3 kimi_watchdog.py
```

---

## 调度 (cron)

| 时间 | 任务 | 说明 |
|------|------|------|
| 工作日 02:00 | `evolve_v3_cron.sh` | 夜间进化(系统cron, 无超时限制) |
| 工作日 15:35 | `daily_run.sh` | 在线预测(数据更新→健康检查→预测) |
| 每日 08:00 | `health_monitor.py` | 系统健康监控 |

---

## 数据

- 主数据 `daily.parquet`: ~550只A股, 31列(OHLCV + 大盘/板块指数 + 衍生因子)
- 数据文件不入库(体积大), 通过 `update_data.py` 增量拉取

---

## 免责声明

本项目仅用于量化研究与技术学习, **不构成任何投资建议**。据此交易风险自负。
