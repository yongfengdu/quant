# 量化自动研究系统 — 系统设计文档

> 版本: 2026-08-11 | 项目: `/root/quant-autoresearch`
> 目标: A股 T+1开盘买入 → T+2开盘卖出, 预测收益 >3% 的股票

---

## 1. 系统总览

系统由**两条独立 Pipeline** 组成, 共享一份数据缓存:

```
                    ┌─────────────────────────────────┐
                    │   数据层 (共享)                    │
                    │   ~/.cache/quant-autoresearch/    │
                    │   daily.parquet (31列, 550股)      │
                    └─────────────────────────────────┘
                          ▲                    ▲
                          │ 读                 │ 读
          ┌───────────────┴──────┐    ┌────────┴─────────────────┐
          │  Pipeline A          │    │  Pipeline B              │
          │  自迭代/进化 (离线夜间) │    │  线上预测 (盘后)          │
          │  产出: 新策略 .py      │    │  产出: 每日推荐股票        │
          └──────────┬───────────┘    └──────────────────────────┘
                     │ 写策略文件
                     ▼
          ┌────────────────────────┐
          │  策略层 (岛屿)           │
          │  islands/island_{bull,  │
          │   bear,range}/strategies│
          └────────────────────────┘
                     ▲ 读(路由)
                     │
          Pipeline B 通过 island_router 读取策略
```

**核心设计理念**:
- 人给方向(网格搜索发现有效因子) + 机器搜索参数(确定性回测) + LLM 提策略(K3 写代码)
- 所有策略必须通过**统一 Tiered Gate** 才能上线, LLM 不参与打分(打分完全确定性)

---

## 2. 数据层

### 2.1 主数据 `daily.parquet`
- 位置: `~/.cache/quant-autoresearch/daily.parquet`
- 规模: ~387K 行, 550 只股票, 2020-01-02 至今
- 31 列:
  - **行情**: `date, open, close, high, low, volume, amount, prev_close`
  - **标的**: `code, name, sector, circ_cap, total_cap`
  - **大盘指数**: `mkt_open/close/high/low/volume/amount, mkt_ret_20d`
  - **板块指数**: `sector_open/close/high/low/volume/amount, sector_ret_20d`
  - **衍生**: `turnover_rate, amplitude, limit_up_flag, limit_down_flag`

### 2.2 辅助数据
| 文件 | 用途 |
|------|------|
| `universe.parquet` | 可交易股票池(549只) — 所有 pipeline 的选股范围 |
| `predictions.parquet` / `trades.parquet` | backtest.py 的回测输出 |
| `daily_predictions.csv` | **线上预测输出(island_router 路径)** |
| `daily_predictions.json` | 线上预测输出(旧 strategy.py 路径, hermes 16:30) ⚠️ |

### 2.3 数据更新 `update_data.py`
- 增量拉取(只取比已有数据新的K线), 并发3线程
- 智能跳过(全股最新则采样5只确认)
- 更新后调 `compute_derived_columns()` 补齐衍生列 → 写 parquet
- 校验31列完整性

---

## 3. 策略层(岛屿架构)

### 3.1 你的问题:每个类别都有独立策略吗?
**是的。** 策略按市场状态分三个岛屿, 物理隔离:

```
islands/
├── island_bull/strategies/    牛市策略 (大盘20日涨>3%)
│   ├── BULL_001.py ~ BULL_005.py
│   ├── R006_HH001.py, R007_HH001.py
│   └── archive/               (已淘汰)
├── island_bear/strategies/    熊市策略 (大盘20日跌>3%)
│   ├── BEAR_001.py, BEAR_003.py, BEAR_004.py
│   └── archive/               (BEAR_002/005 已淘汰)
└── island_range/strategies/   震荡市策略 (±3%内)
    ├── RANGE_001.py, RANGE_002.py, RANGE_005.py
    └── archive/               (RANGE_003/004 已淘汰)
```

每个策略是独立 `.py` 文件, 实现统一接口:
```python
def predict_next_day(df: pd.DataFrame) -> pd.Series:
    # 输入: 全量历史(31列), 每股按时间排序
    # 输出: pd.Series(index=code, value=信号强度0.3~1.0)
```

### 3.2 路由: `island_router.py`
线上预测**不直接用单个策略**, 而是通过路由器按市场状态选策略池:

```python
detect_regime(df):
    mkt_ret_20d > +3%  → bull
    mkt_ret_20d < -3%  → bear
    else               → range
    # 内置脏数据防护: abs(mkt_ret)>5 时回退用中位数收益重算

ISLAND_STRATEGIES = {          # 只列通过 Gate 的策略
    "bull":  ["BULL_002"],
    "bear":  ["BEAR_001", "BEAR_003", "BEAR_004"],
    "range": ["RANGE_001", "RANGE_002", "RANGE_005"],
}
```

**信号聚合**: 当前市场对应岛屿的所有策略各自出信号 → 过滤 `>=0.3` → 按股票取**最高分**(并集) → 排序输出。

### 3.3 ⚠️ 已知问题: 三处"真相源"不一致
| 真相源 | 内容 | 状态 |
|--------|------|------|
| 磁盘 active `.py` | 实际存在的策略文件 | ✓ 权威 |
| `island_router.ISLAND_STRATEGIES` | **实际路由用的策略** | ✓ 权威(线上生效) |
| `state.json` | 岛屿状态记录 | ✗ **过时**(还列着已归档的R0xx策略) |

**风险**: `state.json` 与实际路由脱节, 会误导监控和进化。**建议**: 让 `state.json` 从 `ISLAND_STRATEGIES` + 磁盘扫描自动生成, 消除手工维护。

---

## 4. Pipeline A: 自迭代/进化

历史上有三代进化代码, 现状:

| 版本 | 架构 | 状态 | 问题 |
|------|------|------|------|
| `pipeline.py` | 4-agent (Analyst→Engineer→Trader→Scribe) | ⚠️ cron 02:00 在跑 | **翻译层断裂**, 504轮零采纳, 41%崩溃 |
| `pipeline_v2.py` | 岛屿+Pareto | ⚠️ cron 02:30 在跑 | **120秒超时**, 基本没工作 |
| **`evolve_v3.py`** | **单agent(K3)直接写代码** | ✅ 推荐, 未接cron | 新架构, 已验证跑顺 |

### 4.1 evolve_v3.py 流程(推荐架构)
```
1. Watchdog 健康检查 (K3服务探活, 异常则告警中止)
2. for 每个候选:
   a. K3 生成完整策略代码 (单agent, 无翻译层)
   b. 完整性校验 (AST语法 + 占位符检测 + 函数体≥5语句)  ← 堵住 stub 漏洞
   c. 确定性回测 (Top5, 最近N日)
   d. Tiered Gate 判定 (纯确定性, LLM不参与)
   e. 通过 → 写入 islands/island_X/strategies/X_NNN.py
      失败 → 反馈给下个候选的 prompt
3. 汇报 (通过策略 + K3健康 + 失败明细 → 微信)
```

### 4.2 为什么放弃旧 Kimi 进化(pipeline.py)
证据(见调研): 504轮实验, 中位数准确率8%, **41%(208轮)直接0%崩溃**。根因是 **Researcher→Engineer 的自然语言翻译层**: 研究员想出策略, 工程师翻译成代码时反复写坏(KeyError/占位符/Runtime崩溃)。`evolve_v3` 用单agent消除这一层。

### 4.3 K3 服务保障 `kimi_watchdog.py`
- **健康检查**: 调用前探活
- **异常分类**: 超时/认证/配额/网络/空响应/崩溃
- **智能重试**: 可恢复异常指数退避; 崩溃/认证不重试
- **告警**: 连续失败≥3次推送微信
- **状态记录**: `kimi_health.json`

---

## 5. Pipeline B: 线上预测

### 5.1 主路径 `daily_run.sh` (系统 cron 15:35)
```
1. update_data.py           更新数据
2. daily_tracker.py verify  验证昨日预测(T+2已出结果)
3. daily_tracker.py adaptive  ← 用 island_router 出预测 → daily_predictions.csv
   (失败则 fallback: daily_tracker.py predict)
4. 策略健康检查
5. daily_feedback.py        回测反馈
```

### 5.2 ⚠️ 你的问题: strategy.py 共用会冲突吗?

**会, 而且当前正在冲突。** 存在**两条并行的线上预测路径**, 用不同策略源, 写不同文件:

| 路径 | 调度 | 策略源 | 输出 |
|------|------|--------|------|
| `daily_run.sh` → `daily_tracker adaptive` | 系统cron 15:35 | **island_router**(已修好) | `daily_predictions.csv` |
| hermes → `quant_daily_impl.py` | hermes 16:30 | **strategy.py**(旧, HH600) | `daily_predictions.json` |

**冲突与风险**:
1. **两套预测并存, 结果不一致** — CSV用岛屿多策略, JSON用单个旧策略, 推给用户的可能是旧策略的结果
2. **strategy.py 是单点** — 被 `quant_daily_impl.py`, `backtest.py`, `pipeline.py`, `daily_quant_job.py` 等**7处共用**。旧 pipeline.py 进化时会 `git checkout` 改写它, 曾导致 HH587 stub 污染实盘(13行残缺代码跑了很久)
3. **回滚逻辑缺陷** — pipeline.py:1159 `git checkout HEAD~1 strategy.py`, 连续失败时会把工作区钉死在坏版本

### 5.3 strategy.py 共用的根本矛盾
`strategy.py` 承担了两个冲突的角色:
- **进化的画布**: pipeline.py 每轮覆写它做实验
- **线上/回测的策略**: 多个消费者 import 它

两个角色抢同一个文件 → 进化写坏就污染线上。**这是架构缺陷。**

---

## 6. 核心问题与建议方案

### 6.1 你的三个问题的回答

**Q1: 每个类别都有独立策略吗?**
✅ 是。三岛屿物理隔离, 每策略独立文件, 统一接口。路由器按市场状态选池。

**Q2: strategy.py 共用有冲突和风险吗?**
🔴 有。当前 strategy.py 被进化和线上/回测共用, 是单点污染源。且存在两条线上路径(CSV/JSON)用不同策略源, 结果不一致。

**Q3: 怎么保证自迭代和线上预测合理运行?**
需要**解耦 + 单一真相源**, 见下方方案。

### 6.2 建议架构(解决冲突)

```
进化层(写)          策略层(唯一真相源)         线上层(读)
evolve_v3.py  ──写──▶ islands/*/strategies/ ──读──▶ island_router
                            │                          ▲
                     ISLAND_STRATEGIES ────────────────┘
                     (Gate通过清单)

废弃: strategy.py 作为线上策略 (仅留作历史/单策略回测参考)
```

**具体措施**:
1. **统一线上路径**: 让 hermes 16:30 的 `quant_daily_impl.py` 也改用 `island_router`(和 daily_tracker 一致), 或直接停用 16:30、只保留 15:35。消除双路径。
2. **strategy.py 退役**: 进化不再写 strategy.py(evolve_v3 只写岛屿文件)。停用 pipeline.py/pipeline_v2.py 的 cron(它们会 git checkout 污染)。
3. **单一真相源**: `ISLAND_STRATEGIES` 作为唯一"上线清单"。加一个脚本从它自动同步 `state.json`。
4. **进化↔线上隔离**: 进化产出的新策略先进 `strategies/`(候选), 但**不自动进 ISLAND_STRATEGIES**——需人工/独立验证确认后才加入路由清单。这样进化即使出问题也污染不到线上。
5. **Watchdog 全覆盖**: 进化用 kimi_watchdog; 线上预测也应有健康检查(数据是否更新、信号是否为空)。

### 6.3 两 Pipeline 的协调时序
```
夜间 02:00  Pipeline A (evolve_v3) 进化 → 写候选策略到 islands/
           ↓ (人工/自动验证)
           通过验证的策略手动加入 ISLAND_STRATEGIES
盘后 15:35  Pipeline B (daily_run) 读 island_router → 出预测
           ↑ 两者通过"策略文件+路由清单"解耦, 不直接互相调用
```

**关键**: 进化(写)和线上(读)通过**策略文件 + 路由清单**间接通信, 不共享可变的 strategy.py, 不并发写同一文件。

---

## 7. 组件依赖全图

```
数据: update_data.py ──▶ daily.parquet ◀── (所有组件读)

进化(A):
  evolve_v3.py ──▶ kimi_watchdog.py ──▶ K3 CLI
             ──▶ notifier.py ──▶ 微信
             ──▶ islands/*/strategies/*.py (写)
  [废弃] pipeline.py / pipeline_v2.py

线上(B):
  daily_run.sh ──▶ daily_tracker.py adaptive
                ──▶ island_router.py ──▶ islands/*/strategies/ (读)
                ──▶ daily_predictions.csv (写)
  [待统一] quant_daily_impl.py ──▶ strategy.py (旧)

回测/验证:
  batch_backtest.py ──▶ islands/*/strategies/ (Gate评估)
  backtest.py ──▶ strategy.py (单策略回测)

监控/汇报:
  kimi_watchdog.py, notifier.py, weekly_report.py, health_monitor.py
```

---

## 8. Cron 调度现状

| 调度器 | 时间 | 任务 | 状态 | 建议 |
|--------|------|------|------|------|
| 系统cron | 15:35 工作日 | daily_run.sh(线上预测,island_router) | ✅ 正常 | 保留 |
| hermes | 16:30 工作日 | quant_daily_impl.py(线上预测,strategy.py) | ⚠️ 双路径 | 改用router或停用 |
| hermes | 02:00 工作日 | pipeline.py(旧进化) | ⚠️ 零采纳 | **停用** |
| hermes | 02:30 工作日 | pipeline_v2.py(岛屿进化) | 🔴 超时 | **停用** |
| hermes | 08:00 每日 | health_monitor.py | ✅ 正常 | 保留 |
| (未接) | — | evolve_v3.py(新进化) | ✅ 就绪 | **接入夜间** |

---

## 9. Tiered Gate(统一门禁)

所有策略上线前必须通过, 定义一致(evolve_v3 / batch_backtest / pipeline 共用):
```python
def check_gate(trades, acc, avg):
    if avg <= 0:                    return FAIL   # 必须正期望
    if acc >= 30 and trades >= 7:   return PASS(30%)
    if acc >= 25 and trades >= 13:  return PASS(25%)
    if acc >= 20 and trades >= 26:  return PASS(20%)
    if acc >= 15 and trades >= 101: return PASS(15%)
    return FAIL
```
基于统计显著性设计: 准确率越高所需交易样本越少。基准: 随机选股准确率 ~10%。

---

## 10. 当前策略清单(上线中)

| 岛屿 | 策略 | 逻辑 | 回测表现 |
|------|------|------|----------|
| bear | BEAR_003 | 跌停板打开修复 | 32.4% / +0.16% ✓ |
| bear | BEAR_004 | 超跌龙头 | 23.2% / +0.33% ✓ |
| bear | BEAR_001 | 超跌反弹 | 15.5% / +0.02% ✓ |
| bull | BULL_002 | 趋势回踩 | 20.7% / +0.23% ✓ |
| range | RANGE_001 | 温和放量突破 | 17.9% / +0.26% ✓ |
| range | RANGE_002 | 大涨不炸板 | 18.0% / +0.25% ✓ |
| range | RANGE_005 | 放量中位突破 | 15.6% / +0.08% ✓ |

---

## 附录: 关键文件索引

| 文件 | 职责 |
|------|------|
| `island_router.py` | 市场状态路由 + 信号聚合(线上核心) |
| `evolve_v3.py` | 单agent进化(推荐) |
| `kimi_watchdog.py` | K3服务监控 |
| `notifier.py` | 微信告警/汇报 |
| `daily_tracker.py` | 线上预测/验证/汇报主控 |
| `daily_run.sh` | 线上预测调度脚本 |
| `update_data.py` | 数据增量更新 |
| `batch_backtest.py` | 岛屿策略批量Gate评估 |
| `strategy.py` | ⚠️ 旧单策略(建议退役) |
| `pipeline.py` / `pipeline_v2.py` | ⚠️ 旧进化(建议停用) |
