# 技术设计文档（量化自进化信号系统）

> 配套 `MRD.md`。MRD 回答"做什么、为什么"，本文档回答"怎么做、正常/异常/性能"。
> 版本: v1 (2026-09-10)

---

## 一、系统架构总览

```
                 ┌──────────── supervisor.py (编排器) ────────────┐
                 │ init/daily/evolve/monitor/status/promote/retire│
                 └──┬────────────┬──────────────┬────────────────┘
                    │            │              │
        ┌───────────▼─────┐ ┌────▼──────────┐ ┌─▼─────────────┐
        │ 系统A 信号生成    │ │ 系统B 线上预测 │ │ 数据层          │
        │ signal_evolve   │ │ live_strategy │ │ data_fetch     │
        │  (LLM+五闸门)    │ │ (读active信号) │ │ build_data     │
        │                 │ │ portfolio     │ │ data_store     │
        └───────┬─────────┘ └──────┬────────┘ └───────┬───────┘
                │                  │                  │
                └────────┬─────────┘                  │
                    ┌────▼───────────────────────────▼──┐
                    │ signal_registry (状态机/真相源)      │
                    │ factors + signal_lab (因子/验证)     │
                    └───────────────────────────────────┘
                    ┌───────────────────────────────────┐
                    │ monitor (监控) + run_daily/run_evolve│
                    └───────────────────────────────────┘
```

**分层**：数据层 → 因子/验证层 → 信号生命周期 → 信号生成(研究) / 线上预测(生产) → 编排/监控。

**核心原则**：
1. 进化（A）与线上（B）解耦，注册表是唯一真相源
2. LLM 只提假设不打分，五闸门确定性验收
3. 失败安全默认：数据脏了宁可不出信号

---

## 二、数据层

### 2.1 data_fetch.py（拉取 + 复权）

**功能**：拉取不复权 OHLCV + 送转因子，point-in-time 复权。

**接口**：`fetch_raw_with_factor(code, start, end, split_cache)` → DataFrame[date,open,close,high,low,volume,amount,split_factor]

**正常流程**：
1. 腾讯 fqkline 拉不复权（分段 2 年/段，规避 640 行上限）
2. AkShare 拉分红送配 → 构建送转因子（阶梯函数）
3. 复权 = raw × split_factor

**异常情况**：
| 场景 | 处理 |
|------|------|
| 代理断连 | `http_get` 重试 5 次，指数退避 |
| 腾讯失败 | 回退 AkShare（双源兜底） |
| 分红接口失败 | 重试 5 次，仍失败返回空（无送转） |
| 数据为空 | 返回 None，上层跳过 |

**性能**：
- 分红事件缓存到 `split_events.parquet`，避免每日重复拉取
- 并发拉取（build_data 5 workers）

### 2.2 build_data.py（构建 + 增量）

**功能**：全量构建 + 每日增量更新。

**接口**：`pull(codes, ...)` / `update_incremental(out_path, lookback_days=10, workers=5)`

**正常流程（增量）**：
1. 读现有数据，找最新日期
2. 回看 10 天，并发拉取，合并去重

**异常情况**：
- 无现有数据 → 提示先跑全量
- 单股失败 → 跳过，失败计数（不中断整体）
- 缓存未命中 → 拉取并回填

**性能**：并发 5 workers，549 只约 9.5 分钟（原串行 50-70 分钟）。

### 2.3 data_store.py（日历 + 校验 + 审计）

**功能**：交易日历（节假日感知）、完整性校验、审计日志。

**接口**：`next_trading_day(date, cal, n)` / `validate_bars(df)` / `audit_log(event, **kwargs)`

**正常流程**：日历从 AkShare 官方日历构建（8797 交易日），校验通过。

**异常情况（validate_bars 检测）**：
- 重复 (code,date)、OHLC 不合理、非正价格、split_factor<1、缺失交易日、非交易日数据

**性能**：parquet 读 + 向量化校验，秒级。

---

## 三、因子与验证层

### 3.1 factors.py（因子库）

**功能**：~20 个原语因子（动量/反转/量能/波动/价格位置/振幅），无前视。

**接口**：`compute_factor(df, name)` / `compute_all_factors(df, names)` / `add_forward_returns(df, horizons)`

**正常流程**：groupby(code) 滚动计算，前向收益用 split_factor 复权。

**异常情况**：NaN 处理（`fillna`、`min_periods`），避免崩溃。

**性能**：向量化 pandas，549 只 × 8.5 年秒级。

### 3.2 signal_lab.py（IC 引擎 + 五闸门）

**功能**：横截面秩相关 IC + 五道验证闸门。

**接口**：`daily_rank_ic(df, factor_col, fwd_col)` / `validate_signal(df, factor_name, horizon)`

**五闸门**：

| 闸门 | 判据 | 拦截的坑 |
|------|------|---------|
| 1 正确性 | 有效值占比、区分度 | 脏数据/常数因子 |
| 2 IC 显著性 | \|t\|>3.8 | 假通过（多重比较） |
| 3 稳定性 | 符号一致性>60% + 近期\|t\|>2 | edge 不稳定、衰减 |
| 4 Walk-forward | 验证窗口 \|t\|>2.5 | 过拟合 |
| 5 Regime | 全 regime \|t\|>2 | 单一 regime 幻觉 |

**正常流程**：强信号（如 mom_20）全 5 闸门通过。

**异常情况**：噪声/弱信号被闸门 2/3/4 拦截（元验证精度 100%）。

**性能**：横截面秩相关 O(天数 × 股票数)，秒级。

---

## 四、信号生命周期（signal_registry.py）

**功能**：状态机 + 衰减监控 + 退役。

**状态机**：`proposed → validated → paper → active → decayed → retired`

**正常流程**：信号过五闸门 → paper → 纸面期滚 IC 稳定 → active。

**异常情况**：
- 衰减：近期(60日)滚 IC \|t\|<1.5 连续 3 次 → decayed → retired（防抖，不误杀）
- 未过闸门 → 直接 retired

**性能**：JSON 持久化，IC 历史 list 存储。

**注意**：信号退役只改状态，不删代码（可回滚）。

---

## 五、信号生成（signal_evolve.py）

**功能**：LLM 提因子假设 + 确定性验证 + 注册。

**接口**：`discover_signals(df, n, horizon, registry, test)` 

**正常流程**：
1. 构建 prompt（含已有因子、市场规律、反馈）
2. K3 提假设 → 提取代码 → 完整性校验 → 计算因子 → 五闸门 → 注册

**异常情况**：
| 场景 | 处理 |
|------|------|
| K3 失败/超时/配额 | watchdog 重试，可恢复异常自动重试 |
| 未提取到代码块 | 跳过，记录 |
| 完整性校验失败 | 跳过，反馈给下一轮 prompt |
| 因子运行崩溃 | 跳过，记录错误 |

**性能**：K3 单次 1-8 分钟，3 假设约 10-30 分钟。因子计算 + 验证秒级。

**关键**：LLM 只提假设不打分，代码持久化到注册表 definition（否则信号无法复现）。

---

## 六、线上预测（live_strategy.py + portfolio.py）

### 6.1 live_strategy.py

**功能**：读注册表 active 信号，按 regime 合成分数，生成推荐。

**接口**：`detect_regime` / `load_active_signals` / `run_backtest` / `generate_recommendation`

**正常流程**：
1. 读 active 信号（空则回退默认核心信号）
2. 按 regime 过滤（bull → 动量；bear/range → 反转；all 信号两 regime 都参与）
3. 各信号 direction×因子 → z-score 等权求和 → top-N 推荐

**异常情况**：
- 注册表空 → 回退默认核心信号（dist_hi20/dist_ma60）
- 某信号因子计算失败 → 跳过该信号
- 信号无有效值 → 跳过

**性能**：因子向量化计算，推荐秒级。

### 6.2 portfolio.py

**功能**：分档回测 + 指标（年化/Sharpe/回撤/胜率）。

**接口**：`backtest_portfolio(df, signal_col, ...)` / `metrics(result)`

---

## 七、编排与监控

### 7.1 supervisor.py（编排器/上帝服务）

**功能**：统一入口，7 个子命令。

**正常流程**：各子命令独立执行，互不干扰。

**异常情况**：子命令失败不影响其他（daily 的某步失败，其余步继续）。

**性能**：纯调度，复用各模块，无额外开销。

### 7.2 run_daily.py（每日流水线）

**功能**：数据更新 → 生命周期审视 → 生成推荐 + 持久化推荐。

**正常流程**：三步顺序执行，推荐存 `recommendations.csv`。

**异常情况**：`--no-update` 跳过数据更新；各步独立 try/异常。

### 7.3 run_evolve.py（进化）

**功能**：LLM 找信号入口。

### 7.4 monitor.py（监控）

**功能**：5 大类健康检测 + 回测 vs 实盘偏差 + 效果日志。

**接口**：`SystemMonitor.run_all()` / `report()`

**正常流程**：全绿，退出码 0。

**异常情况**：分级 CRITICAL/WARNING，退出码 2/1，`--alert` 推微信。

**性能**：K3 探活 ~30s（主要耗时），其余秒级。

---

## 八、异常与容错总表

| 层 | 故障 | 容错 |
|----|------|------|
| 数据 | 代理断连 | 重试 + 双源兜底 |
| 数据 | 数据脏/丢失 | validate_bars + 审计日志 |
| 因子 | NaN | fillna + min_periods |
| 验证 | 假信号 | 五闸门（多重比较控制） |
| 生命周期 | 信号衰减 | 滚 IC 监控 + 防抖退役 |
| 生成 | K3 失败 | watchdog 重试 + 分类异常 |
| 预测 | 注册表空 | 回退默认信号 |
| 编排 | 子任务失败 | 各子命令独立 |
| 监控 | 检测失败 | try/except 降级 |

---

## 九、性能要点

| 操作 | 耗时 | 瓶颈 |
|------|------|------|
| 全量数据拉取 | ~2.7h（首跑） | 网络（549 只 × 分段） |
| 增量更新 | ~9.5min | 网络（并发 5 workers） |
| 因子计算 | 秒级 | 向量化 pandas |
| 信号验证 | 秒级 | 横截面秩相关 |
| LLM 进化 | 10-30min/轮 | K3 API |
| 监控 | ~30s | K3 探活 |
| 推荐 | 秒级 | 无 |

**优化记录**：增量更新从串行 50-70min 优化到并发 9.5min；分红事件缓存避免重复拉取。

---

## 十、部署与调度

```
30 8  * * *    monitor.py --alert          # 每日健康检测
30 19 * * 1-5  run_daily.py                # 每日流水线(更新→推荐)
0  2  * * 6    evolve_signal_cron.sh       # 每周 LLM 进化
```

**数据文件**：
- `~/.cache/quant-autoresearch/daily_bars.parquet`（主数据）
- `split_events.parquet`（分红缓存）
- `trading_calendar.parquet`（交易日历）
- `data/signal_registry.json`（信号注册表）
- `data/recommendations.csv`（推荐记录）
- `data/prediction_effectiveness.csv`（预测效果日志）

---

## 十一、测试保障

27 个 pytest 测试覆盖：因子正确性、无前视、日历节假日、完整性校验、IC 引擎、五闸门（黄金 + 负例）、生命周期流转、衰减退役、元验证（合成注入）、live_strategy（黄金回测 Sharpe 0.84）。
