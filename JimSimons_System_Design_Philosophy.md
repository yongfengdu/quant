# 量化交易系统设计：从 Medallion 模型到通用架构

> 本文档从 Renaissance Technologies Medallion 基金的技术设计哲学出发，结合现代量化交易系统架构最佳实践，提炼出可落地的系统设计框架。去除人物传记与管理内容，纯粹聚焦技术架构与设计决策。

---

## 一、Medallion 系统的核心技术特征

### 1.1 关键性能指标（硬数据）

| 指标 | 数值 | 说明 |
|------|------|------|
| 年化总回报（1988-2022） | ~68% gross | 34年复合 |
| 年化净回报（扣费后） | ~40% | 5%管理费 + 44%绩效费 |
| Sharpe Ratio | 6.3~7.5 | 普通量化基金最好约3.0 |
| 基金规模 | $10B~$15B | 主动控制在策略容量上限内 |
| 平均持仓时间 | 1~1.5天 | 极短周期 |
| 日交易量 | 数百万笔 | 赌场式高频覆盖 |
| 持仓数量 | 4300+股票 | 13F文件显示极度分散 |
| 最大单仓位 | 1%~2% | Novo Nordisk等，极度分散 |
| 2008年回报 | 152% gross | 金融危机期间，波动率是朋友 |
| 2020年回报 | 149% gross / 76% net | 疫情波动期间持续有效 |

### 1.2 系统的七个技术支柱

**① 数据护城河：清洗后的高频历史数据**
- 从1980年代开始系统清洗 tick 级数据，远早于竞争对手
- 数据质量优先于数据数量：宁可少覆盖，不可有脏数据
- Point-in-time 正确性：确保回测只用决策时刻可获得的信息

**② 统一模型架构（One Model）**
- 全公司只有一个模型、一套策略、一个代码库
- 所有研究员看到全部代码，所有人改进同一个系统
- 任何人的改进直接提升所有人的绩效——消除了内部竞争的激励错位
- 对比：Citadel/D.E. Shaw/Two Sigma 用多策略多团队竞争模型

**③ 赌场式高频统计套利**
- 单笔交易期望值极小（类似赌场0.5%~1% edge）
- 但每天交易数百万笔，大数定律让方差趋近于零
- 核心不是"预测准"，而是"大量独立下注的统计优势"
- 需要极低的交易成本（执行层优势）才能让微小的 edge 不被磨掉

**④ 市场中性策略**
- 每笔交易同时做多做空，对冲掉系统性风险（市场 beta）
- 只赚取相对价值的微小偏离
- 2008年市场崩盘时不但没亏反而赚152%——因为不暴露方向性风险
- Sharpe 6.3+ 的关键：不是收益高，而是波动极低

**⑤ 跨领域方法论迁移**
- 从数学/密码学/语言学迁移到金融：HMM（隐马尔可夫模型）、模式识别、信号处理
- 不使用传统金融方法（基本面分析、技术指标），而是把市场看作信息流
- 核心洞察：市场价格序列中存在人类行为模式的结构性规律

**⑥ 全自动化交易**
- 从信号生成到下单执行全自动化，人工不干预
- 人类只做研究和系统维护，不参与实时交易决策
- 这是高 Sharpe 的前提：消除人类情绪对执行的干扰

**⑦ 策略容量自我约束**
- 2003年主动踢出外部投资者，把基金规模控制在 $5B
- 后来 Peter Brown 逐步提到 $10B~$15B——只在确认有足够 profitable trades 时才扩容
- 关键设计原则：策略有容量上限，超容量的资金会因 slippage 吃掉收益
- 机构基金（RIEF）规模 $60B+ 但 Sharpe 远低于 Medallion，验证了容量约束的重要性

---

## 二、量化交易系统通用架构

### 2.1 系统管线全景

```
┌─────────────────────────────────────────────────────────┐
│                    数据基础设施层                          │
│  市场数据流 → 数据验证 → 历史数据库 → 特征工程             │
│  替代数据（卫星/信用卡/NLP情感）→ ETL → Point-in-Time DB  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    策略引擎层                              │
│  信号计算 → 模型推理 → 目标仓位生成                       │
│  (事件驱动 / 批处理两种模式)                               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              风险管理系统（连续监控层）                      │
│  Pre-trade 检查 → 实时组合风险 → VaR/ES → 熔断器          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              执行管理系统 (EMS)                            │
│  订单路由 → FIX协议 → 智能路由(SOR) → 执行算法(VWAP/TWAP) │
│  → 暗池接入 → 成交追踪                                   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              组合管理系统 (PMS)                            │
│  仓位追踪 → P&L计算 → 回撤监控 → 状态持久化              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              监控与日志层                                  │
│  全链路日志 → 合规审计 → 事后分析 → 异常告警              │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据基础设施层

**数据质量是第一性原则：** 策略在回测中看起来盈利但在实盘中失败，最常见原因是生产数据与研究数据有细微差异。

#### 市场数据接入

| 数据源类型 | 延迟 | 成本 | 适用场景 |
|-----------|------|------|---------|
| 交易所直连 | 微秒级 | 极高（需co-location） | HFT做市、微秒级套利 |
| SIP聚合流 | 毫秒级 | 中 | 日内策略（分钟级持仓） |
| 第三方API | 十毫秒级 | 低 | 日级策略、研究 |

#### 数据验证（生产必备）

```python
# 核心验证逻辑
def _validate_tick(tick):
    # 1. 交叉市场检测（bid > ask = 坏数据）
    if tick.bid_price >= tick.ask_price:
        return False
    # 2. 零/负价格检测
    if tick.bid_price <= 0 or tick.ask_price <= 0:
        return False
    # 3. 异常价差检测（>10% = 可疑）
    if tick.spread_bps > 1000:
        return False
    return True

# 数据陈旧检测
def is_data_stale(symbol, current_time, threshold_seconds=5.0):
    age = (current_time - last_update[symbol]).total_seconds()
    return age > threshold_seconds
```

**关键原则：** 数据源断线时，系统不能假设最后价格仍然有效。应暂停交易而非用陈旧数据做决策。

#### Point-in-Time 正确性

这是量化研究中最常见的陷阱：

```python
@dataclass
class AlternativeDataPoint:
    symbol: str
    observation_date: datetime    # 数据发生时间
    available_date: datetime      # 数据可用时间
    value: float

    @property
    def publication_lag(self):
        return self.available_date - self.observation_date

class PointInTimeDatabase:
    def query(self, symbol, data_type, as_of):
        # 只返回 available_date <= as_of 的数据
        # 防止 look-ahead bias
        return [d for d in self.data
                if d.symbol == symbol
                and d.available_date <= as_of]
```

**实际场景：** 信用卡交易数据周一发生 → 周三被供应商聚合 → 周四才能获取。如果回测中用"周一的数据"在周一做交易决策，就引入了严重的 look-ahead bias。

#### 历史数据存储

| 数据库 | 特点 | 适用场景 |
|--------|------|---------|
| kdb+ | 量化金融标配，tick级数据极快 | 专业量化 |
| InfluxDB | 开源TSDB，易上手 | 中小团队 |
| TimescaleDB | PostgreSQL扩展 | 已有PG基础设施 |

**存储设计：** 按 资产类别→标的→日期 分区。查 AAPL 2024年1月数据时可直接跳过其他分区。

---

### 2.3 策略引擎层

#### 事件驱动 vs 批处理

| 模式 | 响应方式 | 延迟要求 | 适用场景 |
|------|---------|---------|---------|
| 事件驱动 | 每个tick触发计算 | 微秒~毫秒 | HFT、做市 |
| 批处理 | 固定间隔计算 | 秒~分钟 | 日级策略、因子模型 |

**核心设计原则：** 大部分代码不是性能关键路径。从收到数据到发单的路径只占代码量的一小部分，但决定了策略可行性。优化关键路径，其余用高效开发语言（Python）即可。

#### 策略状态管理

生产策略必须维护的状态：

| 状态类型 | 说明 | 失败后果 |
|---------|------|---------|
| 当前仓位 | 必须与券商记录一致 | 交易失败或意外成交 |
| 未完成订单 | 提交但未成交的订单 | 重启后丢失导致重复下单 |
| 策略内部状态 | 模型参数、累积信号 | 指标值丢失导致错误信号 |
| 风险状态 | 当前敞口、回撤水平 | 风控失效 |

**状态持久化：** 系统在下午2点崩溃后重启，必须能恢复所有仓位和未完成订单。定期 checkpoint 提供审计追踪。

#### 延迟预算分析

| 策略类型 | 目标延迟 | 语言 | 基础设施 |
|---------|---------|------|---------|
| HFT做市 | 10 μs | C++/FPGA | Co-located服务器+直连 |
| 统计套利 | 1 ms | C++/Java | 低延迟数据中心 |
| 日内动量 | 100 ms | Python/Java | 云或标准数据中心 |
| 日级再平衡 | 60 s | Python | 任意 |

**关键洞察：** 日级策略投资 co-location 和 C++ 毫无意义——执行质量提升可忽略。反之，HFT没有这些投资无法运作。

---

### 2.4 风险管理系统

风险是多层防御，每层防御不同类型的失败：

#### 第一层：Pre-Trade 检查（订单提交前）

```python
class RiskLimits:
    max_position_size: float = 100000    # 单仓位最大名义值
    max_order_size: float = 50000        # 单笔订单最大名义值
    max_portfolio_exposure: float = 500000  # 总多空名义值
    max_concentration: float = 0.2       # 单标的最大占比
    max_daily_loss: float = 10000        # 日亏损上限（触发暂停）
    max_drawdown: float = 0.05           # 5%回撤限制

class PreTradeRiskChecker:
    def check_order(self, order, current_position, portfolio_value,
                    current_price, total_exposure, daily_pnl):
        # 1. 单笔订单大小检查
        order_value = order.quantity * current_price
        if order_value > self.limits.max_order_size:
            return (False, f"订单 {order_value} 超限 {self.limits.max_order_size}")

        # 2. 仓位限制检查
        resulting_position = current_position + order.quantity * (1 if order.side == 'BUY' else -1)
        if abs(resulting_position * current_price) > self.limits.max_position_size:
            return (False, f"仓位 {resulting_position * current_price} 超限")

        # 3. 总敞口检查
        if total_exposure + order_value > self.limits.max_portfolio_exposure:
            return (False, "总敞口超限")

        # 4. 日亏损熔断
        if daily_pnl < -self.limits.max_daily_loss:
            return (False, "日亏损超限，交易暂停")

        return (True, "通过")
```

#### 第二层：实时组合风险监控

```python
class PortfolioRiskMonitor:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.returns_history = []
        self.peak_value = 0
        self.current_drawdown = 0

    def update(self, portfolio_value, portfolio_return):
        self.returns_history.append(portfolio_return)
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value

    def calculate_var(self, horizon_days=1):
        """参数化 VaR（假设正态分布）"""
        mu = np.mean(self.returns_history) * horizon_days
        sigma = np.std(self.returns_history) * np.sqrt(horizon_days)
        var = -(mu + sigma * stats.norm.ppf(1 - self.confidence_level))
        return var

    def calculate_expected_shortfall(self, horizon_days=1):
        """ES = 尾部期望损失，比VaR更保守"""
        var = self.calculate_var(horizon_days)
        mu = np.mean(self.returns_history) * horizon_days
        sigma = np.std(self.returns_history) * np.sqrt(horizon_days)
        es = mu - sigma * stats.norm.pdf(stats.norm.ppf(1-self.confidence_level)) / (1-self.confidence_level)
        return -es
```

**VaR vs ES：** 单个仓位可能通过所有独立检查，但组合层面可能暴露于单一风险因子。组合级监控捕获这种聚合风险。

#### 第三层：熔断器（Circuit Breaker）

| 触发条件 | 动作 |
|---------|------|
| 日亏损 > 阈值 | 暂停交易，人工审查 |
| 回撤 > 阈值 | 平掉部分仓位 |
| 数据源断线 | 进入安全模式（停止新开仓） |
| 系统异常 | Kill switch：全部平仓 |

**设计原则（不对称性）：** 错过一个交易机会的成本远小于犯一个灾难性错误的成本。系统应偏向保守——不确定时选择不交易。

---

### 2.5 执行管理系统 (EMS)

#### 执行算法

| 算法 | 逻辑 | 适用场景 |
|------|------|---------|
| TWAP | 等时间间隔均分订单 | 无量价信息时 |
| VWAP | 按预期成交量比例分配 | 有历史成交量模式时 |
| Implementation Shortfall | 最小化市场冲击+时机成本 | 需平衡执行速度与冲击 |
| 自适应 | 根据实时市场条件动态调整 | 复杂策略 |

**Implementation Shortfall 分解：**

```
IS = (P̄ - P₀) × X  总执行短差
   = (P̄ - P̃) × X  市场冲击成本
   + (P̃ - P₀) × X  时机成本

P̄ = 实际平均执行价
P₀ = 决策时价格
P̃ = 基准价（如VWAP）
X = 总股数
```

#### 智能订单路由 (SOR)

```python
@dataclass
class VenueQuote:
    venue_id: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    latency_ms: float
    fee_per_share: float  # 负值=返佣

class SmartOrderRouter:
    def route_order(self, side, quantity, strategy="best_price"):
        if strategy == "best_price":
            # 按最优价格优先填充
            # NASDAQ ask=185.49 先吃，不够再吃 BATS ask=185.50
            return self._route_best_price(side, quantity)
        elif strategy == "minimize_impact":
            # 按流动性比例分散到多个场所
            # 减少信息泄露（不让一个大单扫光一个场所的流动性）
            return self._route_minimize_impact(side, quantity)
```

**路由策略权衡：**
- **Best Price**：优化即时执行成本，但可能暴露意图
- **Minimize Impact**：分散到多场所，减少信息泄露，但执行成本可能略高
- **关键考量**：返佣（maker-taker fee）——NASDAQ可能因 rebate 而实际成本低于表面价格

#### FIX 协议

```
# 新订单消息示例
8=FIX.4.4|35=D|49=QUANT_FIRM|56=EXCHANGE|11=ORD-2024-001|55=AAPL|54=1|38=100|40=2|44=185.50

# 字段说明：
# 8  = BeginString (协议版本)
# 35 = MsgType (D=新订单)
# 49 = SenderCompID
# 56 = TargetCompID
# 11 = ClOrdID (客户端订单ID)
# 55 = Symbol
# 54 = Side (1=买, 2=卖)
# 38 = OrderQty
# 40 = OrdType (2=限价)
# 44 = Price
```

---

### 2.6 软件与硬件选择

#### 语言选型矩阵

| 组件 | 推荐语言 | 原因 |
|------|---------|------|
| 研究/策略开发 | Python | 生态丰富(pandas/numpy/sklearn)，快速迭代 |
| 订单路由/数据处理 | C++/Rust | 微秒级响应，确定性性能 |
| 大型交易基础设施 | Java | 性能与开发效率平衡，内存安全 |
| 数据管道/ETL | Python | 批处理场景延迟不敏感 |

**核心洞察：** 大部分代码不在性能关键路径上。从收数据到发单的路径是一小部分代码，但决定策略可行性。优化这条路径，其余用Python足够。

#### 硬件优化阶梯

| 优化层级 | 延迟改善 | 成本 |
|---------|---------|------|
| 标准云/数据中心 | 基准 | 低 |
| 低延迟数据中心 | 毫秒级改善 | 中 |
| Co-location（交易所机房） | 微秒级改善 | 高 |
| FPGA硬件加速 | 亚微秒级 | 极高 |
| 微波/激光链路 | 城际间毫秒级 | 极高 |

**微波链路示例：** 芝加哥CME到纽约NYSE的微波直线传输比任何光纤都快——电磁波在空气中传播比在玻璃光纤中快，且直线距离更短。

---

## 三、系统设计核心原则

### 3.1 七大设计原则

**1. 关注点分离 (Separation of Concerns)**
- 策略引擎不知道订单如何路由
- 执行系统不关心信号怎么生成
- 组件通过定义良好的接口通信
- 好处：bug隔离、组件可独立测试、更新不级联

**2. 确定性与可复现性 (Determinism)**
- 相同输入必须产生相同输出
- 随机数生成器必须可种子化
- 所有外部依赖必须被记录
- 目的：生产系统行为可重放，debug可复现

**3. 失败安全默认 (Fail-Safe Defaults)**
- 数据源断线 → 不假设最后价格有效
- 风险计算失败 → 暂停交易而非盲目继续
- 不确定时选择最安全状态
- 不对称性：错过机会 << 灾难性错误

**4. 数据优先 (Data First)**
- 数据质量 > 数据数量 > 策略复杂度
- 回测与实盘数据差异是策略失败的首要原因
- Point-in-time 正确性是回测有效性的前提

**5. 容量约束 (Capacity Discipline)**
- 每个策略有容量上限
- 超容量的资金因 slippage 吃掉收益
- 主动控制规模比追求规模更重要

**6. 统一系统 vs 多策略竞争**
- Medallion 选择统一模型：所有人改进同一系统
- 多策略竞争模型在规模更大时可能需要（如 Citadel）
- 统一模型的关键优势：消除内部竞争，信息完全共享
- 代价：单点风险（如果模型整体退化，没有对冲）

**7. 自动化执行**
- 信号生成到下单全自动化
- 人类只做研究和系统维护
- 消除情绪干扰是高 Sharpe 的前提

### 3.2 策略容量与 Alpha 衰减

**Medallion vs 机构基金对比：**

| 维度 | Medallion | RIEF（机构基金） |
|------|-----------|----------------|
| 规模 | $10-15B | $60-70B |
| 持仓时间 | 1-1.5天 | 数月 |
| Sharpe | 6.3-7.5 | 接近市场 |
| 持仓数 | 4300+ | 4300+（但周转慢） |
| 年化净回报 | ~40% | ~8-10% |

**结论：** 同一团队、同样的研究员，不同容量的产品绩效天差地别。容量约束是量化策略的第一性约束。

**Alpha 衰减机制：**
1. 策略被发现 → 更多资金涌入 → 交易冲击增大 → slippage 吃掉 edge
2. 策略被其他量化基金复制 → edge 被竞争者瓜分
3. 市场结构变化 → 历史规律失效
4. 对策：持续发现新策略 + 控制已有策略的资金规模

### 3.3 对 quant-autoresearch 的设计启示

| 层级 | Medallion 原则 | 落地方案 |
|------|---------------|---------|
| **数据层** | Point-in-time 正确性、数据验证 | 所有特征计算只用 available_date <= decision_date 的数据 |
| **策略层** | 统一模型、市场中性 | 单一 pipeline.py 统一调用，多空对冲消除 beta |
| **系统层** | 自动化、容量约束 | pipeline.py 自动执行，设置策略容量上限 |
| **风控层** | 多层防御、熔断器 | Pre-trade 检查 + 日亏损限制 + 回撤熔断 |
| **执行层** | 低 slippage | 日级策略用 VWAP 算法即可，无需 co-location |
| **进化层** | 持续发现新策略 | LESSONS.md 积累 + 每轮验证 + 次日准确率追踪 |
| **哲学层** | 统计优势而非精准预测 | 不追求单次准确，追求大量样本上的统计优势 |

---

## 四、关键数据速查

### 4.1 Medallion 历年表现

| 年份 | Gross 回报 | Net 回报 | 事件 |
|------|-----------|---------|------|
| 1988 | ~20% | - | 基金成立 |
| 1990-1999 | ~30-40% | ~20-25% | 稳定增长期 |
| 2000 | 98.5% | - | 科技泡沫崩盘，波动率红利 |
| 2001 | 100%+ | - | 提carry至36% |
| 2002 | 100%+ | - | 提carry至44% |
| 2003 | 100%+ | - | 踢出外部投资者 |
| 2007 | 136% gross | - | 金融危机前 |
| 2008 | 152% gross | ~80% net | 金融危机，最大波动率红利 |
| 2009 | ~100% | - | Simons 退休 |
| 2010-2022 | ~77% gross avg | ~40% net avg | Brown/Mercer 时代 |
| 2020 | 149% gross | 76% net | 疫情波动 |

### 4.2 系统组件延迟参考

| 操作 | 延迟 |
|------|------|
| 光纤传输（1km） | ~5 μs |
| FIX消息序列化 | ~1-50 μs |
| 策略计算（Python） | ~1-100 ms |
| 策略计算（C++） | ~1-100 μs |
| 网络往返（同数据中心） | ~50-500 μs |
| 网络往返（跨城） | ~1-10 ms |
| 数据库查询（kdb+ tick级） | ~0.1-1 ms |

### 4.3 风险指标参考

| 指标 | 公式 | 用途 |
|------|------|------|
| VaR (95%) | μ - 1.645σ | 95%概率日亏损不超过此值 |
| ES (95%) | E[loss \| loss > VaR] | 超过VaR时的平均损失 |
| Sharpe | (μ - r_f) / σ | 风险调整后收益 |
| Max Drawdown | (Peak - Trough) / Peak | 历史最大回撤 |
| Calmar | CAGR / MaxDD | 回撤调整后收益 |

---

## 五、一句话总结

**Medallion 的本质不是"预测市场"，而是"在大量独立交易中维持微小统计优势 + 用极短持仓和极低波动率控制风险 + 用容量约束保护 edge 不被 slippage 吃掉"——系统设计的胜利，不是天才直觉的胜利。**
