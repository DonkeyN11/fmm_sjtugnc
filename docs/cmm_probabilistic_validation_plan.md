# CMM 后验概率校准性验证实验计划

## 研究问题

**各向异性发射概率（CMM）得到的 HMM 后验概率，是否具有概率学意义上的校准性（calibration），能否可靠地作为定位不确定性的量化指标？**

## 核心假设

- $H_0$：各向同性 HMM 的后验概率不可校准 —— "90% 可信"不代表 90% 正确
- $H_1$：各向异性 CMM 的后验概率可校准 —— 后验概率值直接反映实际正确率
- 预期：CMM 的后验在校准性上显著优于各向同性 HMM

---

## 实验 1：可靠性图（Reliability Diagram）

### 目标
直接验证后验概率是否等于实际正确率。

### 数据
- 输入：`dataset-hainan-06/cmm_traj11.csv`（含协方差）
- 真值：ogeom（GNSS 原始位置，精度~1m）
- 匹配结果：`dataset-hainan-06/mr/cmm_0508_traj11_delta_entropy.csv`
- 基线：同一轨迹用各向同性 HMM 运行的结果（需额外生成）

### 方法

1. 取出每个 GPS 点的最优匹配候选的**后验概率**（即 Viterbi 回溯后该点的条件概率）
2. 计算该点的**匹配误差** = haversine(ogeom, pgeom)
3. 定义"正确匹配" = 误差 $\le 5\text{m}$（可另设 3m / 10m 做稳健性检验）
4. 将后验概率分 10 个 bin：$[0,0.1), [0.1,0.2), \dots, [0.9,1.0]$
5. 对每个 bin 内的点，统计实际正确率 = bin 内正确点数 / bin 内总点数
6. 画出 **后验概率均值（x 轴）vs 实际正确率（y 轴）**

### 预期输出

一张图，两条曲线：
- **CMM** 曲线应紧贴对角线 `y=x`
- **各向同性基线** 曲线偏右上（过于自信）或偏左下（过于保守）

### 成功标准

| 指标 | 各向同性 | CMM (预期) |
|------|---------|-----------|
| 可靠性曲线与 `y=x` 的偏离 | 显著 | 轻微 |
| 最大 bin 偏差 | > 0.15 | < 0.10 |
| 可视化定性判断 | 非对角线 | 近似对角线 |

### 实现提示

```python
bins = [(i/10, (i+1)/10) for i in range(10)]
for lo, hi in bins:
    mask = (posteriors >= lo) & (posteriors < hi)
    observed_acc = sum(mask & correct) / sum(mask)
    mean_conf = posteriors[mask].mean()
    plot(mean_conf, observed_acc)
```

---

## 实验 2：Expected Calibration Error (ECE)

### 目标
用一个单一数值量化后验概率的校准偏差。

### 方法

$$ECE = \sum_{b=1}^{B} \frac{n_b}{N} \cdot \left| \text{mean\_conf}_b - \text{acc}_b \right|$$

- $B = 10$ 个等宽 bin
- $n_b$ = 第 $b$ 个 bin 中的样本数
- $\text{mean\_conf}_b$ = 该 bin 内后验概率的均值
- $\text{acc}_b$ = 该 bin 内的实际正确率

同时计算 **MCE（Maximum Calibration Error）** 作为最坏情况：

$$MCE = \max_{b} \left| \text{mean\_conf}_b - \text{acc}_b \right|$$

### 预期输出

一张表格：

| 方法 | ECE ↓ | MCE ↓ | 样本数 |
|------|------|------|------|
| 各向同性 HMM | ? | ? | 2696 |
| 各向异性 CMM | ? | ? | 2696 |

### 成功标准
- CMM 的 ECE $\le 0.05$（优秀校准）
- CMM 的 ECE $<$ 各向同性的 ECE（显著改善）

---

## 实验 3：Proper Scoring Rules（Brier Score + Log-Loss）

### 目标
用严格恰当评分规则（strictly proper scoring rules）评估概率预测质量。两个金标准指标：

**Brier Score（MSE of probabilities）：**

$$BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

其中 $p_i$ 为后验概率，$y_i \in \{0,1\}$ 为实际正误。

**Log-Loss（对数损失）：**

$$LL = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

### 方法
1. 对每个 GPS 点：$p_i$ = 最优候选后验概率，$y_i$ = 误差 $\le 5\text{m}$ 则 1 否则 0
2. 分别计算两种方法的 Brier Score 和 Log-Loss
3. 做 bootstrap（1000 次重采样）计算 95% 置信区间

### 预期输出

| 方法 | Brier Score ↓ | Log-Loss ↓ |
|------|-------------|-----------|
| 各向同性 HMM | ? ± ? | ? ± ? |
| 各向异性 CMM | ? ± ? | ? ± ? |
| 参考：Always 0.5 | 0.250 | 0.693 |
| 参考：Always 0.9 | — | — |

### 成功标准
- CMM 在 Brier Score 和 Log-Loss 上均优于各向同性基线
- 差异在 95% bootstrap CI 下统计显著
- CMM 的 Brier Score 远低于 naive baseline (0.25)

### 理论依据
Strictly proper scoring rules 是概率预测评估的黄金标准。如果评分低于基线，说明概率具有信息量；如果差距显著，说明各向异性发射概率确实带来了信息增益。

---

## 实验 4：决策论验证（Decision-Theoretic Validation）

### 目标
模拟真实下游任务，验证后验概率作为决策阈值的实用性。

### 场景设计

> **里程收费场景**：ETC / 按行驶路径收费系统需要在匹配充足可信时才记入里程，否则拒绝该段并要求人工审计。

### 方法

1. 定义决策规则：
   ```
   if 后验概率 ≥ θ:
       接受匹配（用于计费）
   else:
       拒绝匹配（标记为"不确定"，需人工校验）
   ```

2. 定义错误：
   - **错误接受（False Accept）**：接受了但实际误差 > 容忍值 $\tau$（设 $\tau = 5\text{m}, 10\text{m}, 20\text{m}$）
   - **错误拒绝（False Reject）**：拒绝了但实际是正确的（浪费人工校验资源）

3. 遍历阈值网格 $\theta \in [0, 1]$：
   - 记录点对（错误接受率, 错误拒绝率）
   - 画出 Decision Error Tradeoff 曲线

4. 对比各向同性 HMM vs 各向异性 CMM

### 预期输出

一张 DET 曲线图：
- x 轴 = 错误接受率
- y 轴 = 错误拒绝率
- 两条曲线，CMM 的曲线整体更低（在相同接受率下拒绝更少，或相同拒绝率下接受错误更少）

加上操作点（operating point）表格：

| $\theta$ | 各向同性 → Accept/Reject/FP/FN | CMM → Accept/Reject/FP/FN |
|----------|-------------------------------|----------------------------|
| 0.6 | ? | ? |
| 0.8 | ? | ? |
| 0.9 | ? | ? |
| 0.95 | ? | ? |

### 成功标准
- 在任意给定错误接受率下，CMM 的错误拒绝率低于各向同性基线
- 典型操作点（如 θ=0.9, τ=10m）下 CMM 同时保持较低的错误接受和错误拒绝
- 曲线下面积（AU-DET 或等效）CMM 更优

---

## 实验 5：多源信息融合的熵分解

### 目标
为未来多传感器融合奠定理论基础——证明信息增益的可加性。

### 理论框架

定义贝叶斯信息增益（互信息）：

$$I(\text{path}; \text{GNSS}) = H(\text{path}) - H(\text{path} \mid \text{GNSS})$$

其中：
- $H(\text{path})$ = 先验熵（路网拓扑的路径不确定性）
- $H(\text{path} \mid \text{GNSS})$ = 后验熵（给定 GNSS 观测后的剩余不确定性）
- $I$ = GNSS 观测带来的**信息增益**（bits）

### 方法

**阶段 A（当前可行）**：单传感器 GNSS 的信息增益

1. 计算先验熵：从候选第一层过渡图的均匀分布开始

   $$H_0 = \log_2(|C_0|) \quad \text{（第一层候选数取对数）}$$

2. 对每个时间步，计算后验熵 $H_t = H(\text{path} \mid \text{GNSS}_{1:t})$

3. 画出：
   - 后验熵随时间变化的曲线（预期单调递减）
   - 信息增益 $\Delta H_t = H_0 - H_t$ 随时间累积曲线

4. 对比：各向同性 HMM vs 各向异性 CMM 的信息增益速率

**阶段 B（需要多传感器数据）**：

1. 单独计算 IMU/视觉的信息增益：$I(\text{path}; \text{IMU})$
2. 计算联合信息增益：$I(\text{path}; \text{GNSS}, \text{IMU})$
3. 验证可加性：$I_{\text{joint}} \approx I_{\text{GNSS}} + I_{\text{IMU}}$（在传感器独立的近似下）

### 预期输出

一张图：**累积信息增益随时间变化**
- x 轴 = 时间步
- y 轴 = 累积互信息（bits）
- 两条曲线：各向同性 HMM 和各向异性 CMM
- 预期：CMM 的信息累积更快（因为各向异性方差对"好"候选赋予更高概率）

一张表：**信息增益分解**

| 信源 | 各向同性 HMM | 各向异性 CMM |
|------|------------|------------|
| GNSS 先验 $H_0$ | ? bits | ? bits |
| GNSS 后验 $H_T$ | ? bits | ? bits |
| GNSS 信息增益 | ? bits | ? bits |

### 成功标准
- CMM 的信息增益累积速率 > 各向同性 HMM
- CMM 的最终后验熵 < 各向同性 HMM（更确定的路径推断）
- 框架可扩展到多传感器场景（为第二阶段研究做准备）

---

## 实验时间线与依赖

| 阶段 | 实验 | 数据需求 | 代码工作量 |
|------|------|---------|-----------|
| 第 1 周 | 实验 1 + 2（可靠性图 + ECE） | 现有 CMM 输出 + 新跑各向同性 HMM | 中 |
| 第 2 周 | 实验 3（Brier / Log-Loss） | 同上 | 小 |
| 第 3 周 | 实验 4（决策论） | 同上 | 中 |
| 第 4 周 | 实验 5（熵分解） | 同上（阶段 A）；传感器数据（阶段 B） | 大 |

**前置条件**：各向同性 HMM 基线需要单独运行一次（关闭 Mahalanobis 候选，使用等向高斯发射概率）。

---

## 预期论文/报告的论证逻辑

```
1. 各向异性发射概率 → 改善匹配精度（已有实验数据支持）
2. 各向异性发射概率 → 改善概率校准性（实验 1+2+3 证明）
3. 校准的后验 → 可用于下游任务决策（实验 4 证明）
4. 校准的后验 + 信息论 → 多传感器融合的理论框架（实验 5 证明）
5. 结论：各向异性 HMM 后验概率是具有概率学意义的可信度量
```

---

## 附录：关键指标速查

| 指标 | 公式 | 含义 | 越低越好？ | 阈值 |
|------|------|------|-----------|------|
| ECE | $\sum \frac{n_b}{N} \| \text{conf}_b - \text{acc}_b \|$ | 概率校准偏差 | ↓ | < 0.05 优秀 |
| MCE | $\max \| \text{conf}_b - \text{acc}_b \|$ | 最坏校准偏差 | ↓ | < 0.10 |
| Brier Score | $\frac{1}{N} \sum (p_i - y_i)^2$ | 概率预测误差平方 | ↓ | < 0.10 优于随机 |
| Log-Loss | $-\frac{1}{N} \sum y \log p + (1-y) \log (1-p)$ | 概率质量 | ↓ | < 0.3 有信息量 |
| $I$ (MI) | $H_0 - H_t$ | 观测带来的信息 | ↑ | — |
