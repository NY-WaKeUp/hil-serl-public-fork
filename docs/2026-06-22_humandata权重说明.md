# humandata 的权重会更大吗？

## 结论（先说答案）

**不是「单条样本 loss 乘更大系数」，而是 RLPD 的「50/50 批次混合采样」——每个训练 batch 里，一半来自在线 `replay_buffer`，一半来自 `demo_buffer`（人类演示 + 介入数据）。**

因此：

| 数据类型 | 是否进 demo_buffer | 是否进 replay_buffer | 在 batch 中的「有效占比」 |
|----------|-------------------|---------------------|--------------------------|
| 离线录制的 demo（`record_demos` pkl） | ✅ | ❌ | **固定占 batch 的 50%**（与 demo_buffer 内其它数据均分这 50%） |
| 在线策略 rollout（无介入） | ❌ | ✅ | 仅占 replay 那 50% 里的一部分 |
| 人类介入步（intervention） | ✅ | ✅ | **双重入口**：既占 replay 50%，又占 demo 50% → **有效被采样概率高于纯策略步** |

**置信度：高**（直接来自 `examples/train_rlpd.py` 源码）。

---

## 1. 代码依据

### 1.1 Actor：数据写到哪里

```python
# examples/train_rlpd.py actor 循环
data_store.insert(transition)          # 所有步 → 在线 replay
if already_intervened:
    intvn_data_store.insert(transition)  # 仅介入步 → demo_buffer
```

- `data_store` → learner 的 `replay_buffer`
- `intvn_data_store` → learner 的 `demo_buffer`（与离线 demo 共用）

### 1.2 Learner：50/50 混合

```python
# examples/train_rlpd.py learner
# 50/50 sampling from RLPD, half from demo and half from online experience
replay_iterator = replay_buffer.get_iterator(
    sample_args={"batch_size": config.batch_size // 2, ...},
)
demo_iterator = demo_buffer.get_iterator(
    sample_args={"batch_size": config.batch_size // 2, ...},
)
# 每次 update：
batch = concat_batches(replay_batch, demo_batch, axis=0)
```

- 总 `batch_size` 不变
- **没有** `weight`、`importance_sampling`、`priority` 等人数据加权字段
- `Dataset.sample` 是**均匀随机**抽样

### 1.3 离线 demo 预填充

```python
for path in FLAGS.demo_path:
  for transition in pkl.load(f):
    demo_buffer.insert(transition)
```

离线 demo **只**进 `demo_buffer`，不进 `replay_buffer`。

---

## 2. 数学直觉

设某次 critic update 的 batch 大小为 $B$，则：

$$
B = B_{\text{online}} + B_{\text{demo}}, \quad B_{\text{online}} = B_{\text{demo}} = B/2
$$

对**一条离线 demo** transition $d$：

- 被采到的概率 $\propto \dfrac{B/2}{|\mathcal{D}_{\text{demo}}|}$
- 与 demo_buffer 里其它 transition **等权**

对**一条纯策略** transition $o$（无介入）：

- 仅来自 $\mathcal{D}_{\text{online}}$
- 被采到的概率 $\propto \dfrac{B/2}{|\mathcal{D}_{\text{online}}|}$

对**一条介入** transition $h$：

- 同时在 $\mathcal{D}_{\text{online}}$ 和 $\mathcal{D}_{\text{demo}}$ 各存一份
- 单次 update 被采到的概率 $\approx \dfrac{B/2}{|\mathcal{D}_{\text{online}}|} + \dfrac{B/2}{|\mathcal{D}_{\text{demo}}|}$（两路独立抽样，近似相加）

因此：**介入数据的有效「曝光率」高于纯在线步**，但不是 loss 里乘系数，而是**双 buffer 重复入库**。

---

## 3. 和「权重更大」的常见误解对比

| 误解 | 实际情况 |
|------|----------|
| humandata loss 乘 2× | ❌ SAC loss 对 batch 内每条样本一视同仁 |
| demo 占 80% / 可配置比例 | ❌ 本仓库写死 50/50，无 CLI 改比例 |
| 离线 demo 比介入更重要 | ❌ 在 demo_buffer 内**均匀随机**，不区分来源 |
| 人类数据一定主导训练 | ⚠️ 仅保证 **batch 一半槽位**来自 demo 池；池子变大后单条 demo 被抽到概率下降 |

---

## 4. RLPD 论文背景（对称采样）

Ball et al., ICML 2023, *Efficient Online Reinforcement Learning with Offline Data*：

- 提出 **symmetric sampling**：每 batch 50% online + 50% offline
- 消融表明对比例不极度敏感，但 **50% 是方差、收敛速度、渐近性能的平衡点**
- 本仓库 HIL-SERL 直接沿用该设计

---

## 5. 术语：humandata 在本项目指什么

仓库内**没有**名为 `humandata` 的变量；通常指：

1. **离线专家演示**：`record_demos.py` → `demo_data/*.pkl` → `--demo_path`
2. **在线人类介入**：遥操作时 `info["intervene_action"]` 覆盖策略动作 → 写入 `demo_buffer`

二者在训练时都走 **demo_buffer 那 50%**。

---

## 6. 优劣势（批判性）

| 优势 | 劣势 / 风险 |
|------|-------------|
| 实现简单，无额外超参 | 50% 不可调，demo 质量差时会拖累一半 batch |
| 防止在线数据淹没少量 demo | 介入步双写可能导致**过拟合介入分布** |
| 与 RLPD / HIL-SERL 论文一致 | demo 与 online 分布偏移时 Critic 可能偏估 |

---

## 7. 若你想「真正加大」人类数据影响

当前代码**不支持**，需改 `train_rlpd.py`，例如：

- 改 `batch_size // 2` 比例为 `// 4` 与 `// 4 * 3`（75% demo）
- 或在 loss 里对 demo 子 batch 乘系数（非标准 RLPD）

默认配置下：**人类数据 = batch 的 50% 槽位 + 介入步额外多一份在 replay 里**，不是更大的 loss 权重。

---

## 8. 常见混淆：「batch 一半一半」≠「两个池子一样大」

很多人会把两件事混为一谈：

| 概念 | 是否恒等 | 含义 |
|------|----------|------|
| **每个 batch 的采样比例** | ✅ 恒为 50/50 | 每次 update 固定从 demo 池抽 `B/2`、从 online 池抽 `B/2` |
| **demo 池 vs online 池里存了多少条数据** | ❌ **不相等** | 两个 buffer 的 `len()` 随训练变化，通常 online 远大于 demo |

### 两个池子各自装什么

```
online 池 (replay_buffer)：
  每一步 env.step 都 insert（策略步 + 介入步）

demo 池 (demo_buffer)：
  启动时：离线 record_demos 的 pkl
  训练中：仅介入步 insert（intvn_data_store）
```

### 典型数量级（仿真举例）

- 离线 demo：20 条 episode × ~100 步 ≈ **2000 条**
- 训练 5 万步，介入 10%：介入 ≈ **5000 条** → demo 池合计 ~7000 条
- online 池：**50000 条**（几乎全部步）

两者 **capacity** 都是 `replay_buffer_capacity=200000`（上限相同），但 **实际存量** online 通常大一个数量级以上。

### 因此

- 「一半一半」说的是：**每个 batch 的槽位分配**，不是池子大小。
- 从 **7000 条的 demo 池**里均匀抽 128 条，比从 **50000 条的 online 池**里均匀抽 128 条，**单条被抽中的概率更高**。
- 介入步还在 online 池里再占一份 → 所以介入步的期望曝光率 > 纯策略步。

**置信度：高**（`train_rlpd.py` actor insert 逻辑 + 两 buffer 同 capacity 不同填充）。
