# 2026-06-23 · Learner 里 Orbax checkpoint 日志是啥？

## 一句话

**这是 learner 在正常存盘**：把当前训练到第 **195000** 步的策略网络权重，写到  
`checkpoints/aubo_pick_cube_sim_rsa/checkpoint_195000/`。  
**不是报错**，训练会继续。

---

## 日志在说什么（按行）

| 日志片段 | 含义 |
|----------|------|
| `Using Orbax as backend to save Flax checkpoints` | JAX/Flax 用 **Orbax** 做 checkpoint（Flax 新版默认后端） |
| `Started saving checkpoint to .../checkpoint_195000` | 开始保存 **全局 step = 195000** 的 ckpt |
| `Creating tmp directory ...orbax-checkpoint-tmp-3` | 先写**临时目录**，写完再原子改名，避免存一半断电损坏 |
| `blocking_bytes_per_sec: 2.8 GiB/s` / `bytes_per_sec: 643 MiB/s` | 写入速度统计（约 **305 MiB** 权重） |
| `Validating params (raw input=1658, unique=829)` | 存完后校验 TensorStore 里的参数树 |
| `Renaming ...tmp-3 to .../checkpoint_195000` | 临时目录改名为正式 checkpoint → **保存成功** |

---

## 为什么是 195000？

`examples/experiments/aubo_pick_cube_sim/config.py`：

```python
checkpoint_period = 5000
```

`train_rlpd.py` learner 循环里：每 `step % 5000 == 0` 调一次 `checkpoints.save_checkpoint(...)`。

所以你会看到 `checkpoint_5000`、`10000`、…、`195000` 等目录。

---

## 目录里有什么？

典型结构（Orbax + RLPD）：

- `checkpoint_195000/` — 该 step 的 **agent 网络权重**（actor/critic 等）
- 同级的 `buffer/`、`demo_buffer/` — replay 数据（另存，不在这几条日志里）

`run_eval.sh` 里可指定：

```bash
--eval_checkpoint_step=195000
```

从该目录恢复策略做评估。

---

## 需要担心吗？

| 情况 | 判断 |
|------|------|
| 只有 INFO（`I0623`），最后有 `Renaming ... to checkpoint_195000` | **正常**，存盘完成 |
| 卡住很久无 `Renaming` | 可能磁盘慢或 IO 堵，看磁盘空间与 `df` |
| `E` 开头 ERROR / 无 rename | 存盘失败，需看完整 traceback |

你贴的日志 **已有 rename** → 195000 步 checkpoint **已成功落盘**。

---

## 和 actor 的关系

- **Learner**：算梯度 + **周期性 save checkpoint**（你看到的日志）
- **Actor**：跑环境、采数据；一般从 learner 同步权重，不自己写这个大 ckpt

存 checkpoint 时 learner 可能短暂变慢，属正常现象。
