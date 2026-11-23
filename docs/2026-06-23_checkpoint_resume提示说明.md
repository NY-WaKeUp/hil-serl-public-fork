# 2026-06-23 · 为啥提示「Checkpoint path already exists. Press Enter to resume training」？

## 一句话

**不是报错。** `train_rlpd.py` 发现 `--checkpoint_path` 目录**已存在且里面已有 `checkpoint_xxxxx`**，在加载旧权重/旧 buffer 之前，**故意停住等你确认**是否断点续训。

你改了环境（cube 随机朝向）≠ 自动换新 checkpoint 目录；**路径没变或目录里已有 ckpt，就会弹这句。**

---

## 触发条件（三个同时满足）

```452:459:examples/train_rlpd.py
    if (
        FLAGS.checkpoint_path is not None
        and os.path.exists(FLAGS.checkpoint_path)
        and not FLAGS.eval_checkpoint_step
    ):
        ckpt_dir = os.path.abspath(FLAGS.checkpoint_path)
        if checkpoints.latest_checkpoint(ckpt_dir) is not None:
            input("Checkpoint path already exists. Press Enter to resume training.")
```

| 条件 | 含义 |
|------|------|
| `checkpoint_path` 已设置 | `run_learner.sh` 里 `CHECKPOINT_PATH=...` |
| 目录存在 | 之前跑过 learner/actor，或手动建过目录 |
| `latest_checkpoint()` 非空 | 里面有 `checkpoint_5000`、`checkpoint_195000` 等 |

若目录存在但**空的**（没有任何 `checkpoint_*`），会打印 `Starting fresh.`，**不会**要按 Enter。

---

## 按 Enter 之后会发生什么？

1. **恢复最新一步的网络权重**（`restore_checkpoint` → `Loaded previous checkpoint at step XXXXX`）
2. Learner 还会从同目录加载：
   - `buffer/*.pkl` — 在线 replay
   - `demo_buffer/*.pkl` — 训练过程中积累的干预数据  

→ **从旧实验接着训**，不是从零开始。

---

## 和前面 ResNet 日志的关系

```
replaced ResNetBlock_0 in pretrained_encoder
...
```

这是 **ImageNet 预训练 ResNet-10 灌进 encoder**，每次启动 learner 都会做，**与 checkpoint 提示无关**。

顺序是：先建 agent（含 ResNet 替换）→ 再检查 checkpoint 目录 → 再问你按不按 Enter。

---

## 你说是「新实验」但仍弹窗 — 常见原因

当前 `run_learner.sh` 默认：

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/aubo_pick_cube_sim_rsa_random_cube}"
```

只要这个目录**曾经跑过一次**（哪怕几分钟），里面就会有 `checkpoint_5000` 等，下次启动就会提示。

| 你的意图 | 做法 |
|----------|------|
| **接着上次训** | 直接 **Enter** |
| **全新实验（新随机 cube 从零训）** | 换**从未用过**的路径，例如：<br>`export CHECKPOINT_PATH=$REPO_ROOT/checkpoints/aubo_pick_cube_sim_rsa_random_cube_v2`<br>再 `./run_learner.sh` 和 `./run_actor.sh`（**两边路径必须一致**） |
| **删掉旧 run 重来** | `rm -rf checkpoints/aubo_pick_cube_sim_rsa_random_cube`（确认无需要保留的 ckpt 再删） |

仓库里已有旧目录示例：`checkpoints/aubo_pick_cube_sim_rsa`（上次训练到 19.5 万步等）。

---

## Actor 侧

`run_actor.sh` 用**同一个** `CHECKPOINT_PATH` 连 learner。新实验换路径时 **learner 和 actor 都要 export 同一个 `CHECKPOINT_PATH`**。

---

## `DeprecationWarning: jax.tree_map`

无害警告，JAX 新版本建议改用 `jax.tree.map`；**不影响训练**，与 resume 提示无关。
