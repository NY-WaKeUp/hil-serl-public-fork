# 2026-06-22 · eval protobuf 崩溃（sitecustomize 修复）

## 现象

`bash run_eval.sh` 在导入 `robotcontrol` → `libpyauboi5` 时崩溃：

```
File already exists in database: google/protobuf/descriptor.proto
Fatal Python error: Aborted
```

## 根因

1. **C++ 层冲突**：TensorFlow 与 `libpyauboi5` 各自静态注册同一 protobuf descriptor，导入顺序错误即崩溃。
2. **预加载未生效**：脚本用 `usercustomize.py` + `ctypes.CDLL` 预加载，但 **venv 下 `ENABLE_USER_SITE=False`**，`site.py` 不执行 `execusercustomize()`，预加载从未运行。

## 修改

| 文件 | 变更 |
|------|------|
| `examples/experiments/aubo_pick_cube_real/run_eval.sh` | `usercustomize` → **`sitecustomize`**；`import libpyauboi5`；`rm -rf $_PRELOAD_DIR` |
| `examples/experiments/aubo_pick_cube_real/run_actor.sh` | 同上 |
| `docs/2026-06-22_eval_protobuf冲突修复.md` | 更新根因与验证说明 |

## 验证

本地复现：无预加载时 `jax` → `tensorflow` → `libpyauboi5` 必崩；加 `sitecustomize` 后通过。

`bash run_eval.sh`（90s 超时截断）已越过 protobuf 阶段，日志含：

- `robotcontrol.py:560` 连接 `127.0.0.1:32085`
- image/gripper TCP ready
- `Restoring orbax checkpoint ... checkpoint_175000`

**无** `Aborted (core dumped)`。

## 你可直接重跑

```bash
cd ~/hil-serl/examples/experiments/aubo_pick_cube_real
source ~/hil-serl/.venv/bin/activate
export DISPLAY=:52.0
export CHECKPOINT_PATH=~/hil-serl/checkpoints/aubo_pick_cube_sim_rsa
export EVAL_STEP=175000 EVAL_N_TRAJS=1
bash run_eval.sh
```
