# 2026-06-23 · 夹爪 `[`/`]` 报 `accessing private attribute '_trigger_gripper' is prohibited`

## 现象

按 `]` 开夹爪时终端打印 `Gripper open`，但 pynput 回调抛错：

```
AttributeError: accessing private attribute '_trigger_gripper' is prohibited
```

堆栈：`aubo_hil_wrappers.py` → `_trigger_gripper_immediate` → `unwrapped._trigger_gripper(close)` → `gymnasium/core.py` `Wrapper.__getattr__`。

## 根因（置信度：高）

Gymnasium `Wrapper` **禁止**通过 `env._xxx` 访问下层私有属性（`name.startswith("_")` 直接 `raise`）。

`_find_aubo_gym_env` 原逻辑：

```python
hasattr(cur, "velocity_integrator") and hasattr(cur, "_gripper_closed")
```

在 **sim** wrapper 栈中会误命中 `AuboGripperPenaltyWrapper`：

| 层 | 为何被误匹配 |
|----|-------------|
| `AuboGripperPenaltyWrapper` | 自身有 `_gripper_closed`；`velocity_integrator` 经 `__getattr__` 转发到内层 |
| 真正的 `AuboPickCubeGymEnv` | 才有 `_trigger_gripper` / 夹爪控制，但在外层之后 |

对误命中的 **Wrapper** 调 `_trigger_gripper` → 触发 Gymnasium 保护 → 报错。

## 修复

1. **`_find_aubo_gym_env`**：用 **类上是否定义** `trigger_gripper` 识别控制 env，跳过仅带 `_gripper_closed` 的 penalty wrapper。
2. **`AuboPickCubeRealEnv` / `AuboPickCubeGymEnv`**：新增公开方法 `trigger_gripper(close)`（真机发 TCP / 仿真更新 latch + `_apply_gripper_target`）。
3. **`_trigger_gripper_immediate`**：改为 `ctrl.trigger_gripper(close)`，不再碰 `_trigger_gripper`。

## 修改文件

- `serl_robot_infra/franka_env/envs/aubo_hil_wrappers.py`
- `serl_robot_infra/franka_env/envs/aubo_i10_real_env.py`
- `serl_robot_infra/franka_env/envs/aubo_i10_gym_env.py`

## 验证

重启 `./run_eval.sh`（sim 或 real），按 `]` / `[` 应无 AttributeError；sim 夹爪 latch 立即更新，真机应发 gripper TCP。
