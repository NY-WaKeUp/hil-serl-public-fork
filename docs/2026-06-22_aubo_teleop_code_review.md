# Aubo 仿真遥操作代码审查（2026-06-22）

## 结论

**夹爪必须按住 `[`/`]` 才能有力**，根因不是 XML 里 `kp/forcerange`，而是 **控制循环架构与 MyI10Tele / Panda 不一致**，叠加了早期错误实现（运行时 `_snap_gripper_qpos`）。

---

## 三方对比

| 机制 | MyI10Tele `0.tele.py` | Panda `gym_hil` | 之前 Aubo 实现（错误点） |
|------|----------------------|-----------------|------------------------|
| 主循环 | **每帧先 `step_env()`**（保持 ctrl） | 每步 `apply_action` + `mj_step` | 只在状态变化时跑 physics |
| 夹爪按键 | **`[`/`]` 按一次 latch** | 按住时 `grasp_command≠0`，松开 `ctrl` **保持** | 曾用 gym_hil **hold** 语义 + qpos snap |
| action[6] | **0=开 / 1=关**（绝对 openpi） | ±1 增量式 grasp | ±1 与 0 混用 |
| 夹爪力来源 | position servo **每 tick 写 ctrl** | tendon motor **ctrl 积分保持** | 曾 snap qpos 破坏 constraint solver |

---

## 已修复项

### 1. 删除运行时 `_snap_gripper_qpos`（夹飞 cube 根因）

手动把 mimic 关节 qpos 设到目标值 → 与 cube 穿插 → contact solver 产生巨大回复力弹飞。

**现在**：只在 `reset` 时 snap 一次；运行时完全交给 MuJoCo equality + contact。

### 2. 每步都跑 `step_env`（夹爪持续有力）

对齐 MyI10Tele：`step_env()` 每 tick 都执行，不是只在按键变化时。

```python
def _advance_physics_hold(self):
    self._apply_gripper_target()  # 每步写 rh_r1_servo ctrl
    ...
    for _ in range(can_substeps):
        self._ctrl.step_env(nstep=nstep_per_substep)
```

### 3. 夹爪 latch 对齐 MyI10Tele

- 键盘：`[` 关 / `]` 开，**按一次切换**，不需要按住
- `get_action()[6] = float(gripper_closed)` → **0.0 或 1.0**
- env：`action[6]≈0`（policy 中性动作）→ **保持 latch**；`>0.5` 关；`<0.5` 且非 0 → 开

### 4. XML 恢复原始 gripper 参数

`kp=5000, forcerange="-75 75"` — 与 MyI10Tele 一致，不再错误降力。

### 5. `AuboKeyboardWrapper.reset` 同步 latch

`env._gripper_closed = controller.gripper_closed`，避免 reset 后键盘与环境状态不一致。

---

## 验证

```
open rh_r1=0.6677
after 15 neutral steps: rh_r1=0.8183, gripper_ctrl=0.8663  ✓
keyboard action[6]: closed=1.0, open=0.0  ✓
neutral action[6]=0 → latch unchanged  ✓
```

---

## 使用注意

1. **必须重启** `record_demos.py` 才能加载新代码
2. 启动 log 应显示：`gripper=MyI10Tele latch`（不是旧的 `gym_hil hold`）
3. `[` 关 / `]` 开：**按一次即可**，不需要按住
4. Space 切换 intervention；intervention ON 时每步键盘 action 带绝对 gripper 状态

---

## Panda 夹爪（更正）

**之前说「Panda 必须 hold `[`/`]`」是错的。**

Panda 键盘在 `intervention_utils.py` 里确实是 **按下时** `close_gripper_command=True`、**松开时** False。但用户体感是「按一下就开/关」，原因是 env 层 **ctrl 会保持**：

```python
# InputsControlWrapper：没按键时 action[-1] = 1.0（不是 0）
gripper_command == "no-op" → action[-1] = 1.0

# EEActionWrapper
grasp_command = action[-1] - 1.0  # no-op → 0

# apply_action
ng = clip(g + grasp_command, 0, 1)  # grasp_command=0 → ng 不变 → ctrl 保持
```

按一下：若干帧 `grasp_command=±1` 积分 → 松开瞬间 `grasp_command=0` → **夹爪停在当前开合度**，不需要继续按住。

| | Panda | Aubo（position servo） |
|--|-------|------------------------|
| 力/位置保持机制 | `data.ctrl` 积分后不变 | 必须每步写 `rh_r1_servo` 目标 |
| 按键语义 | 按住时积分，松开保持 ctrl | **`[`/`]` 按一次 latch** + 每步 `step_env` |
| 目标 UX | 按一下即可 | 同样：按一次 latch，每 tick 维持 ctrl |

## Bugfix（2026-06-22 下午）：按 `[`/`]` 没反应

**根因**：`get_action()` 发 `action[6]=0.0` 表示开，但 `_update_gripper_latch` 把 `|g|<0.05` 当成「policy 中性动作 → 保持 latch」。**开夹爪的 0.0 被当成 hold，latch 永远不变。**

终端里能看到 `Gripper open` 打印（键盘收到了），但 sim 不动——正是这个原因。

**修复**：
- teleop 改回 `action[6] = +1（关）/ -1（开）`，`0` 仅表示 policy 中性 hold
- `AuboKeyboardWrapper` 在 intervention 时每步 `sync controller.gripper_closed → env._gripper_closed`
