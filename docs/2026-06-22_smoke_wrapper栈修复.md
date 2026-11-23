# 2026-06-22 smoke 阻塞修复记录

## 现象（按时间顺序）

### 1. QoS 不匹配 → `_queue.Empty`（已修）

ZED **BEST_EFFORT** vs 旧订阅 **RELIABLE** → `image_tcp_server` 收不到图。  
修复：`qos_profile_sensor_data` + deploy 到 Manjaro。

### 2. wrapper 栈（已修）

```
RuntimeError: AuboPickCubeGymEnv not found in wrapper stack
```

`_find_aubo_gym_env` 只认 `render_teleop_view`（仿真专用）。  
修复：改为识别 `velocity_integrator` + `_gripper_closed`。

### 3. 端口占用（本次）

```
OSError: [Errno 98] Address already in use   # Manjaro :8767
RuntimeError: image tcp server exited (code 1)
```

上次 smoke 被 Ctrl+C 打断，`env.close()` 未跑完，Manjaro 上旧 `image_tcp_server` 仍占 **8767**。

## 修复

`python3.x/image_tcp_bridge.py`：

1. 启动前先连 `127.0.0.1:32087` — 隧道仍活则**复用**
2. 否则 SSH：`fuser -k 8767/tcp` + `pkill -f image_tcp_server.py`
3. 本地 `fuser -k 32087/tcp` 清旧 SSH 隧道
4. 再起 server + `-L 32087`

## 验证

```bash
cd ~/hil-serl/examples/experiments/aubo_pick_cube_real
source ~/hil-serl/.venv/bin/activate
./smoke_env.sh
```

期望：`SMOKE OK`；或 `(reused existing tunnel/server)`。

手动兜底：

```bash
ssh Manjaro 'fuser -k 8767/tcp; pkill -f image_tcp_server.py'
fuser -k 32087/tcp
```

## 优劣势

| | |
|--|--|
| **优势** | 重复 smoke 不必手杀进程；活隧道零重启 |
| **风险** | `pkill -f` 会杀所有同名脚本（本机仅一条 image server，可接受） |

**置信度**：高。
