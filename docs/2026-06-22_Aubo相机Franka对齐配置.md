# 2026-06-22 · Aubo 真机相机 Franka 对齐配置

## 做了什么

将 Aubo 真机相机链路对齐 Franka 惯例，并接入 Manjaro 上的 ZED + 鱼眼 ROS topic。

| 层级 | Franka | Aubo 真机（现配置） |
|------|--------|---------------------|
| 采集标称分辨率 | 1280×720 | 1280×720（`CAMERA_CAPTURE_DIM`） |
| 采集标称帧率 | 15 fps（RealSense 默认） | 由 ROS 发布端决定（ZED ~30Hz）；文档记 `CAMERA_CAPTURE_FPS=15` |
| 策略观测图像 | 128×128 RGB | 128×128 RGB（`IMAGE_SIZE`） |
| 策略步频 | 10 Hz | 10 Hz（`POLICY_HZ`） |
| 后端 | `RSCapture` USB | **`RosImageCapture`** 订阅 ROS |

## 修改文件

| 文件 | 变更 |
|------|------|
| `serl_robot_infra/franka_env/camera/ros_capture.py` | 新增：订阅 `sensor_msgs/Image`，`read()` 接口与 `RSCapture` 一致 |
| `serl_robot_infra/franka_env/envs/aubo_control_config.py` | `ROS_IMAGE_CAMERAS`（ZED/鱼眼 topic）；`REALSENSE_CAMERAS` 置空 |
| `serl_robot_infra/franka_env/envs/aubo_i10_real_env.py` | 优先 `ROS_IMAGE_CAMERAS`；共享 `rclpy` spin 线程 |
| `examples/experiments/aubo_pick_cube_real/config.py` | tunnel 机械臂/夹爪 + ROS 相机 |

## 相机 topic 映射

| key | topic | 硬件 |
|-----|-------|------|
| `front` | `/zed/left/image_raw` | ZED 2i 左目 |
| `wrist` | `/image_raw` | LRCP U3-JX02 鱼眼 |

鱼眼原生 1280×1024 时，env 内仍 **resize 到 128×128**（与 Franka 在任意 crop 后 resize 相同）。

## 运行前提

1. Manjaro 已起 `zed_camera_node` + 鱼眼 `v4l2_camera_node`（鱼眼设备 `/dev/video2`）
2. Ubuntu actor 与 Manjaro **同一 `ROS_DOMAIN_ID`**，能 `ros2 topic list` 看到上述 topic
3. SSH 隧道：`127.0.0.1:32085`（臂）、`32086`（夹爪）

## 烟雾测试

```bash
source ~/hil-serl/.venv/bin/activate
export ROS_DOMAIN_ID=<与 Manjaro 相同>
cd ~/hil-serl/examples
python -c "
from experiments.mappings import get_train_config
cfg = get_train_config('aubo_pick_cube_real')
env = cfg.get_environment(fake_env=False)
obs, _ = env.reset()
print({k: v.shape for k, v in obs['images'].items()})
env.close()
"
```

期望：`front` / `wrist` 均为 `(128, 128, 3)`。

## 若改回 USB RealSense

在 `EnvConfig` 中设 `ROS_IMAGE_CAMERAS = {}`，并填写 `REALSENSE_CAMERAS`（含 `serial_number`、`dim=(1280,720)`，可选 `fps=15`）。
