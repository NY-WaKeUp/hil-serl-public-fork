"""Aubo i10 real robot env — CAN velocity control aligned with python3.x/smooth_teleop.py."""

# 想法1:轨迹帧评分，反复做梦， 延伸：等级分类器，
import os
import sys
import threading
import time
from typing import Callable, Dict, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.tcp_capture import TcpImageCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.envs.aubo_control_config import (
    CAN_SUBSTEPS_PER_POLICY_STEP,
    COLLISION_CLASS,
    GRIPPER_SERVICE,
    IK_MAX_JOINT_STEP,
    IMAGE_KEY_FRONT,
    IMAGE_KEY_WRIST,
    IMAGE_SIZE,
    MAX_ANG_VEL,
    MAX_EPISODE_LENGTH,
    MAX_LIN_VEL,
    POLICY_DT,
    POLICY_HZ,
    ROBOT_IP,
    ROBOT_PORT,
    SAFE_JOINT_MAXACC,
    SAFE_JOINT_MAXVELC,
    TELEOP_JOINT_MAXACC,
    TELEOP_JOINT_MAXVELC,
    TOOL_DYNAMICS_POSITION,
    TOOL_PAYLOAD_KG,
    TELEOP_KEY_MAP,
    VEL_EMA_ALPHA,
    CAN_VEL_EMA_ALPHA,
)
from franka_env.envs.aubo_velocity_control import AuboVelocityIntegrator

PYTHON3X_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "python3.x",
)
if PYTHON3X_ROOT not in sys.path:
    sys.path.insert(0, PYTHON3X_ROOT)

from robotcontrol import Auboi5Robot, RobotErrorType  # noqa: E402


class AuboRealEnvConfig:
    """Real Aubo env configuration (fill before deployment)."""

    ROBOT_IP: str = ROBOT_IP
    ROBOT_PORT: int = ROBOT_PORT
    ROS_IMAGE_CAMERAS: Dict = {}
    TCP_IMAGE_CAMERAS: Dict = {}
    REALSENSE_CAMERAS: Dict = {}
    IMAGE_CROP: Dict[str, Callable] = {}
    USE_TCP_IMAGES: bool = True
    IMAGE_TCP_SSH_HOST: str = "Manjaro"
    IMAGE_TCP_LOCAL_PORT: int = 32087
    IMAGE_TCP_REMOTE_PORT: int = 8767
    IMAGE_TCP_MANUAL_TUNNEL: bool = False
    IMAGE_TCP_SERVER_SCRIPT: str = (
        "~/workspace2025/src/inspire_gripper/image_tcp_server.py"
    )
    IMAGE_TCP_SERVER_TOPICS: str = "f:/zed/left/image_raw w:/image_raw"
    MAX_EPISODE_LENGTH: int = MAX_EPISODE_LENGTH
    # RPY limits [roll,pitch,yaw] in base frame; XYZ is handled by safety sphere.
    ABS_POSE_LIMIT_LOW = np.array(
        [-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi], dtype=np.float64
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [np.inf, np.inf, np.inf, np.pi, np.pi, np.pi], dtype=np.float64
    )
    # Sphere centered at reset TCP, radius in meters.
    ENABLE_RELATIVE_SAFETY_BOX: bool = True
    SAFETY_SPHERE_RADIUS: float = 0.15
    # Hard floor on +X toward operator side (scene -X); None disables.
    ABS_X_MIN: float | None = None
    MAX_LIN_VEL: float = MAX_LIN_VEL
    MAX_ANG_VEL: float = MAX_ANG_VEL
    GRIPPER_SERVICE: str = GRIPPER_SERVICE
    USE_ROS_GRIPPER: bool = True
    USE_TCP_GRIPPER: bool = False
    GRIPPER_SSH_HOST: str = "Manjaro"
    GRIPPER_TCP_LOCAL_PORT: int = 32086
    GRIPPER_TCP_REMOTE_PORT: int = 8765
    GRIPPER_TCP_MANUAL_TUNNEL: bool = False
    TELEOP_JOINT_LIMITS: bool = True


class AuboPickCubeRealEnv(gym.Env):
    """Real Aubo i10 with same 7D velocity action semantics as sim."""

    def __init__(
        self,
        config: AuboRealEnvConfig = None,
        fake_env: bool = False,
        hz: float = POLICY_HZ,
    ):
        self.config = config or AuboRealEnvConfig()
        self.fake_env = fake_env
        self.hz = hz
        self.policy_dt = 1.0 / hz
        self.max_episode_steps = self.config.MAX_EPISODE_LENGTH
        self._step_count = 0
        self._gripper_closed = True
        self._last_gripper_cmd = 0.0
        self._vel = AuboVelocityIntegrator(
            max_lin_vel=self.config.MAX_LIN_VEL,
            max_ang_vel=self.config.MAX_ANG_VEL,
            policy_dt=self.policy_dt,
        )
        self._can_cmd_pos = None
        self._can_cmd_rpy = None
        self._can_cmd_joint = None
        self._in_canbus_mode = False
        self.robot = None
        self.cap = None
        self._gripper_cli = None
        self._ros_node = None
        self._ros_spin_active = False
        self._ros_spin_thread = None
        self._gripper_tcp = None
        self._image_tcp_bridge = None
        self._teleop_keys_fn: Callable[[], tuple[set[str], bool]] | None = None
        self._safety_center: np.ndarray | None = None
        self._rpy_low = np.array(self.config.ABS_POSE_LIMIT_LOW[3:], dtype=np.float64)
        self._rpy_high = np.array(self.config.ABS_POSE_LIMIT_HIGH[3:], dtype=np.float64)

        image_keys = (
            list(self.config.TCP_IMAGE_CAMERAS.keys())
            if self.config.USE_TCP_IMAGES and self.config.TCP_IMAGE_CAMERAS
            else list(self.config.ROS_IMAGE_CAMERAS.keys())
            or list(self.config.REALSENSE_CAMERAS.keys())
            or [IMAGE_KEY_FRONT, IMAGE_KEY_WRIST]
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        k: spaces.Box(
                            0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                        )
                        for k in image_keys
                    }
                ),
            }
        )

        if fake_env:
            return

        self._connect_robot()
        self._init_cameras()
        if self.config.USE_TCP_GRIPPER:
            self._init_tcp_gripper()
        elif self.config.USE_ROS_GRIPPER:
            self._init_ros_gripper()

    @property
    def velocity_integrator(self) -> AuboVelocityIntegrator:
        return self._vel

    def _connect_robot(self) -> None:
        Auboi5Robot.initialize()
        self.robot = Auboi5Robot()
        self.robot.create_context()
        result = self.robot.connect(self.config.ROBOT_IP, self.config.ROBOT_PORT)
        if result != RobotErrorType.RobotError_SUCC:
            raise RuntimeError(
                f"Failed to connect Aubo at {self.config.ROBOT_IP}:{self.config.ROBOT_PORT}"
            )
        self.robot.robot_startup()
        self.robot.set_collision_class(COLLISION_CLASS)
        self.robot.set_tool_dynamics_param(
            {
                "position": TOOL_DYNAMICS_POSITION,
                "payload": TOOL_PAYLOAD_KG,
                "inertia": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            }
        )
        self.robot.init_profile()
        if self.config.TELEOP_JOINT_LIMITS:
            self.robot.set_joint_maxacc(TELEOP_JOINT_MAXACC)
            self.robot.set_joint_maxvelc(TELEOP_JOINT_MAXVELC)
        else:
            self.robot.set_joint_maxacc(SAFE_JOINT_MAXACC)
            self.robot.set_joint_maxvelc(SAFE_JOINT_MAXVELC)

        wp = self.robot.get_current_waypoint()
        self._can_cmd_pos = np.array(wp["pos"], dtype=np.float64)
        self._can_cmd_rpy = np.array(
            self.robot.quaternion_to_rpy(wp["ori"]), dtype=np.float64
        )
        self._can_cmd_joint = np.array(wp["joint"], dtype=np.float64)

    def _ensure_ros_node(self) -> None:
        if self._ros_node is not None:
            return
        import rclpy
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self._ros_node = Node("aubo_hil_env")
        self._ros_spin_active = True

        def _spin() -> None:
            while self._ros_spin_active and rclpy.ok():
                rclpy.spin_once(self._ros_node, timeout_sec=0.01)

        self._ros_spin_thread = threading.Thread(target=_spin, daemon=True)
        self._ros_spin_thread.start()

    def _init_cameras(self) -> None:
        self.cap = {}
        if self.config.USE_TCP_IMAGES and self.config.TCP_IMAGE_CAMERAS:
            from image_tcp_bridge import ImageTcpBridge

            self._image_tcp_bridge = ImageTcpBridge(
                self.config.IMAGE_TCP_SSH_HOST,
                local_port=self.config.IMAGE_TCP_LOCAL_PORT,
                remote_port=self.config.IMAGE_TCP_REMOTE_PORT,
                server_script=self.config.IMAGE_TCP_SERVER_SCRIPT,
                server_topics=self.config.IMAGE_TCP_SERVER_TOPICS,
                manual_tunnel=self.config.IMAGE_TCP_MANUAL_TUNNEL,
                verbose=True,
            )
            self._image_tcp_bridge.start()
            for key in self.config.TCP_IMAGE_CAMERAS:
                tcp_cap = TcpImageCapture(key, self._image_tcp_bridge, image_key=key)
                self.cap[key] = VideoCapture(tcp_cap)
            return
        if self.config.ROS_IMAGE_CAMERAS:
            from franka_env.camera.ros_capture import RosImageCapture

            self._ensure_ros_node()
            for key, cfg in self.config.ROS_IMAGE_CAMERAS.items():
                ros_cap = RosImageCapture(name=key, ros_node=self._ros_node, **cfg)
                self.cap[key] = VideoCapture(ros_cap)
            return
        for key, cfg in self.config.REALSENSE_CAMERAS.items():
            rs = RSCapture(name=key, **cfg)
            self.cap[key] = VideoCapture(rs)

    def _init_tcp_gripper(self) -> None:
        from gripper_tcp_bridge import GripperTcpBridge

        self._gripper_tcp = GripperTcpBridge(
            self.config.GRIPPER_SSH_HOST,
            local_port=self.config.GRIPPER_TCP_LOCAL_PORT,
            remote_port=self.config.GRIPPER_TCP_REMOTE_PORT,
            manual_tunnel=self.config.GRIPPER_TCP_MANUAL_TUNNEL,
            verbose=True,
        )
        self._gripper_tcp.start()

    def _init_ros_gripper(self) -> None:
        from std_srvs.srv import SetBool

        self._ensure_ros_node()
        self._gripper_cli = self._ros_node.create_client(
            SetBool, self.config.GRIPPER_SERVICE
        )

    def trigger_gripper(self, close: bool) -> None:
        """Public API for keyboard / wrappers; gymnasium blocks _trigger_gripper on wrappers."""
        self._gripper_closed = close
        self._trigger_gripper(close)

    def _trigger_gripper(self, close: bool) -> None:
        if self._gripper_tcp is not None:
            self._gripper_tcp.send(close)
            return
        if not self.config.USE_ROS_GRIPPER or self._gripper_cli is None:
            return
        from std_srvs.srv import SetBool

        if not self._gripper_cli.service_is_ready():
            return
        req = SetBool.Request()
        req.data = close
        self._gripper_cli.call_async(req)

    def _safety_box_enabled(self) -> bool:
        return self.config.ENABLE_RELATIVE_SAFETY_BOX

    def _record_safety_center(self) -> None:
        """Record current TCP XYZ as safety sphere center; apply RPY limits."""
        self._safety_center = self._can_cmd_pos.copy()
        abs_low = self.config.ABS_POSE_LIMIT_LOW
        abs_high = self.config.ABS_POSE_LIMIT_HIGH
        self._rpy_low = abs_low[3:].copy()
        self._rpy_high = abs_high[3:].copy()

    def _clip_safety_tcp(self) -> None:
        """Clamp TCP XYZ to safety sphere; clip RPY; zero velocity on boundary hit."""
        if not self._safety_box_enabled():
            return
        pos_before = self._can_cmd_pos.copy()
        rpy_before = self._can_cmd_rpy.copy()
        # Sphere clamp XYZ
        if self._safety_center is not None:
            delta = self._can_cmd_pos - self._safety_center
            dist = np.linalg.norm(delta)
            r = self.config.SAFETY_SPHERE_RADIUS
            if dist > r:
                self._can_cmd_pos[:] = self._safety_center + delta * (r / dist)
        # Hard X floor toward operator
        if self.config.ABS_X_MIN is not None:
            if self._can_cmd_pos[0] < self.config.ABS_X_MIN:
                self._can_cmd_pos[0] = float(self.config.ABS_X_MIN)
        # RPY limits
        self._can_cmd_rpy[:] = np.clip(self._can_cmd_rpy, self._rpy_low, self._rpy_high)
        if not np.allclose(self._can_cmd_pos, pos_before) or not np.allclose(
            self._can_cmd_rpy, rpy_before
        ):
            self._vel.v_filtered[:] = 0.0

    def _sync_cartesian_from_joints(self) -> None:
        fk = self.robot.forward_kin(self._can_cmd_joint.tolist())
        self._can_cmd_pos = np.array(fk["pos"], dtype=np.float64)
        self._can_cmd_rpy = np.array(
            self.robot.quaternion_to_rpy(fk["ori"]), dtype=np.float64
        )

    def _try_apply_ik(self, cmd_ori: tuple) -> None:
        ik_res = self.robot.inverse_kin(
            self._can_cmd_joint.tolist(),
            self._can_cmd_pos.tolist(),
            cmd_ori,
        )
        if not ik_res or "joint" not in ik_res:
            self._sync_cartesian_from_joints()
            self._vel.v_filtered[:] = 0.0
            return
        new_joint = np.array(ik_res["joint"], dtype=np.float64)
        delta = float(np.linalg.norm(new_joint - self._can_cmd_joint))
        if delta > IK_MAX_JOINT_STEP:
            self._sync_cartesian_from_joints()
            self._vel.v_filtered[:] = 0.0
            return
        self._can_cmd_joint = new_joint

    def _execute_can_substep(self, gripper_cmd: float, dt: float) -> None:
        self._can_cmd_pos += self._vel.v_filtered[:3] * dt
        self._can_cmd_rpy += self._vel.v_filtered[3:] * dt
        self._clip_safety_tcp()
        cmd_ori = self.robot.rpy_to_quaternion(self._can_cmd_rpy.tolist())
        if cmd_ori is not None:
            self._try_apply_ik(cmd_ori)
        self.robot.set_waypoint_to_canbus(self._can_cmd_joint.tolist())

        if gripper_cmd > 0.5 and not self._gripper_closed:
            self._gripper_closed = True
            self._trigger_gripper(close=True)
        elif gripper_cmd < -0.5 and self._gripper_closed:
            self._gripper_closed = False
            self._trigger_gripper(close=False)

    def _integrate_velocity_teleop(
        self, keys_fn: Callable[[], tuple[set[str], bool]]
    ) -> None:
        if not self._in_canbus_mode:
            self.robot.enter_tcp2canbus_mode()
            self._in_canbus_mode = True
        for _ in range(CAN_SUBSTEPS_PER_POLICY_STEP):
            loop_start = time.time()
            active_keys, gripper_closed = keys_fn()
            self._vel.update_from_active_keys(
                active_keys, TELEOP_KEY_MAP, alpha=CAN_VEL_EMA_ALPHA
            )
            gripper_cmd = 1.0 if gripper_closed else -1.0
            self._execute_can_substep(gripper_cmd, self._vel.sub_dt)
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self._vel.sub_dt - elapsed))

    def _integrate_velocity_action(self, action: np.ndarray) -> None:
        gripper_cmd = self._vel.update_from_action(action)
        if not self._in_canbus_mode:
            self.robot.enter_tcp2canbus_mode()
            self._in_canbus_mode = True
        for _ in range(CAN_SUBSTEPS_PER_POLICY_STEP):
            loop_start = time.time()
            self._execute_can_substep(gripper_cmd, self._vel.sub_dt)
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self._vel.sub_dt - elapsed))

    def _get_images(self) -> dict:
        images = {}
        for key, cap in self.cap.items():
            rgb = cap.read_latest()
            cropped = (
                self.config.IMAGE_CROP[key](rgb)
                if key in self.config.IMAGE_CROP
                else rgb
            )
            resized = cv2.resize(
                cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
            )
            images[key] = resized[..., ::-1]
        return images

    def _gripper_openpi(self) -> float:
        return 1.0 if self._gripper_closed else 0.0

    def get_cached_proprio_state(self) -> dict[str, np.ndarray]:
        """Proprio from last CAN command — avoids extra SDK round-trip per step."""
        if self._can_cmd_pos is None:
            wp = self.robot.get_current_waypoint()
            pos = np.asarray(wp["pos"], dtype=np.float32)
            rpy = np.asarray(self.robot.quaternion_to_rpy(wp["ori"]), dtype=np.float32)
            q = np.asarray(wp["joint"], dtype=np.float32)
        else:
            pos = np.asarray(self._can_cmd_pos, dtype=np.float32)
            rpy = np.asarray(self._can_cmd_rpy, dtype=np.float32)
            q = np.asarray(self._can_cmd_joint, dtype=np.float32)
        return {
            "tcp_pose": np.concatenate([pos, rpy]),
            "tcp_vel": self._vel.v_filtered.astype(np.float32),
            "tcp_force": np.zeros(3, dtype=np.float32),
            "tcp_torque": np.zeros(3, dtype=np.float32),
            "gripper_pose": np.array([self._gripper_openpi()], dtype=np.float32),
            "q": q[:6],
            "dq": np.zeros(6, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self._step_count = 0
        self._vel.reset()
        if self.fake_env:
            return self.observation_space.sample(), {}
        wp = self.robot.get_current_waypoint()
        self._can_cmd_pos = np.array(wp["pos"], dtype=np.float64)
        self._can_cmd_rpy = np.array(
            self.robot.quaternion_to_rpy(wp["ori"]), dtype=np.float64
        )
        self._can_cmd_joint = np.array(wp["joint"], dtype=np.float64)
        if self._safety_box_enabled():
            self._record_safety_center()
            if self._safety_center is not None:
                print(
                    "[safety] sphere center="
                    f"{self._safety_center.round(3)} r={self.config.SAFETY_SPHERE_RADIUS}m",
                    flush=True,
                )
        if not self.fake_env:
            self.robot.enter_tcp2canbus_mode()
            self._in_canbus_mode = True
        return {"pixels": self._get_images()}, {}

    def step(self, action):
        step_start = time.time()
        if not self.fake_env:
            keys_fn = self._teleop_keys_fn
            if keys_fn is not None:
                self._integrate_velocity_teleop(keys_fn)
            else:
                self._integrate_velocity_action(action)
        self._step_count += 1
        time.sleep(max(0.0, self.policy_dt - (time.time() - step_start)))
        obs = (
            {"pixels": self._get_images()}
            if not self.fake_env
            else self.observation_space.sample()
        )
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.robot is not None:
            if self._in_canbus_mode:
                self.robot.leave_tcp2canbus_mode()
                self._in_canbus_mode = False
            self.robot.move_stop()
            self.robot.disconnect()
            Auboi5Robot.uninitialize()
        if self.cap is not None:
            for cap in self.cap.values():
                cap.close()
            self.cap = None
        if self._image_tcp_bridge is not None:
            self._image_tcp_bridge.stop()
            self._image_tcp_bridge = None
        if self._gripper_tcp is not None:
            self._gripper_tcp.stop()
            self._gripper_tcp = None
        if self._ros_node is not None:
            self._ros_spin_active = False
            if self._ros_spin_thread is not None:
                self._ros_spin_thread.join(timeout=1.0)
            self._ros_node.destroy_node()
            self._ros_node = None
