"""Gymnasium pick-cube sim for Aubo i10 — aligned with gym_hil PandaPickCubeGymEnv.

Mocap EE target + per-substep IK + gravity-compensated position control.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from franka_env.envs.aubo_control_config import (
    ARM_JOINT_NAMES,
    CAN_SUBSTEPS_PER_POLICY_STEP,
    CUBE_BODY_NAME,
    CUBE_FACE_UP_QUATS,
    CUBE_ON_TABLE_Z,
    GRIPPER_RH_R1_CLOSED,
    IMAGE_KEY_FRONT,
    IMAGE_KEY_WRIST,
    IMAGE_SIZE,
    MAX_ANG_VEL,
    MAX_EPISODE_LENGTH,
    MAX_LIN_VEL,
    PICKUP_BLOCK_SAMPLING_BOUNDS_XY,
    PICKUP_CUBE_XYZ,
    PICKUP_HOME_QPOS,
    PICKUP_LIFT_DELTA_Z,
    POLICY_DT,
    TABLE_TOP_Z,
    TELEOP_CAMERA,
    TELEOP_DISPLAY_SIZE,
    VEL_EMA_ALPHA,
)
from franka_env.envs.aubo_mujoco_controller import (
    MYI10TELE_ROOT,
    AuboMujocoController,
)
from franka_env.envs.aubo_velocity_control import AuboVelocityIntegrator

SENSOR_AGENTVIEW_POS = "agentview_pos"
SENSOR_WRIST_CAM_POS = "wrist_cam_pos"
DEFAULT_PICKUP_XML = os.path.join(
    MYI10TELE_ROOT, "assets/aubo_i10_inspire/pickup_scene.xml"
)

GRIPPER_CTRL_CLOSED = GRIPPER_RH_R1_CLOSED  # 0.8663
GRIPPER_CTRL_OPEN = 0.0
_BLOCK_HALF_Z = 0.025


class AuboPickCubeGymEnv(gym.Env):
    """Pick up cube below TCP — Aubo i10 + Inspire gripper in MuJoCo."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        render_mode: str = "rgb_array",
        image_size: int = IMAGE_SIZE,
        max_episode_steps: int = MAX_EPISODE_LENGTH,
        max_lin_vel: float = MAX_LIN_VEL,
        max_ang_vel: float = MAX_ANG_VEL,
        policy_dt: float = POLICY_DT,
        can_substeps: int = CAN_SUBSTEPS_PER_POLICY_STEP,
        vel_alpha: float = VEL_EMA_ALPHA,
        random_block_position: bool = False,
        random_block_orientation: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        del seed
        if xml_path is None:
            xml_path = DEFAULT_PICKUP_XML
        self._ctrl = AuboMujocoController(xml_path, ARM_JOINT_NAMES)
        self._model = self._ctrl.model
        self._data = self._ctrl.data

        # Cube freejoint
        self._cube_jnt_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "cube"
        )
        if self._cube_jnt_id < 0:
            raise RuntimeError("pickup_scene.xml must define freejoint named 'cube'")
        self._cube_qpos_adr = int(self._model.jnt_qposadr[self._cube_jnt_id])

        # Gripper
        self._rh_r1_jid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "rh_r1"
        )
        self._rh_r1_qpos_adr = int(self._model.jnt_qposadr[self._rh_r1_jid])

        # Mocap + pinch site (like panda mocap target + pinch_site_id)
        self._mocap_id = int(self._model.body("ee_target").mocapid[0])
        self._pinch_site_id = int(self._model.site("ee_site").id)

        # Policy cameras via MJCF sensors (like cube_pos)
        self._camera_front_id = int(self._model.sensor(SENSOR_AGENTVIEW_POS).objid[0])
        self._camera_wrist_id = int(self._model.sensor(SENSOR_WRIST_CAM_POS).objid[0])

        # Block sampling
        self._random_block_position = random_block_position
        self._random_block_orientation = random_block_orientation
        self._block_rng: np.random.Generator | None = None

        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.policy_dt = policy_dt
        self._step_count = 0
        self._vel = AuboVelocityIntegrator(
            max_lin_vel=max_lin_vel,
            max_ang_vel=max_ang_vel,
            vel_alpha=vel_alpha,
            policy_dt=policy_dt,
            can_substeps=can_substeps,
        )
        self.physics_dt = float(self._model.opt.timestep)
        self.sub_dt = policy_dt / can_substeps
        self.nstep_per_substep = max(1, int(round(self.sub_dt / self.physics_dt)))

        self._model.vis.global_.offwidth = max(image_size, TELEOP_DISPLAY_SIZE)
        self._model.vis.global_.offheight = max(image_size, TELEOP_DISPLAY_SIZE)
        self._renderer = mujoco.Renderer(self._model, image_size, image_size)
        self._teleop_renderer = mujoco.Renderer(
            self._model, TELEOP_DISPLAY_SIZE, TELEOP_DISPLAY_SIZE
        )

        self._z_init = CUBE_ON_TABLE_Z
        self._z_success = self._z_init + PICKUP_LIFT_DELTA_Z
        self._gripper_closed = True

        # Observation space: pixels + agent_pos (aligned with PandaPickCubeGymEnv)
        agent_dim = 6 + 6 + 1 + 3  # qpos(6) qvel(6) gripper(1) tcp_xyz(3)
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        IMAGE_KEY_FRONT: spaces.Box(
                            0, 255, (image_size, image_size, 3), dtype=np.uint8
                        ),
                        IMAGE_KEY_WRIST: spaces.Box(
                            0, 255, (image_size, image_size, 3), dtype=np.uint8
                        ),
                    }
                ),
                "agent_pos": spaces.Box(
                    -np.inf, np.inf, (agent_dim,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    # -- Accessors --------------------------------------------------------

    @property
    def myenv(self) -> AuboMujocoController:
        return self._ctrl

    @property
    def velocity_integrator(self) -> AuboVelocityIntegrator:
        return self._vel

    # -- Cube helpers -----------------------------------------------------

    def _set_cube_pose(self, xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
        adr = self._cube_qpos_adr
        self._data.qpos[adr : adr + 3] = xyz
        self._data.qpos[adr + 3 : adr + 7] = quat_wxyz
        self._data.qvel[adr : adr + 6] = 0.0

    def _cube_center_z_for_quat(self, quat_wxyz: np.ndarray) -> float:
        """Body-center Z so the lowest cube corner rests on TABLE_TOP_Z."""
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat_wxyz)
        support = _BLOCK_HALF_Z * np.sum(np.abs(mat.reshape(3, 3)[2, :]))
        return float(TABLE_TOP_Z + support)

    def _sample_block_quat(self) -> np.ndarray:
        if self._block_rng is None:
            self._block_rng = np.random.default_rng()
        face_quat = CUBE_FACE_UP_QUATS[
            int(self._block_rng.integers(len(CUBE_FACE_UP_QUATS)))
        ]
        yaw = float(self._block_rng.uniform(0.0, 2.0 * np.pi))
        yaw_quat = np.array(
            [np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float64
        )
        out = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(out, yaw_quat, face_quat)
        return out

    def _sample_block_xy(self) -> np.ndarray:
        low, high = PICKUP_BLOCK_SAMPLING_BOUNDS_XY
        if self._block_rng is None:
            self._block_rng = np.random.default_rng()
        return self._block_rng.uniform(low, high)

    def _cube_pos(self) -> np.ndarray:
        return self._data.body(CUBE_BODY_NAME).xpos.copy()

    def _cube_z(self) -> float:
        return float(self._data.body(CUBE_BODY_NAME).xpos[2])

    # -- Robot state (like panda get_robot_state) -------------------------

    def get_robot_state(self) -> np.ndarray:
        qpos = self._ctrl.env.get_qpos_joints(joint_names=list(ARM_JOINT_NAMES)).astype(
            np.float32
        )
        qvel = self._data.qvel[self._ctrl.arm_dof_ids].astype(np.float32)
        gripper_pose = self.get_gripper_pose()
        tcp_pos = self._data.site_xpos[self._pinch_site_id].astype(np.float32)
        return np.concatenate([qpos, qvel, gripper_pose, tcp_pos])

    def get_gripper_pose(self) -> np.ndarray:
        openpi = float(self._data.qpos[self._rh_r1_qpos_adr]) / GRIPPER_RH_R1_CLOSED
        return np.array([openpi], dtype=np.float32)

    # -- Ctrl + gripper (MyI10Tele: latch + step_env every tick, no qpos snap) ---

    def _apply_ctrl_target(self) -> None:
        nctrl = int(self._model.nu)
        ctrl = np.zeros(nctrl, dtype=np.float64)
        ctrl[self._ctrl.arm_actuator_ids] = (
            self._ctrl.arm_ctrl + self._ctrl.gravity_torque * self._ctrl._arm_kp_inv
        )
        ctrl[self._ctrl.gripper_actuator_id] = self._ctrl.gripper_cmd
        self._data.ctrl[:] = ctrl

    def _snap_gripper_qpos(self, rh_r1_val: float) -> None:
        """Reset-only: sync mimic joints before first mj_step."""
        self._data.qpos[self._rh_r1_qpos_adr] = rh_r1_val
        for name, sign in [("rh_r2", -1), ("rh_l1", -1), ("rh_l2", 1)]:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._data.qpos[self._model.jnt_qposadr[jid]] = sign * rh_r1_val

    def _reset_robot(self) -> None:
        """Like panda reset_robot: set home qpos + sync mocap to TCP."""
        q_home = np.concatenate(
            [
                PICKUP_HOME_QPOS.copy(),
                np.array([GRIPPER_RH_R1_CLOSED], dtype=np.float64),
            ]
        )
        self._ctrl.env.forward(
            q=q_home,
            joint_names=list(ARM_JOINT_NAMES) + ["rh_r1"],
            increase_tick=False,
        )
        self._snap_gripper_qpos(GRIPPER_RH_R1_CLOSED)
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        self._ctrl.arm_ctrl = PICKUP_HOME_QPOS.copy()
        self._ctrl.gripper_ctrl = GRIPPER_CTRL_CLOSED
        self._ctrl.compute_q = PICKUP_HOME_QPOS.copy()
        self._ctrl.last_q = PICKUP_HOME_QPOS.copy()
        self._ctrl._compute_gravity_torque_at_current_pose()
        self._apply_ctrl_target()
        self._ctrl.sync_flange_frame_from_sim()

        # Sync mocap to TCP (like panda mocap_pos[0] = tcp_pos)
        self._data.mocap_pos[self._mocap_id] = self._ctrl.p0.copy()

    # -- Rendering --------------------------------------------------------

    def _render_camera(self, cam: str | int, renderer: mujoco.Renderer) -> np.ndarray:
        renderer.update_scene(self._data, camera=cam)
        rgb = renderer.render()
        h = w = renderer.height
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.uint8)

    def render_teleop_view(self) -> np.ndarray:
        return self._render_camera(TELEOP_CAMERA, self._teleop_renderer)

    def _compute_observation(self) -> dict:
        front = self._render_camera(self._camera_front_id, self._renderer)
        wrist = self._render_camera(self._camera_wrist_id, self._renderer)
        return {
            "pixels": {
                IMAGE_KEY_FRONT: front,
                IMAGE_KEY_WRIST: wrist,
            },
            "agent_pos": self.get_robot_state(),
        }

    # -- Per-substep IK (like panda per-substep OSC) ----------------------

    def trigger_gripper(self, close: bool) -> None:
        """Public API for keyboard / wrappers; applies latch without waiting for step()."""
        self._gripper_closed = close
        self._apply_gripper_target()

    def _update_gripper_latch(self, action: np.ndarray) -> None:
        """Update latch from teleop action[6]: +1=close, -1=open, 0=hold."""
        g = float(action[6])
        if abs(g) < 0.05:
            return
        if g > 0.5:
            self._gripper_closed = True
        elif g < -0.5:
            self._gripper_closed = False

    def _gripper_openpi(self) -> float:
        return 1.0 if self._gripper_closed else 0.0

    def _apply_gripper_target(self) -> None:
        self._ctrl.gripper_ctrl = (
            GRIPPER_CTRL_CLOSED if self._gripper_closed else GRIPPER_CTRL_OPEN
        )

    def _advance_physics_hold(self) -> None:
        """MyI10Tele step_env(): hold arm qpos + gripper servo every tick."""
        self._apply_gripper_target()
        self._ctrl.arm_ctrl = np.asarray(
            self._ctrl.env.get_qpos_joints(joint_names=list(ARM_JOINT_NAMES)),
            dtype=np.float64,
        )
        self._ctrl._compute_gravity_torque_at_current_pose()
        for _ in range(self._vel.can_substeps):
            self._ctrl.step_env(nstep=self.nstep_per_substep)

    def _integrate_velocity_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(7)
        vel_action = np.concatenate([action[:6], np.zeros(1)])
        self._vel.update_from_action(vel_action)
        gripper_openpi = self._gripper_openpi()

        for _ in range(self._vel.can_substeps):
            sub_delta = self._vel.velocity_to_delta(0.0, self._vel.sub_dt)
            sub_delta = np.concatenate([sub_delta[:6], [gripper_openpi]])

            self._data.mocap_pos[self._mocap_id] += sub_delta[:3]
            self._ctrl.step(sub_delta, gravity_comp=True)
            self._ctrl.step_env(nstep=self.nstep_per_substep)

    # -- Core gym API -----------------------------------------------------

    def step(self, action):
        step_start = time.time()
        action = np.asarray(action, dtype=np.float32).reshape(7)
        self._update_gripper_latch(action)

        if np.allclose(action[:6], 0.0, atol=1e-6):
            self._advance_physics_hold()
        else:
            self._integrate_velocity_action(action)

        self._step_count += 1
        time.sleep(max(0.0, self.policy_dt - (time.time() - step_start)))

        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()
        terminated = bool(success or self._block_out_of_bounds())
        truncated = self._step_count >= self.max_episode_steps

        info: dict = {}
        if success:
            info["succeed"] = True
        return obs, rew, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        if seed is not None:
            super().reset(seed=seed)
            self._block_rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)
        self._ctrl.env.tick = 0

        self._reset_robot()

        # Place cube (random or fixed, like panda)
        if self._random_block_position:
            block_xy = self._sample_block_xy()
        else:
            block_xy = PICKUP_CUBE_XYZ[:2]
        if self._random_block_orientation:
            quat = self._sample_block_quat()
            block_z = self._cube_center_z_for_quat(quat)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            block_z = CUBE_ON_TABLE_Z
        self._set_cube_pose(np.array([block_xy[0], block_xy[1], block_z]), quat)
        mujoco.mj_forward(self._model, self._data)

        # Cache initial block Z for lift threshold
        self._z_init = self._cube_z()
        self._z_success = self._z_init + PICKUP_LIFT_DELTA_Z

        self._step_count = 0
        self._vel.reset()
        self._gripper_closed = True
        return self._compute_observation(), {}

    # -- Reward & success (aligned with PandaPickCubeGymEnv) ---------------

    def _compute_reward(self) -> float:
        lift = self._cube_z() - self._z_init
        return float(lift > PICKUP_LIFT_DELTA_Z)

    def _is_success(self) -> bool:
        cube_pos = self._cube_pos()
        tcp_pos = self._data.site_xpos[self._pinch_site_id]
        dist = float(np.linalg.norm(cube_pos - tcp_pos))
        lift = self._cube_z() - self._z_init
        return dist < 0.05 and lift > PICKUP_LIFT_DELTA_Z

    def _block_out_of_bounds(self) -> bool:
        low, high = PICKUP_BLOCK_SAMPLING_BOUNDS_XY
        cube_xy = self._cube_pos()[:2]
        margin = 0.08
        return bool(
            np.any(cube_xy < (low - margin)) or np.any(cube_xy > (high + margin))
        )

    def close(self):
        for renderer in (self._renderer, self._teleop_renderer):
            if renderer is not None and hasattr(renderer, "close"):
                renderer.close()
        self._renderer = None
        self._teleop_renderer = None
