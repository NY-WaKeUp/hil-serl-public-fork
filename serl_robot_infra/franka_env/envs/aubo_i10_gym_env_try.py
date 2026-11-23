"""Gymnasium pick-cube sim for Aubo i10 — native MuJoCo (Route B).

Reset sets qpos/ctrl directly (no long settle loops). Policy obs uses agentview +
wrist_cam; teleop_view is for operator display only.
"""

import os
import time
from typing import Literal, Optional

import cv2
from gym_hil import MujocoGymEnv, mujoco_gym_env
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gym_hil.mujoco_gym_env import GymRenderingSpec

from franka_env.envs.aubo_control_config import (
    ARM_JOINT_NAMES,
    CAN_SUBSTEPS_PER_POLICY_STEP,
    CUBE_BODY_NAME,
    CUBE_ON_TABLE_Z,
    GRIPPER_RH_R1_CLOSED,
    IMAGE_KEY_FRONT,
    IMAGE_KEY_WRIST,
    IMAGE_SIZE,
    MAX_ANG_VEL,
    MAX_LIN_VEL,
    PICKUP_CUBE_XYZ,
    PICKUP_HOME_QPOS,
    PICKUP_LIFT_DELTA_Z,
    POLICY_DT,
    TELEOP_CAMERA,
    TELEOP_DISPLAY_SIZE,
    VEL_EMA_ALPHA,
)
from franka_env.envs.aubo_velocity_control import AuboVelocityIntegrator

CAMERA_FRONT = "agentview"
CAMERA_WRIST = "wrist_cam"


class AuboGymEnv(MujocoGymEnv):
    """Base class for Aubo i10 robot environments."""

    def __init__(
        self,
        xml_path: Optional[str] = None,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        image_obs: bool = False,
        home_position: np.ndarray = np.asarray(
            (0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4)
        ),  # noqa: B08
        cartesian_bounds: np.ndarray = np.asarray(
            [[0.2, -0.3, 0], [0.6, 0.3, 0.5]]
        ),  # noqa: B08
    ):
        if xml_path is None:
            xml_path = DEFAULT_PICKUP_XML

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_mode=render_mode,
            render_spec=render_spec,
            image_obs=image_obs,
        )
        self._home_position = home_position
        self._cartesian_bounds = cartesian_bounds

        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }
        self.render_mode = render_mode
        self.image_obs = image_obs

        # Setup cameras
        camera_name_1 = CAMERA_FRONT
        camera_name_2 = CAMERA_WRIST
        camera_id_1 = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1
        )
        camera_id_2 = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2
        )
        self.camera_id = (camera_id_1, camera_id_2)

        # Cache robot IDs
        self._aubo_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._aubo_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self._pinch_site_id = self._model.site("pinch").id

    @property
    def velocity_integrator(self) -> AuboVelocityIntegrator:
        return self._vel

    def _set_cube_pose(self, xyz: np.ndarray) -> None:
        adr = self._cube_qpos_adr
        self._data.qpos[adr : adr + 3] = xyz
        self._data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])

    def _apply_ctrl_target(self, q_target: np.ndarray) -> None:
        self._data.ctrl[self._ctrl_idxs] = q_target

    def _snap_gripper(self, *, closed: bool) -> None:
        target = GRIPPER_RH_R1_CLOSED if closed else 0.0
        self._data.qpos[self._rh_r1_qpos_adr] = target
        arm_q = np.asarray(
            self._ctrl.env.get_qpos_joints(joint_names=list(ARM_JOINT_NAMES)),
            dtype=np.float64,
        )
        self._ctrl.q = np.concatenate([arm_q, np.array([target], dtype=np.float64)])
        self._ctrl.compute_q = arm_q.copy()
        self._apply_ctrl_target(self._ctrl.q)
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def _reset_pickup_scene(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        self._ctrl.env.tick = 0
        q_home = np.concatenate(
            [
                PICKUP_HOME_QPOS.copy(),
                np.array([GRIPPER_RH_R1_CLOSED], dtype=np.float64),
            ]
        )
        self._ctrl.env.forward(
            q=q_home,
            joint_names=_ARM_GRIPPER_JOINTS,
            increase_tick=False,
        )
        self._set_cube_pose(PICKUP_CUBE_XYZ)
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        self._ctrl.q = q_home.copy()
        self._ctrl.compute_q = PICKUP_HOME_QPOS.copy()
        self._ctrl.last_q = PICKUP_HOME_QPOS.copy()
        self._apply_ctrl_target(self._ctrl.q)
        self._ctrl.sync_flange_frame_from_sim()

    def _render_camera(self, cam_name: str, renderer: mujoco.Renderer) -> np.ndarray:
        renderer.update_scene(self._data, camera=cam_name)
        rgb = renderer.render()
        h = w = renderer.height
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.uint8)

    def render_teleop_view(self) -> np.ndarray:
        """Scene -X overview for operator display (not in policy obs)."""
        return self._render_camera(TELEOP_CAMERA, self._teleop_renderer)

    def _build_observation(self) -> dict:
        return {
            "pixels": {
                IMAGE_KEY_FRONT: self._render_camera(CAMERA_FRONT, self._renderer),
                IMAGE_KEY_WRIST: self._render_camera(CAMERA_WRIST, self._renderer),
            },
        }

    def _integrate_velocity_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(7)
        if self._gripper_wants_closed(action) is None:
            gripper_openpi = (
                float(self._data.qpos[self._rh_r1_qpos_adr]) / GRIPPER_RH_R1_CLOSED
            )
            gripper_cmd = gripper_openpi * 2.0 - 1.0
            vel_action = np.concatenate([action[:6], np.array([gripper_cmd])])
        else:
            gripper_cmd = float(action[6])
            vel_action = action
        self._vel.update_from_action(vel_action)
        gripper_openpi = (gripper_cmd + 1.0) * 0.5
        for _ in range(self._vel.can_substeps):
            delta = self._vel.velocity_to_delta(gripper_cmd, self._vel.sub_dt)
            delta = delta.copy()
            delta[-1] = gripper_openpi
            self._ctrl.step(delta)
            self._ctrl.step_env()

    @staticmethod
    def _gripper_wants_closed(action: np.ndarray) -> bool | None:
        """None = hold current gripper (gym_hil-style, no grasp delta)."""
        if action[6] > 0.5:
            return True
        if action[6] < -0.5:
            return False
        return None

    def step(self, action):
        step_start = time.time()
        action = np.asarray(action, dtype=np.float32).reshape(7)
        arm_idle = bool(np.allclose(action[:6], 0.0, atol=1e-6))
        want_closed = self._gripper_wants_closed(action)

        if arm_idle:
            if want_closed is not None and want_closed != self._gripper_closed:
                self._snap_gripper(closed=want_closed)
                self._gripper_closed = want_closed
        else:
            self._integrate_velocity_action(action)
            if want_closed is not None:
                self._gripper_closed = want_closed
            elif action[6] != 0.0:
                self._gripper_closed = bool(action[6] > 0.5)

        self._step_count += 1
        time.sleep(max(0.0, self.policy_dt - (time.time() - step_start)))
        obs = self._build_observation()
        success = self._check_pickup_success()
        reward = 1.0 if success else 0.0
        terminated = success
        truncated = self._step_count >= self.max_episode_steps
        info: dict = {}
        if success:
            info["succeed"] = True
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self._reset_pickup_scene()
        self._step_count = 0
        self._vel.reset()
        self._gripper_closed = True
        self._cube_z0 = self._cube_z()
        self._pickup_z_thresh = self._cube_z0 + PICKUP_LIFT_DELTA_Z
        return self._build_observation(), {}

    def _cube_z(self) -> float:
        return float(self._data.body(CUBE_BODY_NAME).xpos[2])

    def _check_pickup_success(self) -> bool:
        return self._cube_z() > self._pickup_z_thresh and self._gripper_closed

    def close(self):
        for renderer in (self._renderer, self._teleop_renderer):
            if renderer is not None and hasattr(renderer, "close"):
                renderer.close()
        self._renderer = None
        self._teleop_renderer = None
