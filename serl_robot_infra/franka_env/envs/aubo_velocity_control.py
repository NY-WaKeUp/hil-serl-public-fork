"""Velocity integration shared by Aubo sim and real envs (smooth_teleop semantics)."""

import numpy as np

from franka_env.envs.aubo_control_config import (
    CAN_SUBSTEPS_PER_POLICY_STEP,
    CAN_VEL_EMA_ALPHA,
    MAX_ANG_VEL,
    MAX_LIN_VEL,
    POLICY_DT,
    VEL_EMA_ALPHA,
)


class AuboVelocityIntegrator:
    """EMA-filtered Cartesian velocity, integrated over CAN substeps."""

    def __init__(
        self,
        max_lin_vel: float = MAX_LIN_VEL,
        max_ang_vel: float = MAX_ANG_VEL,
        vel_alpha: float = VEL_EMA_ALPHA,
        policy_dt: float = POLICY_DT,
        can_substeps: int = CAN_SUBSTEPS_PER_POLICY_STEP,
    ):
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.vel_alpha = vel_alpha
        self.policy_dt = policy_dt
        self.can_substeps = can_substeps
        self.v_filtered = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        self.v_filtered[:] = 0.0

    @property
    def sub_dt(self) -> float:
        return self.policy_dt / self.can_substeps

    def update_from_action(self, action: np.ndarray) -> float:
        """Map normalized 7D action to filtered velocity; return gripper command in [-1, 1]."""
        action = np.asarray(action, dtype=np.float64).reshape(7)
        v_target = np.zeros(6, dtype=np.float64)
        v_target[:3] = action[:3] * self.max_lin_vel
        v_target[3:] = action[3:6] * self.max_ang_vel
        self.v_filtered = (
            self.vel_alpha * v_target + (1.0 - self.vel_alpha) * self.v_filtered
        )
        return float(action[6])

    def velocity_to_delta(self, gripper_cmd: float, dt: float) -> np.ndarray:
        """EE delta for one CAN substep (OpenPI gripper on last dim)."""
        dpos = self.v_filtered[:3] * dt
        drpy = self.v_filtered[3:] * dt
        gripper_openpi = (gripper_cmd + 1.0) * 0.5
        return np.concatenate([dpos, drpy, [gripper_openpi]], dtype=np.float64)

    def update_from_active_keys(
        self,
        active_keys: set[str],
        key_map: dict,
        *,
        alpha: float | None = None,
    ) -> None:
        """EMA velocity from held keys (smooth_teleop semantics at CAN rate)."""
        a = self.vel_alpha if alpha is None else alpha
        v_target = np.zeros(6, dtype=np.float64)
        for key in active_keys:
            axis, sign = key_map[key]
            vel_mag = self.max_lin_vel if axis < 3 else self.max_ang_vel
            v_target[axis] += sign * vel_mag
        self.v_filtered = a * v_target + (1.0 - a) * self.v_filtered

    def normalized_from_keys(
        self,
        active_keys: set[str],
        key_map: dict,
    ) -> np.ndarray:
        """Build normalized 7D velocity action; gripper channel left at 0 (hold)."""
        self.update_from_active_keys(active_keys, key_map)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = self.v_filtered[:3] / self.max_lin_vel
        action[3:6] = self.v_filtered[3:] / self.max_ang_vel
        return np.clip(action, -1.0, 1.0)
