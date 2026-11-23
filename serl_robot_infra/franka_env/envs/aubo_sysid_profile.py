"""Open-loop velocity profile for Aubo sim vs real system identification."""

from dataclasses import dataclass

import numpy as np

from franka_env.envs.aubo_control_config import MAX_ANG_VEL, MAX_LIN_VEL, VEL_EMA_ALPHA


@dataclass(frozen=True)
class AuboSysidProfile:
    """Fixed 7D normalized velocity sequence @ 10 Hz (same on sim and real)."""

    name: str
    actions: np.ndarray  # (T, 7) in [-1, 1]
    max_lin_vel: float = MAX_LIN_VEL
    max_ang_vel: float = MAX_ANG_VEL
    vel_ema_alpha: float = VEL_EMA_ALPHA
    policy_hz: float = 10.0

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_steps / self.policy_hz


def _hold_steps(n: int, action: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(action, dtype=np.float32).reshape(1, 7), (n, 1))


def _axis_step(axis: int, sign: float, gripper: float = 1.0) -> np.ndarray:
    a = np.zeros(7, dtype=np.float32)
    a[axis] = sign
    a[6] = gripper
    return a


def build_default_sysid_profile() -> AuboSysidProfile:
    """Multi-axis excitation: settle → XYZ → roll → gripper → settle.

    Each motion block is long enough for EMA (alpha=0.08) to reach ~steady speed.
    Gripper: action[6] +1=close, -1=open (hil-serl / smooth_teleop convention).
    """
    segments: list[np.ndarray] = [
        _hold_steps(20, np.zeros(7, dtype=np.float32)),  # settle @ zero velocity
        _hold_steps(30, _axis_step(0, 1.0)),  # +X
        _hold_steps(10, np.zeros(7, dtype=np.float32)),
        _hold_steps(30, _axis_step(1, -1.0)),  # -Y (matches key "d")
        _hold_steps(10, np.zeros(7, dtype=np.float32)),
        _hold_steps(30, _axis_step(2, 1.0)),  # +Z
        _hold_steps(10, np.zeros(7, dtype=np.float32)),
        _hold_steps(25, _axis_step(3, 1.0)),  # +roll
        _hold_steps(10, np.zeros(7, dtype=np.float32)),
        _hold_steps(15, np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)),  # open
        _hold_steps(15, np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)),  # close
        _hold_steps(20, np.zeros(7, dtype=np.float32)),
    ]
    actions = np.concatenate(segments, axis=0)
    return AuboSysidProfile(name="default_v1", actions=actions)


DEFAULT_SYSID_PROFILE = build_default_sysid_profile()
