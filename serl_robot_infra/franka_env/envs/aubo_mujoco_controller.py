"""Native MuJoCo EE-delta controller for Aubo sim (Route B — no MyEnv).

Gravity-compensated position control (ctrl = q_target + g/kp).
"""

import os
import sys
import types
from typing import Optional, Sequence

import mujoco
import numpy as np

MYI10TELE_ROOT = os.environ.get("MYI10TELE_ROOT", os.path.expanduser("~/MyI10Tele"))
for _path in (os.path.join(MYI10TELE_ROOT, "src"), MYI10TELE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if "tkinter" not in sys.modules:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["tkinter"] = types.ModuleType("tkinter")

from core.my_env import (  # noqa: E402
    gripper_qpos_to_openpi,
)
from utils.MujocoParser import MuJoCoParserClass  # noqa: E402
from utils.transforms import rpy2r  # noqa: E402
from utils.utils import solve_ik  # noqa: E402

# Gripper: rh_r1 position actuator, joint range [0, 0.8663].
# openpi=1.0 (closed) → ctrl=0.8663; openpi=0.0 (open) → ctrl=0.0.
GRIPPER_RANGE_MAX = 0.8663


class AuboMujocoController:
    """Delta EE pose + gravity-compensated gripper via IK."""

    def __init__(
        self,
        xml_path: str,
        joint_names: Sequence[str],
        *,
        verbose: bool = False,
    ):
        self.joint_names = tuple(joint_names)
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        xml_base = os.path.basename(xml_path)
        os.chdir(xml_dir)
        self.env = MuJoCoParserClass(
            name="aubo_pickup",
            rel_xml_path=xml_base,
            verbose=verbose,
        )
        self.model = self.env.model
        self.data = self.env.data
        self.physics_dt = float(self.model.opt.timestep)
        # Parser __init__ calls reset(step=True); avoid one mj_step at default pose.
        self.env.reset(step=False)

        # Arm DOF indices (for gravity compensation via qfrc_bias)
        dof_joint_names = (
            joint_names  # 6 arm joints; gripper joints are unactuated/mimicked
        )
        self.arm_dof_ids: np.ndarray = np.array(
            [self.model.joint(name).dofadr[0] for name in dof_joint_names],
            dtype=np.int32,
        )

        # Actuator indices: 6 arm position servos + 1 gripper position servo
        # MyI10Tele naming: {joint_name}_servo
        self.arm_actuator_ids = np.array(
            [self.model.actuator(f"{name}_servo").id for name in self.joint_names],
            dtype=np.int32,
        )
        self.gripper_actuator_id = int(self.model.actuator("rh_r1_servo").id)

        # Position actuator kp (gainprm[0]) for gravity compensation scaling.
        arm_kp = np.array(
            [self.model.actuator_gainprm[i, 0] for i in self.arm_actuator_ids],
            dtype=np.float64,
        )
        # Clamp near-zero kp to avoid division blowup; default 1.0 should never trigger here.
        arm_kp[arm_kp < 1e-3] = 1.0
        self._arm_kp_inv = 1.0 / arm_kp

        self.p0 = np.zeros(3, dtype=np.float64)
        self.R0 = np.eye(3, dtype=np.float64)
        self.arm_ctrl = np.zeros(len(self.joint_names), dtype=np.float64)
        self.gravity_torque = np.zeros(len(self.joint_names), dtype=np.float64)
        self.gripper_cmd = GRIPPER_RANGE_MAX  # closed default
        self.compute_q = np.zeros(len(self.joint_names), dtype=np.float64)
        self.last_q = np.zeros(len(self.joint_names), dtype=np.float64)

    def sync_flange_frame_from_sim(self) -> None:
        p0, R0 = self.env.get_pR_body(body_name="i10_inspire_flange_link")
        self.p0 = np.asarray(p0, dtype=np.float64).copy()
        self.R0 = np.asarray(R0, dtype=np.float64).copy()

    def _compute_gravity_torque_at_current_pose(self) -> None:
        """Compute gravity torques for the current arm qpos via mj_rne(qacc=0)."""
        data = self.data
        data.qacc[:] = 0.0
        qfrc = np.zeros(self.model.nv, dtype=np.float64)
        mujoco.mj_rne(self.model, data, 0, qfrc)
        self.gravity_torque = qfrc[self.arm_dof_ids].copy()

    def step(self, action: np.ndarray, *, gravity_comp: bool = True) -> np.ndarray:
        """IK + optional gravity compensation. Returns joint state (arm q + gripper openpi)."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        q = self.env.get_qpos_joints(joint_names=self.joint_names)
        self.p0 += action[:3]
        self.R0 = self.R0.dot(rpy2r(action[3:6]))
        q_ik, _, _ = solve_ik(
            env=self.env,
            joint_names_for_ik=list(self.joint_names),
            body_name_trgt="i10_inspire_flange_link",
            q_init=q,
            p_trgt=self.p0,
            R_trgt=self.R0,
            max_ik_tick=50,
            ik_stepsize=1.0,
            ik_eps=1e-2,
            ik_th=np.radians(5.0),
            render=False,
            verbose_warning=False,
        )
        self.compute_q = q_ik
        self.last_q = q.copy()
        self.arm_ctrl = q_ik.copy()

        if gravity_comp:
            self._compute_gravity_torque_at_current_pose()
        else:
            self.gravity_torque[:] = 0.0

        # Map gripper openpi to joint position [0, GRIPPER_RANGE_MAX]
        gripper_openpi = action[-1]
        self.gripper_cmd = float(np.clip(gripper_openpi, 0.0, 1.0) * GRIPPER_RANGE_MAX)

        return self.get_joint_state()

    def step_env(self, *, nstep: int = 1) -> None:
        """Apply gravity-compensated arm ctrl + gripper ctrl, advance nstep physics steps.

        Position actuator:  f = kp * (ctrl - q).  To cancel gravity g:
            ctrl = q_target + g / kp
        so that  f = kp * (q_target + g/kp - q) = kp * (q_target - q) + g.
        """
        nctrl = int(self.model.nu)
        ctrl = np.zeros(nctrl, dtype=np.float64)
        ctrl[self.arm_actuator_ids] = (
            self.arm_ctrl + self.gravity_torque * self._arm_kp_inv
        )
        ctrl[self.gripper_actuator_id] = self.gripper_cmd
        self.env.step(ctrl=ctrl, nstep=nstep)

    def get_joint_state(self) -> np.ndarray:
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper_openpi = gripper_qpos_to_openpi(
            float(self.env.get_qpos_joint("rh_r1")[0])
        )
        return np.concatenate([qpos, [gripper_openpi]], dtype=np.float32)

    @property
    def gripper_ctrl(self) -> float:
        """Gripper actuator ctrl (joint position in [0, GRIPPER_RANGE_MAX])."""
        return self.gripper_cmd

    @gripper_ctrl.setter
    def gripper_ctrl(self, value: float) -> None:
        self.gripper_cmd = float(np.clip(value, 0.0, GRIPPER_RANGE_MAX))
