"""hil-serl wrappers for Aubo i10 — aligned with gym_hil where possible."""

import time
from typing import Callable, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from franka_env.envs.aubo_control_config import (
    GRIPPER_CLOSE_KEY,
    GRIPPER_OPEN_KEY,
    MAX_ANG_VEL,
    MAX_LIN_VEL,
    TELEOP_KEY_MAP,
    VEL_EMA_ALPHA,
)
from franka_env.envs.aubo_velocity_control import AuboVelocityIntegrator
from franka_env.envs.gym_hil_wrappers import GymHilCameraDisplayWrapper

HIL_ACTION_DIM = 7


class AuboKeyboardController:
    """Velocity keyboard teleop; gripper latch on [ / ] like MyI10Tele 0.tele.py."""

    def __init__(
        self,
        max_lin_vel: float = MAX_LIN_VEL,
        max_ang_vel: float = MAX_ANG_VEL,
        vel_ema_alpha: float = VEL_EMA_ALPHA,
    ):
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self._vel = AuboVelocityIntegrator(
            max_lin_vel=max_lin_vel,
            max_ang_vel=max_ang_vel,
            vel_alpha=vel_ema_alpha,
        )
        self.active_keys: set[str] = set()
        self.gripper_closed = True  # latch: toggled once by [ / ], like MyI10Tele
        self.episode_end_status = None
        self.key_states = {"intervention": False}
        self.listener = None
        self._gripper_fn: Callable[[bool], None] | None = None

    def start(self):
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.enter:
                self.episode_end_status = "success"
                return
            if key == keyboard.Key.esc:
                self.episode_end_status = "failure"
                return
            if key == keyboard.Key.space:
                self.key_states["intervention"] = not self.key_states["intervention"]
                return
            if not hasattr(key, "char") or key.char is None:
                return
            ch = key.char
            c = ch.lower()
            if c in ("=", "+"):
                self.max_lin_vel = min(0.1, self.max_lin_vel + 0.005)
                self.max_ang_vel = min(0.5, self.max_ang_vel + 0.02)
                self._vel.max_lin_vel = self.max_lin_vel
                self._vel.max_ang_vel = self.max_ang_vel
                print(
                    f"\n[+] max_lin_vel={self.max_lin_vel:.3f} m/s "
                    f"max_ang_vel={self.max_ang_vel:.3f} rad/s",
                    flush=True,
                )
                return
            if c in ("-", "_"):
                self.max_lin_vel = max(0.002, self.max_lin_vel - 0.005)
                self.max_ang_vel = max(0.01, self.max_ang_vel - 0.02)
                self._vel.max_lin_vel = self.max_lin_vel
                self._vel.max_ang_vel = self.max_ang_vel
                print(
                    f"\n[-] max_lin_vel={self.max_lin_vel:.3f} m/s "
                    f"max_ang_vel={self.max_ang_vel:.3f} rad/s",
                    flush=True,
                )
                return
            if ch == GRIPPER_CLOSE_KEY:
                self.gripper_closed = True
                print("Gripper close", flush=True)
                if self._gripper_fn is not None:
                    self._gripper_fn(True)
                return
            if ch == GRIPPER_OPEN_KEY:
                self.gripper_closed = False
                print("Gripper open", flush=True)
                if self._gripper_fn is not None:
                    self._gripper_fn(False)
                return
            if c in TELEOP_KEY_MAP:
                self.active_keys.add(c)

        def on_release(key):
            if not hasattr(key, "char") or key.char is None:
                return
            c = key.char.lower()
            if c in self.active_keys:
                self.active_keys.remove(c)

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
        print("Aubo keyboard (arm=velocity, gripper=MyI10Tele latch):")
        print("  W/S(X) A/D(Y) R/F(Z) | J/L(Roll) K/I(Pitch) U/O(Yaw)")
        print(f"  {GRIPPER_CLOSE_KEY!r}=close {GRIPPER_OPEN_KEY!r}=open | +/- speed")
        print("  Enter=success ESC=failure Space=intervention")

    def stop(self):
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def reset(self):
        self.active_keys.clear()
        self._vel.reset()
        self.gripper_closed = True
        self.episode_end_status = None
        for key in self.key_states:
            self.key_states[key] = False

    def should_intervene(self) -> bool:
        return self.key_states["intervention"]

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None
        return status

    def get_action(self) -> np.ndarray:
        action = self._vel.normalized_from_keys(self.active_keys, TELEOP_KEY_MAP)
        # ±1 avoids collision with neutral policy action[6]=0 (hold latch).
        action[6] = 1.0 if self.gripper_closed else -1.0
        return action


def build_aubo_state_from_sdk(robot, gripper_openpi: float) -> dict[str, np.ndarray]:
    """Build hil-serl state dict from Auboi5Robot waypoint (real robot)."""
    wp = robot.get_current_waypoint()
    pos = np.asarray(wp["pos"], dtype=np.float32)
    rpy = np.asarray(robot.quaternion_to_rpy(wp["ori"]), dtype=np.float32)
    q = np.asarray(wp["joint"], dtype=np.float32)
    return {
        "tcp_pose": np.concatenate([pos, rpy]),
        "tcp_vel": np.zeros(6, dtype=np.float32),
        "tcp_force": np.zeros(3, dtype=np.float32),
        "tcp_torque": np.zeros(3, dtype=np.float32),
        "gripper_pose": np.array([gripper_openpi], dtype=np.float32),
        "q": q[:6],
        "dq": np.zeros(6, dtype=np.float32),
    }


def build_aubo_state(myenv) -> dict[str, np.ndarray]:
    q = np.asarray(myenv.get_joint_state(), dtype=np.float32).reshape(-1)
    p, _ = myenv.env.get_pR_body("i10_inspire_flange_link")
    # RPY hardcoded to zero — aligned with gym_hil PandaPickCubeGymEnv
    # where parse_agent_pos constructs tcp_pose = [xyz, 0, 0, 0].
    return {
        "tcp_pose": np.concatenate(
            [p.astype(np.float32), np.zeros(3, dtype=np.float32)]
        ),
        "tcp_vel": np.zeros(6, dtype=np.float32),
        "tcp_force": np.zeros(3, dtype=np.float32),
        "tcp_torque": np.zeros(3, dtype=np.float32),
        "gripper_pose": q[6:7],
        "q": q[:6],
        "dq": np.zeros(6, dtype=np.float32),
    }


class AuboObsWrapper(gym.ObservationWrapper):
    """Convert env obs to hil-serl {images, state}."""

    def __init__(self, env, image_keys=None, state_builder=None):
        super().__init__(env)
        self.image_keys = image_keys or ["front", "wrist"]
        self.state_builder = state_builder or self._default_state_builder
        sample_state = self.state_builder(self.env.unwrapped)
        pixels_space = cast(spaces.Dict, self.env.observation_space["pixels"])
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        k: spaces.Box(-np.inf, np.inf, v.shape, dtype=np.float32)
                        for k, v in sample_state.items()
                    }
                ),
                "images": spaces.Dict({k: pixels_space[k] for k in self.image_keys}),
            }
        )

    def _default_state_builder(self, unwrapped):
        return build_aubo_state(unwrapped.myenv)

    def observation(self, observation):
        return {
            "state": self.state_builder(self.env.unwrapped),
            "images": {k: observation["pixels"][k] for k in self.image_keys},
        }


class AuboKeyboardWrapper(gym.Wrapper):
    """Keyboard intervention; sync velocity filter with env on intervene."""

    def __init__(self, env, controller: AuboKeyboardController):
        super().__init__(env)
        self.controller = controller
        controller._gripper_fn = self._trigger_gripper_immediate

    def _trigger_gripper_immediate(self, close: bool) -> None:
        _find_aubo_gym_env(self.env).trigger_gripper(close)

    def _sync_velocity_to_env(self, unwrapped, *, sync_filter: bool) -> None:
        if not hasattr(unwrapped, "velocity_integrator"):
            return
        vi = unwrapped.velocity_integrator
        vi.max_lin_vel = self.controller.max_lin_vel
        vi.max_ang_vel = self.controller.max_ang_vel
        if sync_filter:
            vi.v_filtered[:] = self.controller._vel.v_filtered

    def reset(self, **kwargs):
        self.controller.reset()
        obs, info = self.env.reset(**kwargs)
        _find_aubo_gym_env(self.env)._gripper_closed = self.controller.gripper_closed
        self._sync_velocity_to_env(self.env.unwrapped, sync_filter=True)
        info["intervention_enabled"] = self.controller.key_states["intervention"]
        return obs, info

    def step(self, action):
        status = self.controller.get_episode_end_status()
        is_intervention = self.controller.should_intervene()
        teleop_action = None
        unwrapped = _find_aubo_gym_env(self.env)
        if is_intervention:
            unwrapped._teleop_keys_fn = lambda: (
                self.controller.active_keys,
                self.controller.gripper_closed,
            )
            self._sync_velocity_to_env(unwrapped, sync_filter=True)
            teleop_action = self.controller.get_action()
            action = teleop_action
        else:
            unwrapped._teleop_keys_fn = None
            # Lock roll/pitch only; policy may output yaw (action[5], keyboard u/o).
            action = np.asarray(action, dtype=np.float32).copy()
            action[3:5] = 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        unwrapped._teleop_keys_fn = None
        info["is_intervention"] = is_intervention
        info["intervention_enabled"] = is_intervention
        if teleop_action is not None:
            info["teleop_action"] = teleop_action

        if status == "success":
            reward = 1.0
            terminated = True
            info["succeed"] = True
        elif status == "failure":
            terminated = True
        return obs, reward, terminated, truncated, info

    def close(self):
        self.controller.stop()
        return self.env.close()


def _find_keyboard_controller(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "controller"):
            return cur.controller
        cur = getattr(cur, "env", None)
    return None


class AuboAutoInterventionWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        controller = _find_keyboard_controller(self.env)
        if controller is not None:
            controller.key_states["intervention"] = True
            info["intervention_enabled"] = True
        return obs, info


class AuboResetDelayWrapper(gym.Wrapper):
    """Same as gym_hil ResetDelayWrapper — pause before env.reset()."""

    def __init__(self, env, delay_seconds: float = 1.0):
        super().__init__(env)
        self.delay_seconds = delay_seconds

    def reset(self, **kwargs):
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        return self.env.reset(**kwargs)


class AuboGripperPenaltyWrapper(gym.Wrapper):
    """Same semantics as gym_hil GripperPenaltyWrapper for discrete open/close."""

    def __init__(self, env, penalty: float = -0.05):
        super().__init__(env)
        self.penalty = penalty
        self._gripper_closed = True

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env
        while hasattr(base, "env"):
            if hasattr(base, "_gripper_closed"):
                break
            base = base.env
        self._gripper_closed = bool(getattr(base, "_gripper_closed", True))
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["discrete_penalty"] = 0.0
        if (action[6] < -0.5 and self._gripper_closed) or (
            action[6] > 0.5 and not self._gripper_closed
        ):
            info["discrete_penalty"] = self.penalty
        base = self.env
        while hasattr(base, "env"):
            if hasattr(base, "_gripper_closed"):
                break
            base = base.env
        if hasattr(base, "_gripper_closed"):
            self._gripper_closed = bool(base._gripper_closed)
        return obs, reward, terminated, truncated, info


class AuboHilInterventionInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_intervention", False) and "teleop_action" in info:
            info["intervene_action"] = np.asarray(
                info["teleop_action"], dtype=np.float32
            )
        return obs, reward, terminated, truncated, info


def _find_aubo_gym_env(env):
    """Sim (AuboPickCubeGymEnv) or real (AuboPickCubeRealEnv) — keyboard gripper/velocity sync."""
    cur = env
    while cur is not None:
        # Class-level check: skip wrappers like AuboGripperPenaltyWrapper that own
        # _gripper_closed but are not the control env (gymnasium blocks _foo on wrappers).
        if callable(getattr(type(cur), "trigger_gripper", None)):
            return cur
        cur = getattr(cur, "env", None)
    raise RuntimeError("Aubo control env not found in wrapper stack")


class AuboTeleopDisplayWrapper(gym.Wrapper):
    """OpenCV window: front|wrist on top, large teleop (-X) below."""

    def __init__(
        self,
        env,
        window_name: str = "aubo_pick_cube_sim",
        policy_panel_size: int = 200,
        teleop_height: int = 480,
        display_scale: int = 1,
    ):
        super().__init__(env)
        import cv2

        self._cv2 = cv2
        self.window_name = window_name
        self.policy_panel_size = policy_panel_size
        self.teleop_height = teleop_height
        self.display_scale = display_scale
        self._window_ready = False

    def _resize(self, rgb: np.ndarray, width: int, height: int) -> np.ndarray:
        return self._cv2.resize(
            rgb, (width, height), interpolation=self._cv2.INTER_LINEAR
        )

    def _label(self, frame: np.ndarray, text: str, x: int, y: int = 24) -> None:
        self._cv2.putText(
            frame,
            text,
            (x + 8, y),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            self._cv2.LINE_AA,
        )
        self._cv2.putText(
            frame,
            text,
            (x + 8, y),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            self._cv2.LINE_AA,
        )

    def _show(self, obs, info=None):
        ps = self.policy_panel_size
        top_w = ps * 2
        front = self._resize(obs["images"]["front"], ps, ps)
        wrist = self._resize(obs["images"]["wrist"], ps, ps)
        top_row = np.concatenate([front, wrist], axis=1)
        self._label(top_row, "front", 0)
        self._label(top_row, "wrist", ps)

        teleop = self._resize(
            _find_aubo_gym_env(self.env).render_teleop_view(),
            top_w,
            self.teleop_height,
        )
        self._label(teleop, "teleop (-X)", 0)

        frame = np.concatenate([top_row, teleop], axis=0)
        if info is not None and (
            info.get("is_intervention") or info.get("intervention_enabled")
        ):
            self._cv2.putText(
                frame,
                "INTERVENTION ON",
                (10, ps + self.teleop_height - 12),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                self._cv2.LINE_AA,
            )
        elif info is not None and info.get("intervention_enabled") is False:
            self._cv2.putText(
                frame,
                "Press Space to control",
                (10, ps + self.teleop_height - 12),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                self._cv2.LINE_AA,
            )
        if self.display_scale != 1:
            frame = self._cv2.resize(
                frame,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=self._cv2.INTER_LINEAR,
            )
        if not self._window_ready:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
            self._window_ready = True
        self._cv2.imshow(
            self.window_name, self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        )
        self._cv2.waitKey(1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._show(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._show(obs, info)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._cv2.destroyAllWindows()
        return self.env.close()


__all__ = [
    "AuboAutoInterventionWrapper",
    "AuboGripperPenaltyWrapper",
    "AuboHilInterventionInfoWrapper",
    "AuboKeyboardController",
    "AuboKeyboardWrapper",
    "AuboObsWrapper",
    "AuboResetDelayWrapper",
    "AuboTeleopDisplayWrapper",
    "GymHilCameraDisplayWrapper",
    "build_aubo_state",
    "build_aubo_state_from_sdk",
]
