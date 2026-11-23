"""Bridge gym_hil MuJoCo sim envs to hil-serl observation / intervention format."""

from typing import cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MAX_GRIPPER_COMMAND = 255.0
HIL_ACTION_DIM = 7


def gymhil_action_to_hil(action: np.ndarray) -> np.ndarray:
    """Map gym_hil EE action (4D: xyz + gripper in [0, 2]) to hil-serl 7D."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == HIL_ACTION_DIM:
        return action
    hil = np.zeros(HIL_ACTION_DIM, dtype=np.float32)
    hil[:3] = action[:3]
    hil[6] = action[3] - 1.0
    return hil


def hil_action_to_gymhil(action: np.ndarray) -> np.ndarray:
    """Map hil-serl 7D action to gym_hil EE action (4D)."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == 4:
        return action
    gymhil = np.zeros(4, dtype=np.float32)
    gymhil[:3] = action[:3]
    gymhil[3] = action[6] + 1.0
    return gymhil


def find_gym_hil_inputs_control(env):
    """Return gym_hil InputsControlWrapper if present in the wrapper stack."""
    cur = env
    while cur is not None:
        if type(cur).__name__ == "InputsControlWrapper":
            return cur
        cur = getattr(cur, "env", None)
    return None


def set_keyboard_intervention(env, enabled: bool) -> bool:
    """Toggle gym_hil keyboard intervention flag. Returns True if controller found."""
    ctrl_env = find_gym_hil_inputs_control(env)
    if ctrl_env is None:
        return False
    ctrl_env.controller.key_states["intervention"] = enabled
    return True


class WasdKeyboardController:
    """Keyboard teleop: WASD xy, F/R z down/up, [/] gripper, Space toggle."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.episode_end_status = None
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "success": False,
            "failure": False,
            "intervention": False,
            "rerecord": False,
        }
        self.listener = None

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
            c = key.char.lower()
            if c == "w":
                self.key_states["forward_x"] = True
            elif c == "s":
                self.key_states["backward_x"] = True
            elif c == "a":
                self.key_states["forward_y"] = True
            elif c == "d":
                self.key_states["backward_y"] = True
            elif c == "f":
                self.key_states["backward_z"] = True
            elif c == "r":
                self.key_states["forward_z"] = True
            elif c == "[":
                self.open_gripper_command = True
            elif c == "]":
                self.close_gripper_command = True

        def on_release(key):
            if not hasattr(key, "char") or key.char is None:
                return
            c = key.char.lower()
            if c == "w":
                self.key_states["forward_x"] = False
            elif c == "s":
                self.key_states["backward_x"] = False
            elif c == "a":
                self.key_states["forward_y"] = False
            elif c == "d":
                self.key_states["backward_y"] = False
            elif c == "f":
                self.key_states["backward_z"] = False
            elif c == "r":
                self.key_states["forward_z"] = False
            elif c == "[":
                self.open_gripper_command = False
            elif c == "]":
                self.close_gripper_command = False

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
        print("Keyboard controls (WASD layout):")
        print("  W/S/A/D: move in X/Y")
        print("  F/R: move down/up (Z)")
        print("  [/]: open/close gripper")
        print("  Enter: success | ESC: failure | Space: toggle control")

    def stop(self):
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def update(self):
        pass

    def get_deltas(self):
        delta_x = delta_y = delta_z = 0.0
        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size
        return delta_x, delta_y, delta_z

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None
        return status

    def should_intervene(self):
        return self.key_states["intervention"]

    def gripper_command(self):
        if self.open_gripper_command == self.close_gripper_command:
            return "no-op"
        if self.open_gripper_command:
            return "open"
        return "close"

    def reset(self):
        for key in self.key_states:
            self.key_states[key] = False
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.episode_end_status = None


def install_wasd_keyboard_controller(env) -> bool:
    """Replace gym_hil default arrow-key controller with WASD layout."""
    import time

    ctrl_env = find_gym_hil_inputs_control(env)
    if ctrl_env is None:
        return False
    old = ctrl_env.controller
    # Wait a tiny bit to let X11 listener initialize before stopping it,
    # otherwise pynput Xlib backend throws AttributeError on request_queue.append
    time.sleep(0.1)
    try:
        old.stop()
    except Exception as e:
        print(f"Note: ignored error stopping default keyboard listener: {e}")

    was_intervention = getattr(old, "key_states", {}).get("intervention", False)
    new_ctrl = WasdKeyboardController(
        x_step_size=old.x_step_size,
        y_step_size=old.y_step_size,
        z_step_size=old.z_step_size,
    )
    new_ctrl.key_states["intervention"] = was_intervention
    new_ctrl.start()
    ctrl_env.controller = new_ctrl
    return True


class GymHilAutoInterventionWrapper(gym.Wrapper):
    """Enable keyboard intervention after every reset (sim demo recording)."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        enabled = set_keyboard_intervention(self.env, True)
        info["intervention_enabled"] = enabled
        return obs, info


class GymHilSevenDimActionWrapper(gym.ActionWrapper):
    """Expose 7D hil-serl action space; gym_hil env still receives 4D EE actions."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-np.ones(HIL_ACTION_DIM, dtype=np.float32),
            high=np.ones(HIL_ACTION_DIM, dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        return hil_action_to_gymhil(action)

    def step(self, action):
        return self.env.step(self.action(action))


def parse_agent_pos(agent_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Parse flat agent_pos vector from gym_hil into hil-serl state dict."""
    qpos = agent_pos[:7]
    qvel = agent_pos[7:14]
    gripper = agent_pos[14:15] / MAX_GRIPPER_COMMAND
    tcp_pos = agent_pos[15:18]
    tcp_pose = np.concatenate([tcp_pos, np.zeros(3, dtype=np.float32)])
    return {
        "tcp_pose": tcp_pose.astype(np.float32),
        "tcp_vel": np.zeros(6, dtype=np.float32),
        "tcp_force": np.zeros(3, dtype=np.float32),
        "tcp_torque": np.zeros(3, dtype=np.float32),
        "gripper_pose": gripper.astype(np.float32),
        "q": qpos.astype(np.float32),
        "dq": qvel.astype(np.float32),
    }


class GymHilObsWrapper(gym.ObservationWrapper):
    """Convert gym_hil obs ({pixels, agent_pos}) to hil-serl ({images, state})."""

    def __init__(self, env, image_keys=None):
        super().__init__(env)
        self.image_keys = image_keys or ["front", "wrist"]

        raw_obs_space = cast(spaces.Dict, self.env.observation_space)
        agent_pos = np.asarray(raw_obs_space["agent_pos"].sample(), dtype=np.float32)
        pixels_space = cast(spaces.Dict, raw_obs_space["pixels"])
        sample_state = parse_agent_pos(agent_pos)
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

    def observation(self, observation):
        return {
            "state": parse_agent_pos(observation["agent_pos"]),
            "images": {k: observation["pixels"][k] for k in self.image_keys},
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class GymHilCameraDisplayWrapper(gym.Wrapper):
    """Show camera images via OpenCV (works with MUJOCO_GL=egl + TensorFlow)."""

    def __init__(
        self,
        env,
        image_keys=None,
        window_name="gym_hil",
        display_scale=3,
        display_every_n=1,
    ):
        super().__init__(env)
        import cv2

        self._cv2 = cv2
        self.image_keys = image_keys or ["front", "wrist"]
        self.window_name = window_name
        self.display_scale = display_scale
        self.display_every_n = max(1, display_every_n)
        self._window_ready = False
        self._step_i = 0

    def _show(self, obs, info=None):
        self._step_i += 1
        if self._step_i % self.display_every_n != 0:
            return
        frame = np.concatenate([obs["images"][k] for k in self.image_keys], axis=1)
        if self.display_scale != 1:
            frame = self._cv2.resize(
                frame,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=self._cv2.INTER_LINEAR,
            )
        if info is not None and (
            info.get("is_intervention") or info.get("intervention_enabled")
        ):
            self._cv2.putText(
                frame,
                "INTERVENTION ON",
                (10, 30),
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
                (10, 30),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                self._cv2.LINE_AA,
            )
        if not self._window_ready:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
            self._window_ready = True
        frame_bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        self._cv2.imshow(self.window_name, frame_bgr)
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


class HilSerlInterventionInfoWrapper(gym.Wrapper):
    """Map gym_hil teleop info keys to hil-serl intervene_action convention."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("is_intervention", False):
            info["intervene_action"] = gymhil_action_to_hil(
                np.asarray(info["teleop_action"], dtype=np.float32)
            )
        if "discrete_penalty" in info:
            info["grasp_penalty"] = info["discrete_penalty"]
        return obs, reward, terminated, truncated, info
