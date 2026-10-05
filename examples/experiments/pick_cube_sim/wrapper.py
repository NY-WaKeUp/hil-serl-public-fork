import atexit
import select
import sys
import termios
import time
import tty

import gymnasium as gym
import numpy as np

# Translation keys in the same 6D EE-frame convention as JoystickIntervention.
# W/S: +x/-x, A/D: +y/-y, F/R: +z/-z. Space toggles gripper. Rotations stay zero.
_MOVE_DELTA = {
    "w": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "s": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "a": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "d": np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
    "f": np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
    "r": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
}

# Stdin has no key-up. A move key stays active this long after the last char (OS repeat).
_HOLD_S = 0.02


class KeyBoardIntervention(gym.ActionWrapper):
    """Terminal stdin teleop. MuJoCo windows do not receive these keys."""

    def __init__(self, env, action_indices=None, action_length=0.3):
        super().__init__(env)
        self._stdin_teleop = sys.stdin.isatty()
        self.gripper_enabled = self.action_space.shape != (6,)
        self.action_indices = action_indices
        self.action_length = action_length
        self.held = {k: False for k in _MOVE_DELTA}
        self._last_seen = {k: -1e9 for k in _MOVE_DELTA}
        self.gripper_closed = False
        self._space_down = False
        self.withdraw = False
        self._q_down = False
        self.human_fail = False
        self._b_down = False
        self._intervening = False
        self._i_down = False
        self.left = False
        self.right = False
        self._fd = None
        self._term_old = None
        if self._stdin_teleop:
            self._fd = sys.stdin.fileno()
            self._term_old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            atexit.register(self._restore_term)
            print(
                "Type in THIS terminal (not the MuJoCo window): "
                "I toggle intervention, W/S x, A/D y, F/R z, Space gripper, "
                "Q discard episode, B mark episode failed."
            )
        else:
            print("KeyBoardIntervention: stdin is not a TTY, teleop disabled")

    def _drain_stdin(self):
        chars = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            chars.append(sys.stdin.read(1))
        return chars

    def _poll_keys(self):
        if not self._stdin_teleop:
            return
        now = time.monotonic()
        chars = self._drain_stdin()
        i_now = any(raw.lower() == "i" for raw in chars)
        if i_now and not self._i_down:
            self._intervening = not self._intervening
            print("intervention ON" if self._intervening else "intervention OFF")
            if not self._intervening:
                self.held = {k: False for k in _MOVE_DELTA}
                self._last_seen = {k: -1e9 for k in _MOVE_DELTA}
        self._i_down = i_now
        if self._intervening:
            for raw in chars:
                ch = raw.lower()
                if ch in self.held:
                    self._last_seen[ch] = now
            space_now = any(raw == " " for raw in chars)
            if space_now and not self._space_down:
                self.gripper_closed = not self.gripper_closed
            self._space_down = space_now
        q_now = any(raw.lower() == "q" for raw in chars)
        if q_now and not self._q_down:
            self.withdraw = True
        self._q_down = q_now
        b_now = any(raw.lower() == "b" for raw in chars)
        if b_now and not self._b_down and not self.withdraw:
            self.human_fail = True
        self._b_down = b_now
        for k in self.held:
            self.held[k] = self._intervening and (now - self._last_seen[k]) < _HOLD_S

    def _expert_action(self):
        deltas = np.stack([_MOVE_DELTA[k] for k in self.held], axis=0)
        held_mask = np.array([self.held[k] for k in self.held], dtype=np.float64)
        expert_a = held_mask @ deltas * self.action_length
        if not self.gripper_enabled:
            return expert_a
        gripper = np.array([1.0 if self.gripper_closed else -1.0])
        return np.concatenate([expert_a, gripper], axis=0)

    def action(self, action: np.ndarray):
        self._poll_keys()
        intervened = self._intervening
        expert_a = self._expert_action()
        if self.action_indices is not None:
            filtered = np.zeros_like(expert_a)
            filtered[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered
        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if self.human_fail:
            rew = 0.0
            done = True
            info["succeed"] = False
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        info["withdraw"] = self.withdraw
        info["human_fail"] = self.human_fail
        self.withdraw = False
        self.human_fail = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        self.held = {k: False for k in _MOVE_DELTA}
        self._last_seen = {k: -1e9 for k in _MOVE_DELTA}
        self.gripper_closed = False
        self._space_down = False
        self.withdraw = False
        self._q_down = False
        self.human_fail = False
        self._b_down = False
        self._i_down = False
        return self.env.reset(**kwargs)

    def _restore_term(self):
        if self._term_old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._term_old)

    def close(self):
        self._restore_term()
        super().close()


# This is copied from the USB task.
# It's confusing that the penalty is a negative number but I've verified this is what is intended.
# Search for `grasp_rewards = batch["rewards"] + batch["grasp_penalty"]`
# Also, the reward is not modified by this wrapper. Instead a "grasp_penalty" key is added to the info and used
# downstream where needed.

class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, -1]
        return obs, info

    def step(self, action):
        """Adds grasp_penalty to info; does not change the env reward."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, -1]
        return observation, reward, terminated, truncated, info
