import sys
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from experiments.pick_cube_sim.wrapper import KeyBoardIntervention
from serl_launcher.utils.timer_utils import Timer


class _FakeEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 1.0, False, False, {"succeed": True}

    def close(self):
        pass


def _wrap():
    with patch.object(sys.stdin, "isatty", return_value=False):
        env = KeyBoardIntervention(_FakeEnv())
    env._stdin_teleop = True
    return env


class KeyboardLabelTests(unittest.TestCase):
    def test_q_discards_without_fail_label(self):
        env = _wrap()
        env._drain_stdin = lambda: ["q"]
        _, rew, done, _, info = env.step(np.zeros(7, dtype=np.float32))
        self.assertTrue(info["withdraw"])
        self.assertFalse(info["human_fail"])
        self.assertFalse(done)
        self.assertEqual(rew, 1.0)
        self.assertTrue(info["succeed"])

    def test_b_ends_episode_as_fail(self):
        env = _wrap()
        env._drain_stdin = lambda: ["b"]
        _, rew, done, _, info = env.step(np.zeros(7, dtype=np.float32))
        self.assertFalse(info["withdraw"])
        self.assertTrue(info["human_fail"])
        self.assertTrue(done)
        self.assertEqual(rew, 0.0)
        self.assertFalse(info["succeed"])

    def test_q_wins_over_b(self):
        env = _wrap()
        env._drain_stdin = lambda: ["q", "b"]
        _, _, done, _, info = env.step(np.zeros(7, dtype=np.float32))
        self.assertTrue(info["withdraw"])
        self.assertFalse(info["human_fail"])
        self.assertFalse(done)


class TimerWithdrawTests(unittest.TestCase):
    def test_tock_still_runs_after_withdraw_branch(self):
        timer = Timer()
        withdrawn = True
        timer.tick("total")
        with timer.context("step_env"):
            if withdrawn:
                pass
        timer.tock("total")
        timer.tick("total")
        timer.tock("total")


if __name__ == "__main__":
    unittest.main()
