# Shared evaluation primitives (agent loading, single-preset evaluation).
# Kept separate from the CLI scripts so absl flag definitions don't collide.

import os
import time

import jax
import numpy as np
from flax.training import checkpoints

from experiments.pick_cube_sim.lighting_utils import LightingConfig, apply_lighting


def wilson_ci(sr: float, n: int, z: float = 1.96):
    denom = 1 + z**2 / n
    center = (sr + z**2 / (2 * n)) / denom
    half = z * np.sqrt(sr * (1 - sr) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def make_env_and_agent():
    from serl_launcher.utils.launcher import (
        make_sac_pixel_agent,
        make_sac_pixel_agent_hybrid_single_arm,
    )

    from experiments.pick_cube_sim.config import TrainConfig

    config = TrainConfig()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    make_agent = {
        "single-arm-learned-gripper": make_sac_pixel_agent_hybrid_single_arm,
        "single-arm-fixed-gripper": make_sac_pixel_agent,
    }[config.setup_mode]
    agent = make_agent(
        seed=0,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
    )
    return env, agent, config


def eval_preset(agent, env, model, cfg: LightingConfig, n_episodes: int, seed: int):
    apply_lighting(model, cfg)
    success = 0
    times = []
    rng = np.random.RandomState(seed)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.randint(2**31 - 1)))
        done = False
        start = time.time()
        while not done:
            sampling_key = jax.random.PRNGKey(rng.randint(2**31 - 1))
            actions = agent.sample_actions(
                observations=jax.device_put(obs), argmax=False, seed=sampling_key
            )
            actions = np.asarray(jax.device_get(actions))
            obs, reward, done, truncated, info = env.step(actions)
        if info.get("succeed"):
            success += 1
            times.append(time.time() - start)
    sr = success / n_episodes
    lo, hi = wilson_ci(sr, n_episodes)
    return sr, (lo, hi), (np.mean(times) if times else float("nan"))


def load_agent_at(agent, run_dir: str, step: int):
    ckpt = checkpoints.restore_checkpoint(os.path.abspath(run_dir), agent.state, step=step)
    return agent.replace(state=ckpt)
