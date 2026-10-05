# Evaluate a trained checkpoint under lighting perturbations.
# Phase 0: measure baseline collapse; also the shared evaluation routine for
# recovery-curve experiments (Phase 2).

import os
import sys

# train_rlpd.py lives in examples/, so script mode puts examples/ on sys.path[0]
# and `experiments` resolves as a top-level package. The relative '../../../'
# insert (as in train_rlpd.py) additionally exposes repo-root packages like
# franka_env. Reproduce both here, path-independent.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))      # examples/
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))  # repo root

import csv
import time

import jax
import numpy as np
from absl import app, flags
from flax.training import checkpoints

from experiments.pick_cube_sim.config import TrainConfig
from experiments.pick_cube_sim.lighting_utils import (
    LightingConfig,
    PRESETS,
    apply_lighting,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None,
                    "Checkpoint dir to evaluate (e.g. .../pick_cube_sim/first_run).")
flags.DEFINE_integer("eval_checkpoint_step", 30000, "Checkpoint step to load.")
flags.DEFINE_integer("n_episodes", 20, "Episodes per lighting preset.")
flags.DEFINE_string("out_csv", "lighting_eval.csv", "Output CSV path.")
flags.DEFINE_string("only_preset", None,
                    "If set, evaluate only the preset whose label contains this substring.")
flags.mark_flag_as_required("checkpoint_path")


def wilson_ci(sr: float, n: int, z: float = 1.96):
    denom = 1 + z**2 / n
    center = (sr + z**2 / (2 * n)) / denom
    half = z * np.sqrt(sr * (1 - sr) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


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


def main(_):
    # Mirror train_rlpd.py's setup_mode -> agent constructor mapping.
    from serl_launcher.utils.launcher import (
        make_sac_pixel_agent,
        make_sac_pixel_agent_hybrid_single_arm,
    )

    config = TrainConfig()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    model = env.unwrapped.model

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
    ckpt = checkpoints.restore_checkpoint(
        os.path.abspath(FLAGS.checkpoint_path), agent.state, step=FLAGS.eval_checkpoint_step
    )
    agent = agent.replace(state=ckpt)
    print(f"Loaded checkpoint step {FLAGS.eval_checkpoint_step} from {FLAGS.checkpoint_path}")

    presets = [p for p in PRESETS
               if FLAGS.only_preset is None or FLAGS.only_preset in p.label()]

    rows = []
    for cfg in presets:
        sr, (lo, hi), t = eval_preset(agent, env, model, cfg, FLAGS.n_episodes, seed=0)
        print(f"{cfg.label():>40s}  SR={sr:5.1%}  CI95=[{lo:.3f},{hi:.3f}]  t={t:.1f}s")
        rows.append({"preset": cfg.label(), "success_rate": sr, "ci_lo": lo,
                     "ci_hi": hi, "avg_time": t})

    with open(FLAGS.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["preset", "success_rate", "ci_lo", "ci_hi",
                                               "avg_time"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {FLAGS.out_csv}")


if __name__ == "__main__":
    app.run(main)
