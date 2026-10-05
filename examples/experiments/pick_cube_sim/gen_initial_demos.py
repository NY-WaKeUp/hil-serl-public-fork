# Generate the initial demo buffer by self-rollout: run the 99% checkpoint in
# the BASELINE lighting and record successful episodes. Replaces human teleop
# demos for the recovery experiments (both arms share the same initial buffer).

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))      # examples/
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))  # repo root

import copy
import datetime
import pickle as pkl

import jax
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from tqdm import tqdm

from experiments.pick_cube_sim.config import TrainConfig
from experiments.pick_cube_sim.lighting_utils import LightingConfig, apply_lighting

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Checkpoint dir (e.g. .../first_run).")
flags.DEFINE_integer("eval_checkpoint_step", 30000, "Checkpoint step to load.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful episodes to record.")
flags.DEFINE_integer("max_episodes", 200, "Safety cap on total episodes.")
flags.mark_flag_as_required("checkpoint_path")


def main(_):
    from serl_launcher.utils.launcher import (
        make_sac_pixel_agent,
        make_sac_pixel_agent_hybrid_single_arm,
    )

    config = TrainConfig()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    model = env.unwrapped.model

    # Baseline lighting: the demos must come from the ORIGINAL illumination.
    apply_lighting(model, LightingConfig())

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
    print(f"Loaded checkpoint step {FLAGS.eval_checkpoint_step}")

    transitions = []
    success_count = 0
    episode_count = 0
    pbar = tqdm(total=FLAGS.successes_needed)
    while success_count < FLAGS.successes_needed and episode_count < FLAGS.max_episodes:
        obs, info = env.reset()
        done = False
        trajectory = []
        while not done:
            sampling_key = jax.random.PRNGKey(int(np.random.randint(2**31 - 1)))
            actions = agent.sample_actions(
                observations=jax.device_put(obs), argmax=False, seed=sampling_key
            )
            actions = np.asarray(jax.device_get(actions))
            next_obs, rew, done, truncated, info = env.step(actions)
            trajectory.append(copy.deepcopy(dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )))
            obs = next_obs
        episode_count += 1
        if info.get("succeed"):
            transitions.extend(trajectory)
            success_count += 1
            pbar.update(1)
    pbar.close()

    out_dir = os.path.join(_HERE, "demo_data")
    os.makedirs(out_dir, exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(out_dir, f"self_demo_{FLAGS.successes_needed}_{uuid}.pkl")
    with open(out, "wb") as f:
        pkl.dump(transitions, f)
    print(f"saved {success_count} successful episodes "
          f"({len(transitions)} transitions, {episode_count} episodes total) to {out}")


if __name__ == "__main__":
    app.run(main)
