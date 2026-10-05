import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING
from franka_sim.utils.viewer_utils import DualMujocoViewer

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "pick_cube_sim", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    # Create the dual viewer
    dual_viewer = DualMujocoViewer(env.unwrapped.model, env.unwrapped.data)

    print(
        "Look at the MuJoCo windows; type WASD/FR/Space in THIS terminal. "
        "Q discards the current episode. B ends it as failure (not saved as demo). "
        "Only successful episodes are saved."
    )
    with dual_viewer as viewer:
        while viewer.is_running():
            actions = np.zeros(env.action_space.sample().shape) 
            next_obs, rew, done, truncated, info = env.step(actions)
            viewer.sync()
            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if info.get("withdraw"):
                print("withdraw: discard episode and reset")
                trajectory = []
                returns = 0
                obs, info = env.reset()
                continue
            if done:
                if info.get("human_fail"):
                    print("human fail: drop demo episode and reset")
                elif info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    print(f"Success count: {success_count}")
                    pbar.update(1)
                trajectory = []
                returns = 0
                obs, info = env.reset()
            if success_count >= success_needed:
                break

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if FLAGS.exp_name == "pick_cube_sim":
        out_dir = os.path.join(
            os.path.dirname(__file__),
            "experiments",
            "pick_cube_sim",
            "demo_data",
        )
        os.makedirs(out_dir, exist_ok=True)
        file_name = os.path.join(out_dir, f"self_demo_{success_needed}_{uuid}.pkl")
    else:
        os.makedirs("./demo_data", exist_ok=True)
        file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)