import copy
import datetime
import glob
import os
import pickle as pkl
import sys

import numpy as np
from absl import app, flags
from tqdm import tqdm

from experiments.mappings import CONFIG_MAPPING, get_train_config

DEMO_DATA_DIR = "./demo_data"
SIM_EXP_NAMES = frozenset({"panda_pick_cube_sim", "aubo_pick_cube_sim"})

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "exp_name",
    None,
    "Name of experiment corresponding to folder (e.g. panda_pick_cube_sim, ram_insertion).",
)
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_boolean(
    "record",
    True,
    "If True, save demos to demo_data/ in hil-serl pkl format.",
)


def episode_succeeded(reward, done, truncated, info) -> bool:
    if not (done or truncated):
        return False
    return bool(reward > 0 or info.get("succeed") or info.get("next.success"))


def use_classifier_for_exp(exp_name: str) -> bool:
    return exp_name not in SIM_EXP_NAMES


def existing_demo_paths(exp_name: str) -> list[str]:
    pattern = os.path.join(DEMO_DATA_DIR, f"{exp_name}_*.pkl")
    return sorted(glob.glob(pattern))


def load_transitions(paths: list[str]) -> list:
    transitions = []
    for path in paths:
        with open(path, "rb") as f:
            transitions.extend(pkl.load(f))
    return transitions


def prepare_demo_recording(exp_name: str) -> tuple[list, list[str]]:
    """Return (initial_transitions, old_paths_to_remove_after_save)."""
    existing = existing_demo_paths(exp_name)
    if not existing:
        return [], []

    print(f"{DEMO_DATA_DIR}/ already has {len(existing)} {exp_name} demo file(s):")
    for path in existing:
        print(f"  {path}")

    while True:
        print("Choose: [1] delete old files and re-record  [2] append  [3] quit")
        choice = input("> ").strip()
        if choice == "1":
            for path in existing:
                os.remove(path)
            print(f"Removed {len(existing)} file(s). Starting fresh.")
            return [], []
        if choice == "2":
            transitions = load_transitions(existing)
            print(f"Loaded {len(transitions)} transitions. Continue recording.")
            return transitions, existing
        if choice == "3":
            print("Cancelled.")
            sys.exit(0)
        print("Invalid input. Enter 1, 2, or 3.")


def save_transitions(
    exp_name: str,
    successes_needed: int,
    transitions: list,
    old_paths: list[str],
) -> None:
    os.makedirs(DEMO_DATA_DIR, exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(
        DEMO_DATA_DIR,
        f"{exp_name}_{successes_needed}_demos_{uuid}.pkl",
    )
    with open(path, "wb") as f:
        pkl.dump(transitions, f)
    for old_path in old_paths:
        if os.path.exists(old_path):
            os.remove(old_path)
    print(f"Saved {len(transitions)} transitions to {path}")
    if old_paths:
        print(f"Removed {len(old_paths)} old demo file(s).")


def print_teleop_instructions(exp_name: str) -> None:
    if exp_name in SIM_EXP_NAMES:
        print("Sim demo recording: keyboard control is ON after each reset.")
        print("Use WASD / F-R / [/] to move the arm (see startup log for full map).")
        print("Space toggles control off/on. Enter = success, ESC = failure.")
        print("Look for the OpenCV window (aubo_pick_cube_sim or gym_hil) on DISPLAY.")
        print("Keys must be pressed in the graphical desktop (not SSH terminal).")
    else:
        print("Teleop-only mode: use SpaceMouse to control the robot.")
        print("Episode ends when the reward classifier detects success or times out.")


def main(_):
    if FLAGS.exp_name is None:
        raise app.UsageError(
            "Missing --exp_name. Example: python record_demos.py "
            "--exp_name=panda_pick_cube_sim --successes_needed=10"
        )
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."

    old_demo_paths: list[str] = []
    transitions: list = []
    if FLAGS.record:
        transitions, old_demo_paths = prepare_demo_recording(FLAGS.exp_name)

    config = get_train_config(FLAGS.exp_name)
    teleop_record = FLAGS.exp_name in SIM_EXP_NAMES
    env_kwargs = dict(
        fake_env=False,
        save_video=False,
        classifier=use_classifier_for_exp(FLAGS.exp_name),
    )
    if teleop_record:
        env_kwargs["teleop_record"] = True
    env = config.get_environment(**env_kwargs)

    obs, info = env.reset()
    print_teleop_instructions(FLAGS.exp_name)

    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0.0
    neutral_action = np.zeros(env.action_space.sample().shape, dtype=np.float32)

    while success_count < success_needed:
        next_obs, rew, done, truncated, info = env.step(neutral_action)
        reward = float(rew)
        returns += reward

        actions = neutral_action
        if "intervene_action" in info:
            actions = info["intervene_action"]

        if FLAGS.record:
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - float(done or truncated),
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)

        pbar.set_description(f"return={returns:.2f} success={success_count}")
        obs = next_obs

        if done or truncated:
            if episode_succeeded(reward, done, truncated, info):
                if FLAGS.record:
                    transitions.extend(copy.deepcopy(trajectory))
                success_count += 1
                pbar.update(1)
                print(f"Success {success_count}/{success_needed}")
            trajectory = []
            returns = 0.0
            obs, info = env.reset()

    pbar.close()
    env.close()

    if FLAGS.record and transitions:
        save_transitions(
            FLAGS.exp_name,
            success_needed,
            transitions,
            old_demo_paths,
        )


if __name__ == "__main__":
    app.run(main)
