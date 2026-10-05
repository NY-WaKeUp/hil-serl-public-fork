#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
import mujoco.viewer

from franka_sim.utils.viewer_utils import DualMujocoViewer

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging
flags.DEFINE_boolean(
    "freeze_actor", False,
    "Diagnostic mode: update critics only and keep the pretrained actor/temperature fixed.",
)
flags.DEFINE_boolean(
    "freeze_shared_encoder", False,
    "Keep the shared visual encoder fixed while allowing actor and critic heads to update.",
)
flags.DEFINE_float(
    "actor_anchor_weight", 0.0,
    "Weight of the actor action-MSE anchor to batch behavior actions.",
)
flags.DEFINE_boolean(
    "inject_success_demos", True,
    "Insert successful online episodes into the demo buffer.",
)
flags.DEFINE_float(
    "injected_rank_exponent", 1.0,
    "Rank-based recency exponent inside the injected-demo subpool.",
)


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def find_metric(data, name):
    if isinstance(data, dict):
        if name in data:
            return float(np.asarray(data[name]))
        for value in data.values():
            metric = find_metric(value, name)
            if metric is not None:
                return metric
    return None


class TensorBoardLogger:
    def __init__(self, log_dir):
        self._writer = EventFileWriter(log_dir)

    def log(self, data, step):
        values = []
        pending = list(data.items())
        while pending:
            name, value = pending.pop()
            if isinstance(value, dict):
                pending.extend((f"{name}/{key}", child) for key, child in value.items())
                continue
            value = np.asarray(value)
            if value.ndim == 0:
                values.append(summary_pb2.Summary.Value(tag=name, simple_value=float(value)))
        if values:
            self._writer.add_event(
                event_pb2.Event(wall_time=time.time(), step=step, summary=summary_pb2.Summary(value=values))
            )
            self._writer.flush()


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += reward
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        buf_pkls = natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))
        if buf_pkls:
            start_step = int(os.path.basename(buf_pkls[-1])[12:-4]) + 1

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    network_update_count = 0

    def update_params(params):
        nonlocal agent, network_update_count
        agent = agent.replace(state=agent.state.replace(params=params))
        network_update_count += 1
        print_green(f"actor received network update #{network_update_count}")

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    episode_transitions = []
    episode_intvn = []

    obs, _ = env.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    episode_count = 0
    success_count = 0
    max_episode_steps = round(env.unwrapped._time_limit / env.unwrapped.control_dt)
    episode_pbar = tqdm.tqdm(
        total=max_episode_steps,
        desc="episode",
        leave=False,
        dynamic_ncols=True,
    )

    # Create the dual viewer
    dual_viewer = DualMujocoViewer(env.model, env.data)

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    with dual_viewer as viewer:
        for step in pbar:
            timer.tick("total")
            viewer.sync()

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))

            # Step environment
            with timer.context("step_env"):

                next_obs, reward, done, truncated, info = env.step(actions)
                episode_pbar.update(1)

                if "left" in info:
                    info.pop("left")
                if "right" in info:
                    info.pop("right")

                # override the action with the intervention action
                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    # print("intervened!!!")
                    intervention_steps += 1
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                else:
                    already_intervened = False

                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                if 'grasp_penalty' in info:
                    transition['grasp_penalty']= info['grasp_penalty']

                if info.get("withdraw"):
                    print("withdraw: discard episode and reset")
                    episode_transitions = []
                    episode_intvn = []
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    already_intervened = False
                    obs, _ = env.reset()
                    episode_pbar.reset()
                else:
                    episode_transitions.append(copy.deepcopy(transition))
                    if already_intervened:
                        episode_intvn.append(copy.deepcopy(transition))

                    obs = next_obs
                    if done or truncated:
                        if info.get("human_fail"):
                            print("human fail: insert episode as failure and reset")
                        for item in episode_transitions:
                            data_store.insert(item)
                            transitions.append(item)
                        for item in episode_intvn:
                            intvn_data_store.insert(item)
                            demo_transitions.append(item)
                        episode_transitions = []
                        episode_intvn = []
                        info["episode"]["intervention_count"] = intervention_count
                        info["episode"]["intervention_steps"] = intervention_steps
                        episode_count += 1
                        success_count += int(info.get("succeed", False))
                        info["episode"]["success_rate"] = success_count / episode_count
                        stats = {"environment": info}  # send stats to the learner to log
                        client.request("send-stats", stats)
                        pbar.set_description(f"last return: {running_return}")
                        running_return = 0.0
                        intervention_count = 0
                        intervention_steps = 0
                        already_intervened = False
                        client.update()
                        obs, _ = env.reset()
                        episode_pbar.reset()

            if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
                # dump to pickle file
                buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
                demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                if not os.path.exists(buffer_path):
                    os.makedirs(buffer_path)
                if not os.path.exists(demo_buffer_path):
                    os.makedirs(demo_buffer_path)
                with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(transitions, f)
                    transitions = []
                with open(
                    os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
                ) as f:
                    pkl.dump(demo_transitions, f)
                    demo_transitions = []

            timer.tock("total")

            if step % config.log_period == 0:
                stats = {"timer": timer.get_average_times()}
                client.request("send-stats", stats)

    episode_pbar.close()


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest is not None:
            start_step = int(os.path.basename(latest)[11:]) + 1
    step = start_step

    latest_update_info = None

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        nonlocal latest_update_info
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        episode = payload.get("environment", {}).get("episode")
        if episode is not None:
            def display_metric(name):
                value = find_metric(latest_update_info, name)
                return "n/a" if value is None else f"{value:.4g}"

            print_green(
                "episode "
                f"success_rate={episode['success_rate']:.3f} "
                f"interventions={episode['intervention_count']} "
                f"intervention_steps={episode['intervention_steps']} "
                f"critic_loss={display_metric('critic_loss')} "
                f"grasp_critic_loss={display_metric('grasp_critic_loss')} "
                f"actor_loss={display_metric('actor_loss')} "
                f"temperature={display_metric('temperature')}"
            )
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    if FLAGS.freeze_actor:
        train_networks_to_update = train_critic_networks_to_update
        print_green("[diagnostic] freeze_actor=True: actor and temperature updates disabled")

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                critic_update_kwargs = {
                    "networks_to_update": train_critic_networks_to_update,
                }
                if isinstance(agent, SACAgentHybridSingleArm):
                    critic_update_kwargs["freeze_shared_encoder"] = (
                        FLAGS.freeze_actor or FLAGS.freeze_shared_encoder
                    )
                agent, critics_info = agent.update(batch, **critic_update_kwargs)

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            update_kwargs = {"networks_to_update": train_networks_to_update}
            if isinstance(agent, SACAgentHybridSingleArm):
                update_kwargs["freeze_shared_encoder"] = (
                    FLAGS.freeze_actor or FLAGS.freeze_shared_encoder
                )
            agent, update_info = agent.update(
                batch,
                **update_kwargs,
            )
        latest_update_info = update_info
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)
            print_green(f"published network at learner step {step}")

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    light_diffuse = float(os.environ.get("HILSERL_LIGHT_DIFFUSE", "1.0"))

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=bool(FLAGS.learner),
        save_video=FLAGS.save_video,
        classifier=False,
    )
    env = RecordEpisodeStatistics(env)

    if light_diffuse != 1.0:
        from experiments.pick_cube_sim.lighting_utils import LightingConfig, apply_lighting
        apply_lighting(env.unwrapped.model, LightingConfig(diffuse_scale=light_diffuse))
        print_green(f"Applied lighting perturbation: diffuse_scale={light_diffuse}")

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            actor_anchor_weight=FLAGS.actor_anchor_weight,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        if os.environ.get("HILSERL_AUTO_RESUME") != "1":
            input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt_dir = os.path.abspath(FLAGS.checkpoint_path)
        latest = checkpoints.latest_checkpoint(ckpt_dir)
        if latest is not None:
            ckpt = checkpoints.restore_checkpoint(ckpt_dir, agent.state)
            agent = agent.replace(state=ckpt)
            ckpt_number = os.path.basename(latest)[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
        else:
            print_green(f"No checkpoint in {ckpt_dir}; starting from scratch.")

    def create_replay_buffer_and_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        log_dir = os.path.join(FLAGS.checkpoint_path, "tensorboard")
        print_green(f"TensorBoard logs: {log_dir}")
        return replay_buffer, TensorBoardLogger(log_dir)

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
