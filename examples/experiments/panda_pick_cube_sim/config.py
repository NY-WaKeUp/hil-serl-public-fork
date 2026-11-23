import os

import gymnasium as gym
import gym_hil  # noqa: F401

from franka_env.envs.gym_hil_wrappers import (
    GymHilAutoInterventionWrapper,
    GymHilCameraDisplayWrapper,
    GymHilObsWrapper,
    GymHilSevenDimActionWrapper,
    HilSerlInterventionInfoWrapper,
    install_wasd_keyboard_controller,
)
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig

# Slower EE steps for keyboard teleop (default gym_hil is 0.025 m/step).
SIM_EE_STEP_SIZE = {"x": 0.005, "y": 0.005, "z": 0.005}


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"]
    classifier_keys = ["front"]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose", "q", "dq"]
    buffer_period = 1000
    checkpoint_period = 5000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    max_traj_length = 500
    discount = 0.97

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        teleop_record=False,
    ):
        del save_video, classifier
        reset_delay_seconds = 0.2 if teleop_record else 1.0
        if fake_env:
            # Learner has no display; EGL avoids MuJoCo GL segfault.
            os.environ.setdefault("MUJOCO_GL", "egl")
            env_id = "gym_hil/PandaPickCubeBase-v0"
            env = gym.make(
                env_id,
                render_mode="rgb_array",
                image_obs=True,
                max_episode_steps=1000,
            )
        else:
            # EGL offscreen: stable when TF/JAX already loaded (record_demos, train_rlpd).
            # MuJoCo PassiveViewer (GLFW) segfaults in that setup; use OpenCV display.
            os.environ["MUJOCO_GL"] = "egl"
            env = gym.make(
                "gym_hil/PandaPickCubeKeyboard-v0",
                render_mode="rgb_array",
                image_obs=True,
                use_viewer=False,
                ee_step_size=SIM_EE_STEP_SIZE,
                reset_delay_seconds=reset_delay_seconds,
                max_episode_steps=1000,
            )
            install_wasd_keyboard_controller(env)

        env = GymHilSevenDimActionWrapper(env)
        env = GymHilObsWrapper(env, image_keys=self.image_keys)
        if not fake_env:
            if teleop_record:
                env = GymHilAutoInterventionWrapper(env)
            env = GymHilCameraDisplayWrapper(
                env, image_keys=self.image_keys, display_scale=3
            )
        env = HilSerlInterventionInfoWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
