"""Aubo i10 sim trainer config — aligned with panda_pick_cube_sim / gym_hil."""

import os

from franka_env.envs.aubo_control_config import (
    IMAGE_KEY_FRONT,
    IMAGE_KEY_WRIST,
    MAX_EPISODE_LENGTH,
)
from franka_env.envs.aubo_hil_wrappers import (
    AuboAutoInterventionWrapper,
    AuboGripperPenaltyWrapper,
    AuboKeyboardController,
    AuboKeyboardWrapper,
    AuboObsWrapper,
    AuboResetDelayWrapper,
    AuboTeleopDisplayWrapper,
)
from franka_env.envs.aubo_i10_gym_env import AuboPickCubeGymEnv
from franka_env.envs.gym_hil_wrappers import HilSerlInterventionInfoWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig


class TrainConfig(DefaultTrainingConfig):
    """Aubo i10 sim — wrapper stack aligned with panda_pick_cube_sim / gym_hil."""

    image_keys = [IMAGE_KEY_FRONT, IMAGE_KEY_WRIST]
    rsa_image_keys = [IMAGE_KEY_WRIST]
    # rsa_image_keys = None
    classifier_keys = [IMAGE_KEY_FRONT]
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose", "q", "dq"]
    buffer_period = 1000
    checkpoint_period = 5000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    max_traj_length = MAX_EPISODE_LENGTH
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
        os.environ.setdefault("MUJOCO_GL", "egl")

        env = AuboPickCubeGymEnv(
            render_mode="rgb_array",
            max_episode_steps=MAX_EPISODE_LENGTH,
            random_block_position=True,
            random_block_orientation=True,
        )
        env = AuboGripperPenaltyWrapper(env, penalty=-0.05)
        env = AuboResetDelayWrapper(env, delay_seconds=reset_delay_seconds)

        if not fake_env:
            controller = AuboKeyboardController()
            controller.start()
            env = AuboKeyboardWrapper(env, controller)

        env = AuboObsWrapper(env, image_keys=self.image_keys)

        if not fake_env:
            if teleop_record:
                env = AuboAutoInterventionWrapper(env)
            env = AuboTeleopDisplayWrapper(
                env,
                window_name="aubo_pick_cube_sim",
                policy_panel_size=200,
                teleop_height=480,
                display_scale=1,
            )

        env = HilSerlInterventionInfoWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
