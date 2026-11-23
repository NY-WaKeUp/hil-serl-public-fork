import numpy as np

from franka_env.envs.aubo_control_config import (
    IMAGE_CROP,
    IMAGE_KEY_FRONT,
    IMAGE_KEY_WRIST,
    IMAGE_TCP_LOCAL_PORT,
    IMAGE_TCP_MANUAL_TUNNEL,
    IMAGE_TCP_REMOTE_PORT,
    IMAGE_TCP_SERVER_SCRIPT,
    IMAGE_TCP_SERVER_TOPICS,
    IMAGE_TCP_SSH_HOST,
    MAX_EPISODE_LENGTH,
    ROBOT_IP,
    ROBOT_PORT,
    TCP_IMAGE_CAMERAS,
)
from franka_env.envs.aubo_hil_wrappers import (
    AuboAutoInterventionWrapper,
    AuboHilInterventionInfoWrapper,
    AuboKeyboardController,
    AuboKeyboardWrapper,
    AuboObsWrapper,
    GymHilCameraDisplayWrapper,
)
from franka_env.envs.aubo_i10_real_env import AuboPickCubeRealEnv, AuboRealEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig


class EnvConfig(AuboRealEnvConfig):
    """Manjaro cameras via SSH+TCP JPEG relay; arm/gripper via SSH tunnel."""

    ROBOT_IP = ROBOT_IP
    ROBOT_PORT = ROBOT_PORT

    USE_ROS_GRIPPER = False
    USE_TCP_GRIPPER = True
    GRIPPER_SSH_HOST = "Manjaro"
    GRIPPER_TCP_LOCAL_PORT = 32086
    GRIPPER_TCP_MANUAL_TUNNEL = False

    USE_TCP_IMAGES = True
    IMAGE_TCP_SSH_HOST = IMAGE_TCP_SSH_HOST
    IMAGE_TCP_LOCAL_PORT = IMAGE_TCP_LOCAL_PORT
    IMAGE_TCP_REMOTE_PORT = IMAGE_TCP_REMOTE_PORT
    IMAGE_TCP_MANUAL_TUNNEL = IMAGE_TCP_MANUAL_TUNNEL
    IMAGE_TCP_SERVER_SCRIPT = IMAGE_TCP_SERVER_SCRIPT
    IMAGE_TCP_SERVER_TOPICS = IMAGE_TCP_SERVER_TOPICS
    TCP_IMAGE_CAMERAS = TCP_IMAGE_CAMERAS
    ROS_IMAGE_CAMERAS = {}
    REALSENSE_CAMERAS = {}
    IMAGE_CROP = IMAGE_CROP

    # --- Real HIL-RL safety (see docs/2026-06-22_真机RL安全探索范围.md) ---
    # Slower policy + SAFE joint caps (not teleop-fast limits).
    TELEOP_JOINT_LIMITS = False
    MAX_LIN_VEL = 0.01
    MAX_ANG_VEL = 0.05
    # Sphere centered at each reset TCP; keep arm on scene side.
    ENABLE_RELATIVE_SAFETY_BOX = True
    SAFETY_SPHERE_RADIUS = 0.15
    # Hard X floor (base frame); tune with print_tcp_pose.py after measuring layout.
    ABS_X_MIN = 0.02
    # RPY limits only [roll, pitch, yaw]; XYZ is handled by safety sphere.
    ABS_POSE_LIMIT_LOW = np.array(
        [-np.inf, -np.inf, -np.inf, -np.pi, -0.6, -np.pi], dtype=np.float64
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [np.inf, np.inf, np.inf, np.pi, 0.6, np.pi], dtype=np.float64
    )


class TrainConfig(DefaultTrainingConfig):
    """Aubo i10 real robot — same action/obs/camera keys as aubo_pick_cube_sim."""

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
        env = AuboPickCubeRealEnv(config=EnvConfig(), fake_env=fake_env)

        if not fake_env:
            controller = AuboKeyboardController()
            controller.start()
            env = AuboKeyboardWrapper(env, controller)

        def state_builder(unwrapped):
            if unwrapped.fake_env:
                return {
                    "tcp_pose": np.zeros(6, dtype=np.float32),
                    "tcp_vel": np.zeros(6, dtype=np.float32),
                    "tcp_force": np.zeros(3, dtype=np.float32),
                    "tcp_torque": np.zeros(3, dtype=np.float32),
                    "gripper_pose": np.zeros(1, dtype=np.float32),
                    "q": np.zeros(6, dtype=np.float32),
                    "dq": np.zeros(6, dtype=np.float32),
                }
            return unwrapped.get_cached_proprio_state()

        env = AuboObsWrapper(
            env, image_keys=self.image_keys, state_builder=state_builder
        )

        if not fake_env:
            if teleop_record:
                env = AuboAutoInterventionWrapper(env)
            env = GymHilCameraDisplayWrapper(
                env,
                image_keys=self.image_keys,
                window_name="aubo_pick_cube_real",
                display_scale=2,
                display_every_n=2,
            )

        env = AuboHilInterventionInfoWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
