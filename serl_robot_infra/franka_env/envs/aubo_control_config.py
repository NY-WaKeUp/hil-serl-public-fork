"""Shared Aubo i10 parameters — aligned with python3.x/smooth_teleop.py (real CAN teleop)."""

import numpy as np

# CAN bus loop (RosDesktopCanbusTeleop.loop_hz)
CAN_LOOP_HZ = 100.0
CAN_DT = 1.0 / CAN_LOOP_HZ

# hil-serl / FrankaEnv policy rate
POLICY_HZ = 10.0
POLICY_DT = 1.0 / POLICY_HZ
CAN_SUBSTEPS_PER_POLICY_STEP = int(CAN_LOOP_HZ / POLICY_HZ)

# Velocity limits at |action|=1 (smooth_teleop max_lin_vel / max_ang_vel defaults)
MAX_LIN_VEL = 0.02  # m/s
MAX_ANG_VEL = 0.1  # rad/s
VEL_EMA_ALPHA = 0.5  # at 10Hz policy rate matches real 100Hz α=0.08 (τ≈0.2s)
CAN_VEL_EMA_ALPHA = 0.08  # smooth_teleop loop_hz=100 default

# IK guard per CAN tick (smooth_teleop ik_max_joint_step)
IK_MAX_JOINT_STEP = 0.025  # rad

# Per policy step displacement at full action (for docs / discrete fallback)
EE_STEP_SIZE = MAX_LIN_VEL * POLICY_DT
RPY_STEP_SIZE = MAX_ANG_VEL * POLICY_DT

# Aubo SDK connection (smooth_teleop main)
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 32085

# Joint limits: teleop mode (smooth_teleop) vs safe startup (my_aubo_control.set_and_startup)
TELEOP_JOINT_MAXACC = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
TELEOP_JOINT_MAXVELC = (1.5, 1.5, 1.5, 1.5, 1.5, 1.5)
SAFE_JOINT_MAXACC = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
SAFE_JOINT_MAXVELC = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
COLLISION_CLASS = 6
TOOL_DYNAMICS_POSITION = (0.0, 0.0, 0.1)
TOOL_PAYLOAD_KG = 1.0

# ROS2 services (smooth_teleop)
GRIPPER_SERVICE = "gripper_setting"
RECORDER_START_SERVICE = "/my_aubo_recorder/start_recording"
RECORDER_END_SERVICE = "/my_aubo_recorder/end_episode"

# Keyboard — same as smooth_teleop declare_parameter defaults
GRIPPER_CLOSE_KEY = "["
GRIPPER_OPEN_KEY = "]"
RECORDER_START_KEY = "q"
RECORDER_END_KEY = "e"

# axis: 0=X 1=Y 2=Z 3=Roll 4=Pitch 5=Yaw
TELEOP_KEY_MAP = {
    "w": (0, 1),
    "s": (0, -1),
    "a": (1, 1),
    "d": (1, -1),
    "r": (2, 1),
    "f": (2, -1),
    "l": (3, 1),
    "j": (3, -1),
    "k": (4, -1),
    "i": (4, 1),
    "o": (5, 1),
    "u": (5, -1),
}

# hil-serl image keys (global pinhole + wrist wide-angle / fisheye)
IMAGE_KEY_FRONT = "front"
IMAGE_KEY_WRIST = "wrist"
IMAGE_KEY_TELEOP = "teleop"
IMAGE_SIZE = 128
TELEOP_DISPLAY_SIZE = 480

# FrankaEnv-aligned capture: 1280×720 color @ ~15fps from device, policy @ POLICY_HZ
CAMERA_CAPTURE_DIM = (1280, 720)
CAMERA_CAPTURE_FPS = 15

# Manjaro ROS topics (when actor machine has ROS2)
ROS_IMAGE_CAMERAS = {
    IMAGE_KEY_FRONT: {
        "topic": "/zed/left/image_raw",
        "dim": CAMERA_CAPTURE_DIM,
    },
    IMAGE_KEY_WRIST: {
        "topic": "/image_raw",
        "dim": CAMERA_CAPTURE_DIM,
    },
}

# Ubuntu without ROS2: JPEG relay via SSH+TCP (image_tcp_server on Manjaro)
IMAGE_TCP_SSH_HOST = "Manjaro"
IMAGE_TCP_LOCAL_PORT = 32087
IMAGE_TCP_REMOTE_PORT = 8767
IMAGE_TCP_MANUAL_TUNNEL = False
IMAGE_TCP_SERVER_SCRIPT = "~/workspace2025/src/inspire_gripper/image_tcp_server.py"
IMAGE_TCP_SERVER_TOPICS = "f:/zed/left/image_raw w:/image_raw"

TCP_IMAGE_CAMERAS = {
    IMAGE_KEY_FRONT: {},
    IMAGE_KEY_WRIST: {},
}

# Optional USB RealSense fallback (same dim/fps as Franka ram_insertion)
REALSENSE_CAMERAS: dict = {}

IMAGE_CROP: dict = {}

MAX_EPISODE_LENGTH = 300

# Inspire rh_r1 closed (matches MyI10Tele RH_R1_QPOS_MAX)
GRIPPER_RH_R1_CLOSED = 0.8663

# Pickup-cube task layout (myscene.xml table + cube geoms)
PICKUP_HOME_QPOS = np.array(
    [
        -1.3445706367492676,
        -0.9264265298843384,
        0.4955613911151886,
        -0.11877074837684631,
        1.5727146863937378,
        -1.3323304653167725,
    ],
    dtype=np.float64,
)
TABLE_TOP_Z = 0.23  # table.xml: geom pos.z + size.z = 0.115 + 0.115
TABLE_GEOM_NAME = "front_object_table"
# half-y=0.45 → y∈[-1.70,-0.80] (table center y=-1.25); cube/tcp y≈-1.24 still on table
PICKUP_TABLE_HALF_Y = 0.45
CUBE_HALF_Z = 0.025
CUBE_ON_TABLE_Z = TABLE_TOP_Z + CUBE_HALF_Z
# MuJoCo quat (w,x,y,z): each row puts a different cube face on +Z (stable on table).
CUBE_FACE_UP_QUATS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # +Z face up (default)
        [0.0, 1.0, 0.0, 0.0],  # -Z face up
        [0.70710678, 0.0, 0.70710678, 0.0],  # -X → +Z
        [0.70710678, 0.0, -0.70710678, 0.0],  # +X → +Z
        [0.70710678, -0.70710678, 0.0, 0.0],  # +Y → +Z
        [0.70710678, 0.70710678, 0.0, 0.0],  # -Y → +Z
    ],
    dtype=np.float64,
)
PICKUP_LIFT_DELTA_Z = 0.10
# agentview: front-above cube from table far side (-Y past cube)
PICKUP_AGENTVIEW_POS = np.array([0.13, -1.66, 0.78], dtype=np.float64)
PICKUP_AGENTVIEW_LOOKAT = np.array([0.13, -1.24, 0.28], dtype=np.float64)
# teleop_view: scene -X side, operator seated here during demo recording
TELEOP_VIEW_POS = np.array([-1.05, -1.05, 0.62], dtype=np.float64)
TELEOP_VIEW_LOOKAT = np.array([0.05, -1.05, 0.30], dtype=np.float64)
TELEOP_CAMERA = "teleop_view"
AGENTVIEW_BRACKET_BODY = "agentview_bracket"
AGENTVIEW_CAMERA = "agentview"
TCP_GRIPPER_BODY = "tcp_link"
CUBE_BODY_NAME = "cube"
ARM_FLANGE_BODY = "i10_inspire_flange_link"
# Fixed cube center at PICKUP_HOME_QPOS (FK once); skip per-reset raycast.
PICKUP_CUBE_XYZ = np.array([0.13201245, -1.23968069, 0.255], dtype=np.float64)
# Block randomization range on table (x ± 0.05, y ± 0.05 around cube center)
PICKUP_BLOCK_SAMPLING_BOUNDS_XY = np.array(
    [[0.08, -1.29], [0.18, -1.19]], dtype=np.float64
)
ARM_JOINT_NAMES = (
    "shoulder_joint",
    "upperArm_joint",
    "foreArm_joint",
    "wrist1_joint",
    "wrist2_joint",
    "wrist3_joint",
)
