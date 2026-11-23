#!/usr/bin/env python3
"""Record open-loop Aubo sysid trajectory: joint angles + gripper (sim or real).

Sim and real must use the same profile (aubo_sysid_profile.DEFAULT_SYSID_PROFILE).
After sim run, align the real arm to printed q0 before starting the real run.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from absl import app, flags

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBOT_INFRA = _REPO_ROOT / "serl_robot_infra"
if str(_ROBOT_INFRA) not in sys.path:
    sys.path.insert(0, str(_ROBOT_INFRA))

from franka_env.envs.aubo_control_config import (
    IMAGE_SIZE,
    ROBOT_IP,
    ROBOT_PORT,
)  # noqa: E402
from franka_env.envs.aubo_hil_wrappers import build_aubo_state  # noqa: E402
from franka_env.envs.aubo_i10_gym_env import AuboPickCubeGymEnv
from franka_env.envs.aubo_mujoco_controller import MYI10TELE_ROOT  # noqa: E402
from franka_env.envs.aubo_i10_real_env import (  # noqa: E402
    AuboPickCubeRealEnv,
    AuboRealEnvConfig,
)
from franka_env.envs.aubo_sysid_profile import DEFAULT_SYSID_PROFILE  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_enum("platform", "sim", ["sim", "real"], "sim or real robot")
flags.DEFINE_string(
    "output",
    "",
    "Output .npz path (default: ./sysid_data/aubo_sysid_{platform}.npz)",
)
flags.DEFINE_integer("seed", 0, "MuJoCo reset seed (sim only)")
flags.DEFINE_string("robot_ip", ROBOT_IP, "Aubo controller IP (real only)")
flags.DEFINE_integer("robot_port", ROBOT_PORT, "Aubo controller port (real only)")
flags.DEFINE_boolean(
    "tunnel",
    False,
    "Ubuntu preset: arm via localhost:32085 + TCP gripper via Manjaro SSH",
)
flags.DEFINE_boolean(
    "use_ros_gripper",
    True,
    "Call ROS gripper service on real robot (real only; ignored if --tunnel)",
)
flags.DEFINE_boolean(
    "use_tcp_gripper",
    False,
    "TCP gripper via SSH to Manjaro (same as smooth_teleop tcp mode)",
)
flags.DEFINE_string(
    "gripper_ssh_host",
    "Manjaro",
    "SSH host for gripper tcp bridge",
)
flags.DEFINE_integer(
    "gripper_tcp_local_port",
    32086,
    "Local SSH -L port for gripper tcp",
)
flags.DEFINE_boolean(
    "gripper_tcp_manual_tunnel",
    False,
    "Gripper -L already in your ssh -N session; skip auto tunnel",
)
flags.DEFINE_float(
    "reset_delay_s",
    2.0,
    "Seconds to wait after reset before logging q0 (real: time to confirm pose)",
)


class _SysidRealEnv(AuboPickCubeRealEnv):
    """Real env without cameras — proprio logging only."""

    def _init_cameras(self) -> None:
        self.cap = {}

    def _get_images(self) -> dict:
        return {
            "front": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            "wrist": np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        }


def _read_sim_sample(env: AuboPickCubeGymEnv) -> dict:
    q = np.asarray(env.myenv.get_joint_state(), dtype=np.float64).reshape(-1)
    state = build_aubo_state(env.myenv)
    return {
        "q": q,
        "q_arm": q[:6],
        "gripper_openpi": float(q[6]),
        "tcp_pose": np.asarray(state["tcp_pose"], dtype=np.float64),
        "v_filtered": np.asarray(env.velocity_integrator.v_filtered, dtype=np.float64),
    }


def _read_real_sample(env: AuboPickCubeRealEnv) -> dict:
    wp = env.robot.get_current_waypoint()
    q_arm = np.asarray(wp["joint"], dtype=np.float64).reshape(-1)[:6]
    gripper_openpi = 1.0 if env._gripper_closed else 0.0
    q = np.concatenate([q_arm, [gripper_openpi]])
    rpy = np.asarray(env.robot.quaternion_to_rpy(wp["ori"]), dtype=np.float64)
    pos = np.asarray(wp["pos"], dtype=np.float64)
    return {
        "q": q,
        "q_arm": q_arm,
        "gripper_openpi": gripper_openpi,
        "tcp_pose": np.concatenate([pos, rpy]),
        "v_filtered": np.asarray(env.velocity_integrator.v_filtered, dtype=np.float64),
    }


def _print_q0_banner(q0: np.ndarray, platform: str) -> None:
    arm = ", ".join(f"{v:.4f}" for v in q0[:6])
    print("\n" + "=" * 72)
    print(
        f"  SYSID q0 ({platform}) — align real arm to these joint angles (rad) before real run"
    )
    print(f"  q_arm = [{arm}]")
    print(f"  gripper_openpi = {q0[6]:.4f}  (0=open, 1=closed)")
    print("=" * 72 + "\n")


def _make_sim_env(seed: int) -> AuboPickCubeGymEnv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    xml_path = os.path.join(MYI10TELE_ROOT, "assets/aubo_i10_inspire/pickup_scene.xml")
    return AuboPickCubeGymEnv(
        xml_path=xml_path,
        max_episode_steps=DEFAULT_SYSID_PROFILE.num_steps + 10,
        reset_delay_seconds=1.0,
        seed=seed,
    )


def _apply_tunnel_flags() -> None:
    if not FLAGS.tunnel:
        return
    FLAGS.robot_ip = "127.0.0.1"
    FLAGS.robot_port = 32085
    FLAGS.use_ros_gripper = False
    FLAGS.use_tcp_gripper = True


def _make_real_env() -> AuboPickCubeRealEnv:
    _apply_tunnel_flags()
    cfg = AuboRealEnvConfig()
    cfg.ROBOT_IP = FLAGS.robot_ip
    cfg.ROBOT_PORT = FLAGS.robot_port
    cfg.REALSENSE_CAMERAS = {}
    cfg.USE_ROS_GRIPPER = FLAGS.use_ros_gripper and not FLAGS.use_tcp_gripper
    cfg.USE_TCP_GRIPPER = FLAGS.use_tcp_gripper
    cfg.GRIPPER_SSH_HOST = FLAGS.gripper_ssh_host
    cfg.GRIPPER_TCP_LOCAL_PORT = FLAGS.gripper_tcp_local_port
    cfg.GRIPPER_TCP_MANUAL_TUNNEL = FLAGS.gripper_tcp_manual_tunnel
    return _SysidRealEnv(config=cfg, fake_env=False)


def run_sysid_record(platform: str, output_path: Path) -> None:
    profile = DEFAULT_SYSID_PROFILE
    reader = _read_sim_sample if platform == "sim" else _read_real_sample

    if platform == "sim":
        env = _make_sim_env(FLAGS.seed)
    else:
        env = _make_real_env()

    actions = profile.actions
    t0 = time.time()
    env.reset(seed=FLAGS.seed if platform == "sim" else None)
    time.sleep(FLAGS.reset_delay_s)

    samples = [reader(env)]
    samples[0]["step"] = -1
    samples[0]["t"] = 0.0
    _print_q0_banner(samples[0]["q"], platform)

    if platform == "real":
        _apply_tunnel_flags()
        if FLAGS.tunnel:
            print(
                "Tunnel mode: arm 127.0.0.1:32085 (need ssh -L), "
                "gripper TCP via Manjaro SSH"
            )
        print("Real run starts in 3 s — confirm pose and clear workspace.")
        time.sleep(3.0)

    for step_idx in range(profile.num_steps):
        action = actions[step_idx]
        env.step(action)
        sample = reader(env)
        sample["step"] = step_idx
        sample["t"] = time.time() - t0
        samples.append(sample)
        if (step_idx + 1) % 10 == 0:
            q = sample["q"]
            print(
                f"  step {step_idx + 1:3d}/{profile.num_steps} "
                f"t={sample['t']:.2f}s "
                f"q0={q[0]:+.3f} gripper={q[6]:.2f}"
            )

    env.close()

    q_log = np.stack([s["q"] for s in samples], axis=0)
    tcp_log = np.stack([s["tcp_pose"] for s in samples], axis=0)
    v_log = np.stack([s["v_filtered"] for s in samples], axis=0)
    ts = np.array([s["t"] for s in samples], dtype=np.float64)

    meta = {
        "platform": platform,
        "profile_name": profile.name,
        "seed": FLAGS.seed,
        "policy_hz": profile.policy_hz,
        "max_lin_vel": profile.max_lin_vel,
        "max_ang_vel": profile.max_ang_vel,
        "vel_ema_alpha": profile.vel_ema_alpha,
        "num_steps": profile.num_steps,
        "duration_s": profile.duration_s,
        "q0": samples[0]["q"].tolist(),
        "robot_ip": FLAGS.robot_ip if platform == "real" else None,
        "robot_port": FLAGS.robot_port if platform == "real" else None,
        "tunnel": FLAGS.tunnel if platform == "real" else None,
        "use_ros_gripper": FLAGS.use_ros_gripper if platform == "real" else None,
        "use_tcp_gripper": FLAGS.use_tcp_gripper if platform == "real" else None,
        "gripper_ssh_host": FLAGS.gripper_ssh_host if platform == "real" else None,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        actions=actions,
        q=q_log,
        q_arm=q_log[:, :6],
        gripper_openpi=q_log[:, 6],
        tcp_pose=tcp_log,
        v_filtered=v_log,
        timestamps=ts,
        meta_json=json.dumps(meta),
    )

    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved {output_path}")
    print(f"Saved {meta_path}")
    print(
        f"Profile: {profile.name}, {profile.num_steps} steps, {profile.duration_s:.1f} s"
    )


def main(_argv):
    platform = FLAGS.platform
    output = FLAGS.output or f"./sysid_data/aubo_sysid_{platform}.npz"
    run_sysid_record(platform, Path(output))


if __name__ == "__main__":
    app.run(main)
