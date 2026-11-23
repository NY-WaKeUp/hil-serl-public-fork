#!/usr/bin/env python3
"""Print Aubo TCP pose and suggested safety-box config (run on real arm at hover pose)."""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_EXAMPLES = os.path.join(_REPO, "examples")
for p in (_REPO, _EXAMPLES):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("AUBO_SDK_DIR", os.path.join(_REPO, "python3.x"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from experiments.aubo_pick_cube_real.config import EnvConfig  # noqa: E402
from franka_env.envs.aubo_i10_real_env import AuboPickCubeRealEnv  # noqa: E402


def main() -> None:
    env = AuboPickCubeRealEnv(config=EnvConfig(), fake_env=False)
    wp = env.robot.get_current_waypoint()
    pos = wp["pos"]
    rpy = env.robot.quaternion_to_rpy(wp["ori"])
    q = wp["joint"]
    tcp = list(pos) + list(rpy)

    print("\n=== Current TCP (base frame) ===")
    print(f"pos  = {pos}")
    print(f"rpy  = {rpy}")
    print(f"q    = {q}")
    print("\nPaste into config.py if using absolute TARGET_POSE:")
    print(f"TARGET_POSE = np.array({tcp!r}, dtype=np.float64)")

    margin_lo = EnvConfig.WORKSPACE_MARGIN_LOW
    margin_hi = EnvConfig.WORKSPACE_MARGIN_HIGH
    lo = [tcp[i] - margin_lo[i] for i in range(6)]
    hi = [tcp[i] + margin_hi[i] for i in range(6)]
    print("\nRelative box at this pose (ENABLE_RELATIVE_SAFETY_BOX=True):")
    print(f"  xyz low  = {[round(lo[i], 4) for i in range(3)]}")
    print(f"  xyz high = {[round(hi[i], 4) for i in range(3)]}")
    print(
        f"  ABS_X_MIN suggestion = {max(0.0, lo[0]):.3f}  "
        "# block motion toward operator (-X)"
    )
    env.close()


if __name__ == "__main__":
    main()
