#!/bin/bash
# Smoke test: TCP image bridge + Aubo env reset (Ubuntu needs no ROS2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export DISPLAY="${DISPLAY:-:52.0}"

echo "=== Aubo real env smoke test (TCP images) ==="
echo "DISPLAY=${DISPLAY}"
echo "Prerequisites:"
echo "  Manjaro: zed + fisheye ROS topics @ ~15Hz"
echo "  Manjaro: ~/workspace2025/src/inspire_gripper/image_tcp_server.py"
echo "           (deploy: bash deploy_manjaro_image_server.sh)"
echo "  Ubuntu:  ssh -L 32085 (arm); image 32087 auto via ImageTcpBridge"
echo ""

cd "$REPO_ROOT/examples"
python -c "
from experiments.mappings import get_train_config

cfg = get_train_config('aubo_pick_cube_real')
env = cfg.get_environment(fake_env=False)
obs, _ = env.reset()
# SERLObsWrapper flattens images to top-level keys; ChunkingWrapper adds batch dim.
shapes = {k: obs[k].shape for k in cfg.image_keys}
print('images', shapes)
print('state shape', obs['state'].shape)
env.close()
assert all(s[-3:] == (128, 128, 3) for s in shapes.values()), shapes
print('SMOKE OK')
"
