#!/bin/bash
# Actor node: Aubo i10 real robot + ROS cameras + keyboard intervention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export DISPLAY="${DISPLAY:-:52.0}"
export AUBO_SDK_DIR="$REPO_ROOT/python3.x"

# RTLD_DEEPBIND isolates libpyauboi5's statically-linked protobuf from TF/JAX's.
export LD_PRELOAD="$REPO_ROOT/python3.x/protobuf_deepbind.so:${LD_PRELOAD:-}"

OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/aubo_pick_cube_real_rsa}"
LEARNER_IP="${LEARNER_IP:-localhost}"

echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "LEARNER_IP=${LEARNER_IP}"
echo "Images: SSH+TCP JPEG via Manjaro (no local ROS2)"
echo "Ensure SSH tunnels: 32085 (arm), 32086 (gripper); image uses 32087"

cd "$SCRIPT_DIR"
python ../../train_rlpd.py \
    --exp_name=aubo_pick_cube_real \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor \
    --ip="${LEARNER_IP}" \
    "$@"
