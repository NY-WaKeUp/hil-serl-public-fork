#!/bin/bash
# Actor node: Aubo i10 sim policy rollout + keyboard intervention.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export MUJOCO_GL=egl
export DISPLAY="${DISPLAY:-:52.0}"
export MYI10TELE_ROOT="${MYI10TELE_ROOT:-$HOME/MyI10Tele}"

OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/aubo_pick_cube_sim_rsa_random_cube}"

cd "$SCRIPT_DIR"
python ../../train_rlpd.py \
    --exp_name=aubo_pick_cube_sim \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor \
    --ip=localhost \
    "$@"
