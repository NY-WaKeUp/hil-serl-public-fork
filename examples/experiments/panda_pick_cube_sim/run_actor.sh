#!/bin/bash
# Actor node: policy rollout + keyboard intervention in MuJoCo sim.
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export MUJOCO_GL=egl
export DISPLAY="${DISPLAY:-:52.0}"

OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-../../checkpoints/panda_pick_cube_sim}"

cd "$(dirname "$0")"
python ../../train_rlpd.py \
    --exp_name=panda_pick_cube_sim \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor \
    --ip=localhost \
    "$@"
