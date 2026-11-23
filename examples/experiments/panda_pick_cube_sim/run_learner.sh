#!/bin/bash
# Learner node for sim HIL-RLPD training.
# Edit DEMO_PATH to your recorded demo pkl before first run.
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export MUJOCO_GL=egl
OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-../../checkpoints/panda_pick_cube_sim}"
DEMO_PATH="${DEMO_PATH:-../../demo_data/panda_pick_cube_sim_10_demos_2026-06-21_21-11-29.pkl}"

cd "$(dirname "$0")"
python ../../train_rlpd.py \
    --exp_name=panda_pick_cube_sim \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --demo_path="${DEMO_PATH}" \
    --learner \
    "$@"
