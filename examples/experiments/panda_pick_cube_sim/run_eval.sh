#!/bin/bash
# Evaluate a saved checkpoint (no learner / no keyboard intervention needed).
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export DISPLAY="${DISPLAY:-:52.0}"

OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-../../checkpoints/panda_pick_cube_sim}"
EVAL_STEP="${EVAL_STEP:-5000}"
EVAL_N_TRAJS="${EVAL_N_TRAJS:-20}"

cd "$(dirname "$0")"
python ../../train_rlpd.py \
    --exp_name=panda_pick_cube_sim \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor \
    --eval_checkpoint_step="${EVAL_STEP}" \
    --eval_n_trajs="${EVAL_N_TRAJS}" \
    "$@"
