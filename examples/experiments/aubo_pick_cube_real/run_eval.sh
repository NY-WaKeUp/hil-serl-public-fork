#!/bin/bash
# Sim2Real or real-checkpoint eval on hardware.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
export DISPLAY="${DISPLAY:-:52.0}"
export AUBO_SDK_DIR="$REPO_ROOT/python3.x"

# RTLD_DEEPBIND isolates libpyauboi5's statically-linked protobuf from TF/JAX's.
# Without it, the second protobuf static init crashes with:
#   "File already exists in database: google/protobuf/descriptor.proto"
export LD_PRELOAD="$REPO_ROOT/python3.x/protobuf_deepbind.so:${LD_PRELOAD:-}"

OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

# Sim2Real default: load sim checkpoint; override for real-trained ckpt
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/aubo_pick_cube_sim_rsa}"
EVAL_STEP="${EVAL_STEP:-5000}"
EVAL_N_TRAJS="${EVAL_N_TRAJS:-5}"

echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "EVAL_STEP=${EVAL_STEP} EVAL_N_TRAJS=${EVAL_N_TRAJS}"

cd "$SCRIPT_DIR"
python ../../train_rlpd.py \
    --exp_name=aubo_pick_cube_real \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --actor \
    --eval_checkpoint_step="${EVAL_STEP}" \
    --eval_n_trajs="${EVAL_N_TRAJS}" \
    "$@"
