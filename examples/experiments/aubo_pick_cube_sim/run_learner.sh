#!/bin/bash
# Learner node for Aubo i10 sim HIL-RLPD training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
export MUJOCO_GL=egl
export MYI10TELE_ROOT="${MYI10TELE_ROOT:-$HOME/MyI10Tele}"

OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/aubo_pick_cube_sim_rsa_random_cube}"

_resolve_demo_path() {
  local -a files=()
  shopt -s nullglob
  files=(
    "$REPO_ROOT/demo_data"/aubo_pick_cube_sim_*_demos_*.pkl
    "$REPO_ROOT/examples/demo_data"/aubo_pick_cube_sim_*_demos_*.pkl
  )
  shopt -u nullglob
  if [ ${#files[@]} -eq 0 ]; then
    return 1
  fi
  ls -t "${files[@]}" | head -1
}

if [ -n "${DEMO_PATH:-}" ] && [ ! -f "${DEMO_PATH}" ]; then
  echo "WARNING: DEMO_PATH is set but missing: ${DEMO_PATH}" >&2
  echo "         Unset DEMO_PATH or point it to an existing .pkl file." >&2
  unset DEMO_PATH
fi

if [ -z "${DEMO_PATH:-}" ]; then
  DEMO_PATH="$(_resolve_demo_path || true)"
fi

if [ -z "${DEMO_PATH}" ] || [ ! -f "${DEMO_PATH}" ]; then
  echo "No Aubo demo pkl found. Searched:" >&2
  echo "  $REPO_ROOT/demo_data/aubo_pick_cube_sim_*_demos_*.pkl" >&2
  echo "  $REPO_ROOT/examples/demo_data/aubo_pick_cube_sim_*_demos_*.pkl" >&2
  echo "Record demos first, or: export DEMO_PATH=/path/to/your.pkl" >&2
  exit 1
fi
echo "Using DEMO_PATH=${DEMO_PATH}"

cd "$SCRIPT_DIR"
python ../../train_rlpd.py \
    --exp_name=aubo_pick_cube_sim \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --demo_path="${DEMO_PATH}" \
    --learner \
    "$@"
