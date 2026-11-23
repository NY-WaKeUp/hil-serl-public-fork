#!/bin/bash
# Learner node for Aubo i10 real HIL-RLPD (fake_env; no robot/cameras).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

OPENSSL_LIB="$(brew --prefix openssl@3 2>/dev/null)/lib"
if [ -d "${OPENSSL_LIB}" ]; then
  export LD_LIBRARY_PATH="${OPENSSL_LIB}:${LD_LIBRARY_PATH:-}"
fi

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/checkpoints/aubo_pick_cube_real_rsa}"

_resolve_demo_path() {
  local pattern="$1"
  local -a files=()
  shopt -s nullglob
  files=(
    "$REPO_ROOT/demo_data"/${pattern}
    "$REPO_ROOT/examples/demo_data"/${pattern}
  )
  shopt -u nullglob
  if [ ${#files[@]} -eq 0 ]; then
    return 1
  fi
  ls -t "${files[@]}" | head -1
}

if [ -n "${DEMO_PATH:-}" ] && [ ! -f "${DEMO_PATH}" ]; then
  echo "WARNING: DEMO_PATH is set but missing: ${DEMO_PATH}" >&2
  unset DEMO_PATH
fi

if [ -z "${DEMO_PATH:-}" ]; then
  DEMO_PATH="$(_resolve_demo_path "aubo_pick_cube_real_*_demos_*.pkl" || true)"
  if [ -n "${DEMO_PATH}" ]; then
    echo "Using latest real demo: ${DEMO_PATH}"
  fi
fi

if [ -z "${DEMO_PATH:-}" ]; then
  DEMO_PATH="$(_resolve_demo_path "aubo_pick_cube_sim_*_demos_*.pkl" || true)"
  if [ -n "${DEMO_PATH}" ]; then
    echo "WARNING: No real demos; falling back to sim demo: ${DEMO_PATH}" >&2
  fi
fi

if [ -z "${DEMO_PATH}" ] || [ ! -f "${DEMO_PATH}" ]; then
  echo "No demo pkl found. Searched:" >&2
  echo "  aubo_pick_cube_real_*_demos_*.pkl" >&2
  echo "  aubo_pick_cube_sim_*_demos_*.pkl" >&2
  echo "Record demos or: export DEMO_PATH=/path/to/your.pkl" >&2
  exit 1
fi
echo "Using DEMO_PATH=${DEMO_PATH}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"

cd "$SCRIPT_DIR"
python ../../train_rlpd.py \
    --exp_name=aubo_pick_cube_real \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --demo_path="${DEMO_PATH}" \
    --learner \
    "$@"
