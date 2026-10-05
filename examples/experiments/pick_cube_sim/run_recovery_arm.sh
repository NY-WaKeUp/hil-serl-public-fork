#!/usr/bin/env bash
# Deprecated compatibility wrapper. Use train_course.sh directly for new runs.
set -euo pipefail

BASE=/home/agiuser/mani/hil-serl/examples/experiments/pick_cube_sim
exec bash "$BASE/train_course.sh" start "$@"
