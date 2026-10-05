#!/usr/bin/env bash
# systemd --user wrapper for course_controller.py.
set -euo pipefail

BASE=/home/agiuser/mani/hil-serl/examples/experiments/pick_cube_sim
PYTHON=/home/agiuser/miniconda3/envs/hilserl-sim/bin/python
CMD=${1:?usage: course_control.sh <start|foreground|logs|status|tb-url|stop> [arm seed tau stages]}
shift || true
ARM=${1:-mech}
SEED=${2:-0}
TAU=${3:-3000}
STAGES=${4:-0.8,0.6,0.4,0.2,0.1}
UNIT=hilserl-course-${ARM}-seed${SEED}-controller.service

case "$CMD" in
    start)
        if systemctl --user is-active --quiet "$UNIT"; then
            echo "ERROR: $UNIT is already active"; exit 1
        fi
        if [ "$(loginctl show-user "$USER" --property=Linger --value)" != "yes" ]; then
            echo "ERROR: run 'loginctl enable-linger $USER' first"; exit 1
        fi
        systemd_env=(--setenv=PYTHONUNBUFFERED=1)
        for env_name in PATH VIRTUAL_ENV LD_LIBRARY_PATH PYTHONPATH CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES ${!XLA_@} ${!WANDB_@}; do
            if [[ -v "$env_name" ]]; then
                systemd_env+=("--setenv=${env_name}=${!env_name}")
            fi
        done
        systemd-run --user --unit="$UNIT" --collect --property=Type=exec --working-directory="$BASE" \
            "${systemd_env[@]}" /usr/bin/env bash "$0" foreground "$ARM" "$SEED" "$TAU" "$STAGES"
        echo "Logs:   bash $0 logs $ARM $SEED"
        echo "Status: bash $0 status $ARM $SEED"
        echo "Board:  bash train_course.sh tb-url $ARM $SEED"
        echo "Stop:   bash $0 stop $ARM $SEED"
        ;;
    foreground)
        export DISPLAY=:10 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
        exec "$PYTHON" course_controller.py --apply --watch --arm "$ARM" --seed "$SEED" --tau "$TAU" --stages "$STAGES"
        ;;
    logs) journalctl --user -u "$UNIT" -f ;;
    status)
        systemctl --user status "$UNIT" --no-pager || true
        bash train_course.sh status "$ARM" "$SEED"
        ;;
    tb-url) bash train_course.sh tb-url "$ARM" "$SEED" ;;
    stop)
        systemctl --user stop "$UNIT" || true
        bash train_course.sh stop "$ARM" "$SEED"
        ;;
    *) echo "unknown command: $CMD"; exit 1 ;;
esac