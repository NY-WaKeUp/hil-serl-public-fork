#!/usr/bin/env bash
# systemd --user launcher for lighting-course training on Ke-Wang hil-serl.
#
# Usage:
#   bash train_course.sh start <arm> <seed> <diffuse> [tau] [src_run_dir src_step]
#   bash train_course.sh foreground <arm> <seed> <diffuse> [tau]
#   bash train_course.sh actor <arm> <seed> <diffuse>
#   bash train_course.sh logs [arm] [seed]
#   bash train_course.sh status [arm] [seed]
#   bash train_course.sh tb-url
#   bash train_course.sh stop
#
# Learner runs in the systemd unit (survives disconnect). Actor must run in an
# interactive TTY (`actor`) so KeyBoardIntervention can read WASD/FR/Space/Q.
set -euo pipefail

BASE=/home/agiuser/mani/hil-serl/examples/experiments/pick_cube_sim
PYTHON=/home/agiuser/miniconda3/envs/hilserl-sim/bin/python
DEMO_PKL=$(ls -t "$BASE"/demo_data/self_demo_20_*.pkl | head -1)

CMD=${1:?usage: train_course.sh <start|foreground|actor|logs|status|tb-url|stop> [arm seed diffuse]}
shift || true

ARM=${1:-mech}
SEED=${2:-0}
DIFFUSE=${3:-0.8}
SRC_RUN_DIR=${4:-}
SRC_STEP=${5:-}

if { [ -n "$SRC_RUN_DIR" ] && [ -z "$SRC_STEP" ]; } || { [ -z "$SRC_RUN_DIR" ] && [ -n "$SRC_STEP" ]; }; then
    echo "ERROR: src_run_dir and src_step must be given together"; exit 1
fi

UNIT=hilserl-course-${ARM}-seed${SEED}-train.service
TB_UNIT=hilserl-course-tensorboard.service
RUN_DIR=$BASE/course_${ARM}_diffuse${DIFFUSE}_seed${SEED}
LEARNER_LOG=/tmp/course_${ARM}_seed${SEED}_learner.log
ACTOR_PID_FILE=/tmp/course_${ARM}_seed${SEED}_actor.pid

if [ "$ARM" = "mech" ] || [ "$ARM" = "rank" ] || [ "$ARM" = "rankhalf" ] || [ "$ARM" = "vanilla" ]; then
    :
else
    echo "arm must be mech|rank|rankhalf|vanilla"; exit 1
fi

export_train_env() {
    REPO=/home/agiuser/mani/hil-serl
    export PYTHONPATH="$REPO:$REPO/examples:$REPO/franka_sim:$REPO/serl_robot_infra:$REPO/serl_launcher${PYTHONPATH:+:$PYTHONPATH}"
    export DISPLAY=:10
    export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
    export HILSERL_AUTO_RESUME=1
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export HILSERL_LIGHT_DIFFUSE=$DIFFUSE
}

learner_flags() {
    FREEZE_ACTOR_FLAG=()
    if [ "${HILSERL_FREEZE_ACTOR:-0}" = "1" ]; then
        FREEZE_ACTOR_FLAG=(--freeze_actor)
    fi
    FREEZE_ENCODER_FLAG=()
    if [ "${HILSERL_FREEZE_SHARED_ENCODER:-0}" = "1" ]; then
        FREEZE_ENCODER_FLAG=(--freeze_shared_encoder)
    fi
    ACTOR_ANCHOR_FLAG=()
    if [ -n "${HILSERL_ACTOR_ANCHOR_WEIGHT:-}" ]; then
        ACTOR_ANCHOR_FLAG=(--actor_anchor_weight="$HILSERL_ACTOR_ANCHOR_WEIGHT")
    fi
    RANK_EXPONENT_FLAG=()
    if [ "$ARM" = "rankhalf" ]; then
        RANK_EXPONENT_FLAG=(--injected_rank_exponent=0.5)
    fi
    INJECT_DEMOS_FLAG=()
    if [ "${HILSERL_INJECT_SUCCESS_DEMOS:-1}" = "0" ]; then
        INJECT_DEMOS_FLAG=(--noinject_success_demos)
    fi
}

kill_actor() {
    if [ -f "$ACTOR_PID_FILE" ]; then
        ACTOR_PID=$(cat "$ACTOR_PID_FILE")
        if [ -n "$ACTOR_PID" ] && kill -0 "$ACTOR_PID" 2>/dev/null; then
            kill "$ACTOR_PID"
        fi
        rm -f "$ACTOR_PID_FILE"
    fi
    pkill -f "train_rlpd_sim.py --exp_name=pick_cube_sim --actor" 2>/dev/null || true
}

cd "$BASE"

case "$CMD" in
    start)
        if systemctl --user is-active --quiet "$UNIT"; then
            echo "ERROR: $UNIT is already active. Stop it first: bash $0 stop"; exit 1
        fi
        if ! systemctl --user is-active --quiet "$TB_UNIT"; then
            systemd-run --user --unit="$TB_UNIT" --collect \
                --property=Type=exec \
                "$PYTHON" -m tensorboard.main --logdir "$BASE" --host 127.0.0.1 --port 0
        fi
        if [ "$(loginctl show-user "$USER" --property=Linger --value)" != "yes" ]; then
            echo "ERROR: run 'loginctl enable-linger $USER' first, or the unit dies on logout."; exit 1
        fi
        if [ -d "$RUN_DIR" ]; then
            if [ -n "$SRC_RUN_DIR" ]; then
                echo "WARN: $RUN_DIR already exists; source run $SRC_RUN_DIR@$SRC_STEP ignored."
            fi
        elif [ -n "$SRC_RUN_DIR" ]; then
            SRC_CKPT="$SRC_RUN_DIR/checkpoint_$SRC_STEP"
            if [ ! -d "$SRC_CKPT" ]; then
                echo "ERROR: source checkpoint dir not found: $SRC_CKPT"; exit 1
            fi
            TMP_RUN="$RUN_DIR.tmp.$$"
            mkdir -p "$TMP_RUN"
            cp -r "$SRC_CKPT" "$TMP_RUN/checkpoint_$SRC_STEP"
            if compgen -G "$SRC_RUN_DIR/demo_buffer/injected_*.pkl" > /dev/null; then
                mkdir -p "$TMP_RUN/demo_buffer"
                cp "$SRC_RUN_DIR"/demo_buffer/injected_*.pkl "$TMP_RUN/demo_buffer/"
            fi
            mv "$TMP_RUN" "$RUN_DIR"
            echo "seeded $RUN_DIR from $SRC_CKPT (+ injected demo pkls)"
        else
            mkdir -p "$RUN_DIR"
            if [ -d "$BASE/first_run" ]; then
                cp -r "$BASE/first_run"/. "$RUN_DIR"/
            fi
        fi
        systemd_env=(--setenv=PYTHONUNBUFFERED=1)
        for env_name in PATH VIRTUAL_ENV LD_LIBRARY_PATH PYTHONPATH CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES DISPLAY XAUTHORITY HOME MUJOCO_GL; do
            if [[ -v "$env_name" ]]; then
                systemd_env+=("--setenv=${env_name}=${!env_name}")
            fi
        done
        for env_name in ${!XLA_@} ${!WANDB_@} ${!HILSERL_@}; do
            systemd_env+=("--setenv=${env_name}=${!env_name}")
        done
        systemd-run --user --unit="$UNIT" --collect \
            --property=Type=exec --working-directory="$BASE" \
            "${systemd_env[@]}" \
            /usr/bin/env bash "$0" foreground "$ARM" "$SEED" "$DIFFUSE"
        echo "started learner-only $UNIT"
        echo "Logs:   bash $0 logs $ARM $SEED"
        echo "Status: bash $0 status $ARM $SEED"
        echo "Board:  bash $0 tb-url"
        echo "Stop:   bash $0 stop $ARM $SEED"
        echo "HIL actor (must be an interactive terminal, not systemd):"
        echo "  bash $0 actor $ARM $SEED $DIFFUSE"
        ;;
    foreground)
        export_train_env
        learner_flags
        echo "learner -> $LEARNER_LOG"
        exec "$PYTHON" ../../train_rlpd_sim.py --exp_name=pick_cube_sim --learner --seed="$SEED" \
            "${FREEZE_ACTOR_FLAG[@]}" "${FREEZE_ENCODER_FLAG[@]}" \
            "${ACTOR_ANCHOR_FLAG[@]}" "${RANK_EXPONENT_FLAG[@]}" "${INJECT_DEMOS_FLAG[@]}" \
            --demo_path="$DEMO_PKL" --checkpoint_path="$RUN_DIR" \
            > "$LEARNER_LOG" 2>&1
        ;;
    actor)
        if [ ! -t 0 ]; then
            echo "ERROR: actor stdin is not a TTY. Run this in a real terminal so WASD/FR/Space/Q work."
            exit 1
        fi
        export_train_env
        echo $$ > "$ACTOR_PID_FILE"
        echo "actor TTY teleop: I toggle intervene, W/S x, A/D y, F/R z, Space gripper, Q discard, B fail"
        echo "keys go in THIS terminal, not the MuJoCo windows"
        exec "$PYTHON" ../../train_rlpd_sim.py --exp_name=pick_cube_sim --actor --seed="$SEED" \
            --checkpoint_path="$RUN_DIR"
        ;;
    logs)
        journalctl --user -u "$UNIT" -f
        ;;
    status)
        systemctl --user status "$UNIT" --no-pager || true
        systemctl --user status "$TB_UNIT" --no-pager || true
        echo '--- GPU ---'
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
        echo '--- actor pidfile ---'
        if [ -f "$ACTOR_PID_FILE" ]; then
            cat "$ACTOR_PID_FILE"
            ps -p "$(cat "$ACTOR_PID_FILE")" -o pid,cmd || true
        else
            echo none
        fi
        ;;
    tb-url)
        if ! systemctl --user is-active --quiet "$TB_UNIT"; then
            echo "ERROR: $TB_UNIT is not running. Start training with: bash $0 start <arm> <seed> <diffuse>"
            exit 1
        fi
        TB_URL=$(journalctl --user -u "$TB_UNIT" --no-pager -l | grep -oE 'http://127\.0\.0\.1:[0-9]+' | tail -1 || true)
        if [ -z "$TB_URL" ]; then
            echo "ERROR: TensorBoard is starting; run this command again shortly."
            exit 1
        fi
        echo "$TB_URL"
        ;;
    stop)
        systemctl --user stop "$UNIT" || true
        kill_actor
        sleep 3
        if systemctl --user is-active --quiet "$UNIT"; then
            systemctl --user kill "$UNIT" || true
        fi
        echo "$UNIT stopped. GPU check:"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
        ;;
    *)
        echo "unknown command: $CMD"; exit 1
        ;;
esac
