#!/bin/bash
# Push image_tcp_server.py from Ubuntu dev tree → Manjaro runtime path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SSH_HOST="${SSH_HOST:-Manjaro}"
REMOTE_PATH="${REMOTE_PATH:-~/workspace2025/src/inspire_gripper/image_tcp_server.py}"

SRC="${REPO_ROOT}/python3.x/image_tcp_server.py"
if [ ! -f "${SRC}" ]; then
  echo "missing ${SRC}" >&2
  exit 1
fi

echo "scp ${SRC}"
echo " -> ${SSH_HOST}:${REMOTE_PATH}"
scp "${SRC}" "${SSH_HOST}:${REMOTE_PATH}"
ssh "${SSH_HOST}" "ls -la ${REMOTE_PATH}"
