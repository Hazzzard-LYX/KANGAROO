#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export MUJOCO_GL=glfw
exec "$TASK_ROOT/pal_mjlab/.venv/bin/python" -u training/keyboard_play.py "$@"
