#!/usr/bin/env bash
# Viewer for the loaded-carry policy: fixed tray mass, commanded vs actual velocity overlay.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export MUJOCO_GL=glfw
exec "$TASK_ROOT/pal_mjlab/.venv/bin/python" -u training/play_loaded_carry.py "$@"
