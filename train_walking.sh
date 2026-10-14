#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export MUJOCO_GL=egl
exec "$TASK_ROOT/pal_mjlab/.venv/bin/python" training/train.py Mjlab-Velocity-Flat-Kangaroo-Walking --env.scene.num-envs 1024 "$@"
