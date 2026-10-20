#!/usr/bin/env bash
# Loaded walking from scratch (tray 1 -> 6 kg adaptive curriculum). See TRAINING_DESIGN.md.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
export MUJOCO_GL=egl
exec "$TASK_ROOT/pal_mjlab/.venv/bin/python" training/train_loaded_carry.py "$@"
