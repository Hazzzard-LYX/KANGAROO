#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
exec "$TASK_ROOT/pal_mjlab/.venv/bin/python" training/play.py Mjlab-Velocity-Flat-Kangaroo-Walking "$@"
