#!/usr/bin/env bash
set -euo pipefail
TOOL="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${TOOL}:${TOOL}/birdy_runtime${PYTHONPATH:+:$PYTHONPATH}"
cd "$TOOL"

if [[ ! -d "${TOOL}/birdy_runtime/gpx_track" ]]; then
  echo "[BIRDY Track Map] Missing birdy_runtime. Run: python sync_runtime.py" >&2
  exit 1
fi

exec python -m birdy_track_map
