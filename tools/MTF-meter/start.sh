#!/usr/bin/env bash
set -euo pipefail
TOOL="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${TOOL}${PYTHONPATH:+:$PYTHONPATH}"
cd "$TOOL"

python -c "import PyQt5, cv2, numpy, PIL, matplotlib" >/dev/null 2>&1 || {
  echo "[MTF-meter] Missing dependencies. Run: python -m pip install -r requirements.txt" >&2
  exit 1
}

exec python -m mtf_meter
