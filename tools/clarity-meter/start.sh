#!/usr/bin/env bash
set -euo pipefail
TOOL="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${TOOL}${PYTHONPATH:+:$PYTHONPATH}"
cd "$TOOL"

python -c "import PyQt5, cv2, numpy" >/dev/null 2>&1 || {
  echo "[clarity-meter] Missing dependencies. Run: python -m pip install -r requirements.txt" >&2
  exit 1
}

exec python -m clarity_meter "$@"
