# -*- coding: utf-8 -*-
"""子进程生成轨迹图（避免 matplotlib 与 Qt 同进程死锁）。"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: generate_worker <kwargs.json> <result.json>", file=sys.stderr)
        return 2
    kin = Path(sys.argv[1])
    kout = Path(sys.argv[2])
    try:
        kwargs = json.loads(kin.read_text(encoding="utf-8"))
        from gpx_track.track_map import generate_track_maps

        written = generate_track_maps(**kwargs)
        kout.write_text(
            json.dumps(written, ensure_ascii=False), encoding="utf-8"
        )
        return 0
    except Exception:
        err = traceback.format_exc()
        try:
            kout.write_text(
                json.dumps({"error": err}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
