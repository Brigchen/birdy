# -*- coding: utf-8 -*-
"""本地 config.json 读写。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .paths import config_path, default_output_dir

DEFAULT_CONFIG: Dict[str, Any] = {
    "photo_folder": "",
    "gpx_file_path": "",
    "gpx_file_paths": [],
    "output_folder": "",
    "location_name": "",
    "use_gpx_track": True,
    "use_exif_gps": True,
    "radius_km": 1.0,
    "include_elevation": True,
    "wm_logo_path": "",
    "wm_logo_width_ratio": 0.30,
    "gpx_match_exif_tz": "Asia/Shanghai",
    "gpx_match_gpx_tz": "UTC",
}


def load_config(path: Path | None = None) -> Dict[str, Any]:
    p = path or config_path()
    out = deepcopy(DEFAULT_CONFIG)
    if not p.is_file():
        out["output_folder"] = str(default_output_dir())
        return out
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        out["output_folder"] = str(default_output_dir())
        return out
    if isinstance(raw, dict):
        out.update(raw)
    if not (out.get("output_folder") or "").strip():
        out["output_folder"] = str(default_output_dir())
    return out


def save_config(config: Dict[str, Any], path: Path | None = None) -> None:
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
