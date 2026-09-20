# -*- coding: utf-8 -*-
"""本地 config.json。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from .paths import config_path, default_output_dir

DEFAULT_CONFIG: Dict[str, Any] = {
    "iso_folder": "",
    "scene_folder": "",
    "output_folder": "",
    "group_by_lens": True,
    "group_by_focal": True,
    "group_by_iso": False,
    "group_by_aperture": False,
    "group_by_camera": False,
    "scene_mode": "time",
    "scene_gap_sec": 180,
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
