# -*- coding: utf-8 -*-
"""工具目录路径。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parents[1]


def tool_dir() -> Path:
    return _TOOL_DIR


def default_output_dir() -> Path:
    return _TOOL_DIR / "output"


def config_path() -> Path:
    return _TOOL_DIR / "config.json"


def setup_import_paths() -> None:
    tool = str(_TOOL_DIR)
    if tool not in sys.path:
        sys.path.insert(0, tool)
    os.environ["MTF_METER_DIR"] = tool


def find_window_icon() -> Path | None:
    assets = _TOOL_DIR / "assets"
    repo_res = _TOOL_DIR.parent.parent / "resources"
    for base in (assets, _TOOL_DIR, repo_res):
        for name in ("birdy_logo_128.png", "birdy_logo_640.png", "logo.png"):
            p = base / name
            if p.is_file():
                return p
    return None
