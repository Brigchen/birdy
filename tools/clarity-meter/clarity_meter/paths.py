# -*- coding: utf-8 -*-
"""工具目录与 Birdy src 路径。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TOOL_DIR.parent.parent
_SRC = _REPO_ROOT / "src"


def tool_dir() -> Path:
    return _TOOL_DIR


def repo_root() -> Path:
    return _REPO_ROOT


def birdy_src() -> Path:
    return _SRC


def default_output_dir() -> Path:
    return _TOOL_DIR / "output"


def setup_import_paths() -> None:
    tool = str(_TOOL_DIR)
    src = str(_SRC)
    if tool not in sys.path:
        sys.path.insert(0, tool)
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ["CLARITY_METER_DIR"] = tool
