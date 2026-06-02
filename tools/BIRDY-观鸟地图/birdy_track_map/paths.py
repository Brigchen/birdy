# -*- coding: utf-8 -*-
"""工具目录路径与 import 引导（独立运行，不依赖 monorepo 布局）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parents[1]
_RUNTIME_DIR = _TOOL_DIR / "birdy_runtime"
_MONOREPO_SRC = _TOOL_DIR.parent.parent / "src"
_ASSETS_DIR = _TOOL_DIR / "assets"


def tool_dir() -> Path:
    return _TOOL_DIR


def runtime_dir() -> Path:
    """Python 模块目录：优先工具内 birdy_runtime，开发时可回退 monorepo src。"""
    if (_RUNTIME_DIR / "gpx_track").is_dir():
        return _RUNTIME_DIR
    if (_MONOREPO_SRC / "gpx_track").is_dir():
        return _MONOREPO_SRC
    return _RUNTIME_DIR


def src_dir() -> Path:
    """兼容旧名：即 runtime_dir()。"""
    return runtime_dir()


def repo_root() -> Path | None:
    if (_MONOREPO_SRC / "gpx_track").is_dir():
        return _MONOREPO_SRC.parent
    return None


def default_output_dir() -> Path:
    return _TOOL_DIR / "output"


def config_path() -> Path:
    return _TOOL_DIR / "config.json"


def setup_import_paths() -> None:
    rt = str(runtime_dir())
    if rt not in sys.path:
        sys.path.insert(0, rt)
    tool = str(_TOOL_DIR)
    if tool not in sys.path:
        sys.path.insert(0, tool)
    os.environ["BIRDY_TOOL_DIR"] = str(_TOOL_DIR)
    os.environ.pop("BIRDY_AMAP_CONFIG", None)


def find_window_icon() -> Path | None:
    for base in (_ASSETS_DIR, _TOOL_DIR, runtime_dir()):
        for name in ("birdy_logo_128.png", "birdy_logo_640.png", "logo.png"):
            p = base / name
            if p.is_file():
                return p
    root = repo_root()
    if root is not None:
        for name in ("birdy_logo_128.png", "birdy_logo_640.png", "logo.png"):
            p = root / "resources" / name
            if p.is_file():
                return p
    return None
