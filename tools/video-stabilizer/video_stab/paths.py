# -*- coding: utf-8 -*-
"""工具目录与 Birdy 仓库资源路径。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TOOL_DIR.parent.parent


def tool_dir() -> Path:
    return _TOOL_DIR


def repo_root() -> Path:
    return _REPO_ROOT


def default_output_dir() -> Path:
    """稳定结果固定放在这个文件夹，方便和加水印成片放在一起找。"""
    d = Path(r"C:\Users\brigc\Pictures\watermarked")
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_import_paths() -> None:
    tool = str(_TOOL_DIR)
    if tool not in sys.path:
        sys.path.insert(0, tool)


def bundled_ffmpeg() -> Path:
    return _REPO_ROOT / "bin" / "ffmpeg.exe"


def default_bird_seg_model() -> Path:
    return _REPO_ROOT / "models" / "bird-seg.pt"


def find_window_icon() -> Path | None:
    for base in (_TOOL_DIR / "assets", _TOOL_DIR, _REPO_ROOT / "resources"):
        for name in ("birdy_logo_128.png", "birdy_logo_640.png", "logo.png"):
            p = base / name
            if p.is_file():
                return p
    return None
