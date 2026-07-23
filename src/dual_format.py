# -*- coding: utf-8 -*-
"""RAW+JPG 双格式文件夹：主流程仅 JPG，可选同步复制对应 RAW。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

from image_io import RAW_EXTENSIONS, all_supported_extensions, is_raw_path

DUAL_FORMAT_OFF = "off"
DUAL_FORMAT_JPG_ONLY = "jpg_only"
DUAL_FORMAT_JPG_COPY_RAW = "jpg_copy_raw"

_JPEG_EXTENSIONS: FrozenSet[str] = frozenset({".jpg", ".jpeg"})


def normalize_dual_format_mode(value: object) -> str:
    s = (str(value or "").strip().lower())
    if s in (DUAL_FORMAT_JPG_ONLY, "jpg", "jpg-only", "jpg only"):
        return DUAL_FORMAT_JPG_ONLY
    if s in (
        DUAL_FORMAT_JPG_COPY_RAW,
        "jpg_copy_raw",
        "jpg+copy_raw",
        "jpg + copy raw",
        "jpg_and_raw",
    ):
        return DUAL_FORMAT_JPG_COPY_RAW
    return DUAL_FORMAT_OFF


def extensions_for_dual_mode(mode: object) -> FrozenSet[str]:
    m = normalize_dual_format_mode(mode)
    if m in (DUAL_FORMAT_JPG_ONLY, DUAL_FORMAT_JPG_COPY_RAW):
        return _JPEG_EXTENSIONS
    return all_supported_extensions()


def path_allowed_for_dual_mode(path: str, mode: object) -> bool:
    return Path(path).suffix.lower() in extensions_for_dual_mode(mode)


def filter_paths_for_dual_mode(paths: List[str], mode: object) -> List[str]:
    m = normalize_dual_format_mode(mode)
    if m == DUAL_FORMAT_OFF:
        return list(paths)
    return [p for p in paths if path_allowed_for_dual_mode(p, mode)]


def find_raw_companion(image_path: str) -> Optional[str]:
    """同目录、同主文件名（不同扩展名）的 RAW，供 JPG 配对复制。"""
    p = Path(image_path)
    if is_raw_path(str(p)):
        return None
    parent = p.parent
    stem = p.stem
    if not parent.is_dir():
        return None
    for ext in RAW_EXTENSIONS:
        cand = parent / f"{stem}{ext}"
        if cand.is_file():
            return str(cand.resolve())
    return None


def screened_raw_dir_for(screened_dir: str) -> str:
    """与 Screened_images 同级的 Screened_raw_images。"""
    base = os.path.dirname(os.path.abspath(screened_dir))
    return os.path.join(base, "Screened_raw_images")


def copy_kept_raw_companions_to_screened(
    result: Dict,
    image_folder: str,
    raw_screened_dir: str,
    *,
    get_kept_paths,
) -> int:
    """
    对已保留的 JPG，将原库中同名 RAW 复制到 raw_screened_dir（保持相对路径）。
    get_kept_paths: 通常为 burst_grouping.get_kept_images。
    """
    image_folder = os.path.abspath(image_folder)
    os.makedirs(raw_screened_dir, exist_ok=True)
    n = 0
    seen_dest: set[str] = set()
    for path in get_kept_paths(result):
        abs_p = os.path.abspath(path)
        if is_raw_path(abs_p):
            continue
        raw_p = find_raw_companion(abs_p)
        if not raw_p or not os.path.isfile(raw_p):
            continue
        try:
            rel = os.path.relpath(raw_p, image_folder)
        except ValueError:
            rel = os.path.basename(raw_p)
        rel = rel.replace("\\", "/")
        dest = os.path.join(raw_screened_dir, rel)
        if dest in seen_dest:
            continue
        dest_dir = os.path.dirname(dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(raw_p, dest)
        seen_dest.add(dest)
        n += 1
    return n
