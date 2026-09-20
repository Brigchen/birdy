# -*- coding: utf-8 -*-
"""读取普通照片为 BGR（兼容 Windows 中文路径）。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def collect_images(root: str, recursive: bool = True) -> List[str]:
    p = Path(root)
    if not p.is_dir():
        return []
    it: Iterable[Path] = p.rglob("*") if recursive else p.glob("*")
    out: List[str] = []
    for f in it:
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            if any(part.startswith(".") for part in f.relative_to(p).parts):
                continue
            out.append(str(f))
    return sorted(out)


def read_bgr(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def to_luma(bgr: np.ndarray) -> np.ndarray:
    """Rec.709 亮度，float64 0–255。"""
    b = bgr[:, :, 0].astype(np.float64)
    g = bgr[:, :, 1].astype(np.float64)
    r = bgr[:, :, 2].astype(np.float64)
    return 0.0722 * b + 0.7152 * g + 0.2126 * r
