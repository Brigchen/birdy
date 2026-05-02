# -*- coding: utf-8 -*-
"""
统一图像扩展名与解码（含常见相机 RAW）。

RAW 依赖 LibRaw（通过 rawpy）；未安装 rawpy 时仍可识别扩展名，
但打开 RAW 会失败并应在调用处提示安装： pip install rawpy
"""

from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, Optional

import cv2
import numpy as np
from PIL import Image

try:
    import rawpy

    _RAWPY_OK = True
except Exception:  # pragma: no cover
    rawpy = None  # type: ignore
    _RAWPY_OK = False

# 与 LibRaw 支持的常见扩展名对齐（小写，含点）
RAW_EXTENSIONS: FrozenSet[str] = frozenset(
    {
        ".3fr",
        ".ari",
        ".arw",
        ".bay",
        ".cr2",
        ".cr3",
        ".crw",
        ".cs1",
        ".dcr",
        ".dcs",
        ".dng",
        ".drf",
        ".eip",
        ".erf",
        ".fff",
        ".iiq",
        ".k25",
        ".kdc",
        ".mdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".orf",
        ".pef",
        ".ptx",
        ".pxn",
        ".r3d",
        ".raf",
        ".raw",
        ".rw2",
        ".rwl",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
)

STANDARD_IMAGE_EXTENSIONS: FrozenSet[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".gif",
    }
)


def all_supported_extensions() -> FrozenSet[str]:
    return RAW_EXTENSIONS | STANDARD_IMAGE_EXTENSIONS


def is_raw_path(path: str) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def rawpy_available() -> bool:
    return bool(_RAWPY_OK)


def read_raw_bgr(path: str, *, half_size: bool = True) -> np.ndarray:
    """将 RAW 解码为 BGR uint8（优先 JPEG 嵌入缩略图，否则 demosaic）。"""
    if not _RAWPY_OK:
        raise RuntimeError("未安装 rawpy，无法读取 RAW。请执行: pip install rawpy")
    with rawpy.imread(path) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                buf = np.frombuffer(thumb.data, dtype=np.uint8)
                im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if im is not None:
                    return im
        except Exception:
            pass
        rgb = raw.postprocess(
            half_size=half_size,
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8,
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def imread_bgr(path: str, *, raw_half_size: bool = True) -> Optional[np.ndarray]:
    """以 BGR 读取普通图或 RAW；失败返回 None。"""
    if is_raw_path(path):
        try:
            return read_raw_bgr(path, half_size=raw_half_size)
        except Exception:
            return None
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im


def open_pil_rgb(path: str, *, raw_half_size: bool = False) -> Optional[Image.Image]:
    """
    打开为 RGB 的 PIL.Image。
    RAW 默认 full demosaic（raw_half_size=False）便于水印/导出；较慢时可传 True。
    """
    if is_raw_path(path):
        try:
            bgr = read_raw_bgr(path, half_size=raw_half_size)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        except Exception:
            return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def file_filter_all_images() -> str:
    """供 QFileDialog 使用的通配字符串。"""
    globs = sorted({("*" + ext) for ext in all_supported_extensions()})
    return "图像 (" + " ".join(globs) + ")"
