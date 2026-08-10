# -*- coding: utf-8 -*-
"""
统一图像扩展名与解码（含常见相机 RAW）。

RAW 依赖 LibRaw（通过 rawpy）；未安装 rawpy 时仍可识别扩展名，
但打开 RAW 会失败并应在调用处提示安装： pip install rawpy
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import FrozenSet, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

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


# ==================== DNG 转换（适用于 rawpy 不支持的新相机） ====================

_DNG_CONVERTER_PATH: Optional[str] = None


def find_dng_converter() -> Optional[str]:
    """查找 Adobe DNG Converter 可执行文件路径。

    查找顺序：
    1. 环境变量 DNG_CONVERTER_PATH
    2. Windows 常见安装路径
    3. macOS 常见安装路径

    Returns:
        找到返回路径字符串，未找到返回 None。结果会被缓存。
    """
    global _DNG_CONVERTER_PATH
    if _DNG_CONVERTER_PATH is not None:
        return _DNG_CONVERTER_PATH if _DNG_CONVERTER_PATH else None

    candidates = [
        os.environ.get("DNG_CONVERTER_PATH", ""),
        r"C:\Program Files\Adobe\Adobe DNG Converter.exe",
        r"C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe",
        r"C:\Program Files (x86)\Adobe\Adobe DNG Converter.exe",
        r"C:\Program Files (x86)\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe",
    ]
    if sys.platform == "darwin":
        candidates.append(
            "/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter"
        )

    for p in candidates:
        if p and os.path.isfile(p):
            _DNG_CONVERTER_PATH = p
            return p

    _DNG_CONVERTER_PATH = ""  # 标记已查找但未找到
    return None


def convert_raw_to_dng(
    raw_path: str,
    output_dir: str,
    *,
    timeout: int = 120,
    force: bool = False,
) -> Optional[str]:
    """使用 Adobe DNG Converter 将 RAW 转换为 DNG。

    适用于 rawpy/LibRaw 不支持的新相机（如 Sony A7R VI）。
    DNG 是通用格式，rawpy 可完美读取并做全 demosaic。

    Args:
        raw_path: RAW 文件绝对路径
        output_dir: DNG 输出目录
        timeout: 超时秒数（单张转换通常 2-5 秒）
        force: True 时强制重新转换，False 时若 DNG 已存在且比 RAW 新则跳过

    Returns:
        成功返回 DNG 文件绝对路径，失败返回 None
    """
    converter = find_dng_converter()
    if converter is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    raw_name = os.path.splitext(os.path.basename(raw_path))[0]
    dng_path = os.path.join(output_dir, raw_name + ".dng")

    # 缓存复用：DNG 已存在且比原 RAW 新 → 跳过转换
    if not force and os.path.isfile(dng_path):
        try:
            if os.path.getmtime(dng_path) >= os.path.getmtime(raw_path):
                return dng_path
        except OSError:
            pass

    # Adobe DNG Converter 命令行：
    #   -c   压缩 DNG
    #   -d   输出目录
    cmd = [converter, "-c", "-d", output_dir, raw_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")[:200]
            print(
                f"[image_io] DNG 转换失败 {Path(raw_path).name}: "
                f"returncode={result.returncode} {stderr}",
                flush=True,
            )
            return None
    except subprocess.TimeoutExpired:
        print(
            f"[image_io] DNG 转换超时 {Path(raw_path).name}（{timeout}s）",
            flush=True,
        )
        return None
    except Exception as e:
        print(f"[image_io] DNG 转换异常 {Path(raw_path).name}: {e}", flush=True)
        return None

    if os.path.isfile(dng_path):
        print(
            f"[image_io] DNG 转换成功 {Path(raw_path).name} → {Path(dng_path).name}",
            flush=True,
        )
        return dng_path
    return None


def read_raw_bgr_via_dng(
    raw_path: str,
    *,
    half_size: bool = False,
    dng_cache_dir: Optional[str] = None,
) -> Optional[np.ndarray]:
    """通过 DNG 转换读取 RAW（适用于 rawpy 不支持的新相机如 Sony A7R VI）。

    流程：
      1. 调用 Adobe DNG Converter 将 RAW 转为 DNG（有缓存）
      2. 用 rawpy 读取 DNG 做 demosaic

    Args:
        raw_path: RAW 文件绝对路径
        half_size: 是否半尺寸 demosaic（True 快但分辨率减半）
        dng_cache_dir: DNG 缓存目录，None 则用原 RAW 同目录的 .dng_cache

    Returns:
        BGR uint8 图像，失败返回 None
    """
    if not _RAWPY_OK:
        return None

    if dng_cache_dir is None:
        dng_cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(raw_path)), ".dng_cache"
        )

    dng_path = convert_raw_to_dng(raw_path, dng_cache_dir)
    if dng_path is None:
        return None

    try:
        with rawpy.imread(dng_path) as raw:
            rgb = raw.postprocess(
                half_size=half_size,
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[image_io] DNG 读取失败 {Path(raw_path).name}: {e}", flush=True)
        return None


def copy_exif_from_raw_to_jpeg(raw_path: str, jpg_path: str) -> bool:
    """从 RAW 文件复制 EXIF 到 JPEG。

    使用 piexif 读取 RAW 的 EXIF（拍摄时间、相机参数、GPS 等），写入 JPEG。
    适用于 ARW/CR2/NEF 等基于 TIFF 的 RAW 文件。

    Args:
        raw_path: RAW 文件路径
        jpg_path: JPEG 文件路径（必须已存在）

    Returns:
        成功返回 True，失败返回 False
    """
    try:
        import piexif

        exif_dict = piexif.load(raw_path)
        # 清理 thumbnail 避免写入无关数据
        if "thumbnail" in exif_dict and exif_dict["thumbnail"] is not None:
            exif_dict["thumbnail"] = None
        exif_bytes = piexif.dump(exif_dict)
        im = Image.open(jpg_path)
        im.save(jpg_path, "JPEG", exif=exif_bytes, quality=95)
        return True
    except Exception as e:
        print(f"[image_io] EXIF 复制失败 {Path(raw_path).name}: {e}", flush=True)
        return False


def _extract_embedded_jpeg(path: str) -> Optional[np.ndarray]:
    """从 RAW 文件中提取最大的嵌入式 JPEG 预览图（不依赖 LibRaw）。

    适用于 rawpy/LibRaw 不支持的新相机（如 Sony A7R VI）。
    扫描文件中的 FFD8 (JPEG SOI) 标记，解码所有 JPEG 块，返回最大的有效图像。
    """
    try:
        with open(path, "rb") as fp:
            data = fp.read()
    except Exception:
        return None

    best_img: Optional[np.ndarray] = None
    best_pixels = 0
    i = 0
    while True:
        idx = data.find(b"\xff\xd8", i)
        if idx == -1:
            break
        # 合法 JPEG marker：SOI 后跟 FFE0/FFE1/FFDB/FFC0/FFC2/FFC4
        if idx + 3 < len(data) and data[idx + 2] == 0xFF and data[idx + 3] in (
            0xE0, 0xE1, 0xDB, 0xC0, 0xC2, 0xC4,
        ):
            eoi = data.find(b"\xff\xd9", idx + 2)
            if eoi != -1:
                jpg = data[idx : eoi + 2]
                if len(jpg) > 1024:  # 跳过太小的块（如 160x120 缩略图）
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        pixels = img.shape[0] * img.shape[1]
                        if pixels > best_pixels:
                            best_img = img
                            best_pixels = pixels
        i = idx + 1

    if best_img is not None:
        print(
            f"[image_io] 嵌入式 JPEG fallback 成功 {Path(path).name}: "
            f"{best_img.shape[1]}x{best_img.shape[0]}",
            flush=True,
        )
    return best_img


def read_raw_bgr(path: str, *, half_size: bool = True) -> np.ndarray:
    """将 RAW 解码为 BGR uint8（优先 JPEG 嵌入缩略图，否则 demosaic）。

    rawpy/LibRaw 不支持的新相机（如 Sony A7R VI）会抛 LibRawFileUnsupportedError，
    此时 fallback 到从文件中提取嵌入式 JPEG 预览图。

    注意：本函数优先取嵌入式 JPEG 缩略图（速度快但画质受限），
    适用于连拍筛选评分等对画质要求不高的场景。
    物种检测/裁剪等需要全画质的场景请使用 read_raw_bgr_full。
    """
    if not _RAWPY_OK:
        raise RuntimeError("未安装 rawpy，无法读取 RAW。请执行: pip install rawpy")

    rawpy_error: Optional[Exception] = None
    try:
        with rawpy.imread(path) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    buf = np.frombuffer(thumb.data, dtype=np.uint8)
                    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if im is not None:
                        return im
            except Exception as _e:
                print(f"[image_io] extract_thumb 失败 {Path(path).name}: {_e}", flush=True)
            # 缩略图不可用 → 走 demosaic
            rgb = raw.postprocess(
                half_size=half_size,
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        rawpy_error = e

    # rawpy 失败（如新相机 ARW 不支持）→ fallback 提取嵌入式 JPEG
    print(
        f"[image_io] rawpy 读取失败 {Path(path).name}: {rawpy_error}，尝试嵌入式 JPEG fallback",
        flush=True,
    )
    img = _extract_embedded_jpeg(path)
    if img is not None:
        return img

    # 嵌入式 JPEG 也失败 → 抛原始错误
    raise rawpy_error


def read_raw_bgr_full(path: str, *, half_size: bool = False) -> np.ndarray:
    """将 RAW 解码为 BGR uint8（强制全 demosaic，不取嵌入式 JPEG 缩略图）。

    与 read_raw_bgr 的区别：跳过 extract_thumb()，直接走 raw.postprocess()
    做 demosaic，获得完整画质的图像。适用于物种检测、裁剪、水印等需要
    全画质的场景。

    rawpy 失败时 fallback 到 read_raw_bgr（含嵌入式 JPEG fallback）。

    Args:
        path: RAW 文件路径
        half_size: 是否半尺寸 demosaic，默认 False（全尺寸）
    """
    if not _RAWPY_OK:
        raise RuntimeError("未安装 rawpy，无法读取 RAW。请执行: pip install rawpy")

    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                half_size=half_size,
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(
            f"[image_io] read_raw_bgr_full demosaic 失败 {Path(path).name}: {e}，"
            f"fallback 到 read_raw_bgr",
            flush=True,
        )
        # demosaic 失败 → fallback 到 read_raw_bgr（含嵌入式 JPEG fallback）
        return read_raw_bgr(path, half_size=half_size)


def imread_bgr(path: str, *, raw_half_size: bool = True) -> Optional[np.ndarray]:
    """以 BGR 读取普通图或 RAW；失败返回 None。"""
    if is_raw_path(path):
        try:
            return read_raw_bgr(path, half_size=raw_half_size)
        except Exception as _e:
            print(f"[image_io] read_raw_bgr 失败 {Path(path).name}: {_e}", flush=True)
            return None
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im


def open_pil_rgb(path: str, *, raw_half_size: bool = False) -> Optional[Image.Image]:
    """
    打开为 RGB 的 PIL.Image，并按 EXIF orientation 自动旋转/翻转。
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
        im = Image.open(path)
        # 应用 EXIF orientation，避免相机竖拍图片显示为旋转状态
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        return im.convert("RGB")
    except Exception:
        return None


def file_filter_all_images() -> str:
    """供 QFileDialog 使用的通配字符串。"""
    globs = sorted({("*" + ext) for ext in all_supported_extensions()})
    return "图像 (" + " ".join(globs) + ")"
