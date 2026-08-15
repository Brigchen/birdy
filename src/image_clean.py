# -*- coding: utf-8 -*-
"""分类/待识别鸟图清洗：未检出鸟体、模糊、高度重复。"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from image_io import all_supported_extensions, imread_bgr

ProgressCB = Optional[Callable[[Dict], None]]
CancelCB = Optional[Callable[[], bool]]


@dataclass
class ImageCleanOptions:
    remove_no_bird: bool = True
    remove_blurry: bool = True
    dedupe: bool = True
    # 0~100：最低清晰度，低于则判为模糊（越大越严）
    min_clarity: float = 35.0
    # 0~100：重复相似度，高于则判为重复（越大越严，删得越多）
    dup_similarity: float = 92.0
    bird_conf: float = 0.35
    # 删除后清理空目录
    prune_empty_dirs: bool = True


@dataclass
class ImageCleanResult:
    total: int = 0
    kept: int = 0
    removed_no_bird: int = 0
    removed_blurry: int = 0
    removed_duplicate: int = 0
    failed: int = 0
    removed_paths: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, int]:
        return {
            "total": self.total,
            "kept": self.kept,
            "removed_no_bird": self.removed_no_bird,
            "removed_blurry": self.removed_blurry,
            "removed_duplicate": self.removed_duplicate,
            "failed": self.failed,
            "removed": (
                self.removed_no_bird
                + self.removed_blurry
                + self.removed_duplicate
            ),
        }


def collect_images_recursive(root: str) -> List[str]:
    exts = all_supported_extensions()
    out: List[str] = []
    root_p = Path(root)
    if not root_p.is_dir():
        return out
    for p in root_p.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root_p)
        except Exception:
            continue
        if any(part.startswith(".") or part.startswith("_") for part in rel.parts):
            continue
        if p.suffix.lower() in exts:
            out.append(str(p))
    return sorted(out)


def clarity_score_0_100(bgr: np.ndarray) -> float:
    """
    清晰度 0~100（越高越清晰）。

    先轻度高斯抑噪再算 Laplacian 方差，避免传感器噪点被当成「锐利边缘」
    （失焦+高 ISO 噪点图否则会虚高到 80~90，无法用合理阈值剔除）。
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr
    # 过大图缩小，加速且稳定
    h, w = gray.shape[:2]
    max_side = max(h, w)
    if max_side > 640:
        scale = 640.0 / max_side
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    # σ≈1.5：压掉细粒度噪点，保留真正轮廓/羽枝结构
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # 抑噪后 lap 量级变小，映射参考取 100
    return float(min(100.0, 100.0 * math.log1p(lap) / math.log1p(100.0)))


def dhash64(bgr: np.ndarray, hash_size: int = 8) -> int:
    """差值哈希，返回 64-bit 整数。"""
    if bgr is None or bgr.size == 0:
        return 0
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr
    small = cv2.resize(
        gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA
    )
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            bits |= 1 << i
    return int(bits)


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count()) if hasattr(int, "bit_count") else bin(a ^ b).count("1")


def similarity_to_max_hamming(similarity_0_100: float, bits: int = 64) -> int:
    """相似度阈值 → 允许的最大汉明距离（含）。"""
    s = max(0.0, min(100.0, float(similarity_0_100)))
    # 100% → dist 0；0% → dist bits
    return int(round((1.0 - s / 100.0) * bits))


def _bird_bbox_crop(
    bgr: np.ndarray, bird: Dict
) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    bbox = bird.get("bbox") or []
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2]


def _center_bird_crop(
    bgr: np.ndarray, birds: Sequence[Dict]
) -> Optional[np.ndarray]:
    """多鸟时取框心最靠近画面中心的个体裁剪，供模糊判定。"""
    if not birds:
        return None
    h, w = bgr.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    best = None
    best_dist2 = float("inf")
    for bird in birds:
        bbox = bird.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        bx = 0.5 * (x1 + x2)
        by = 0.5 * (y1 + y2)
        dist2 = (bx - cx) ** 2 + (by - cy) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best = bird
    if best is None:
        return None
    return _bird_bbox_crop(bgr, best)


def _clarity_crop(
    bgr: np.ndarray, birds: Sequence[Dict]
) -> np.ndarray:
    """模糊判定用裁剪：优先中央鸟体，否则全图。"""
    crop = _center_bird_crop(bgr, birds) if birds else None
    return crop if crop is not None else bgr


def _emit(cb: ProgressCB, payload: Dict) -> None:
    if not cb:
        return
    try:
        cb(payload)
    except Exception:
        pass


def _safe_unlink(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except Exception:
        return False


def _prune_empty_dirs(root: str) -> int:
    removed = 0
    root_p = Path(root)
    if not root_p.is_dir():
        return 0
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        p = Path(dirpath)
        if p == root_p:
            continue
        try:
            if not any(p.iterdir()):
                p.rmdir()
                removed += 1
        except Exception:
            pass
    return removed


class _BirdDetector:
    """轻量包装：仅鸟体检测，不加载物种模型。"""

    def __init__(self, bird_conf: float = 0.35):
        from detect_bird_and_eye import BirdAndEyeDetector

        self._det = BirdAndEyeDetector(
            bird_conf=float(bird_conf),
            enable_species=False,
            enable_eye=False,
        )

    def detect(self, bgr: np.ndarray) -> List[Dict]:
        return self._det.detect_birds(bgr)


def clean_bird_images(
    root_folder: str,
    options: Optional[ImageCleanOptions] = None,
    *,
    progress_callback: ProgressCB = None,
    should_cancel: CancelCB = None,
    detector: Optional[_BirdDetector] = None,
) -> ImageCleanResult:
    """
    清洗目录内鸟图（递归）。默认直接删除不合格文件。

    步骤顺序：未检出鸟体 → 模糊 → 同目录高度重复（保留更清晰的一张）。
    """
    opts = options or ImageCleanOptions()
    result = ImageCleanResult()
    root_folder = os.path.normpath(root_folder or "")
    if not root_folder or not os.path.isdir(root_folder):
        raise ValueError(f"清洗目录不存在: {root_folder}")

    images = collect_images_recursive(root_folder)
    result.total = len(images)
    _emit(
        progress_callback,
        {"kind": "start", "done": 0, "total": max(1, result.total)},
    )

    need_detect = bool(opts.remove_no_bird or opts.remove_blurry)
    det = detector
    if need_detect and det is None:
        det = _BirdDetector(bird_conf=opts.bird_conf)

    survivors: List[Tuple[str, float, int]] = []  # path, clarity, dhash
    done = 0

    for path in images:
        if should_cancel and should_cancel():
            break
        done += 1
        bgr = imread_bgr(path, raw_half_size=True)
        if bgr is None:
            result.failed += 1
            _emit(
                progress_callback,
                {
                    "kind": "tick",
                    "done": done,
                    "total": max(1, result.total),
                    "phase": "scan",
                },
            )
            continue

        birds: List[Dict] = []
        if need_detect and det is not None:
            try:
                birds = det.detect(bgr)
            except Exception:
                birds = []

        if opts.remove_no_bird and not birds:
            if _safe_unlink(path):
                result.removed_no_bird += 1
                result.removed_paths.append(path)
            else:
                result.failed += 1
            _emit(
                progress_callback,
                {
                    "kind": "tick",
                    "done": done,
                    "total": max(1, result.total),
                    "phase": "no_bird",
                },
            )
            continue

        # 模糊：多鸟时以最靠近画面中央的鸟体为准
        crop = _clarity_crop(bgr, birds)
        clarity = clarity_score_0_100(crop)

        if opts.remove_blurry and clarity < float(opts.min_clarity):
            if _safe_unlink(path):
                result.removed_blurry += 1
                result.removed_paths.append(path)
            else:
                result.failed += 1
            _emit(
                progress_callback,
                {
                    "kind": "tick",
                    "done": done,
                    "total": max(1, result.total),
                    "phase": "blurry",
                },
            )
            continue

        ph = dhash64(crop)
        survivors.append((path, clarity, ph))
        _emit(
            progress_callback,
            {
                "kind": "tick",
                "done": done,
                "total": max(1, result.total),
                "phase": "keep_scan",
            },
        )

    if opts.dedupe and survivors:
        max_dist = similarity_to_max_hamming(opts.dup_similarity)
        by_dir: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        for item in survivors:
            by_dir[str(Path(item[0]).parent)].append(item)

        kept_items: List[Tuple[str, float, int]] = []
        for _dir, group in by_dir.items():
            # 清晰度从高到低，优先保留清晰图
            group_sorted = sorted(group, key=lambda x: x[1], reverse=True)
            kept_hashes: List[Tuple[str, float, int]] = []
            for path, clarity, ph in group_sorted:
                dup = False
                for _kp, _kc, kh in kept_hashes:
                    if hamming64(ph, kh) <= max_dist:
                        dup = True
                        break
                if dup:
                    if _safe_unlink(path):
                        result.removed_duplicate += 1
                        result.removed_paths.append(path)
                    else:
                        result.failed += 1
                else:
                    kept_hashes.append((path, clarity, ph))
            kept_items.extend(kept_hashes)
        survivors = kept_items

    result.kept = len(survivors)

    if opts.prune_empty_dirs:
        _prune_empty_dirs(root_folder)

    _emit(
        progress_callback,
        {
            "kind": "done",
            "done": max(1, result.total),
            "total": max(1, result.total),
            "result": result.as_dict(),
        },
    )
    return result


def clean_image_list(
    image_paths: Sequence[str],
    options: Optional[ImageCleanOptions] = None,
    *,
    progress_callback: ProgressCB = None,
    should_cancel: CancelCB = None,
) -> ImageCleanResult:
    """
    清洗给定文件列表（主流程识别前用）：按父目录分组去重，直接删文件。
    返回结果后调用方应刷新残留路径列表。
    """
    opts = options or ImageCleanOptions()
    # 按根聚合调用 clean_bird_images 更干净；这里对列表做等价逻辑
    result = ImageCleanResult(total=len(image_paths))
    _emit(
        progress_callback,
        {"kind": "start", "done": 0, "total": max(1, result.total)},
    )

    need_detect = bool(opts.remove_no_bird or opts.remove_blurry)
    det = _BirdDetector(bird_conf=opts.bird_conf) if need_detect else None

    survivors: List[Tuple[str, float, int]] = []
    done = 0

    for path in image_paths:
        if should_cancel and should_cancel():
            break
        done += 1
        bgr = imread_bgr(path, raw_half_size=True)
        if bgr is None:
            result.failed += 1
            _emit(
                progress_callback,
                {"kind": "tick", "done": done, "total": max(1, result.total)},
            )
            continue

        birds: List[Dict] = []
        if det is not None:
            try:
                birds = det.detect(bgr)
            except Exception:
                birds = []

        if opts.remove_no_bird and not birds:
            if _safe_unlink(path):
                result.removed_no_bird += 1
                result.removed_paths.append(path)
            else:
                result.failed += 1
            _emit(
                progress_callback,
                {"kind": "tick", "done": done, "total": max(1, result.total)},
            )
            continue

        crop = _clarity_crop(bgr, birds)
        clarity = clarity_score_0_100(crop)
        if opts.remove_blurry and clarity < float(opts.min_clarity):
            if _safe_unlink(path):
                result.removed_blurry += 1
                result.removed_paths.append(path)
            else:
                result.failed += 1
            _emit(
                progress_callback,
                {"kind": "tick", "done": done, "total": max(1, result.total)},
            )
            continue

        survivors.append((path, clarity, dhash64(crop)))
        _emit(
            progress_callback,
            {"kind": "tick", "done": done, "total": max(1, result.total)},
        )

    if opts.dedupe and survivors:
        max_dist = similarity_to_max_hamming(opts.dup_similarity)
        by_dir: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
        for item in survivors:
            by_dir[str(Path(item[0]).parent)].append(item)
        kept: List[Tuple[str, float, int]] = []
        for group in by_dir.values():
            group_sorted = sorted(group, key=lambda x: x[1], reverse=True)
            kept_hashes: List[Tuple[str, float, int]] = []
            for path, clarity, ph in group_sorted:
                if any(hamming64(ph, kh) <= max_dist for _kp, _kc, kh in kept_hashes):
                    if _safe_unlink(path):
                        result.removed_duplicate += 1
                        result.removed_paths.append(path)
                    else:
                        result.failed += 1
                else:
                    kept_hashes.append((path, clarity, ph))
            kept.extend(kept_hashes)
        survivors = kept

    result.kept = len(survivors)
    _emit(
        progress_callback,
        {
            "kind": "done",
            "done": max(1, result.total),
            "total": max(1, result.total),
            "result": result.as_dict(),
        },
    )
    return result
