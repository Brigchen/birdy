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

# 清晰度统一在该长边尺度上计算，避免小切割图因像素少而虚高
CLARITY_REF_LONG_SIDE = 640
# 计分前高斯抑噪：略强于旧版 σ=1.5，压掉高 ISO 颗粒，仍保留羽、眼结构
CLARITY_DENOISE_SIGMA = 2.2
# 峰度：高斯噪点 ~3，眼/嘴/羽缘等稀疏真边缘 >3。用于压低「糊+噪」虚高
CLARITY_KURT_REF = 3.0
CLARITY_KURT_EXP = 2.4
CLARITY_KFAC_MIN = 0.40
CLARITY_KFAC_MAX = 1.55
# 高 ISO 残差 MAD 惩罚；膝点取 8，避免轻度颗粒（如尚可的 9.jpg）被一票否决
CLARITY_NOISE_MAD_REF = 8.0
CLARITY_NPEN_MIN = 0.45
# 抑噪后 Laplacian 已经很高时视为密边缘（测试条纹/锐利大结构），不再用峰度往下压
CLARITY_TRUST_LAP_SCORE = 50.0


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
    # True：整图判清晰度（已是鸟体切割图时用，避免再取中央框）
    use_full_frame_for_clarity: bool = False


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


def _map_lap_to_0_100(lap: float) -> float:
    return float(min(100.0, 100.0 * math.log1p(max(0.0, lap)) / math.log1p(100.0)))


def _laplacian_kurtosis(vals: np.ndarray) -> float:
    if vals.size < 16:
        return float(CLARITY_KURT_REF)
    sd = float(vals.std())
    if sd < 1e-6:
        return float(CLARITY_KURT_REF)
    mu = float(vals.mean())
    return float(np.mean(((vals - mu) / sd) ** 4))


def _noise_mad(gray: np.ndarray, core: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    resid = gray.astype(np.float64) - blur.astype(np.float64)
    x = resid[core > 0]
    if x.size < 16:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _structure_noise_factors(lap_vals: np.ndarray, gray: np.ndarray, core: np.ndarray) -> Tuple[float, float]:
    """
    返回 (kfac, npen)。
    kfac：Laplacian 峰度，区分稀疏真边缘与近似高斯的颗粒/拖影。
    npen：高频残差 MAD，高 ISO 颗粒越重越往下压。
    """
    kurt = _laplacian_kurtosis(lap_vals)
    kfac = (max(1e-6, kurt) / float(CLARITY_KURT_REF)) ** float(CLARITY_KURT_EXP)
    kfac = float(min(CLARITY_KFAC_MAX, max(CLARITY_KFAC_MIN, kfac)))
    mad = _noise_mad(gray, core)
    npen = 1.0 / (1.0 + (mad / float(CLARITY_NOISE_MAD_REF)) ** 1.5)
    npen = float(min(1.0, max(CLARITY_NPEN_MIN, npen)))
    return kfac, npen


def _map_clarity_with_quality(lap_var: float, lap_vals: np.ndarray, gray: np.ndarray, core: np.ndarray) -> float:
    base = _map_lap_to_0_100(lap_var)
    if base <= 0.0:
        return 0.0
    kfac, npen = _structure_noise_factors(lap_vals, gray, core)
    if base >= float(CLARITY_TRUST_LAP_SCORE):
        kfac = max(kfac, 1.0)
        npen = max(npen, 0.9)
    return float(min(100.0, max(0.0, base * kfac * npen)))


def _scale_gray_and_mask(
    gray: np.ndarray, mask: Optional[np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = gray.shape[:2]
    max_side = max(h, w)
    if max_side <= 0:
        return gray, mask
    if max_side == CLARITY_REF_LONG_SIDE:
        return gray, mask
    if h >= w:
        nh = CLARITY_REF_LONG_SIDE
        nw = max(1, int(round(w * CLARITY_REF_LONG_SIDE / float(h))))
    else:
        nw = CLARITY_REF_LONG_SIDE
        nh = max(1, int(round(h * CLARITY_REF_LONG_SIDE / float(w))))
    interp = cv2.INTER_AREA if max_side > CLARITY_REF_LONG_SIDE else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (nw, nh), interpolation=interp)
    if mask is not None and mask.size > 0:
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return gray, mask


def _crop_to_mask_bbox(
    gray: np.ndarray, mask: np.ndarray, pad: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """只保留掩膜外接框，再交给 640 缩放，避免整图里小鸟被缩太小、峰度被剪影顶满。"""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return gray, mask
    h, w = gray.shape[:2]
    p = max(0, int(pad))
    y0 = max(0, int(ys.min()) - p)
    y1 = min(h, int(ys.max()) + 1 + p)
    x0 = max(0, int(xs.min()) - p)
    x1 = min(w, int(xs.max()) + 1 + p)
    if y1 <= y0 or x1 <= x0:
        return gray, mask
    return gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def clarity_score_0_100(
    bgr: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """
    清晰度 0~100（越高越清晰）。

    长边缩放到 CLARITY_REF_LONG_SIDE 后：较强高斯抑噪 → 掩膜内核 Laplacian 方差，
    再按边缘峰度与高频残差压低「高 ISO 颗粒 / 运动模糊剪影」虚高。
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr
    if mask is not None:
        if mask.shape[:2] != gray.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask = mask.astype(np.uint8)
        if int(mask.sum()) >= 50:
            gray, mask = _crop_to_mask_bbox(gray, mask)
    gray, mask = _scale_gray_and_mask(gray, mask)
    core = np.ones(gray.shape[:2], np.uint8)
    if mask is not None:
        m = (mask > 0).astype(np.uint8)
        if int(m.sum()) >= 80:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            m_e = cv2.erode(m, k, iterations=1)
            core = m_e if int(m_e.sum()) >= 50 else m
    gray_s = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(CLARITY_DENOISE_SIGMA))
    lap = cv2.Laplacian(gray_s, cv2.CV_64F)
    vals = lap[core > 0]
    if vals.size > 1:
        return _map_clarity_with_quality(float(np.var(vals)), vals, gray, core)
    return _map_clarity_with_quality(float(lap.var()), lap.reshape(-1), gray, core)


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


def _clip_bbox(bird: Dict, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    bbox = bird.get("bbox") or []
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _pick_center_bird(
    birds: Sequence[Dict], w: int, h: int
) -> Optional[Dict]:
    """多鸟时取框心最靠近画面中心的个体。"""
    if not birds:
        return None
    cx, cy = w * 0.5, h * 0.5
    best = None
    best_dist2 = float("inf")
    for bird in birds:
        bb = _clip_bbox(bird, w, h)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        bx = 0.5 * (x1 + x2)
        by = 0.5 * (y1 + y2)
        dist2 = (bx - cx) ** 2 + (by - cy) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            best = bird
    return best


def _mask_u8_for_bird(
    bird: Optional[Dict],
    img_w: int,
    img_h: int,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    if not bird:
        return None
    xy = bird.get("mask_xy") or []
    if len(xy) < 3:
        return None
    m = np.zeros((img_h, img_w), np.uint8)
    try:
        pts = np.round(np.asarray(xy, dtype=np.float32)).astype(np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            return None
        cv2.fillPoly(m, [pts], 1)
    except Exception:
        return None
    if roi is not None:
        x1, y1, x2, y2 = roi
        m = m[y1:y2, x1:x2]
    if m.size == 0 or int(m.sum()) < 50:
        return None
    return m


def _center_bird_crop(
    bgr: np.ndarray, birds: Sequence[Dict]
) -> Optional[np.ndarray]:
    """多鸟时取框心最靠近画面中心的个体裁剪，供模糊判定。"""
    h, w = bgr.shape[:2]
    bird = _pick_center_bird(birds, w, h)
    if bird is None:
        return None
    return _bird_bbox_crop(bgr, bird)


def _clarity_crop(
    bgr: np.ndarray, birds: Sequence[Dict]
) -> np.ndarray:
    """模糊判定用裁剪：优先中央鸟体，否则全图。"""
    crop = _center_bird_crop(bgr, birds) if birds else None
    return crop if crop is not None else bgr


def subject_for_clarity(
    bgr: np.ndarray,
    birds: Sequence[Dict],
    *,
    use_full_frame: bool = False,
) -> np.ndarray:
    """切割图已是单只鸟时用整图；大图仍取中央鸟体。"""
    if use_full_frame:
        return bgr
    return _clarity_crop(bgr, birds)


def subject_clarity_score(
    bgr: np.ndarray,
    birds: Sequence[Dict],
    *,
    use_full_frame: bool = False,
) -> float:
    """
    主体清晰度：优先用分割掩膜在鸟体内计分。
    切割图（use_full_frame）在整张切割图上套掩膜；大图则取中央鸟框。
    """
    h, w = bgr.shape[:2]
    bird = _pick_center_bird(birds, w, h)
    if use_full_frame:
        return clarity_score_0_100(bgr, _mask_u8_for_bird(bird, w, h))
    if bird is None:
        return clarity_score_0_100(bgr)
    roi = _clip_bbox(bird, w, h)
    crop = _bird_bbox_crop(bgr, bird)
    if crop is None or roi is None:
        return clarity_score_0_100(bgr)
    return clarity_score_0_100(crop, _mask_u8_for_bird(bird, w, h, roi=roi))


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

        crop = subject_for_clarity(
            bgr, birds, use_full_frame=opts.use_full_frame_for_clarity
        )
        clarity = subject_clarity_score(
            bgr, birds, use_full_frame=opts.use_full_frame_for_clarity
        )

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
    清洗给定文件列表（主流程切割后再识别时用）：按父目录分组去重，直接删文件。
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

        crop = subject_for_clarity(
            bgr, birds, use_full_frame=opts.use_full_frame_for_clarity
        )
        clarity = subject_clarity_score(
            bgr, birds, use_full_frame=opts.use_full_frame_for_clarity
        )
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
