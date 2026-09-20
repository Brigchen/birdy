# -*- coding: utf-8 -*-
"""调用 Birdy 认种前清晰度打分，不删除任何文件。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .paths import setup_import_paths

setup_import_paths()

from image_clean import (  # noqa: E402
    collect_images_recursive,
    subject_clarity_score,
    subject_for_clarity,
    clarity_score_0_100,
    _pick_center_bird,
    _mask_u8_for_bird,
    _clip_bbox,
)
from image_io import imread_bgr  # noqa: E402

ProgressCB = Optional[Callable[[Dict[str, Any]], None]]
CancelCB = Optional[Callable[[], bool]]

_PREVIEW_MAX_SIDE = 360
_RED = (0, 0, 255)
_ORANGE = (0, 165, 255)


def preview_with_mask(
    bgr: np.ndarray,
    birds: List[Dict[str, Any]],
    max_side: int = _PREVIEW_MAX_SIDE,
) -> np.ndarray:
    """缩小预览并画掩膜轮廓：计分个体红线，其它检出橙色。不改原图。"""
    h, w = bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w, 1)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    vis = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    if not birds:
        return vis
    center = _pick_center_bird(birds, w, h)
    thick = max(2, int(round(max(nw, nh) * 0.008)))
    for bird in birds:
        color = _RED if bird is center else _ORANGE
        m = _mask_u8_for_bird(bird, w, h)
        if m is not None:
            m_s = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
            fill = vis.copy()
            fill[m_s > 0] = color
            cv2.addWeighted(fill, 0.22, vis, 0.78, 0, vis)
            cnts, _ = cv2.findContours(m_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, thick)
            continue
        bb = _clip_bbox(bird, w, h)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        cv2.rectangle(
            vis,
            (int(round(x1 * scale)), int(round(y1 * scale))),
            (int(round(x2 * scale)), int(round(y2 * scale))),
            color,
            thick,
        )
    return vis


def _jpeg_data_uri(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _verdict(n_birds: int, mask_score: Optional[float], min_clarity: float, error: str) -> str:
    if error:
        return "读图失败"
    if n_birds <= 0:
        return "未检出鸟体"
    if mask_score is None:
        return "无分数"
    if float(mask_score) < float(min_clarity):
        return "模糊"
    return "通过"


def score_one(
    path: str,
    detector,
    *,
    min_clarity: float = 35.0,
    use_full_frame: bool = False,
) -> Dict[str, Any]:
    rel = Path(path).name
    row: Dict[str, Any] = {
        "path": path,
        "name": rel,
        "width": 0,
        "height": 0,
        "n_birds": 0,
        "bbox_score": None,
        "mask_score": None,
        "min_clarity": float(min_clarity),
        "verdict": "读图失败",
        "error": "",
        "preview_uri": "",
    }
    bgr = imread_bgr(path, raw_half_size=True)
    if bgr is None:
        row["error"] = "无法读取"
        return row
    h, w = bgr.shape[:2]
    row["width"] = int(w)
    row["height"] = int(h)
    birds: List[Dict[str, Any]] = []
    if detector is not None:
        try:
            birds = detector.detect(bgr) or []
        except Exception as e:
            row["error"] = f"检测失败: {e}"
            birds = []
    row["n_birds"] = len(birds)
    try:
        crop = subject_for_clarity(bgr, birds, use_full_frame=use_full_frame)
        row["bbox_score"] = round(float(clarity_score_0_100(crop)), 2)
        row["mask_score"] = round(
            float(
                subject_clarity_score(
                    bgr, birds, use_full_frame=use_full_frame
                )
            ),
            2,
        )
    except Exception as e:
        row["error"] = str(e) or row["error"]
    row["verdict"] = _verdict(
        row["n_birds"], row["mask_score"], min_clarity, row["error"]
    )
    if row["verdict"] not in ("读图失败", "无分数"):
        row["error"] = ""
    try:
        row["preview_uri"] = _jpeg_data_uri(preview_with_mask(bgr, birds))
    except Exception:
        row["preview_uri"] = ""
    return row


def score_folder(
    root: str,
    *,
    min_clarity: float = 35.0,
    use_full_frame: bool = False,
    progress: ProgressCB = None,
    should_cancel: CancelCB = None,
    detector=None,
) -> Dict[str, Any]:
    """
    递归打分。detector 为 None 时加载 Birdy `_BirdDetector`。
    绝不删除源文件。
    """
    root = os_norm(root)
    paths = collect_images_recursive(root)
    total = len(paths)
    if progress:
        progress({"kind": "start", "done": 0, "total": max(1, total)})

    det = detector
    det_error = ""
    if det is None and paths:
        try:
            from image_clean import _BirdDetector

            det = _BirdDetector(bird_conf=0.35)
        except Exception as e:
            det_error = str(e)
            det = None

    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(paths, start=1):
        if should_cancel and should_cancel():
            break
        row = score_one(
            path, det, min_clarity=min_clarity, use_full_frame=use_full_frame
        )
        try:
            row["name"] = str(Path(path).resolve().relative_to(Path(root).resolve()))
        except Exception:
            row["name"] = Path(path).name
        if det_error and not row["error"] and row["n_birds"] == 0:
            row["error"] = det_error
        rows.append(row)
        if progress:
            progress(
                {
                    "kind": "tick",
                    "done": i,
                    "total": max(1, total),
                    "path": path,
                    "name": row["name"],
                }
            )

    n_pass = sum(1 for r in rows if r.get("verdict") == "通过")
    n_blur = sum(1 for r in rows if r.get("verdict") == "模糊")
    n_nobird = sum(1 for r in rows if r.get("verdict") == "未检出鸟体")
    n_fail = sum(1 for r in rows if r.get("verdict") == "读图失败")
    return {
        "root": root,
        "min_clarity": float(min_clarity),
        "use_full_frame": bool(use_full_frame),
        "total": len(rows),
        "n_pass": n_pass,
        "n_blur": n_blur,
        "n_nobird": n_nobird,
        "n_fail": n_fail,
        "rows": rows,
    }


def os_norm(root: str) -> str:
    return str(Path(root).expanduser().resolve())
