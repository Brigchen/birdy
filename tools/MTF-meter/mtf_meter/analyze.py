# -*- coding: utf-8 -*-
"""单张测量与批量编排（无 Qt）。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .edges import find_edge_rois
from .exif_meta import read_photo_meta
from .image_load import read_bgr, to_luma
from .relative import laplacian_var, tenengrad
from .slanted_edge import cy_px_to_lp_mm, slanted_edge_sfr

ProgressCB = Optional[Callable[[Dict[str, Any]], None]]
CancelCB = Optional[Callable[[], bool]]


def _median(vals: List[float]) -> Optional[float]:
    arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not arr:
        return None
    return float(np.median(arr))


def center_edge_ratio(
    center: Optional[float], edge: Optional[float]
) -> Optional[float]:
    """中心 MTF50 ÷ 四周 MTF50；两边都有效时才有值。"""
    if center is None or edge is None:
        return None
    try:
        c = float(center)
        e = float(edge)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(c) and np.isfinite(e)) or e <= 1e-12:
        return None
    return c / e


def measure_image(path: str) -> Dict[str, Any]:
    meta = read_photo_meta(path)
    bgr = read_bgr(path)
    if bgr is None:
        return {
            "ok": False,
            "error": "无法读取图片",
            "path": path,
            "meta": meta,
        }
    gray = to_luma(bgr)
    rois = find_edge_rois(gray)
    edge_rows: List[Dict[str, Any]] = []
    for roi in rois:
        patch = gray[roi["y1"] : roi["y2"], roi["x1"] : roi["x2"]]
        sfr = slanted_edge_sfr(patch)
        if not sfr:
            continue
        rec = dict(sfr)
        rec["zone"] = roi.get("zone", "")
        rec["roi"] = [roi["x1"], roi["y1"], roi["x2"], roi["y2"]]
        rec["angle_deg"] = roi.get("angle_deg")
        rec["slot"] = roi.get("slot") or rec["zone"]
        edge_rows.append(rec)
    mtf50_all = [e["mtf50_cy_px"] for e in edge_rows]
    mtf50_c = [e["mtf50_cy_px"] for e in edge_rows if e.get("zone") == "center"]
    mtf50_k = [e["mtf50_cy_px"] for e in edge_rows if e.get("zone") == "corner"]
    mtf30_all = [e.get("mtf30_cy_px") for e in edge_rows]
    mtf10_all = [e.get("mtf10_cy_px") for e in edge_rows]
    peaks = [e.get("mtf_peak") for e in edge_rows]
    mtf50 = _median(mtf50_all)
    center_only = _median(mtf50_c) if mtf50_c else None
    corner_only = _median(mtf50_k) if mtf50_k else None
    ppm = meta.get("pixels_per_mm")
    return {
        "ok": True,
        "error": "",
        "path": path,
        "meta": meta,
        "n_edges": len(edge_rows),
        "mtf50_cy_px": mtf50,
        "mtf50_center": center_only if center_only is not None else mtf50,
        "mtf50_corner": corner_only,
        "mtf30_cy_px": _median(mtf30_all),
        "mtf10_cy_px": _median(mtf10_all),
        "mtf_peak": _median(peaks),
        "center_edge_ratio": center_edge_ratio(center_only, corner_only),
        "mtf50_lp_mm": cy_px_to_lp_mm(mtf50, ppm),
        "tenengrad": tenengrad(gray),
        "lap_var": laplacian_var(gray),
        "edges": edge_rows,
    }


def measure_many(
    paths: List[str],
    *,
    progress: ProgressCB = None,
    should_cancel: CancelCB = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = len(paths)
    for p in paths:
        if should_cancel and should_cancel():
            break
        try:
            rec = measure_image(p)
        except Exception as e:
            rec = {
                "ok": False,
                "error": str(e),
                "path": p,
                "meta": read_photo_meta(p),
            }
        rows.append(rec)
        if progress:
            progress(
                {
                    "kind": "row",
                    "done": len(rows),
                    "total": max(1, n),
                    "path": p,
                    "row": rec,
                }
            )
    if progress:
        progress({"kind": "done", "done": len(rows), "total": max(1, n)})
    return rows
