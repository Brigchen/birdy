# -*- coding: utf-8 -*-
"""无标板时的相对清晰度，以及同场景组内归一化比较。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def tenengrad(gray: np.ndarray) -> float:
    g = np.asarray(gray, dtype=np.float64)
    gx = np.gradient(g, axis=1)
    gy = np.gradient(g, axis=0)
    return float(np.mean(gx * gx + gy * gy))


def laplacian_var(gray: np.ndarray) -> float:
    g = np.asarray(gray, dtype=np.float64)
    # 简易 4-邻域拉普拉斯
    k = (
        np.roll(g, 1, 0)
        + np.roll(g, -1, 0)
        + np.roll(g, 1, 1)
        + np.roll(g, -1, 1)
        - 4.0 * g
    )
    return float(np.var(k))


def _finite_median(vals: List[Optional[float]]) -> Optional[float]:
    arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not arr:
        return None
    return float(np.median(arr))


def relative_score(row: Dict[str, Any]) -> float:
    """综合分：优先 MTF50，否则 Tenengrad。"""
    m = row.get("mtf50_cy_px")
    if m is not None and np.isfinite(m) and float(m) > 0:
        return float(m)
    t = row.get("tenengrad")
    if t is not None and np.isfinite(t):
        return float(t)
    return 0.0


def rank_scene(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    同一场景内相对比较：最优为 1.00，其余为相对值。
    不修改原行，返回带 relative / rank 的副本摘要。
    """
    if not rows:
        return []
    scored = []
    for r in rows:
        scored.append((relative_score(r), r))
    best = max(v for v, _ in scored) if scored else 0.0
    best = best if best > 1e-12 else 1.0
    ranked = sorted(scored, key=lambda x: -x[0])
    out = []
    for i, (sc, r) in enumerate(ranked, start=1):
        meta = r.get("meta") or {}
        out.append(
            {
                "path": meta.get("path") or r.get("path"),
                "file": meta.get("file") or r.get("file"),
                "lens": meta.get("lens"),
                "focal_mm": meta.get("focal_mm"),
                "iso": meta.get("iso"),
                "aperture": meta.get("aperture"),
                "camera": meta.get("camera"),
                "mtf50_cy_px": r.get("mtf50_cy_px"),
                "mtf30_cy_px": r.get("mtf30_cy_px"),
                "mtf10_cy_px": r.get("mtf10_cy_px"),
                "mtf_peak": r.get("mtf_peak"),
                "mtf50_center": r.get("mtf50_center"),
                "mtf50_corner": r.get("mtf50_corner"),
                "center_edge_ratio": r.get("center_edge_ratio"),
                "tenengrad": r.get("tenengrad"),
                "lap_var": r.get("lap_var"),
                "n_edges": r.get("n_edges"),
                "edges": r.get("edges") or [],
                "ok": r.get("ok"),
                "error": r.get("error"),
                "score": sc,
                "relative": sc / best,
                "rank": i,
            }
        )
    return out


def median_metric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    return _finite_median([r.get(key) for r in rows])
