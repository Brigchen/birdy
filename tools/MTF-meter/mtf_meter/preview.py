# -*- coding: utf-8 -*-
"""把自动斜边 ROI 画到预览图上，便于核对测量框是否落在标板上。"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

_CENTER = (46, 160, 60)
_CORNER = (0, 140, 255)
_OTHER = (200, 200, 40)


def overlay_rois(
    bgr: np.ndarray,
    edges: Optional[Sequence[Dict[str, Any]]] = None,
    max_side: int = 900,
) -> np.ndarray:
    """
    在图上画测量框：绿色=中下斜方块，橙色=上/左/右。
    框内再画一条与检测角度一致的斜线。
    """
    if bgr is None or bgr.size == 0:
        return np.zeros((120, 160, 3), dtype=np.uint8)
    vis = np.ascontiguousarray(bgr.copy())
    h, w = vis.shape[:2]
    scale = 1.0
    m = max(h, w)
    if m > max_side > 0:
        scale = float(max_side) / float(m)
        vis = cv2.resize(
            vis,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    thick = max(2, int(round(max(vis.shape[:2]) * 0.0035)))
    font_sc = max(0.38, thick * 0.22)
    for e in edges or []:
        roi = e.get("roi") or []
        if len(roi) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in roi]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        zone = str(e.get("zone") or "")
        slot = str(e.get("slot") or "")
        tag_map = {
            "center": "C",
            "tl": "TL",
            "top": "T",
            "tr": "TR",
            "left": "L",
            "right": "R",
            "bl": "BL",
            "bottom": "B",
            "br": "BR",
        }
        if slot in tag_map:
            tag = tag_map[slot]
            color = _CENTER if slot == "center" else _CORNER
        elif zone == "center":
            color = _CENTER
            tag = "C"
        elif zone == "corner":
            color = _CORNER
            tag = "K"
        else:
            color = _OTHER
            tag = "E"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        ang = e.get("angle_deg")
        if ang is not None:
            try:
                rad = math.radians(float(ang))
                half = 0.42 * math.hypot(x2 - x1, y2 - y1)
                dx = half * math.cos(rad)
                dy = half * math.sin(rad)
                p1 = (int(round(cx - dx)), int(round(cy - dy)))
                p2 = (int(round(cx + dx)), int(round(cy + dy)))
                cv2.line(vis, p1, p2, color, max(1, thick - 1), cv2.LINE_AA)
            except (TypeError, ValueError):
                pass
        mtf = e.get("mtf50_cy_px")
        if mtf is not None:
            try:
                tag = f"{tag} {float(mtf):.3f}"
            except (TypeError, ValueError):
                pass
        ty = y1 - 6 if y1 > 18 else y1 + 16
        cv2.putText(
            vis,
            tag,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_sc,
            color,
            max(1, thick // 2),
            cv2.LINE_AA,
        )
    # 图例
    cv2.putText(
        vis,
        "C=below-center  T/L/R",
        (8, max(16, vis.shape[0] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.4, font_sc),
        (220, 220, 220),
        max(1, thick // 2),
        cv2.LINE_AA,
    )
    return vis
