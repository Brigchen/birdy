# -*- coding: utf-8 -*-
"""鸟眼关键帧校准：用 models/birdeye.pt 在锚点附近找眼，供动图锁眼校准使用。

只负责"在给定位置附近找一只眼"，不改变用户锚点的语义：
调用方用"眼-锚偏移"把眼中心换算回锚点，见 burst_anchor.track_layouts_features。
模型缺失或推理失败时静默返回 None，不影响原有导出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EYE_MODEL = PROJECT_ROOT / "models" / "birdeye.pt"


class BirdEyeCalibrator:
    """懒加载 birdeye.pt；每次只在锚点附近的小 ROI 上推理。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.35,
        min_roi: int = 96,
    ):
        self._model_path = str(model_path or DEFAULT_EYE_MODEL)
        self._conf = float(conf)
        self._min_roi = int(min_roi)
        self._model = None
        self._failed = False

    def available(self) -> bool:
        """模型文件是否存在（不代表一定能加载）。"""
        try:
            return Path(self._model_path).is_file()
        except OSError:
            return False

    def _ensure_model(self):
        if self._model is not None or self._failed:
            return self._model
        if not self.available():
            print(
                f"[Birdy 鸟眼校准] 未找到鸟眼模型：{self._model_path}，已跳过校准。",
                flush=True,
            )
            self._failed = True
            return None
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
        except Exception as ex:
            print(f"[Birdy 鸟眼校准] 鸟眼模型加载失败：{ex}", flush=True)
            self._failed = True
            self._model = None
        return self._model

    def detect_anchor(
        self,
        frame_bgr: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        在 (cx, cy) 附近找眼，返回像素坐标 (eye_cx, eye_cy, conf)。

        选择规则：优先离期望位置近的检出（避免左右眼跳变），再比置信度。
        没有可用检出时返回 None。
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        model = self._ensure_model()
        if model is None:
            return None
        h, w = frame_bgr.shape[:2]
        r = max(float(radius), float(self._min_roi) * 0.5, 24.0)
        x0 = max(0, int(round(float(cx) - r)))
        y0 = max(0, int(round(float(cy) - r)))
        x1 = min(w, int(round(float(cx) + r)))
        y1 = min(h, int(round(float(cy) + r)))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        roi = frame_bgr[y0:y1, x0:x1]
        try:
            results = model(roi, conf=float(self._conf), verbose=False)
        except Exception as ex:
            print(f"[Birdy 鸟眼校准] 推理失败：{ex}", flush=True)
            return None

        best = None  # (score, ex, ey, conf)
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            try:
                xy = boxes.xyxy.cpu().numpy()
                cf = boxes.conf.cpu().numpy()
            except Exception:
                continue
            for i in range(len(xy)):
                x1b, y1b, x2b, y2b = [float(v) for v in xy[i]]
                ex = x0 + (x1b + x2b) * 0.5
                ey = y0 + (y1b + y2b) * 0.5
                dist = float(np.hypot(ex - float(cx), ey - float(cy)))
                if dist > r * 1.15:
                    continue
                c = float(cf[i])
                # 距离惩罚让靠近期望位置的眼优先，避免左右眼来回切换
                score = c - 0.35 * (dist / max(r, 1.0))
                if best is None or score > best[0]:
                    best = (score, ex, ey, c)
        if best is None:
            return None
        return float(best[1]), float(best[2]), float(best[3])
