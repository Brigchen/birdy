# -*- coding: utf-8 -*-
"""首帧预览：拖拽框选跟踪目标。"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QRect, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget

NormBox = Tuple[float, float, float, float]


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return image.copy()


_BOX_COLORS = {
    "bird": QColor(0, 200, 255),
    "view": QColor(255, 170, 0),
}


class BoxCanvas(QWidget):
    """按画面比例显示视频帧。可分别拖拽鸟体框和输出画面框。"""

    box_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._boxes: dict[str, Optional[NormBox]] = {"bird": None, "view": None}
        self._active = "bird"
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_now: Optional[Tuple[float, float]] = None
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def has_frame(self) -> bool:
        return self._image is not None and not self._image.isNull()

    def set_active(self, name: str) -> None:
        if name in self._boxes:
            self._active = name
            self._drag_start = None
            self._drag_now = None
            self.update()

    def active(self) -> str:
        return self._active

    def norm_box(self, name: Optional[str] = None) -> Optional[NormBox]:
        return self._boxes.get(name or self._active)

    def set_bgr(self, bgr: np.ndarray) -> None:
        self._image = bgr_to_qimage(bgr)
        self.update()

    def clear_frame(self) -> None:
        self._image = None
        self._boxes = {"bird": None, "view": None}
        self._drag_start = None
        self._drag_now = None
        self.update()
        self.box_changed.emit()

    def clear_box(self, name: Optional[str] = None) -> None:
        key = name or self._active
        if key in self._boxes:
            self._boxes[key] = None
        self._drag_start = None
        self._drag_now = None
        self.update()
        self.box_changed.emit()

    def _dest(self):
        if not self.has_frame():
            return None
        iw, ih = self._image.width(), self._image.height()
        lw, lh = max(1, self.width()), max(1, self.height())
        scale = min(lw / iw, lh / ih)
        dw, dh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox, oy = (lw - dw) // 2, (lh - dh) // 2
        return ox, oy, dw, dh

    def _to_norm(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        dest = self._dest()
        if dest is None:
            return None
        ox, oy, dw, dh = dest
        nx = (x - ox) / dw
        ny = (y - oy) / dh
        if nx < -0.02 or ny < -0.02 or nx > 1.02 or ny > 1.02:
            return None
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.has_frame():
            pt = self._to_norm(event.x(), event.y())
            if pt:
                self._drag_start = pt
                self._drag_now = pt
                self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start:
            pt = self._to_norm(event.x(), event.y())
            if pt:
                self._drag_now = pt
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or not self._drag_start or not self._drag_now:
            return
        x1, y1 = self._drag_start
        x2, y2 = self._drag_now
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        self._drag_start = None
        self._drag_now = None
        if w >= 0.01 and h >= 0.01:
            self._boxes[self._active] = (x, y, w, h)
        self.update()
        self.box_changed.emit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(28, 28, 28))
        dest = self._dest()
        if dest is None:
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "选择视频后拖拽：青色框是鸟，橙色框是输出画面",
            )
            return
        ox, oy, dw, dh = dest
        painter.drawImage(QRect(ox, oy, dw, dh), self._image)
        for name in ("view", "bird"):
            box = self._box_to_draw(name)
            if not box:
                continue
            color = _BOX_COLORS[name]
            painter.setPen(QPen(color, 3 if name == self._active else 2))
            fill = QColor(color)
            fill.setAlpha(50 if name == self._active else 25)
            painter.setBrush(fill)
            x, y, w, h = box
            painter.drawRect(QRectF(ox + x * dw, oy + y * dh, w * dw, h * dh))

    def _box_to_draw(self, name: str) -> Optional[NormBox]:
        if name == self._active and self._drag_start and self._drag_now:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_now
            return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        return self._boxes.get(name)
