# -*- coding: utf-8 -*-
"""轨迹图预览：滚轮缩放、拖拽平移（QScrollArea，避免 QGraphicsView 闪退）。"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import QEvent, QPoint, Qt, QTimer
from PyQt5.QtGui import QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _load_pixmap(png_path: str) -> QPixmap:
    path = os.path.normpath(os.path.abspath(png_path))
    max_edge = 2400
    try:
        from PyQt5.QtGui import QImageReader

        reader = QImageReader(path)
        reader.setAutoTransform(True)
        sz = reader.size()
        if sz.isValid():
            w, h = sz.width(), sz.height()
            if w > 0 and h > 0 and max(w, h) > max_edge:
                scale = max_edge / float(max(w, h))
                reader.setScaledSize(
                    sz.scaled(
                        int(w * scale),
                        int(h * scale),
                        Qt.KeepAspectRatio,
                    )
                )
        img = reader.read()
        if not img.isNull():
            return QPixmap.fromImage(img)
    except Exception:
        pass
    pm = QPixmap(path)
    if pm.isNull():
        return pm
    if max(pm.width(), pm.height()) > max_edge:
        pm = pm.scaled(
            max_edge,
            max_edge,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    return pm


class _ZoomPanViewport(QWidget):
    """在滚动区域内显示可缩放图片，滚轮缩放、左键拖拽平移。"""

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._orig = pixmap
        self._scale = 1.0
        self._dragging = False
        self._drag_start = QPoint()
        self._scroll_start_x = 0
        self._scroll_start_y = 0
        self._scroll_area: Optional[QScrollArea] = None

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._apply_scale()

    def set_scroll_area(self, area: QScrollArea) -> None:
        self._scroll_area = area

    def _apply_scale(self) -> None:
        w = max(1, int(self._orig.width() * self._scale))
        h = max(1, int(self._orig.height() * self._scale))
        scaled = self._orig.scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
        self.resize(scaled.size())

    def zoom_by(self, factor: float) -> None:
        self._scale = max(0.05, min(20.0, self._scale * factor))
        self._apply_scale()

    def fit_to_viewport(self) -> None:
        if self._scroll_area is None:
            return
        vp = self._scroll_area.viewport()
        vw = max(1, vp.width())
        vh = max(1, vp.height())
        ow = max(1, self._orig.width())
        oh = max(1, self._orig.height())
        self._scale = min(vw / ow, vh / oh)
        self._apply_scale()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        self.zoom_by(factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if (
            event.button() == Qt.LeftButton
            and self._scroll_area is not None
        ):
            self._dragging = True
            self._drag_start = event.globalPos()
            hbar = self._scroll_area.horizontalScrollBar()
            vbar = self._scroll_area.verticalScrollBar()
            self._scroll_start_x = hbar.value()
            self._scroll_start_y = vbar.value()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._scroll_area is not None:
            delta = event.globalPos() - self._drag_start
            hbar = self._scroll_area.horizontalScrollBar()
            vbar = self._scroll_area.verticalScrollBar()
            hbar.setValue(self._scroll_start_x - delta.x())
            vbar.setValue(self._scroll_start_y - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class TrackMapPreviewPanel(QWidget):
    """可嵌入主窗口的轨迹图预览（滚轮缩放、拖拽平移）。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._viewport: Optional[_ZoomPanViewport] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self._placeholder = QLabel(
            "观鸟地图预览\n\n"
            "点击「预览」或「生成并保存 PNG」后，地图将显示在此区域\n"
            "滚轮缩放 · 按住左键拖拽平移"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #888; font-size: 13px;")
        lay.addWidget(self._placeholder, 1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setStyleSheet("QScrollArea { background-color: #e8e8e8; }")
        self._scroll.hide()
        lay.addWidget(self._scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._fit_btn = QPushButton("适应窗口")
        self._fit_btn.clicked.connect(self.fit_to_viewport)
        self._fit_btn.hide()
        btn_row.addWidget(self._fit_btn)
        lay.addLayout(btn_row)

        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self.fit_to_viewport)

    def set_image(self, png_path: str) -> bool:
        if not png_path or not os.path.isfile(png_path):
            return False
        pm = _load_pixmap(png_path)
        if pm.isNull():
            self._placeholder.setText(f"无法加载图片：\n{png_path}")
            self._placeholder.show()
            self._scroll.hide()
            self._fit_btn.hide()
            return False

        self._placeholder.hide()
        self._scroll.show()
        self._fit_btn.show()
        if self._viewport is not None:
            self._scroll.viewport().removeEventFilter(self)
        self._viewport = _ZoomPanViewport(pm)
        self._viewport.set_scroll_area(self._scroll)
        self._scroll.setWidget(self._viewport)
        self._scroll.viewport().installEventFilter(self)
        self._fit_timer.start(0)
        return True

    def fit_to_viewport(self) -> None:
        if self._viewport is not None:
            self._viewport.fit_to_viewport()
            self._scroll.viewport().update()

    def eventFilter(self, obj, event) -> bool:
        if (
            self._viewport is not None
            and obj is self._scroll.viewport()
            and event.type() == QEvent.Wheel
        ):
            self._viewport.wheelEvent(event)
            return True
        return super().eventFilter(obj, event)


class TrackMapPreviewDialog(QDialog):
    def __init__(
        self,
        parent: Optional[QWidget],
        png_path: str,
        *,
        window_title: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(window_title or "观鸟地图预览")
        self.resize(920, 840)

        lay = QVBoxLayout(self)
        hint = QLabel("滚轮缩放 · 按住左键拖拽平移 · 可拉大窗口")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        lay.addWidget(hint)

        self._pm = _load_pixmap(png_path)
        if self._pm.isNull():
            lay.addWidget(QLabel(f"无法加载图片：\n{png_path}"))
            btns = QDialogButtonBox(QDialogButtonBox.Ok)
            btns.accepted.connect(self.accept)
            lay.addWidget(btns)
            return

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setStyleSheet("QScrollArea { background-color: #e8e8e8; }")

        self._viewport = _ZoomPanViewport(self._pm)
        self._viewport.set_scroll_area(self._scroll)
        self._scroll.setWidget(self._viewport)
        self._scroll.viewport().installEventFilter(self)
        lay.addWidget(self._scroll, stretch=1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        fit_btn = btns.addButton("适应窗口", QDialogButtonBox.ResetRole)
        fit_btn.clicked.connect(self._viewport.fit_to_viewport)
        btns.accepted.connect(self.accept)
        lay.addWidget(btns)

        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._initial_fit)
        self._fit_timer.start(0)

    def _initial_fit(self) -> None:
        vp = getattr(self, "_viewport", None)
        scroll = getattr(self, "_scroll", None)
        if vp is None or scroll is None:
            return
        vp.fit_to_viewport()
        scroll.viewport().update()

    def closeEvent(self, event) -> None:
        if hasattr(self, "_fit_timer"):
            self._fit_timer.stop()
        if hasattr(self, "_scroll") and hasattr(self, "_viewport"):
            try:
                self._scroll.viewport().removeEventFilter(self)
            except Exception:
                pass
        super().closeEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if (
            hasattr(self, "_viewport")
            and hasattr(self, "_scroll")
            and obj is self._scroll.viewport()
            and event.type() == QEvent.Wheel
        ):
            self._viewport.wheelEvent(event)
            return True
        return super().eventFilter(obj, event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(self, "_viewport", None) is not None and hasattr(self, "_fit_timer"):
            self._fit_timer.start(50)


def show_track_map_preview(
    parent,
    png_path: str,
    *,
    window_title: Optional[str] = None,
) -> None:
    if not png_path or not os.path.isfile(png_path):
        return
    if parent is not None:
        old = getattr(parent, "_track_map_preview_dialog", None)
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
            parent._track_map_preview_dialog = None
    dlg = TrackMapPreviewDialog(parent, png_path, window_title=window_title)
    if parent is not None:
        parent._track_map_preview_dialog = dlg

        def _clear_preview_ref(_result: int = 0) -> None:
            if getattr(parent, "_track_map_preview_dialog", None) is dlg:
                parent._track_map_preview_dialog = None

        dlg.finished.connect(_clear_preview_ref)
    dlg.exec_()
