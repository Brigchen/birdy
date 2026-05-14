# -*- coding: utf-8 -*-
"""连拍 → WebP 动图：弹窗选择、参数与预览（供 birdy_gui 调用）。"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from ecology_jpeg_develop import develop_bgr_ecology_wildlife
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QRect, Qt, QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap

from burst_webp import (
    BurstWebpBuildOptions,
    build_animated_mp4,
    build_animated_webp,
    build_preview_frames_rgb,
    crop_window_rect_pixels,
    gray_world_white_balance,
    infer_crop_center_norm_from_birds,
    infer_shot_interval_ms,
    sort_paths_by_capture_time,
)
from image_io import file_filter_all_images, imread_bgr


def _burst_gui_log(msg: str) -> None:
    """动图弹窗相关操作一律打控制台，GUI 卡住时可对照终端进展。"""
    print(f"[Birdy 动图GUI] {msg}", flush=True)


def _burst_worker_configure_openmp() -> None:
    """
    Windows 上常见：主线程已加载一份 OpenMP（如 NumPy/OpenCV），QThread 内再加载
    PyTorch 会再链入 libiomp5，触发 OMP #15 并整进程退出。在副线程任何 torch/鸟检
    之前调用；KMP 仅在未由用户显式设置时生效（setdefault）。
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def _burst_safe_available_geometry(widget: Optional[QWidget]) -> QRect:
    """
    可用屏幕区域。部分环境（远程桌面、多屏切换、Wayland）下 QApplication.desktop()
    或首屏几何在窗口构造早期无效，若据此 resize 出负宽高会导致 Qt 原生闪退。
    """
    app = QApplication.instance()
    if app is None:
        return QRect(0, 0, 1280, 800)
    g = QRect()
    try:
        ps = app.primaryScreen()
        if ps is not None:
            g = ps.availableGeometry()
    except Exception:
        pass
    if g.width() >= 320 and g.height() >= 240:
        return g
    try:
        dw = app.desktop()
        if dw is not None:
            g = dw.availableGeometry(widget)
    except Exception:
        pass
    if g.width() >= 320 and g.height() >= 240:
        return g
    return QRect(0, 0, 1280, 800)


def _burst_app_state_dir() -> Path:
    """
    本机可写目录（与源码安装位置无关）。
    Windows: %LOCALAPPDATA%\\Birdy；其它: XDG_DATA_HOME/Birdy 或 ~/.local/share/Birdy。
    不可写时退回 burst_webp_dialog.py 所在目录。
    """
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
    else:
        root = os.environ.get("XDG_DATA_HOME") or str(
            Path.home() / ".local" / "share"
        )
    cand = Path(root) / "Birdy"
    try:
        cand.mkdir(parents=True, exist_ok=True)
        if os.access(str(cand), os.W_OK):
            return cand
    except OSError:
        pass
    return Path(__file__).resolve().parent


def _burst_webp_dialog_state_path() -> Path:
    return _burst_app_state_dir() / "burst_webp_dialog_state.json"


def _burst_webp_dialog_state_legacy_path() -> Path:
    """旧版：写在 src/ 旁，在 Program Files 等只读安装下会保存失败。"""
    return Path(__file__).resolve().parent / "burst_webp_dialog_state.json"


def _int_safe_combo_data(combo: QComboBox, default: int) -> int:
    try:
        return int(combo.currentData())
    except (TypeError, ValueError):
        return default


def _burst_align_roi_valid(
    roi: Optional[Tuple[float, float, float, float]], min_span: float = 0.02
) -> bool:
    if roi is None or len(roi) != 4:
        return False
    x0, y0, x1, y1 = (float(t) for t in roi)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (x1 - x0 >= min_span) and (y1 - y0 >= min_span)


def _pil_rgb_to_qimage(pil_img) -> QImage:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    arr = np.ascontiguousarray(np.asarray(pil_img, dtype=np.uint8), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[1] != w or arr.shape[0] != h:
        raise ValueError(f"PIL→QImage：期望 RGB HWC，实际 shape={getattr(arr, 'shape', None)}")
    data = arr.tobytes()
    return QImage(data, w, h, 3 * w, QImage.Format_RGB888).copy()


class BurstCropPreviewWidget(QWidget):
    """
    参考模式：首张 + 红十字（裁剪中心）+ 绿虚线裁剪框 + 可选黄虚线对齐 ROI（拖拽矩形）；
    播放模式：动效帧。点击行为由「红十字 / ROI」模式切换。
    """

    MODE_CROP = 0
    MODE_TRACK = 1

    center_changed = pyqtSignal(float, float)
    track_roi_changed = pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pix: Optional[QPixmap] = None
        self._nx = 0.5
        self._ny = 0.5
        self._retention = 0.94
        self._dest = (0, 0, 1, 1)
        self._iw = 1
        self._ih = 1
        self._playback_frames: List[QImage] = []
        self._playback_idx = 0
        self._mode = BurstCropPreviewWidget.MODE_CROP
        self._roi_norm: Optional[Tuple[float, float, float, float]] = None
        self._drag_a: Optional[Tuple[float, float]] = None
        self._drag_b: Optional[Tuple[float, float]] = None
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self.setMouseTracking(True)

    def stop_playback(self) -> None:
        if self._playback_frames:
            _burst_gui_log("预览播放已停止（退出动效帧轮播）。")
        self._playback_frames = []
        self._playback_idx = 0
        self.update()

    def set_playback_frames(self, frames: List[QImage]) -> None:
        self._playback_frames = list(frames or [])
        self._playback_idx = 0
        n = len(self._playback_frames)
        if n:
            _burst_gui_log(f"进入动效播放模式：共 {n} 帧（更新预览结果）。")
        self.update()

    def set_playback_index(self, idx: int) -> None:
        if not self._playback_frames:
            return
        n = len(self._playback_frames)
        self._playback_idx = int(idx) % n
        self.update()

    def is_playing_back(self) -> bool:
        return len(self._playback_frames) > 0

    def set_reference_bgr(self, bgr: Optional[np.ndarray]) -> None:
        self._base_pix = None
        self._iw = self._ih = 1
        if bgr is None or bgr.size == 0:
            self.update()
            return
        arr = np.ascontiguousarray(bgr, dtype=np.uint8)
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._iw, self._ih = w, h
        qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._base_pix = QPixmap.fromImage(qi)
        self.update()

    def set_center_norm(self, nx: float, ny: float) -> None:
        self._nx = float(np.clip(nx, 0.0, 1.0))
        self._ny = float(np.clip(ny, 0.0, 1.0))
        self.update()

    def set_retention(self, r: float) -> None:
        self._retention = float(np.clip(r, 0.25, 1.0))
        self.update()

    def set_interaction_mode(self, mode: int) -> None:
        self._mode = (
            BurstCropPreviewWidget.MODE_CROP
            if int(mode) == BurstCropPreviewWidget.MODE_CROP
            else BurstCropPreviewWidget.MODE_TRACK
        )
        self._drag_a = self._drag_b = None
        self.update()

    def set_track_roi_norm(
        self, rect: Optional[Tuple[float, float, float, float]]
    ) -> None:
        if rect is None:
            self._roi_norm = None
            self.update()
            return
        x0, y0, x1, y1 = (float(t) for t in rect)
        x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
        y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
        min_sp = 0.02
        if x1 - x0 < min_sp:
            c = (x0 + x1) * 0.5
            x0 = max(0.0, c - min_sp * 0.5)
            x1 = min(1.0, c + min_sp * 0.5)
        if y1 - y0 < min_sp:
            c = (y0 + y1) * 0.5
            y0 = max(0.0, c - min_sp * 0.5)
            y1 = min(1.0, c + min_sp * 0.5)
        self._roi_norm = (x0, y0, x1, y1)
        self.update()

    def clear_track_roi(self) -> None:
        self._roi_norm = None
        self._drag_a = self._drag_b = None
        self.update()

    def _norm_from_widget(self, wx: int, wy: int) -> Optional[Tuple[float, float]]:
        ox, oy, sw, sh = self._dest
        mx, my = wx - ox, wy - oy
        if mx < 0 or my < 0 or mx >= sw or my >= sh or sw <= 0 or sh <= 0:
            return None
        return float(mx) / float(sw), float(my) / float(sh)

    def paintEvent(self, event) -> None:
        del event
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(43, 43, 43))
        pw, ph = self.width(), self.height()
        if pw <= 1 or ph <= 1:
            return

        if self._playback_frames:
            qim = self._playback_frames[self._playback_idx]
            pix = QPixmap.fromImage(qim)
            scal = pix.scaled(pw, ph, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            dox = (pw - scal.width()) // 2
            doy = (ph - scal.height()) // 2
            p.drawPixmap(dox, doy, scal)
            return

        if self._base_pix is None or self._base_pix.isNull():
            p.setPen(QColor(200, 200, 200))
            p.drawText(
                self.rect(),
                Qt.AlignCenter,
                "添加图片后显示首张：红十字=裁剪中心，黄虚线框=连拍对齐 ROI（下方选「对齐 ROI」后在图上拖拽）。\n"
                "绿框=裁剪范围。「更新预览」后此处播动效。",
            )
            return

        iw, ih = self._iw, self._ih
        if iw <= 0 or ih <= 0:
            return
        scale = min(pw / float(iw), ph / float(ih))
        sw = max(1, int(round(iw * scale)))
        sh = max(1, int(round(ih * scale)))
        ox = (pw - sw) // 2
        oy = (ph - sh) // 2
        scal_pix = self._base_pix.scaled(
            sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        dox = ox + (sw - scal_pix.width()) // 2
        doy = oy + (sh - scal_pix.height()) // 2
        dw, dh = scal_pix.width(), scal_pix.height()
        self._dest = (dox, doy, dw, dh)
        p.drawPixmap(dox, doy, scal_pix)

        cx_w = dox + int(round(self._nx * dw))
        cy_w = doy + int(round(self._ny * dh))
        pen_cross = QPen(QColor(255, 60, 60))
        pen_cross.setWidth(2)
        p.setPen(pen_cross)
        p.drawLine(cx_w, doy, cx_w, doy + dh)
        p.drawLine(dox, cy_w, dox + dw, cy_w)

        x0, y0, cw, ch = crop_window_rect_pixels(iw, ih, self._retention, self._nx, self._ny)
        rx0 = dox + int(round(x0 / float(iw) * dw))
        ry0 = doy + int(round(y0 / float(ih) * dh))
        rw = max(1, int(round(cw / float(iw) * dw)))
        rh = max(1, int(round(ch / float(ih) * dh)))
        pen_rect = QPen(QColor(120, 220, 140))
        pen_rect.setWidth(2)
        pen_rect.setStyle(Qt.DashLine)
        p.setPen(pen_rect)
        p.setBrush(Qt.NoBrush)
        p.drawRect(rx0, ry0, rw, rh)

        def _draw_roi_rect(x0n: float, y0n: float, x1n: float, y1n: float) -> None:
            xa = dox + int(round(x0n * dw))
            ya = doy + int(round(y0n * dh))
            xb = dox + int(round(x1n * dw))
            yb = doy + int(round(y1n * dh))
            pen_tr = QPen(QColor(255, 220, 60))
            pen_tr.setWidth(2)
            pen_tr.setStyle(Qt.DashLine)
            p.setPen(pen_tr)
            p.setBrush(Qt.NoBrush)
            p.drawRect(
                min(xa, xb),
                min(ya, yb),
                max(1, abs(xb - xa)),
                max(1, abs(yb - ya)),
            )

        if self._roi_norm is not None:
            _draw_roi_rect(*self._roi_norm)
        if (
            self._mode == BurstCropPreviewWidget.MODE_TRACK
            and self._drag_a is not None
            and self._drag_b is not None
        ):
            x0n, x1n = sorted((self._drag_a[0], self._drag_b[0]))
            y0n, y1n = sorted((self._drag_a[1], self._drag_b[1]))
            _draw_roi_rect(x0n, y0n, x1n, y1n)

    def mousePressEvent(self, e) -> None:
        if self._playback_frames:
            return
        if self._base_pix is None or self._base_pix.isNull():
            return
        if e.button() != Qt.LeftButton:
            return
        pn = self._norm_from_widget(e.x(), e.y())
        if pn is None:
            return
        if self._mode == BurstCropPreviewWidget.MODE_TRACK:
            self._drag_a = pn
            self._drag_b = pn
            self.update()
        else:
            self.set_center_norm(pn[0], pn[1])
            self.center_changed.emit(self._nx, self._ny)

    def mouseMoveEvent(self, e) -> None:
        if self._playback_frames or self._base_pix is None or self._base_pix.isNull():
            return
        if self._mode != BurstCropPreviewWidget.MODE_TRACK or self._drag_a is None:
            return
        pn = self._norm_from_widget(e.x(), e.y())
        if pn is None:
            return
        if e.buttons() & Qt.LeftButton:
            self._drag_b = pn
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._playback_frames or self._base_pix is None or self._base_pix.isNull():
            return
        if e.button() != Qt.LeftButton:
            return
        if self._mode != BurstCropPreviewWidget.MODE_TRACK or self._drag_a is None:
            return
        pn = self._norm_from_widget(e.x(), e.y())
        if pn is None:
            self._drag_a = self._drag_b = None
            self.update()
            return
        self._drag_b = pn
        x0, x1 = sorted((self._drag_a[0], self._drag_b[0]))
        y0, y1 = sorted((self._drag_a[1], self._drag_b[1]))
        self._drag_a = self._drag_b = None
        min_sp = 0.02
        if x1 - x0 < min_sp:
            c = (x0 + x1) * 0.5
            x0, x1 = max(0.0, c - min_sp * 0.5), min(1.0, c + min_sp * 0.5)
        if y1 - y0 < min_sp:
            c = (y0 + y1) * 0.5
            y0, y1 = max(0.0, c - min_sp * 0.5), min(1.0, c + min_sp * 0.5)
        self.set_track_roi_norm((x0, y0, x1, y1))
        if self._roi_norm is not None:
            self.track_roi_changed.emit(*self._roi_norm)
        self.update()


class BurstWebpBuildWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: List[str],
        out_path: str,
        opts: BurstWebpBuildOptions,
        export_format: str = "webp",
        parent=None,
    ):
        super().__init__(parent)
        self._paths = list(paths)
        self._out_path = out_path
        self._opts = opts
        self._export_format = (export_format or "webp").strip().lower()

    def run(self) -> None:
        _burst_worker_configure_openmp()
        try:
            _burst_gui_log(
                f"导出线程开始（{self._export_format}）：{len(self._paths)} 张 → "
                f"{os.path.basename(self._out_path)} "
                f"(WB={self._opts.enable_white_balance}, 显影={self._opts.enable_ecology_develop}, "
                f"对齐={self._opts.enable_align}, 水印={'开' if self._opts.watermark_options else '关'})"
            )

            def _cb(cur: int, tot: int, msg: str) -> None:
                self.progress.emit(cur, tot, msg)

            if self._export_format == "mp4":
                r = build_animated_mp4(
                    self._paths, self._out_path, self._opts, progress=_cb
                )
            else:
                r = build_animated_webp(
                    self._paths, self._out_path, self._opts, progress=_cb
                )
            _burst_gui_log(
                f"导出线程完成：{r.get('n_frames', 0)} 帧 → {r.get('out_path', '')}"
            )
            self.finished_ok.emit(r)
        except Exception as e:
            _burst_gui_log(f"导出线程失败：{e}")
            self.failed.emit(str(e))


class BurstWebpPreviewWorker(QThread):
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(object, float, str, object, float, float)
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: List[str],
        opts: BurstWebpBuildOptions,
        get_detector: Optional[Callable[[], Optional[object]]] = None,
        user_touched_center: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._paths = list(paths)
        self._opts = opts
        self._get_detector = get_detector
        self._user_touched_center = user_touched_center

    def run(self) -> None:
        _burst_worker_configure_openmp()
        try:
            _burst_gui_log(
                f"预览线程开始：{len(self._paths)} 张路径，用户已手调中心={self._user_touched_center}，"
                f"当前中心=({self._opts.crop_center_norm[0]:.4f},{self._opts.crop_center_norm[1]:.4f})"
            )

            opts = self._opts
            ordered = sort_paths_by_capture_time(
                [p for p in self._paths if os.path.isfile(p)]
            )
            if (
                self._get_detector is not None
                and not self._user_touched_center
                and ordered
                and abs(opts.crop_center_norm[0] - 0.5) < 1e-4
                and abs(opts.crop_center_norm[1] - 0.5) < 1e-4
            ):
                _burst_gui_log("预览线程：默认中心，尝试首张鸟检以推断裁剪中心…")
                try:
                    bgr = imread_bgr(ordered[0], raw_half_size=True)
                    if bgr is not None and bgr.size:
                        x = bgr
                        if opts.enable_white_balance:
                            x = gray_world_white_balance(x)
                        if opts.enable_ecology_develop:
                            x = develop_bgr_ecology_wildlife(x)
                        det = self._get_detector()
                        if det is not None:
                            birds = det.detect_birds(x)
                            if birds:
                                nx1, ny1 = infer_crop_center_norm_from_birds(x, birds)
                                opts = replace(
                                    opts, crop_center_norm=(float(nx1), float(ny1))
                                )
                                _burst_gui_log(
                                    f"预览线程：鸟检命中 {len(birds)} 框 → 裁剪中心=({nx1:.4f},{ny1:.4f})"
                                )
                            else:
                                _burst_gui_log("预览线程：首张无鸟框，保持中心 (0.5,0.5)。")
                        else:
                            _burst_gui_log("预览线程：无鸟检测器，跳过首张鸟检。")
                except Exception as ex:
                    _burst_gui_log(f"预览线程：首张鸟检异常（已忽略）：{ex}")
            else:
                _burst_gui_log("预览线程：跳过首张鸟检（用户已设中心或中心非默认）。")

            def _cb(cur: int, tot: int, msg: str) -> None:
                self.progress.emit(cur, tot, f"[预览] {msg}")

            _burst_gui_log("预览线程：调用 build_preview_frames_rgb（终端另有 [Birdy 动图预览] 日志）…")
            pil_list, dur, note, align0 = build_preview_frames_rgb(
                self._paths,
                opts,
                max_long_edge=720,
                max_frames=20,
                progress=_cb,
                log_terminal=True,
            )
            try:
                qimgs = [_pil_rgb_to_qimage(p) for p in pil_list]
            except Exception as ex:
                _burst_gui_log(f"预览线程：PIL→QImage 失败：{ex}")
                self.failed.emit(f"预览转图失败（PIL→QImage）：{ex}")
                return
            cnx, cny = opts.crop_center_norm
            _burst_gui_log(
                f"预览线程完成：{len(qimgs)} 张 QImage，间隔≈{dur:.1f} ms，"
                f"采用裁剪中心=({cnx:.4f},{cny:.4f})"
            )
            self.done.emit(qimgs, float(dur), note, align0, float(cnx), float(cny))
        except Exception as e:
            _burst_gui_log(f"预览线程异常：{e}")
            self.failed.emit(str(e))


class BurstRefBirdWorker(QThread):
    """
    首张参考图上的鸟体检测与裁剪中心推断。
    YOLO/torch 若在 Qt GUI 主线程与 OpenMP 并行库叠加，Windows 上易出现整进程闪退；
    故统一放到后台线程，并通过 job id 丢弃过期结果。
    """

    finished_norm = pyqtSignal(float, float, int)

    def __init__(
        self,
        bgr_u8: np.ndarray,
        get_detector: Callable[[], Optional[object]],
        job_id: int,
        parent=None,
    ):
        super().__init__(parent)
        self._bgr = bgr_u8
        self._get_detector = get_detector
        self._job_id = int(job_id)

    def run(self) -> None:
        nx, ny = 0.5, 0.5
        _burst_worker_configure_openmp()
        try:
            _burst_gui_log("首张参考：鸟检后台线程开始…")
            if self.isInterruptionRequested():
                self.finished_norm.emit(nx, ny, self._job_id)
                return
            det = self._get_detector()
            if det is None or self._bgr.size == 0:
                _burst_gui_log(
                    "首张参考：鸟检线程内无检测器或无图像，使用中心 (0.5,0.5)。"
                )
                self.finished_norm.emit(nx, ny, self._job_id)
                return
            birds = det.detect_birds(self._bgr)
            nx, ny = infer_crop_center_norm_from_birds(self._bgr, birds)
            _burst_gui_log(
                f"首张参考：鸟检线程完成，{len(birds)} 框，中心=({nx:.4f},{ny:.4f})"
            )
        except Exception as ex:
            _burst_gui_log(f"首张参考：鸟检线程异常，使用 (0.5,0.5)：{ex}")
            nx, ny = 0.5, 0.5
        self.finished_norm.emit(float(nx), float(ny), self._job_id)


class BurstWebpDialog(QDialog):
    def __init__(self, parent=None, default_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("连拍 → WebP / MP4")
        self.setMinimumSize(760, 520)
        scr = _burst_safe_available_geometry(self)
        aw = max(int(scr.width()) - 32, 760)
        ah = max(int(scr.height()) - 48, 520)
        self.resize(min(1240, aw), min(820, ah))
        self._default_dir = default_dir or ""
        self._preview_qimages: List[QImage] = []
        self._preview_dur_ms = 200.0
        self._preview_idx = 0
        self._pv_worker: Optional[BurstWebpPreviewWorker] = None
        self._build_worker: Optional[BurstWebpBuildWorker] = None
        self._ref_bird_worker: Optional[BurstRefBirdWorker] = None
        self._ref_bird_job_id = 0
        self._ref_bird_ctx_had_saved = False
        self._crop_nx = 0.5
        self._crop_ny = 0.5
        self._crop_user_touched = False
        self._ref_bgr_cache: Optional[np.ndarray] = None
        self._initial_crop_nx = 0.5
        self._initial_crop_ny = 0.5
        self._align_roi_norm: Optional[Tuple[float, float, float, float]] = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_preview_tick)
        self._ref_debounce = QTimer(self)
        self._ref_debounce.setSingleShot(True)
        self._ref_debounce.setInterval(200)
        self._ref_debounce.timeout.connect(self._refresh_ref_from_first_image)

        # ── 主区域：左（可滚动参数） | 右（可滚动预览 + 底部固定路径/按钮）──
        main_h = QHBoxLayout(self)
        main_h.setSpacing(10)
        main_h.setContentsMargins(8, 8, 8, 8)

        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setFrameShape(QScrollArea.StyledPanel)
        scroll_left.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_left.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_left.setMinimumWidth(360)
        scroll_left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        left_inner = QWidget()
        left_inner.setMinimumWidth(340)
        left_l = QVBoxLayout(left_inner)
        left_l.setSpacing(8)

        left_l.addWidget(QLabel("待合成图片（顺序可调整；建议按时间排序）:"))
        self.list_w = QListWidget()
        self.list_w.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_w.setMinimumHeight(160)
        self.list_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_l.addWidget(self.list_w, stretch=1)

        row_btns = QHBoxLayout()
        self.btn_add = QPushButton("添加图片…")
        self.btn_add.clicked.connect(self._on_add_files)
        self.btn_rm = QPushButton("移除所选")
        self.btn_rm.clicked.connect(self._on_remove_sel)
        self.btn_sort = QPushButton("按拍摄时间排序")
        self.btn_sort.clicked.connect(self._on_sort_time)
        self.btn_up = QPushButton("上移")
        self.btn_up.clicked.connect(lambda: self._move_sel(-1))
        self.btn_dn = QPushButton("下移")
        self.btn_dn.clicked.connect(lambda: self._move_sel(1))
        row_btns.addWidget(self.btn_add)
        row_btns.addWidget(self.btn_rm)
        row_btns.addWidget(self.btn_sort)
        row_btns.addWidget(self.btn_up)
        row_btns.addWidget(self.btn_dn)
        left_l.addLayout(row_btns)

        opt_g = QGroupBox("处理与导出")
        fl = QFormLayout(opt_g)
        self.cb_wb = QCheckBox("灰世界白平衡")
        self.cb_wb.setChecked(True)
        self.cb_eco = QCheckBox("生态显影（曝光 / 分区明暗 / 对比度等，与入库显影同源）")
        self.cb_eco.setChecked(True)
        self.cb_align = QCheckBox("连拍对齐（ROI 内特征平移，非 ROI 留白）")
        self.cb_align.setChecked(True)
        self.cb_align.setToolTip(
            "首张在 ROI 内提特征作全局锚；后续帧在 ROI 外扩区内提点。"
            "自第 3 张起优先与「已对齐全的上一张」配准，再与「只对首张」择优；"
            "链式单步平移异常大时改首张以免顶夹漂移；每 6 帧强制与首张重锚。"
            "未设 ROI 时回退边带相位/ECC（自动场景对齐）。"
        )
        self.cb_debug_align = QCheckBox(
            "测试：导出/预览叠画对齐 ROI 与数值（黄虚线框+ASCII，裁剪前）"
        )
        self.cb_debug_align.setChecked(False)
        self.cb_debug_align.setToolTip(
            "黄虚线=你在首张画的「对齐 ROI」（归一化矩形投到当前分辨率）。\n"
            "橙虚线=该帧在对齐前用于提 ORB/SIFT/BRISK 的「ROI 外扩矩形」，会按本帧估计平移换算后画在"
            "「对齐后的整图」上，便于和黄色锚框对照；不是跟踪框，不保证目标始终在橙框内。\n"
            "字与框线已加粗加大；终端另有 [Birdy 连拍对齐] 每帧位移日志。未设 ROI 的边带 ECC 模式不画黄/橙框。"
        )
        self.cb_wm = QCheckBox("叠加水印（与主界面「水印与分享」当前选项一致，含布局 / Logo / 文字）")
        self.cb_wm.setChecked(True)
        self.cb_wm.setToolTip(
            "连拍 ≥2 张时：日期/相机/GPS/物种等一律按首张图的 EXIF 与路径解析，"
            "画面仍为各帧像素（与连拍白平衡、显影沿用首张参数一致）。单张预览仍按该张元数据。"
        )
        fl.addRow(self.cb_wb)
        fl.addRow(self.cb_eco)
        fl.addRow(self.cb_align)
        fl.addRow(self.cb_debug_align)
        fl.addRow(self.cb_wm)
        self.lbl_wm_theme = QLabel("水印物种 / 动图主题：")
        self.ed_wm_theme = QLineEdit()
        self.ed_wm_theme.setPlaceholderText(
            "选填，例如：白头鹎、东湖晨拍（叠水印时作左侧物种/主题；不填则仍用水印源目录推断）"
        )
        self.ed_wm_theme.setToolTip(
            "动图帧可能来自任意文件夹，叠水印时往往无法从路径得到物种目录名；"
            "填写后优先显示在此处。需主界面「水印与分享」中开启物种/左侧文案相关选项。"
        )
        fl.addRow(self.lbl_wm_theme, self.ed_wm_theme)

        self.slider_ret = QSlider(Qt.Horizontal)
        self.slider_ret.setRange(25, 100)
        self.slider_ret.setValue(94)
        self.slider_ret.setToolTip(
            "保留画幅比例（裁剪窗口大小），约 25%～100%；越小裁边越多、画面越稳。"
            "裁剪中心由红色十字决定（首张自动鸟检或点击预览图），先对准中心再夹紧到图像内。"
        )
        self.lbl_ret = QLabel("保留画幅约 94%（裁剪中心见预览十字）")
        self.slider_ret.valueChanged.connect(self._on_ret_changed)
        fl.addRow(self.lbl_ret, self.slider_ret)

        self.spn_speed = QDoubleSpinBox()
        self.spn_speed.setRange(0.25, 8.0)
        self.spn_speed.setSingleStep(0.25)
        self.spn_speed.setDecimals(2)
        self.spn_speed.setValue(1.0)
        self.spn_speed.setToolTip(
            "1.0 = 按推断拍照间隔播放；大于 1 加快，小于 1 减慢。"
        )
        fl.addRow("播放加速倍率:", self.spn_speed)

        self.cmb_max = QComboBox()
        self.cmb_max.addItem("最长边 1080", 1080)
        self.cmb_max.addItem("最长边 1280", 1280)
        self.cmb_max.addItem("最长边 1600（推荐）", 1600)
        self.cmb_max.addItem("最长边 1920", 1920)
        self.cmb_max.addItem("最长边 2160", 2160)
        self.cmb_max.addItem("不缩放（原分辨率，文件较大）", 0)
        self.cmb_max.setCurrentIndex(2)
        fl.addRow("导出尺寸:", self.cmb_max)

        self.spn_q = QSpinBox()
        self.spn_q.setRange(40, 100)
        self.spn_q.setValue(85)
        fl.addRow("WebP 质量:", self.spn_q)

        self.lbl_interval = QLabel("推断间隔：—")
        self.lbl_interval.setWordWrap(True)
        fl.addRow(self.lbl_interval)

        left_l.addWidget(opt_g)
        left_l.addStretch(0)
        scroll_left.setWidget(left_inner)
        main_h.addWidget(scroll_left, 4)

        right_panel = QWidget()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_v = QVBoxLayout(right_panel)
        right_v.setSpacing(8)
        right_v.setContentsMargins(0, 0, 0, 0)

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setFrameShape(QScrollArea.NoFrame)
        preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        preview_inner = QWidget()
        pv_inner_l = QVBoxLayout(preview_inner)
        pv_inner_l.setContentsMargins(0, 0, 0, 0)
        pv_inner_l.setSpacing(6)

        self._pv_canvas = BurstCropPreviewWidget(self)
        self._pv_canvas.set_retention(self.slider_ret.value() / 100.0)
        self._pv_canvas.center_changed.connect(self._on_crop_center_from_canvas)
        self._pv_canvas.track_roi_changed.connect(self._on_track_roi_from_canvas)
        pv_inner_l.addWidget(self._pv_canvas, stretch=1)

        row_pick = QHBoxLayout()
        self._bg_pick = QButtonGroup(self)
        self.rb_pick_crop = QRadioButton("红十字：裁剪中心")
        self.rb_pick_track = QRadioButton("黄框：对齐 ROI（拖拽矩形）")
        self.rb_pick_crop.setChecked(True)
        self._bg_pick.addButton(self.rb_pick_crop)
        self._bg_pick.addButton(self.rb_pick_track)
        self.rb_pick_crop.clicked.connect(lambda: self._pv_canvas.set_interaction_mode(0))
        self.rb_pick_track.clicked.connect(lambda: self._pv_canvas.set_interaction_mode(1))
        row_pick.addWidget(self.rb_pick_crop)
        row_pick.addWidget(self.rb_pick_track)
        pv_inner_l.addLayout(row_pick)

        row_trk = QHBoxLayout()
        self.lbl_align_feat = QLabel("ROI 特征：")
        self.cmb_align_feat = QComboBox()
        self.cmb_align_feat.addItem("ORB（默认）", "ORB")
        self.cmb_align_feat.addItem("SIFT", "SIFT")
        self.cmb_align_feat.addItem("BRISK", "BRISK")
        self.cmb_align_feat.setToolTip(
            "仅在手动 ROI 内提关键点；首张与当前帧同一像素框内匹配。"
            "SIFT 需 OpenCV 非 contrib 构建支持；不可用时自动退回 ORB。"
        )
        self.cmb_align_feat.currentIndexChanged.connect(self._schedule_burst_state_save)
        row_trk.addWidget(self.lbl_align_feat)
        row_trk.addWidget(self.cmb_align_feat, 1)
        pv_inner_l.addLayout(row_trk)

        row_prev = QHBoxLayout()
        self.btn_prev = QPushButton("更新预览")
        self.btn_prev.clicked.connect(self._start_preview)
        self.btn_reset_view = QPushButton("恢复初始设置")
        self.btn_reset_view.setToolTip(
            "停止动效预览，回到首张参考图，并将裁剪中心恢复为自动鸟检结果（无鸟则为画面中心）。"
        )
        self.btn_reset_view.clicked.connect(self._on_reset_initial)
        row_prev.addWidget(self.btn_prev)
        row_prev.addWidget(self.btn_reset_view)
        pv_inner_l.addLayout(row_prev)

        self.lbl_pv_status = QLabel("")
        self.lbl_pv_status.setWordWrap(True)
        self.lbl_pv_status.setStyleSheet("color: #444444; font-size: 9pt;")
        self.lbl_pv_status.setMinimumHeight(52)
        pv_inner_l.addWidget(self.lbl_pv_status)

        preview_scroll.setWidget(preview_inner)
        right_v.addWidget(preview_scroll, stretch=1)

        row_fmt = QHBoxLayout()
        self._bg_export_fmt = QButtonGroup(self)
        self.rb_export_webp = QRadioButton("WebP 动图")
        self.rb_export_mp4 = QRadioButton("MP4 视频")
        self.rb_export_webp.setChecked(True)
        self.rb_export_webp.setToolTip(
            "动画 WebP，体积小；部分相册或 App 不支持动图 WebP 时无法播放。"
        )
        self.rb_export_mp4.setToolTip(
            "常见兼容编码（mp4v / MPEG-4 Part 2）；便于在不支持动图 WebP 的客户端播放。"
        )
        self._bg_export_fmt.addButton(self.rb_export_webp)
        self._bg_export_fmt.addButton(self.rb_export_mp4)
        self._bg_export_fmt.buttonClicked.connect(self._on_export_format_clicked)
        row_fmt.addWidget(QLabel("导出格式："))
        row_fmt.addWidget(self.rb_export_webp)
        row_fmt.addWidget(self.rb_export_mp4)
        row_fmt.addStretch(1)
        right_v.addLayout(row_fmt)

        out_row = QHBoxLayout()
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("输出 .webp 路径")
        self.btn_out = QPushButton("浏览…")
        self.btn_out.clicked.connect(self._on_browse_out)
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(self.btn_out)
        right_v.addLayout(out_row)

        bot = QHBoxLayout()
        self.btn_go = QPushButton("生成")
        self.btn_go.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.btn_go.clicked.connect(self._on_export)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.reject)
        bot.addStretch(1)
        bot.addWidget(self.btn_go)
        bot.addWidget(self.btn_close)
        right_v.addLayout(bot)

        main_h.addWidget(right_panel, 6)

        self.list_w.model().rowsInserted.connect(self._on_list_changed)
        self.list_w.model().rowsRemoved.connect(self._on_list_changed)
        self.spn_speed.valueChanged.connect(self._refresh_interval_hint)
        self.spn_speed.valueChanged.connect(self._log_speed_changed)
        self.cb_wb.stateChanged.connect(self._schedule_ref_refresh)
        self.cb_eco.stateChanged.connect(self._schedule_ref_refresh)
        self.cb_wb.stateChanged.connect(self._log_wb_toggle)
        self.cb_eco.stateChanged.connect(self._log_eco_toggle)
        self.cb_align.stateChanged.connect(self._log_align_toggle)
        self.cb_wm.stateChanged.connect(self._log_wm_toggle)
        self.cmb_max.currentIndexChanged.connect(self._log_export_size_changed)
        self.spn_q.valueChanged.connect(self._log_webp_quality_changed)

        self._state_window_maximized = True
        self._state_window_geometry: Optional[Tuple[int, int, int, int]] = None
        self._saved_crop_nx = 0.5
        self._saved_crop_ny = 0.5
        self._restore_saved_crop_once = False
        # 必须在 _load_burst_dialog_state 之前：加载状态时会触发控件信号，
        # 进而 _schedule_burst_state_save，依赖本定时器。
        self._save_state_timer = QTimer(self)
        self._save_state_timer.setSingleShot(True)
        self._save_state_timer.setInterval(500)
        self._save_state_timer.timeout.connect(self._save_burst_dialog_state)
        for sig in (
            self.cb_wb.stateChanged,
            self.cb_eco.stateChanged,
            self.cb_align.stateChanged,
            self.cb_wm.stateChanged,
            self.cb_debug_align.stateChanged,
            self.slider_ret.valueChanged,
            self.spn_speed.valueChanged,
            self.cmb_max.currentIndexChanged,
            self.spn_q.valueChanged,
        ):
            sig.connect(self._schedule_burst_state_save)
        self.ed_wm_theme.textChanged.connect(self._schedule_burst_state_save)
        self.ed_out.textChanged.connect(self._schedule_burst_state_save)

        self._load_burst_dialog_state()

        _burst_gui_log(
            f"动图对话框初始化完成；默认相片目录={self._default_dir or '(空)'}，"
            f"列表中 {self.list_w.count()} 张。"
        )
        # 推迟到事件循环：避免 __init__ 末尾与 _load 触发的信号链重入同一刷新逻辑导致不稳定
        QTimer.singleShot(0, self._schedule_ref_refresh)
        QTimer.singleShot(0, self._log_wm_toggle)

    def _schedule_burst_state_save(self, *_args) -> None:
        """参数变更后防抖写入，避免只读安装目录或强关窗口时从未落盘。"""
        self._save_state_timer.stop()
        self._save_state_timer.start(500)

    def _log_wb_toggle(self, _state: int = 0) -> None:
        _burst_gui_log(f"参数：灰世界白平衡 → {'开启' if self.cb_wb.isChecked() else '关闭'}")

    def _log_eco_toggle(self, _state: int = 0) -> None:
        _burst_gui_log(f"参数：生态显影 → {'开启' if self.cb_eco.isChecked() else '关闭'}")

    def _log_align_toggle(self, _state: int = 0) -> None:
        _burst_gui_log(f"参数：连拍对齐 → {'开启' if self.cb_align.isChecked() else '关闭'}")

    def _log_wm_toggle(self, _state: int = 0) -> None:
        en = self.cb_wm.isChecked()
        _burst_gui_log(f"参数：叠加水印 → {'开启' if en else '关闭'}")
        self.ed_wm_theme.setEnabled(en)
        self.lbl_wm_theme.setEnabled(en)

    def _log_export_size_changed(self, _idx: int = 0) -> None:
        v = self.cmb_max.currentData()
        _burst_gui_log(f"参数：导出最长边 → {int(v) if v else '不缩放'}")

    def _log_webp_quality_changed(self, v: int) -> None:
        _burst_gui_log(f"参数：WebP 质量 → {v}")

    def _schedule_ref_refresh(self) -> None:
        if self._pv_worker is not None and self._pv_worker.isRunning():
            _burst_gui_log("首张参考刷新已推迟：预览线程占用中。")
            return
        _burst_gui_log("已安排首张参考图刷新（200ms 防抖）。")
        self._ref_debounce.stop()
        self._ref_debounce.start(200)

    def _on_list_changed(self, *args) -> None:
        del args
        n = self.list_w.count()
        _burst_gui_log(f"图片列表已变化，当前共 {n} 项。")
        self._refresh_interval_hint()
        self._schedule_ref_refresh()

    def _on_ret_changed(self, v: int) -> None:
        self.lbl_ret.setText(f"保留画幅约 {v}%（裁剪中心见预览十字）")
        self._pv_canvas.set_retention(v / 100.0)
        _burst_gui_log(f"参数：保留画幅 → {v}%（参考图绿色虚线框已更新）")

    def _collect_paths(self) -> List[str]:
        return [self.list_w.item(i).text() for i in range(self.list_w.count())]

    def _log_speed_changed(self, v: float) -> None:
        _burst_gui_log(f"参数：播放加速倍率 → {v:g}×")

    def _refresh_interval_hint(self) -> None:
        paths = self._collect_paths()
        if len(paths) < 2:
            self.lbl_interval.setText("推断间隔：请至少添加 2 张图片")
            return
        ordered = sort_paths_by_capture_time(paths)
        ms, note = infer_shot_interval_ms(ordered)
        spd = float(self.spn_speed.value())
        dur = ms / max(0.05, spd)
        self.lbl_interval.setText(
            f"推断拍照间隔 ≈ {ms:.0f} ms（{note}）；"
            f"当前倍率 {spd:g}× → 每帧约 {dur:.0f} ms"
        )

    def _on_add_files(self) -> None:
        _burst_gui_log("操作：添加图片…（文件选择对话框已打开）")
        start = self._default_dir
        if not start or not os.path.isdir(start):
            start = os.path.expanduser("~")
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择连拍图片", start, file_filter_all_images()
        )
        n0 = self.list_w.count()
        added = 0
        for fp in files:
            if fp and os.path.isfile(fp):
                self.list_w.addItem(fp)
                added += 1
        if files and files[0]:
            self._default_dir = os.path.dirname(os.path.abspath(files[0]))
        self._refresh_interval_hint()
        _burst_gui_log(
            f"操作：添加图片结束，本次加入 {added} 张，列表由 {n0} → {self.list_w.count()} 项。"
        )

    def _on_remove_sel(self) -> None:
        n0 = self.list_w.count()
        _burst_gui_log(f"操作：移除所选（移除前列表 {n0} 项）…")
        for it in self.list_w.selectedItems():
            row = self.list_w.row(it)
            self.list_w.takeItem(row)
        self._refresh_interval_hint()
        _burst_gui_log(f"操作：移除完成，列表余 {self.list_w.count()} 项。")

    def _on_sort_time(self) -> None:
        n = self.list_w.count()
        _burst_gui_log(f"操作：按拍摄时间排序（共 {n} 项）…")
        paths = self._collect_paths()
        for _ in range(self.list_w.count()):
            self.list_w.takeItem(0)
        for p in sort_paths_by_capture_time(paths):
            self.list_w.addItem(p)
        self._refresh_interval_hint()
        _burst_gui_log("操作：按拍摄时间排序完成。")

    def _move_sel(self, delta: int) -> None:
        row = self.list_w.currentRow()
        if row < 0:
            _burst_gui_log("操作：上移/下移跳过（未选中行）。")
            return
        nr = row + delta
        if nr < 0 or nr >= self.list_w.count():
            _burst_gui_log(f"操作：上移/下移跳过（目标行 {nr} 越界）。")
            return
        _burst_gui_log(f"操作：列表项移动 delta={delta}，行 {row} → {nr}。")
        item = self.list_w.takeItem(row)
        self.list_w.insertItem(nr, item)
        self.list_w.setCurrentRow(nr)
        self._refresh_interval_hint()
        self._schedule_ref_refresh()

    def _refresh_ref_from_first_image(self) -> None:
        t0 = time.monotonic()
        _burst_gui_log("首张参考图刷新开始（防抖到期）…")
        if self._pv_worker is not None and self._pv_worker.isRunning():
            _burst_gui_log("首张参考图刷新中止：预览线程仍占用。")
            return
        self._timer.stop()
        self._preview_qimages = []
        self._pv_canvas.stop_playback()
        paths = self._collect_paths()
        ordered = sort_paths_by_capture_time([p for p in paths if os.path.isfile(p)])
        if not ordered:
            _burst_gui_log("首张参考：无有效路径，清空参考图与裁剪中心。")
            w0 = getattr(self, "_ref_bird_worker", None)
            if w0 is not None and w0.isRunning():
                w0.requestInterruption()
                w0.wait(2000)
            self._ref_bgr_cache = None
            self._pv_canvas.set_reference_bgr(None)
            self._pv_canvas.clear_track_roi()
            self._crop_nx = self._crop_ny = 0.5
            self._initial_crop_nx = self._initial_crop_ny = 0.5
            self._crop_user_touched = False
            return
        first = ordered[0]
        _burst_gui_log(f"首张参考：排序后首张 → {os.path.basename(first)}")
        self._crop_user_touched = False
        _burst_gui_log("首张参考：解码（半尺寸 raw）…")
        bgr = imread_bgr(first, raw_half_size=True)
        if bgr is None or bgr.size == 0:
            _burst_gui_log(f"首张参考：读取失败 → {first}")
            self._ref_bgr_cache = None
            self._pv_canvas.set_reference_bgr(None)
            return
        h0, w0 = bgr.shape[:2]
        _burst_gui_log(f"首张参考：解码完成 {w0}×{h0}，用时 {time.monotonic() - t0:.2f}s")
        x = bgr
        if self.cb_wb.isChecked():
            _burst_gui_log("首张参考：灰世界白平衡…")
            t1 = time.monotonic()
            x = gray_world_white_balance(x)
            _burst_gui_log(f"首张参考：白平衡完成，用时 {time.monotonic() - t1:.2f}s")
        if self.cb_eco.isChecked():
            _burst_gui_log("首张参考：生态显影（可能较慢）…")
            t1 = time.monotonic()
            x = develop_bgr_ecology_wildlife(x)
            _burst_gui_log(f"首张参考：显影完成，用时 {time.monotonic() - t1:.2f}s")
        had_saved = bool(self._restore_saved_crop_once)
        self._ref_bird_ctx_had_saved = had_saved
        self._initial_crop_nx = 0.5
        self._initial_crop_ny = 0.5
        if self._restore_saved_crop_once:
            self._crop_nx = float(self._saved_crop_nx)
            self._crop_ny = float(self._saved_crop_ny)
            self._crop_user_touched = True
            self._restore_saved_crop_once = False
        else:
            self._crop_nx = 0.5
            self._crop_ny = 0.5
        self._ref_bgr_cache = np.ascontiguousarray(x, dtype=np.uint8).copy()
        self._pv_canvas.set_reference_bgr(self._ref_bgr_cache)
        self._pv_canvas.set_center_norm(self._crop_nx, self._crop_ny)
        self._pv_canvas.set_retention(self.slider_ret.value() / 100.0)
        if _burst_align_roi_valid(self._align_roi_norm):
            self._pv_canvas.set_track_roi_norm(self._align_roi_norm)
        else:
            self._pv_canvas.clear_track_roi()
        _burst_gui_log(
            f"首张参考图主线程阶段结束，用时 {time.monotonic() - t0:.2f}s；"
            f"鸟检在后台线程进行；保留比例={self.slider_ret.value()}%。"
        )
        self._start_ref_bird_worker(self._ref_bgr_cache.copy())

    def _start_ref_bird_worker(self, bgr: np.ndarray) -> None:
        w = self._ref_bird_worker
        if w is not None and w.isRunning():
            self._ref_bird_job_id += 1
            w.requestInterruption()
            if not w.wait(8000):
                _burst_gui_log("首张参考：旧鸟检线程未在 8s 内结束，仍启动新任务。")
        self._ref_bird_job_id += 1
        jid = int(self._ref_bird_job_id)
        self._ref_bird_worker = BurstRefBirdWorker(
            bgr, self._get_burst_detector, jid, parent=self
        )
        self._ref_bird_worker.finished_norm.connect(
            self._on_ref_bird_worker_done, Qt.QueuedConnection
        )
        self._ref_bird_worker.start()
        _burst_gui_log(f"首张参考：鸟检已提交后台线程（job={jid}）。")

    @pyqtSlot(float, float, int)
    def _on_ref_bird_worker_done(self, nx: float, ny: float, job_id: int) -> None:
        if int(job_id) != int(self._ref_bird_job_id):
            _burst_gui_log(
                f"首张参考：忽略过期鸟检结果 job={job_id}（当前={self._ref_bird_job_id}）。"
            )
            return
        self._initial_crop_nx = float(nx)
        self._initial_crop_ny = float(ny)
        if getattr(self, "_ref_bird_ctx_had_saved", False):
            _burst_gui_log(
                f"首张参考：鸟检完成，「恢复初始」将使用该中心（{nx:.4f},{ny:.4f}）；"
                "当前仍显示上次保存的裁剪位置。"
            )
            return
        if self._crop_user_touched:
            _burst_gui_log(
                "首张参考：鸟检完成，但用户已调整裁剪中心，不自动覆盖画布。"
            )
            return
        self._crop_nx = float(nx)
        self._crop_ny = float(ny)
        self._pv_canvas.set_center_norm(self._crop_nx, self._crop_ny)
        _burst_gui_log(f"首张参考：画布已应用鸟检中心（{nx:.4f},{ny:.4f}）。")

    def _on_reset_initial(self) -> None:
        _burst_gui_log(
            f"操作：恢复初始设置（中心 → {self._initial_crop_nx:.4f},{self._initial_crop_ny:.4f}，停止动效）"
        )
        self._timer.stop()
        self._preview_qimages = []
        self._pv_canvas.stop_playback()
        self._crop_user_touched = False
        self._crop_nx = float(self._initial_crop_nx)
        self._crop_ny = float(self._initial_crop_ny)
        self._align_roi_norm = None
        self._pv_canvas.clear_track_roi()
        self.rb_pick_crop.setChecked(True)
        self._pv_canvas.set_interaction_mode(BurstCropPreviewWidget.MODE_CROP)
        if self._ref_bgr_cache is not None and getattr(self._ref_bgr_cache, "size", 0) > 0:
            self._pv_canvas.set_reference_bgr(self._ref_bgr_cache)
            self._pv_canvas.set_center_norm(self._crop_nx, self._crop_ny)
            self._pv_canvas.set_retention(self.slider_ret.value() / 100.0)
            _burst_gui_log("操作：恢复初始设置完成，已回到首张参考 + 十字/虚线框。")
        else:
            self._pv_canvas.set_reference_bgr(None)
            _burst_gui_log("操作：恢复初始设置完成，但无缓存参考图可显示。")

    def _watermark_opts_and_folder(self):
        if not self.cb_wm.isChecked():
            return None, ""
        par = self.parent()
        if par is not None and hasattr(par, "_build_watermark_options"):
            try:
                return (
                    par._build_watermark_options(),
                    par._resolve_watermark_source_folder() or "",
                )
            except Exception as ex:
                _burst_gui_log(f"水印：从主界面读取选项失败，将不叠水印：{ex}")
                return None, ""
        return None, ""

    def _opts_from_ui(self) -> BurstWebpBuildOptions:
        ret_pct = int(self.slider_ret.value()) / 100.0
        wo, wf = self._watermark_opts_and_folder()
        return BurstWebpBuildOptions(
            enable_white_balance=self.cb_wb.isChecked(),
            enable_ecology_develop=self.cb_eco.isChecked(),
            enable_align=self.cb_align.isChecked(),
            stability_center_retention=ret_pct,
            crop_center_norm=(float(self._crop_nx), float(self._crop_ny)),
            speed_multiplier=float(self.spn_speed.value()),
            max_long_edge=_int_safe_combo_data(self.cmb_max, 1600),
            webp_quality=int(self.spn_q.value()),
            watermark_options=wo,
            watermark_source_folder=wf,
            prefer_folder_name_as_species=True,
            watermark_species_or_theme=(
                self.ed_wm_theme.text().strip() if self.cb_wm.isChecked() else ""
            ),
            align_track_roi_norm=(
                tuple(float(x) for x in self._align_roi_norm)
                if _burst_align_roi_valid(self._align_roi_norm)
                else None
            ),
            align_feature_detector=str(
                self.cmb_align_feat.currentData() or "ORB"
            ).upper(),
            debug_export_align_overlay=self.cb_debug_align.isChecked(),
        )

    def _get_burst_detector(self):
        par = self.parent()
        if par is not None and hasattr(par, "get_burst_webp_bird_detector"):
            try:
                return par.get_burst_webp_bird_detector()
            except Exception:
                return None
        return None

    def _on_crop_center_from_canvas(self, nx: float, ny: float) -> None:
        self._crop_user_touched = True
        self._crop_nx = float(nx)
        self._crop_ny = float(ny)
        _burst_gui_log(f"操作：在预览图上点击设置裁剪中心 → ({nx:.4f},{ny:.4f})")
        self._schedule_burst_state_save()

    def _on_track_roi_from_canvas(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        self._align_roi_norm = (float(x0), float(y0), float(x1), float(y1))
        _burst_gui_log(
            f"操作：设置连拍对齐 ROI → ({x0:.4f},{y0:.4f})—({x1:.4f},{y1:.4f})，"
            f"特征={self.cmb_align_feat.currentData()}"
        )
        self._schedule_burst_state_save()

    def _on_preview_progress(self, cur: int, tot: int, msg: str) -> None:
        t0 = getattr(self, "_pv_t0", None)
        elapsed = (time.monotonic() - t0) if t0 is not None else 0.0
        eta_line = ""
        if cur > 0 and tot > cur and elapsed > 0.05:
            rem = (elapsed / float(cur)) * float(tot - cur)
            eta_line = f"\n预计剩余约 {rem:.0f} 秒（按当前步均耗时粗算，显影/水印步波动大）"
        self.lbl_pv_status.setText(
            f"{msg}\n进度 {cur}/{tot} · 已用 {elapsed:.1f} s{eta_line}"
        )
        _burst_gui_log(f"预览进度 {cur}/{tot} · {msg} · 已用 {elapsed:.1f}s")

    def _start_preview(self) -> None:
        paths = self._collect_paths()
        if len(paths) < 1:
            _burst_gui_log("操作：更新预览取消（列表为空）。")
            QMessageBox.information(self, "提示", "请先添加至少一张图片。")
            return
        if self._pv_worker is not None and self._pv_worker.isRunning():
            _burst_gui_log("操作：更新预览忽略（预览线程已在运行）。")
            return
        _burst_gui_log(f"操作：更新预览开始，共 {len(paths)} 张路径…")
        self._ref_debounce.stop()
        self._timer.stop()
        self._pv_canvas.stop_playback()
        self.btn_prev.setEnabled(False)
        self.btn_prev.setText("预览生成中…")
        self._pv_t0 = time.monotonic()
        opts = self._opts_from_ui()
        align_note = ""
        if self.cb_align.isChecked() and not _burst_align_roi_valid(
            self._align_roi_norm
        ):
            align_note = "（未画对齐 ROI：预览将使用边带 ECC 自动对齐；锚点模式请先框 ROI）\n"
        self.lbl_pv_status.setText(
            align_note
            + "已开始…\n解码阶段请查看运行 Birdy 的终端窗口（逐张打印文件名）。"
        )
        _burst_gui_log(
            f"预览参数：WB={opts.enable_white_balance}, 显影={opts.enable_ecology_develop}, "
            f"对齐={opts.enable_align}, 水印={'开' if opts.watermark_options else '关'}, "
            f"中心=({opts.crop_center_norm[0]:.4f},{opts.crop_center_norm[1]:.4f}), "
            f"保留={opts.stability_center_retention:.2f}, "
            f"对齐ROI={opts.align_track_roi_norm}, 特征={opts.align_feature_detector}"
        )
        self._pv_worker = BurstWebpPreviewWorker(
            paths,
            opts,
            get_detector=self._get_burst_detector,
            user_touched_center=self._crop_user_touched,
            parent=self,
        )
        self._pv_worker.progress.connect(
            self._on_preview_progress, Qt.QueuedConnection
        )
        self._pv_worker.done.connect(self._on_preview_done, Qt.QueuedConnection)
        self._pv_worker.failed.connect(self._on_preview_fail, Qt.QueuedConnection)
        self._pv_worker.start()

    def _on_preview_done(
        self, qimgs, dur_ms: float, _note: str, _align0_bgr, cnx: float, cny: float
    ) -> None:
        self.btn_prev.setEnabled(True)
        self.btn_prev.setText("更新预览")
        self._crop_nx = float(cnx)
        self._crop_ny = float(cny)
        self._preview_qimages = list(qimgs or [])
        self._preview_dur_ms = max(1.0, float(dur_ms))
        self._preview_idx = 0
        if not self._preview_qimages:
            _burst_gui_log("预览完成回调：无有效帧，将刷新首张参考。")
            self.lbl_pv_status.setText("预览失败：无有效帧")
            self._pv_canvas.stop_playback()
            self._schedule_ref_refresh()
            return
        _burst_gui_log(
            f"预览完成：{len(self._preview_qimages)} 帧，每帧 {self._preview_dur_ms:.0f} ms，"
            f"裁剪中心=({cnx:.4f},{cny:.4f})；右侧进入动效播放。"
        )
        self._pv_canvas.set_playback_frames(self._preview_qimages)
        self._pv_canvas.set_playback_index(0)
        self._timer.start(max(1, int(round(self._preview_dur_ms))))
        self._refresh_interval_hint()

    def _on_preview_fail(self, msg: str) -> None:
        _burst_gui_log(f"预览失败（GUI 回调）：{msg}")
        self.btn_prev.setEnabled(True)
        self.btn_prev.setText("更新预览")
        self.lbl_pv_status.setText(f"预览失败：{msg}")
        self._pv_canvas.stop_playback()
        self._schedule_ref_refresh()
        QMessageBox.warning(self, "预览失败", msg)

    def _on_preview_tick(self) -> None:
        if not self._preview_qimages:
            return
        self._preview_idx = (self._preview_idx + 1) % len(self._preview_qimages)
        self._show_preview_frame()

    def _show_preview_frame(self) -> None:
        if not self._preview_qimages:
            return
        self._pv_canvas.set_playback_index(self._preview_idx)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._pv_canvas.update()

    def _apply_export_path_placeholder(self) -> None:
        if self.rb_export_mp4.isChecked():
            self.ed_out.setPlaceholderText("输出 .mp4 路径")
        else:
            self.ed_out.setPlaceholderText("输出 .webp 路径")

    def _sync_out_path_extension_to_format(self) -> None:
        t = self.ed_out.text().strip()
        if not t:
            return
        if self.rb_export_mp4.isChecked() and t.lower().endswith(".webp"):
            self.ed_out.setText(t[:-5] + ".mp4")
        elif self.rb_export_webp.isChecked() and t.lower().endswith(".mp4"):
            self.ed_out.setText(t[:-4] + ".webp")

    def _on_export_format_clicked(self, _btn) -> None:
        self._apply_export_path_placeholder()
        self._sync_out_path_extension_to_format()
        self._schedule_burst_state_save()

    def _on_browse_out(self) -> None:
        start = self._default_dir or os.path.expanduser("~")
        if self.rb_export_mp4.isChecked():
            _burst_gui_log("操作：浏览输出 MP4 路径…")
            fp, _ = QFileDialog.getSaveFileName(
                self, "保存 MP4 视频", start, "MP4 (*.mp4);;所有文件 (*.*)"
            )
            ext = ".mp4"
        else:
            _burst_gui_log("操作：浏览输出 WebP 路径…")
            fp, _ = QFileDialog.getSaveFileName(
                self, "保存 WebP 动图", start, "WebP (*.webp);;所有文件 (*.*)"
            )
            ext = ".webp"
        if fp:
            if not fp.lower().endswith(ext):
                fp += ext
            self.ed_out.setText(fp)
            _burst_gui_log(f"操作：已选输出路径 → {fp}")
        else:
            _burst_gui_log("操作：取消选择输出路径。")

    def _on_export_progress(self, cur: int, tot: int, msg: str) -> None:
        t0 = getattr(self, "_export_t0", None)
        elapsed = (time.monotonic() - t0) if t0 is not None else 0.0
        eta_line = ""
        if cur > 0 and tot > cur and elapsed > 0.05:
            rem = (elapsed / float(cur)) * float(tot - cur)
            eta_line = f"\n预计剩余约 {rem:.0f} 秒（按当前步均耗时粗算）"
        fmt_lbl = "MP4" if self.rb_export_mp4.isChecked() else "WebP"
        self.lbl_pv_status.setText(
            f"[导出 {fmt_lbl}] {msg}\n进度 {cur}/{tot} · 已用 {elapsed:.1f} s{eta_line}"
        )
        _burst_gui_log(f"导出进度 {cur}/{tot} · {msg} · 已用 {elapsed:.1f}s")

    def _on_export(self) -> None:
        paths = self._collect_paths()
        if len(paths) < 2:
            _burst_gui_log("操作：生成取消（少于 2 张）。")
            QMessageBox.warning(self, "提示", "至少需要 2 张图片才能合成动图。")
            return
        if self.cb_align.isChecked() and not _burst_align_roi_valid(self._align_roi_norm):
            QMessageBox.warning(
                self,
                "未设置对齐 ROI",
                "已开启「连拍对齐」。请在预览中切换到「黄框：对齐 ROI」，"
                "在首张参考图上拖拽画出黄虚线矩形（包住参考物）后再导出。",
            )
            return
        out = self.ed_out.text().strip()
        if not out:
            _burst_gui_log("操作：生成取消（未设输出路径）。")
            QMessageBox.warning(self, "提示", "请选择输出文件路径。")
            return
        export_fmt = "mp4" if self.rb_export_mp4.isChecked() else "webp"
        if export_fmt == "mp4":
            if not out.lower().endswith(".mp4"):
                if out.lower().endswith(".webp"):
                    out = out[:-5] + ".mp4"
                else:
                    out = out + ".mp4"
                self.ed_out.setText(out)
        else:
            if not out.lower().endswith(".webp"):
                if out.lower().endswith(".mp4"):
                    out = out[:-4] + ".webp"
                else:
                    out = out + ".webp"
                self.ed_out.setText(out)
        if self._build_worker is not None and self._build_worker.isRunning():
            _burst_gui_log("操作：生成忽略（导出线程已在运行）。")
            QMessageBox.information(self, "提示", "正在生成，请稍候。")
            return
        opts = self._opts_from_ui()
        _burst_gui_log(
            f"操作：开始生成 {export_fmt.upper()} → {out}（{len(paths)} 张）；"
            f"最长边={opts.max_long_edge}, WebP质量={opts.webp_quality}"
        )
        self.btn_go.setEnabled(False)
        self.btn_go.setText("生成中…")
        self._export_t0 = time.monotonic()
        log_hint = (
            "[Birdy 视频导出]"
            if export_fmt == "mp4"
            else "[Birdy WebP导出]"
        )
        self.lbl_pv_status.setText(
            f"已开始导出 {export_fmt.upper()}…\n详细步进见终端 {log_hint}。"
        )
        self._build_worker = BurstWebpBuildWorker(
            paths, out, opts, export_format=export_fmt, parent=self
        )
        self._build_worker.progress.connect(
            self._on_export_progress, Qt.QueuedConnection
        )
        self._build_worker.finished_ok.connect(self._on_build_ok, Qt.QueuedConnection)
        self._build_worker.failed.connect(self._on_build_fail, Qt.QueuedConnection)
        self._build_worker.start()

    def _on_build_ok(self, r: dict) -> None:
        fmt = str(r.get("format", "webp")).upper()
        _burst_gui_log(
            f"操作：生成 {fmt} 成功，帧数={r.get('n_frames')}, 输出={r.get('out_path')}"
        )
        self.btn_go.setEnabled(True)
        self.btn_go.setText("生成")
        fps_note = ""
        if r.get("format") == "mp4" and r.get("fps") is not None:
            fps_note = f"，约 {float(r['fps']):.2f} fps"
        self.lbl_pv_status.setText(
            f"导出完成（{fmt}）：{r.get('n_frames', 0)} 帧，每帧约 "
            f"{r.get('frame_duration_ms', 0):.0f} ms{fps_note}\n"
            f"{r.get('out_path', '')}"
        )
        QMessageBox.information(
            self,
            "完成",
            f"已生成 {fmt}：{r.get('n_frames', 0)} 帧，每帧 "
            f"{r.get('frame_duration_ms', 0):.0f} ms{fps_note}\n"
            f"{r.get('out_path', '')}",
        )

    def _on_build_fail(self, msg: str) -> None:
        _burst_gui_log(f"操作：生成失败：{msg}")
        self.btn_go.setEnabled(True)
        self.btn_go.setText("生成")
        self.lbl_pv_status.setText(f"导出失败：{msg}")
        QMessageBox.critical(self, "生成失败", msg)

    def closeEvent(self, e) -> None:
        self._save_state_timer.stop()
        self._save_burst_dialog_state()
        _burst_gui_log("对话框 closeEvent：停止定时器并等待后台线程…")
        self._timer.stop()
        if self._pv_worker and self._pv_worker.isRunning():
            _burst_gui_log("等待预览线程结束（最多 3s）…")
            self._pv_worker.wait(3000)
        if self._build_worker and self._build_worker.isRunning():
            _burst_gui_log("等待导出线程结束（最多 3s）…")
            self._build_worker.wait(3000)
        if self._ref_bird_worker and self._ref_bird_worker.isRunning():
            _burst_gui_log("等待首张鸟检后台线程结束（最多 5s）…")
            self._ref_bird_worker.wait(5000)
        _burst_gui_log("动图对话框关闭流程结束。")
        super().closeEvent(e)

    def done(self, result: int) -> None:
        self._save_state_timer.stop()
        self._save_burst_dialog_state()
        super().done(result)

    def _load_burst_dialog_state(self) -> None:
        """恢复上次关闭时的参数（不含待合成图片列表）。"""
        primary = _burst_webp_dialog_state_path()
        legacy = _burst_webp_dialog_state_legacy_path()
        path = primary if primary.is_file() else legacy
        self._state_window_maximized = True
        self._state_window_geometry = None
        self._restore_saved_crop_once = False
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as ex:
            _burst_gui_log(f"加载上次动图窗口参数失败（将用默认）：{ex}")
            return
        if not isinstance(raw, dict):
            return
        if int(raw.get("version", 1)) < 1:
            return

        def _b(key: str, default: bool = True) -> bool:
            if key not in raw:
                return default
            return bool(raw[key])

        self.cb_wb.setChecked(_b("enable_wb", True))
        self.cb_eco.setChecked(_b("enable_eco", True))
        self.cb_align.setChecked(_b("enable_align", True))
        if "debug_align_overlay" in raw:
            try:
                self.cb_debug_align.setChecked(bool(raw["debug_align_overlay"]))
            except (TypeError, ValueError):
                pass
        self.cb_wm.setChecked(_b("enable_wm", True))
        if "wm_theme" in raw and isinstance(raw["wm_theme"], str):
            self.ed_wm_theme.setText(raw["wm_theme"])
        if "retention_pct" in raw:
            try:
                rp = int(raw["retention_pct"])
                self.slider_ret.setValue(int(np.clip(rp, 25, 100)))
            except (TypeError, ValueError):
                pass
        if "speed" in raw:
            try:
                self.spn_speed.setValue(float(raw["speed"]))
            except (TypeError, ValueError):
                pass
        if "max_long_edge" in raw:
            try:
                want = int(raw["max_long_edge"])
                for i in range(self.cmb_max.count()):
                    idat = self.cmb_max.itemData(i)
                    try:
                        edge = int(idat)
                    except (TypeError, ValueError):
                        continue
                    if edge == want:
                        self.cmb_max.setCurrentIndex(i)
                        break
            except (TypeError, ValueError):
                pass
        if "webp_quality" in raw:
            try:
                q = int(raw["webp_quality"])
                self.spn_q.setValue(int(np.clip(q, 40, 100)))
            except (TypeError, ValueError):
                pass
        if "out_path" in raw and isinstance(raw["out_path"], str):
            self.ed_out.setText(raw["out_path"])
        if str(raw.get("export_format", "webp")).lower() == "mp4":
            self.rb_export_mp4.setChecked(True)
        else:
            self.rb_export_webp.setChecked(True)
        self._apply_export_path_placeholder()
        self._sync_out_path_extension_to_format()
        if "crop_nx" in raw and "crop_ny" in raw:
            try:
                nx = float(raw["crop_nx"])
                ny = float(raw["crop_ny"])
                if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                    self._saved_crop_nx = nx
                    self._saved_crop_ny = ny
                    self._restore_saved_crop_once = bool(
                        raw.get("restore_crop_on_next_ref", True)
                    )
            except (TypeError, ValueError):
                pass
        self._align_roi_norm = None
        roi_raw = raw.get("align_track_roi")
        if isinstance(roi_raw, (list, tuple)) and len(roi_raw) == 4:
            try:
                xr = (
                    float(roi_raw[0]),
                    float(roi_raw[1]),
                    float(roi_raw[2]),
                    float(roi_raw[3]),
                )
                if _burst_align_roi_valid(xr):
                    self._align_roi_norm = xr
            except (TypeError, ValueError):
                pass
        if self._align_roi_norm is None:
            if raw.get("align_track_nx") is not None and raw.get("align_track_ny") is not None:
                try:
                    tnx = float(raw["align_track_nx"])
                    tny = float(raw["align_track_ny"])
                    if 0.0 <= tnx <= 1.0 and 0.0 <= tny <= 1.0:
                        half = 0.08
                        self._align_roi_norm = (
                            max(0.0, tnx - half),
                            max(0.0, tny - half),
                            min(1.0, tnx + half),
                            min(1.0, tny + half),
                        )
                except (TypeError, ValueError):
                    pass
        if "align_feature_detector" in raw:
            det = str(raw["align_feature_detector"]).strip().upper()
            for i in range(self.cmb_align_feat.count()):
                idat = self.cmb_align_feat.itemData(i)
                if str(idat).strip().upper() == det:
                    self.cmb_align_feat.setCurrentIndex(i)
                    break
        if "window_maximized" in raw:
            self._state_window_maximized = bool(raw["window_maximized"])
        wg = raw.get("window_geometry")
        if isinstance(wg, (list, tuple)) and len(wg) == 4:
            try:
                x, y, w, h = int(wg[0]), int(wg[1]), int(wg[2]), int(wg[3])
                if w >= 400 and h >= 320:
                    self._state_window_geometry = (x, y, w, h)
            except (TypeError, ValueError):
                pass
        _burst_gui_log(f"已加载上次的动图窗口参数：{path}")
        if path == legacy and not primary.is_file():
            try:
                shutil.copy2(legacy, primary)
                _burst_gui_log(f"已将动图设置复制到首选路径：{primary}")
            except OSError as ex:
                _burst_gui_log(f"复制到首选路径失败（不影响本次使用）：{ex}")
        self._log_wm_toggle()
        self._on_ret_changed(int(self.slider_ret.value()))
        if _burst_align_roi_valid(self._align_roi_norm):
            self._pv_canvas.set_track_roi_norm(self._align_roi_norm)
        else:
            self._pv_canvas.clear_track_roi()

    def _save_burst_dialog_state(self) -> None:
        """写入参数（不含待合成图片列表）；首选本机 AppData，失败则写 src 旁旧路径。"""
        primary = _burst_webp_dialog_state_path()
        legacy = _burst_webp_dialog_state_legacy_path()
        try:
            maxed = self.isMaximized()
            rg = self.normalGeometry() if maxed else self.geometry()
            geom = [int(rg.x()), int(rg.y()), int(rg.width()), int(rg.height())]
        except Exception:
            maxed, geom = True, [100, 100, 1240, 820]
        roi_save = (
            list(self._align_roi_norm)
            if _burst_align_roi_valid(self._align_roi_norm)
            else None
        )
        data = {
            "version": 3,
            "enable_wb": self.cb_wb.isChecked(),
            "enable_eco": self.cb_eco.isChecked(),
            "enable_align": self.cb_align.isChecked(),
            "debug_align_overlay": self.cb_debug_align.isChecked(),
            "enable_wm": self.cb_wm.isChecked(),
            "wm_theme": self.ed_wm_theme.text(),
            "retention_pct": int(self.slider_ret.value()),
            "speed": float(self.spn_speed.value()),
            "max_long_edge": _int_safe_combo_data(self.cmb_max, 1600),
            "webp_quality": int(self.spn_q.value()),
            "out_path": self.ed_out.text().strip(),
            "crop_nx": float(self._crop_nx),
            "crop_ny": float(self._crop_ny),
            "align_track_roi": roi_save,
            "align_feature_detector": str(
                self.cmb_align_feat.currentData() or "ORB"
            ).upper(),
            "export_format": ("mp4" if self.rb_export_mp4.isChecked() else "webp"),
            "restore_crop_on_next_ref": True,
            "window_maximized": bool(maxed),
            "window_geometry": geom,
        }
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        try:
            primary.parent.mkdir(parents=True, exist_ok=True)
            primary.write_text(text, encoding="utf-8")
            _burst_gui_log(f"已保存动图窗口参数（不含图片列表）：{primary}")
            return
        except OSError as ex1:
            if primary != legacy:
                try:
                    legacy.write_text(text, encoding="utf-8")
                    _burst_gui_log(
                        f"已保存动图窗口参数到备用路径：{legacy}（首选失败：{ex1}）"
                    )
                    return
                except OSError as ex2:
                    _burst_gui_log(
                        f"保存动图窗口参数失败：首选 {ex1}；备用 {ex2}"
                    )
            else:
                _burst_gui_log(f"保存动图窗口参数失败：{ex1}")


def open_burst_webp_dialog(parent, default_dir: str = "") -> None:
    _burst_gui_log(
        f"打开动图对话框：default_dir={repr(default_dir or '')}，"
        f"parent={'有' if parent is not None else '无'}"
    )
    try:
        dlg = BurstWebpDialog(parent, default_dir=default_dir or "")
    except Exception as ex:
        _burst_gui_log(f"动图对话框构造失败：{ex}")
        import traceback

        traceback.print_exc()
        QMessageBox.critical(
            parent,
            "动图",
            f"无法打开连拍动图窗口（构造异常）：\n{ex}",
        )
        return
    try:
        if getattr(dlg, "_state_window_maximized", True):
            dlg.showMaximized()
        else:
            g = getattr(dlg, "_state_window_geometry", None)
            if g is not None and len(g) == 4:
                gx, gy, gw, gh = int(g[0]), int(g[1]), int(g[2]), int(g[3])
                scr = _burst_safe_available_geometry(parent)
                sx0, sy0 = int(scr.x()), int(scr.y())
                sx1, sy1 = sx0 + int(scr.width()), sy0 + int(scr.height())
                if gw >= 400 and gh >= 320:
                    gx = int(np.clip(gx, sx0 - gw + 160, sx1 - 160))
                    gy = int(np.clip(gy, sy0 - gh + 120, sy1 - 120))
                    dlg.setGeometry(gx, gy, gw, gh)
            dlg.show()
        dlg.exec_()
    except Exception as ex:
        _burst_gui_log(f"动图对话框显示/运行失败：{ex}")
        import traceback

        traceback.print_exc()
        QMessageBox.critical(
            parent,
            "动图",
            f"动图窗口异常退出：\n{ex}",
        )
    _burst_gui_log("动图对话框 exec_ 已返回（窗口已关闭）。")
