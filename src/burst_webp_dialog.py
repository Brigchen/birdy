# -*- coding: utf-8 -*-
"""连拍 → WebP 动图：弹窗选择、参数与预览（供 birdy_gui 调用）。"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
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

from burst_anchor import (
    FrameLayout,
    geom_from_first,
    in_bounds_crop_xyxy,
    layout_from_anchor,
    layout_valid,
    merge_propagate,
)
from burst_project import (
    PROJECT_SUFFIX,
    build_project_dict,
    default_project_path_for_images,
    is_burst_project_path,
    load_project_file,
    match_layout_for_path,
    save_project_file,
)
from burst_webp import (
    BurstWebpBuildOptions,
    build_animated_mp4,
    build_animated_webp,
    build_preview_frames_rgb,
    gray_world_white_balance,
    infer_crop_center_norm_from_birds,
    sort_paths_by_capture_time,
)
from image_io import file_filter_all_images, imread_bgr

_DRAG_PX = 8


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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def _int_safe_combo_data(combo: QComboBox, default: int) -> int:
    try:
        return int(combo.currentData())
    except (TypeError, ValueError):
        return default


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
    设置阶段：当前帧 + 红十字（标定点，单击）+ 绿框（裁剪区，拖拽）。
    播放模式：动效帧。
    """

    anchor_changed = pyqtSignal(float, float)
    crop_changed = pyqtSignal(float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pix: Optional[QPixmap] = None
        self._ax = 0.5
        self._ay = 0.5
        self._crop: Optional[Tuple[float, float, float, float]] = None
        self._dest = (0, 0, 1, 1)
        self._iw = 1
        self._ih = 1
        self._playback_frames: List[QImage] = []
        self._playback_idx = 0
        self._press: Optional[Tuple[int, int, float, float]] = None
        self._drag_a: Optional[Tuple[float, float]] = None
        self._drag_b: Optional[Tuple[float, float]] = None
        self._dragging = False
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

    def image_wh(self) -> Tuple[int, int]:
        return int(self._iw), int(self._ih)

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

    def set_layout(self, lay: Optional[FrameLayout]) -> None:
        if lay is None:
            self._ax, self._ay = 0.5, 0.5
            self._crop = None
        else:
            self._ax = float(np.clip(lay.ax, 0.0, 1.0))
            self._ay = float(np.clip(lay.ay, 0.0, 1.0))
            self._crop = (float(lay.x0), float(lay.y0), float(lay.x1), float(lay.y1))
        self.update()

    def set_anchor_norm(self, ax: float, ay: float) -> None:
        self._ax = float(np.clip(ax, 0.0, 1.0))
        self._ay = float(np.clip(ay, 0.0, 1.0))
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
                "添加图片后：单击设标定点（红十字），拖拽矩形设裁剪区（绿框）。\n"
                "用上一张/下一张检查各帧；改首图后自动重算未锁定的后续帧。",
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

        def _draw_crop(x0n: float, y0n: float, x1n: float, y1n: float) -> None:
            xa = dox + int(round(x0n * dw))
            ya = doy + int(round(y0n * dh))
            xb = dox + int(round(x1n * dw))
            yb = doy + int(round(y1n * dh))
            rx = min(xa, xb)
            ry = min(ya, yb)
            rw = max(1, abs(xb - xa))
            rh = max(1, abs(yb - ya))
            pen_rect = QPen(QColor(120, 220, 140))
            pen_rect.setWidth(2)
            p.setPen(pen_rect)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rx, ry, rw, rh)

        crop = self._crop
        if (
            self._dragging
            and self._drag_a is not None
            and self._drag_b is not None
        ):
            x0n, x1n = sorted((self._drag_a[0], self._drag_b[0]))
            y0n, y1n = sorted((self._drag_a[1], self._drag_b[1]))
            _draw_crop(x0n, y0n, x1n, y1n)
        elif crop is not None:
            _draw_crop(*crop)

        cx_w = dox + int(round(self._ax * dw))
        cy_w = doy + int(round(self._ay * dh))
        arm = max(10, min(dw, dh) // 18)
        pen_cross = QPen(QColor(255, 60, 60))
        pen_cross.setWidth(2)
        p.setPen(pen_cross)
        p.drawLine(cx_w - arm, cy_w, cx_w + arm, cy_w)
        p.drawLine(cx_w, cy_w - arm, cx_w, cy_w + arm)

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
        self._press = (int(e.x()), int(e.y()), pn[0], pn[1])
        self._drag_a = pn
        self._drag_b = pn
        self._dragging = False

    def mouseMoveEvent(self, e) -> None:
        if self._playback_frames or self._base_pix is None or self._base_pix.isNull():
            return
        if self._press is None:
            return
        if not (e.buttons() & Qt.LeftButton):
            return
        dx = abs(int(e.x()) - self._press[0])
        dy = abs(int(e.y()) - self._press[1])
        if dx >= _DRAG_PX or dy >= _DRAG_PX:
            self._dragging = True
        pn = self._norm_from_widget(e.x(), e.y())
        if pn is None:
            return
        self._drag_b = pn
        if self._dragging:
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self._playback_frames or self._base_pix is None or self._base_pix.isNull():
            return
        if e.button() != Qt.LeftButton or self._press is None:
            return
        pn = self._norm_from_widget(e.x(), e.y())
        if pn is None:
            pn = (self._press[2], self._press[3])
        dragging = self._dragging
        self._press = None
        self._dragging = False
        if dragging and self._drag_a is not None:
            x0, x1 = sorted((self._drag_a[0], pn[0]))
            y0, y1 = sorted((self._drag_a[1], pn[1]))
            min_sp = 0.02
            if x1 - x0 < min_sp:
                c = (x0 + x1) * 0.5
                x0, x1 = max(0.0, c - min_sp * 0.5), min(1.0, c + min_sp * 0.5)
            if y1 - y0 < min_sp:
                c = (y0 + y1) * 0.5
                y0, y1 = max(0.0, c - min_sp * 0.5), min(1.0, c + min_sp * 0.5)
            self._crop = (x0, y0, x1, y1)
            self._drag_a = self._drag_b = None
            self.crop_changed.emit(x0, y0, x1, y1)
            self.update()
            return
        self._drag_a = self._drag_b = None
        self._ax = float(np.clip(pn[0], 0.0, 1.0))
        self._ay = float(np.clip(pn[1], 0.0, 1.0))
        self.anchor_changed.emit(self._ax, self._ay)
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
                f"(WB={self._opts.enable_white_balance}, "
                f"自动曝光={self._opts.enable_auto_exposure}×{self._opts.auto_exposure_strength:.2f}, "
                f"模式={self._opts.mode}, fps={self._opts.fps:g}, "
                f"水印={'开' if self._opts.watermark_options else '关'})"
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
    done = pyqtSignal(object, float, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: List[str],
        opts: BurstWebpBuildOptions,
        parent=None,
    ):
        super().__init__(parent)
        self._paths = list(paths)
        self._opts = opts

    def run(self) -> None:
        _burst_worker_configure_openmp()
        try:
            _burst_gui_log(
                f"预览线程开始：{len(self._paths)} 张，模式={self._opts.mode}，"
                f"fps={self._opts.fps:g}"
            )

            def _cb(cur: int, tot: int, msg: str) -> None:
                self.progress.emit(cur, tot, f"[预览] {msg}")

            pil_list, dur, note, _ref0 = build_preview_frames_rgb(
                self._paths,
                self._opts,
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
            _burst_gui_log(
                f"预览线程完成：{len(qimgs)} 张 QImage，间隔≈{dur:.1f} ms（{note}）"
            )
            self.done.emit(qimgs, float(dur), note)
        except Exception as e:
            _burst_gui_log(f"预览线程异常：{e}")
            self.failed.emit(str(e))


class BurstAnchorPropagateWorker(QThread):
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(object, int)
    failed = pyqtSignal(str, int)

    def __init__(
        self,
        paths: List[str],
        layouts: List[Optional[FrameLayout]],
        mode: str,
        enable_wb: bool,
        job_id: int,
        parent=None,
        get_detector: Optional[Callable[[], Optional[object]]] = None,
    ):
        super().__init__(parent)
        self._paths = list(paths)
        self._layouts = list(layouts)
        self._mode = mode
        self._enable_wb = bool(enable_wb)
        self._job_id = int(job_id)
        self._get_detector = get_detector

    def run(self) -> None:
        _burst_worker_configure_openmp()
        jid = self._job_id
        try:
            n = len(self._paths)
            frames: List[np.ndarray] = []
            for i, p in enumerate(self._paths):
                if self.isInterruptionRequested():
                    return
                self.progress.emit(i + 1, n + 2, f"传播：加载 {i + 1}/{n}")
                im = imread_bgr(p, raw_half_size=True)
                if im is None:
                    raise RuntimeError(f"无法读取：{p}")
                x = im
                if self._enable_wb:
                    x = gray_world_white_balance(x)
                frames.append(x)
            detect_fn = None
            det = None
            self.progress.emit(n + 1, n + 2, "传播：加载鸟体模型…")
            if self._get_detector is not None:
                try:
                    det = self._get_detector()
                except Exception as ex:
                    _burst_gui_log(f"传播：主界面鸟体模型不可用：{ex}")
                    det = None
            if det is None:
                try:
                    from auto_exposure import _get_bird_detector

                    det = _get_bird_detector()
                except Exception as ex:
                    _burst_gui_log(f"传播：鸟体模型加载失败，改用模板匹配：{ex}")
                    det = None
            if det is not None and hasattr(det, "detect_birds"):

                def detect_fn(bgr, _det=det):
                    try:
                        return _det.detect_birds(bgr) or []
                    except Exception:
                        return []

                _burst_gui_log("传播：已启用 YOLO 鸟体跟踪 + 卡尔曼。")
            else:
                _burst_gui_log("传播：无鸟体模型，仅模板匹配。")

            def _prog(cur: int, tot: int, msg: str) -> None:
                if self.isInterruptionRequested():
                    return
                self.progress.emit(max(1, cur + 1), max(tot + 1, 1), msg)

            self.progress.emit(n + 2, n + 2, "传播：鸟体追踪 / 模板匹配…")
            merged = merge_propagate(
                frames,
                self._layouts,
                self._mode,
                detect_birds_fn=detect_fn,
                progress=_prog,
            )
            self.done.emit(merged, jid)
        except Exception as e:
            _burst_gui_log(f"标定点传播失败：{e}")
            self.failed.emit(str(e), jid)


class BurstRefBirdWorker(QThread):
    """
    首张参考图上的鸟体检测，用于建议标定点。
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
                f"首张参考：鸟检线程完成，{len(birds)} 框，标定点=({nx:.4f},{ny:.4f})"
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
        self._preview_dur_ms = 500.0
        self._preview_idx = 0
        self._pv_worker: Optional[BurstWebpPreviewWorker] = None
        self._build_worker: Optional[BurstWebpBuildWorker] = None
        self._ref_bird_worker: Optional[BurstRefBirdWorker] = None
        self._ref_bird_job_id = 0
        self._prop_worker: Optional[BurstAnchorPropagateWorker] = None
        self._prop_job_id = 0
        self._layouts: List[Optional[FrameLayout]] = []
        self._layout_paths: List[str] = []
        self._layout_by_path: Dict[str, FrameLayout] = {}
        self._frame_idx = 0
        self._project_path: Optional[Path] = None
        self._project_path_user_set = False
        self._project_io_ready = False
        self._loading_project = False
        self._project_bulk_update = False
        self._last_project_path_hint = ""
        self._bgr_cache: Dict[Tuple[str, bool], np.ndarray] = {}
        self._ae_corr_cache: Dict[tuple, np.ndarray] = {}
        self._ref_bird_key: Optional[Tuple[str, bool]] = None
        self._logged_show_path: Optional[str] = None
        self._later_anchor_dirty = False
        self._first_wh: Optional[Tuple[int, int]] = None
        self._suggested_ax = 0.5
        self._suggested_ay = 0.5
        self._anchor_user_touched = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_preview_tick)
        self._ref_debounce = QTimer(self)
        self._ref_debounce.setSingleShot(True)
        self._ref_debounce.setInterval(200)
        self._ref_debounce.timeout.connect(self._show_current_source_frame)
        self._prop_debounce = QTimer(self)
        self._prop_debounce.setSingleShot(True)
        self._prop_debounce.setInterval(350)
        self._prop_debounce.timeout.connect(self._start_propagate)

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

        row_proj = QHBoxLayout()
        self.btn_open_proj = QPushButton("打开项目…")
        self.btn_open_proj.setToolTip("打开相片目录下的 .birdy-burst.json，恢复图片列表与每帧标定点/裁剪区。")
        self.btn_open_proj.clicked.connect(self._on_open_project)
        self.btn_save_proj_as = QPushButton("项目另存为…")
        self.btn_save_proj_as.setToolTip("复制当前图片列表与定位到指定项目文件；之后自动保存到该文件。")
        self.btn_save_proj_as.clicked.connect(self._on_save_project_as)
        row_proj.addWidget(self.btn_open_proj)
        row_proj.addWidget(self.btn_save_proj_as)
        left_l.addLayout(row_proj)
        self.lbl_project = QLabel("项目：导入图片后将自动保存在相片目录")
        self.lbl_project.setWordWrap(True)
        self.lbl_project.setStyleSheet("color: #555; font-size: 9pt;")
        left_l.addWidget(self.lbl_project)

        opt_g = QGroupBox("处理与导出")
        fl = QFormLayout(opt_g)
        self.cb_wb = QCheckBox("灰世界白平衡")
        self.cb_wb.setChecked(True)
        self.cb_ae = QCheckBox("自动曝光")
        self.cb_ae.setChecked(True)
        self.cb_ae.setToolTip(
            "按每帧裁剪框测光，自动提亮或压暗（与水印前自动曝光同类算法）。\n"
            "勾选或拖动强度时，当前图立即预览效果。\n"
            "强度 1=算出的自动曝光；小于 1 减弱；大于 1 在自动曝光上继续加曝光（最大 3）。\n"
            "未画裁剪区时按全图测光；有裁剪区后按绿框测光（与导出一致）。"
        )
        self.cb_wm = QCheckBox("叠加水印（与主界面「水印与分享」当前选项一致，含布局 / Logo / 文字）")
        self.cb_wm.setChecked(True)
        self.cb_wm.setToolTip(
            "连拍 ≥2 张时：日期/相机/GPS/物种等一律按首张图的 EXIF 与路径解析，"
            "画面仍为各帧像素（与连拍白平衡、显影沿用首张参数一致）。单张预览仍按该张元数据。"
        )
        fl.addRow(self.cb_wb)
        fl.addRow(self.cb_ae)
        self.slider_ae = QSlider(Qt.Horizontal)
        self.slider_ae.setRange(0, 300)
        self.slider_ae.setSingleStep(5)
        self.slider_ae.setValue(100)
        self.slider_ae.setToolTip(
            "0=原图，1=按裁剪框算出的自动曝光，>1 在此基础上继续加曝光（2≈+1 档，3≈+2 档）。拖动时当前图同步预览。"
        )
        self.lbl_ae_strength = QLabel("1.00")
        self.lbl_ae_strength.setMinimumWidth(34)
        self.slider_ae.valueChanged.connect(
            lambda v: self.lbl_ae_strength.setText(f"{v / 100:.2f}")
        )
        ae_row = QHBoxLayout()
        ae_row.addWidget(self.slider_ae, 1)
        ae_row.addWidget(self.lbl_ae_strength)
        fl.addRow("曝光强度:", ae_row)
        self.cb_ae.toggled.connect(self.slider_ae.setEnabled)
        self.slider_ae.setEnabled(self.cb_ae.isChecked())
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

        self._bg_mode = QButtonGroup(self)
        self.rb_mode_fixed = QRadioButton("定点模式（三脚架）")
        self.rb_mode_track = QRadioButton("跟踪模式")
        self.rb_mode_fixed.setChecked(True)
        self.rb_mode_fixed.setToolTip(
            "相机相对固定时：后续帧用鸟体检测跟踪你点的那只鸟（卡尔曼预测位置），"
            "找不到鸟再退回首图模板匹配。鸟可在裁剪框内移动。"
        )
        self.rb_mode_track.setToolTip(
            "跟拍时：用鸟体 YOLO 检测 + 卡尔曼滤波跟踪标定点（眼/头相对鸟框位置保持不变）；"
            "漏检时用光流与模板在附近搜索。"
        )
        self._bg_mode.addButton(self.rb_mode_fixed)
        self._bg_mode.addButton(self.rb_mode_track)
        fl.addRow(self.rb_mode_fixed)
        fl.addRow(self.rb_mode_track)

        self.spn_fps = QDoubleSpinBox()
        self.spn_fps.setRange(0.25, 30.0)
        self.spn_fps.setSingleStep(0.5)
        self.spn_fps.setDecimals(2)
        self.spn_fps.setValue(2.0)
        self.spn_fps.setToolTip("播放时每秒显示几张图。默认 2。")
        fl.addRow("播放帧率（张/秒）:", self.spn_fps)

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

        self.lbl_fps_hint = QLabel("播放：每秒 2 张 → 每帧约 500 ms")
        self.lbl_fps_hint.setWordWrap(True)
        fl.addRow(self.lbl_fps_hint)

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
        self._pv_canvas.anchor_changed.connect(self._on_anchor_from_canvas)
        self._pv_canvas.crop_changed.connect(self._on_crop_from_canvas)
        pv_inner_l.addWidget(self._pv_canvas, stretch=1)

        self.lbl_pick_hint = QLabel(
            "单击：标定点（红十字）　拖拽：裁剪区（绿框）。"
            "后续帧单击后，裁剪区按首图相对位置与大小跟随。"
        )
        self.lbl_pick_hint.setStyleSheet("color: #666; font-size: 9pt;")
        pv_inner_l.addWidget(self.lbl_pick_hint)

        row_nav = QHBoxLayout()
        self.btn_frame_prev = QPushButton("上一张")
        self.btn_frame_next = QPushButton("下一张")
        self.lbl_frame_idx = QLabel("当前 — / —")
        self.btn_recompute = QPushButton("按首图重算后续")
        self.btn_recompute.setToolTip("解锁后续帧并用当前首图标定点/裁剪区重新自动传播。")
        self.btn_frame_prev.clicked.connect(lambda: self._step_frame(-1))
        self.btn_frame_next.clicked.connect(lambda: self._step_frame(1))
        self.btn_recompute.clicked.connect(self._on_recompute_later)
        self.list_w.itemClicked.connect(self._on_list_item_clicked)
        row_nav.addWidget(self.btn_frame_prev)
        row_nav.addWidget(self.lbl_frame_idx, 1)
        row_nav.addWidget(self.btn_frame_next)
        row_nav.addWidget(self.btn_recompute)
        pv_inner_l.addLayout(row_nav)

        row_prev = QHBoxLayout()
        self.btn_prev = QPushButton("更新预览")
        self.btn_prev.clicked.connect(self._start_preview)
        self.btn_reset_view = QPushButton("恢复初始设置")
        self.btn_reset_view.setToolTip(
            "停止动效预览，回到首张，并清除各帧标定点/裁剪区（可再标一次）。"
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
        self.spn_fps.valueChanged.connect(self._refresh_fps_hint)
        self.spn_fps.valueChanged.connect(self._log_fps_changed)
        self.cb_wb.stateChanged.connect(self._on_enhance_toggled)
        self.cb_wb.stateChanged.connect(self._log_wb_toggle)
        self.cb_ae.stateChanged.connect(self._log_ae_toggle)
        self.cb_ae.stateChanged.connect(self._on_ae_preview_changed)
        self.slider_ae.valueChanged.connect(self._on_ae_preview_changed)
        self.cb_wm.stateChanged.connect(self._log_wm_toggle)
        self.cmb_max.currentIndexChanged.connect(self._log_export_size_changed)
        self.spn_q.valueChanged.connect(self._log_webp_quality_changed)
        self.rb_mode_fixed.toggled.connect(self._on_mode_toggled)
        self.rb_mode_track.toggled.connect(self._on_mode_toggled)

        self._state_window_maximized = True
        self._state_window_geometry: Optional[Tuple[int, int, int, int]] = None
        self._state_io_ready = False
        self._save_state_timer = QTimer(self)
        self._save_state_timer.setSingleShot(True)
        self._save_state_timer.setInterval(400)
        self._save_state_timer.timeout.connect(self._save_burst_dialog_state)
        self._save_project_timer = QTimer(self)
        self._save_project_timer.setSingleShot(True)
        self._save_project_timer.setInterval(600)
        self._save_project_timer.timeout.connect(self._save_project_now)
        for sig in (
            self.cb_wb.stateChanged,
            self.cb_ae.stateChanged,
            self.slider_ae.valueChanged,
            self.cb_wm.stateChanged,
            self.spn_fps.valueChanged,
            self.cmb_max.currentIndexChanged,
            self.spn_q.valueChanged,
            self.rb_mode_fixed.toggled,
            self.rb_mode_track.toggled,
        ):
            sig.connect(self._schedule_burst_state_save)
            sig.connect(self._schedule_project_save)
        self.ed_wm_theme.textChanged.connect(self._schedule_burst_state_save)
        self.ed_wm_theme.textChanged.connect(self._schedule_project_save)
        self.ed_out.textChanged.connect(self._schedule_burst_state_save)
        self.ed_out.textChanged.connect(self._schedule_project_save)
        self._bg_export_fmt.buttonClicked.connect(self._schedule_burst_state_save)
        self._bg_export_fmt.buttonClicked.connect(self._schedule_project_save)

        self._load_burst_dialog_state()
        self._try_autoload_project_from_default_dir()
        self._state_io_ready = True
        self._project_io_ready = True

        _burst_gui_log(
            f"动图对话框初始化完成；默认相片目录={self._default_dir or '(空)'}，"
            f"列表中 {self.list_w.count()} 张。"
        )
        QTimer.singleShot(0, self._schedule_ref_refresh)
        QTimer.singleShot(0, self._log_wm_toggle)
        QTimer.singleShot(0, self._refresh_fps_hint)

    def _schedule_burst_state_save(self, *_args) -> None:
        """参数变更后防抖写入。初始化加载期间不写盘，避免用默认值覆盖上次设置。"""
        if not getattr(self, "_state_io_ready", False):
            return
        self._save_state_timer.stop()
        self._save_state_timer.start(400)

    def _schedule_project_save(self, *_args) -> None:
        if not getattr(self, "_project_io_ready", False):
            return
        if getattr(self, "_loading_project", False) or getattr(
            self, "_project_bulk_update", False
        ):
            return
        self._save_project_timer.stop()
        self._save_project_timer.start(600)

    def _burst_mode(self) -> str:
        return "track" if self.rb_mode_track.isChecked() else "fixed"

    def _log_wb_toggle(self, _state: int = 0) -> None:
        _burst_gui_log(f"参数：灰世界白平衡 → {'开启' if self.cb_wb.isChecked() else '关闭'}")

    def _log_ae_toggle(self, _state: int = 0) -> None:
        _burst_gui_log(f"参数：自动曝光 → {'开启' if self.cb_ae.isChecked() else '关闭'}")

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

    def _log_fps_changed(self, v: float) -> None:
        _burst_gui_log(f"参数：播放帧率 → {v:g} 张/秒")

    def _on_enhance_toggled(self, *_a) -> None:
        self._bgr_cache.clear()
        self._ae_corr_cache.clear()
        self._schedule_ref_refresh()

    def _on_ae_preview_changed(self, *_a) -> None:
        """勾选自动曝光或拖动强度时，刷新当前图预览（防抖）。"""
        self._schedule_ref_refresh()

    def _on_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        _burst_gui_log(f"参数：动图模式 → {self._burst_mode()}")
        if layout_valid(self._layouts[0] if self._layouts else None):
            self._schedule_propagate()

    def _schedule_ref_refresh(self) -> None:
        if self._pv_worker is not None and self._pv_worker.isRunning():
            _burst_gui_log("首张参考刷新已推迟：预览线程占用中。")
            return
        self._ref_debounce.stop()
        self._ref_debounce.start(200)

    def _flush_layouts_to_sticky(self) -> None:
        for p, lay in zip(self._layout_paths, self._layouts):
            if p and lay is not None:
                self._layout_by_path[p] = lay

    def _sync_layouts_to_list(self) -> None:
        new_paths = self._collect_paths()
        self._flush_layouts_to_sticky()
        if not getattr(self, "_project_bulk_update", False):
            live = set(new_paths)
            self._layout_by_path = {
                k: v for k, v in self._layout_by_path.items() if k in live
            }
        old_paths = list(self._layout_paths)
        old_lays = list(self._layouts)
        if (
            new_paths == old_paths
            and len(old_lays) == len(new_paths)
            and not getattr(self, "_project_bulk_update", False)
        ):
            return
        new_lays: List[Optional[FrameLayout]] = []
        for i, p in enumerate(new_paths):
            if p in self._layout_by_path:
                new_lays.append(self._layout_by_path[p])
            elif i < len(old_lays) and i < len(old_paths) and old_paths[i] == p:
                new_lays.append(old_lays[i])
            elif not old_paths and i < len(old_lays):
                new_lays.append(old_lays[i])
            else:
                new_lays.append(None)
        self._layouts = new_lays
        self._layout_paths = list(new_paths)
        if self._frame_idx >= len(new_paths):
            self._frame_idx = max(0, len(new_paths) - 1)
        if not new_paths:
            self._frame_idx = 0
            self._first_wh = None

    def _begin_list_bulk(self) -> None:
        self._flush_layouts_to_sticky()
        self._project_bulk_update = True

    def _end_list_bulk(self) -> None:
        self._project_bulk_update = False
        self._sync_layouts_to_list()
        self._refresh_fps_hint()
        self._schedule_ref_refresh()
        self._schedule_project_save()

    def _on_list_changed(self, *args) -> None:
        del args
        if getattr(self, "_project_bulk_update", False) or getattr(
            self, "_loading_project", False
        ):
            return
        n = self.list_w.count()
        _burst_gui_log(f"图片列表已变化，当前共 {n} 项。")
        self._sync_layouts_to_list()
        self._refresh_fps_hint()
        self._schedule_ref_refresh()
        self._schedule_project_save()

    def _collect_paths(self) -> List[str]:
        return [self.list_w.item(i).text() for i in range(self.list_w.count())]

    def _refresh_fps_hint(self) -> None:
        fps = float(self.spn_fps.value())
        dur = 1000.0 / max(0.1, fps)
        n = self.list_w.count()
        extra = f"；列表 {n} 张" if n else ""
        self.lbl_fps_hint.setText(
            f"播放：每秒 {fps:g} 张 → 每帧约 {dur:.0f} ms{extra}"
        )
        self._update_frame_nav_label()

    def _update_frame_nav_label(self) -> None:
        n = self.list_w.count()
        if n <= 0:
            self.lbl_frame_idx.setText("当前 — / —")
            return
        i = int(np.clip(self._frame_idx, 0, n - 1))
        lay = self._layouts[i] if i < len(self._layouts) else None
        tag = ""
        if lay is not None:
            if i == 0:
                tag = " · 首图"
            elif not lay.auto:
                if i == self._frame_idx and self._later_anchor_dirty:
                    tag = " · 待锁定"
                else:
                    tag = " · 已锁定"
            else:
                tag = f" · 自动 {lay.conf:.2f}"
        self.lbl_frame_idx.setText(f"当前 {i + 1} / {n}{tag}")

    def _clone_layouts(self) -> List[Optional[FrameLayout]]:
        out: List[Optional[FrameLayout]] = []
        for lay in self._layouts:
            out.append(replace(lay) if lay is not None else None)
        return out

    def _layout_lock_counts(self) -> Tuple[int, int]:
        n_auto = 0
        n_lock = 0
        for i, lay in enumerate(self._layouts):
            if i == 0 or lay is None:
                continue
            if lay.auto:
                n_auto += 1
            else:
                n_lock += 1
        return n_auto, n_lock

    def _refresh_lock_status(self, prefix: str = "") -> None:
        self._update_frame_nav_label()
        n_auto, n_lock = self._layout_lock_counts()
        n_later = max(0, len(self._layouts) - 1)
        body = (
            f"自动 {n_auto} 张，锁定 {n_lock} 张"
            + (f"（共 {n_later} 张后续帧）" if n_later else "")
            + "。可用上一张/下一张检查或改标定点。"
        )
        self.lbl_pv_status.setText((prefix + body) if prefix else body)

    def _on_add_files(self) -> None:
        _burst_gui_log("操作：添加图片…（文件选择对话框已打开）")
        start = self._file_dialog_start_dir()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择连拍图片或动图项目",
            start,
            f"{file_filter_all_images()};;"
            f"动图项目 (*{PROJECT_SUFFIX});;"
            "所有文件 (*.*)",
        )
        if not files:
            _burst_gui_log("操作：添加图片取消。")
            return
        proj_files = [f for f in files if is_burst_project_path(f)]
        img_files = [
            f
            for f in files
            if f not in proj_files and os.path.isfile(f)
        ]
        if proj_files and not img_files:
            self._load_project_from_path(Path(proj_files[0]), user_set=True)
            return
        self._import_image_paths(img_files)

    def _import_image_paths(self, files: List[str]) -> None:
        abs_files = [
            os.path.abspath(fp) for fp in files if fp and os.path.isfile(fp)
        ]
        if not abs_files:
            return
        n0 = self.list_w.count()
        was_empty = n0 == 0
        cand = default_project_path_for_images(abs_files)
        if was_empty and cand is not None and cand.is_file():
            self._load_project_from_path(
                cand, user_set=False, warn_missing=False, apply_options=True
            )
            have = {os.path.abspath(p) for p in self._collect_paths()}
            extra = [p for p in abs_files if p not in have]
            if extra:
                self._begin_list_bulk()
                for p in extra:
                    self.list_w.addItem(p)
                self._end_list_bulk()
                self._save_project_now()
            _burst_gui_log(
                f"操作：已从项目恢复 {self.list_w.count()} 张（本次选择 {len(abs_files)} 张）。"
            )
            return
        existing = {os.path.abspath(p) for p in self._collect_paths()}
        self._begin_list_bulk()
        added = 0
        for fp in abs_files:
            if fp in existing:
                continue
            self.list_w.addItem(fp)
            existing.add(fp)
            added += 1
        self._default_dir = os.path.dirname(abs_files[0])
        self._end_list_bulk()
        self._ensure_default_project_path()
        self._hydrate_layouts_from_project_file()
        self._save_project_now()
        _burst_gui_log(
            f"操作：添加图片结束，本次加入 {added} 张，列表由 {n0} → {self.list_w.count()} 项。"
        )

    def _on_remove_sel(self) -> None:
        n0 = self.list_w.count()
        _burst_gui_log(f"操作：移除所选（移除前列表 {n0} 项）…")
        for it in self.list_w.selectedItems():
            row = self.list_w.row(it)
            self.list_w.takeItem(row)
        self._refresh_fps_hint()
        self._schedule_project_save()
        _burst_gui_log(f"操作：移除完成，列表余 {self.list_w.count()} 项。")

    def _on_sort_time(self) -> None:
        n = self.list_w.count()
        _burst_gui_log(f"操作：按拍摄时间排序（共 {n} 项）…")
        paths = self._collect_paths()
        self._begin_list_bulk()
        for _ in range(self.list_w.count()):
            self.list_w.takeItem(0)
        for p in sort_paths_by_capture_time(paths):
            self.list_w.addItem(p)
        self._frame_idx = 0
        self._end_list_bulk()
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
        self._frame_idx = nr
        self._refresh_fps_hint()
        self._schedule_ref_refresh()
        self._schedule_project_save()

    def _file_dialog_start_dir(self) -> str:
        if self._project_path is not None:
            parent = str(self._project_path.parent)
            if os.path.isdir(parent):
                return parent
        hint = str(self._last_project_path_hint or "").strip()
        if hint:
            hp = Path(hint)
            if hp.is_file():
                return str(hp.parent)
            if hp.is_dir():
                return str(hp)
        start = self._default_dir
        if start and os.path.isdir(start):
            return start
        return os.path.expanduser("~")

    def _update_project_label(self) -> None:
        if self._project_path is not None:
            self.lbl_project.setText(f"项目：{self._project_path.name}")
            self.lbl_project.setToolTip(str(self._project_path))
        else:
            self.lbl_project.setText("项目：导入图片后将自动保存在相片目录")
            self.lbl_project.setToolTip("")

    def _project_options_snapshot(self) -> dict:
        d = self._collect_burst_dialog_state()
        keys = (
            "enable_wb",
            "enable_auto_exposure",
            "auto_exposure_strength",
            "enable_wm",
            "wm_theme",
            "burst_mode",
            "fps",
            "max_long_edge",
            "webp_quality",
            "out_path",
            "export_format",
        )
        return {k: d[k] for k in keys if k in d}

    def _ensure_default_project_path(self) -> None:
        if self._project_path_user_set:
            self._update_project_label()
            return
        paths = self._collect_paths()
        if not paths:
            self._project_path = None
            self._update_project_label()
            return
        p = default_project_path_for_images(paths)
        if p is not None:
            self._project_path = p
        self._update_project_label()

    def _try_autoload_project_from_default_dir(self) -> None:
        d = str(self._default_dir or "").strip()
        if not d or not os.path.isdir(d):
            return
        folder = Path(d)
        cand = folder / f"{folder.name}{PROJECT_SUFFIX}"
        if cand.is_file():
            self._load_project_from_path(
                cand, user_set=False, warn_missing=False, apply_options=True
            )

    def _hydrate_layouts_from_project_file(self) -> None:
        if self._project_path is None or not self._project_path.is_file():
            return
        try:
            data = load_project_file(self._project_path)
        except Exception as ex:
            _burst_gui_log(f"读取已有动图项目失败（将新建/覆盖）：{ex}")
            return
        entries = list(zip(data.paths, data.layouts))
        self._sync_layouts_to_list()
        filled = 0
        for i, p in enumerate(self._collect_paths()):
            if i < len(self._layouts) and self._layouts[i] is not None:
                continue
            lay = match_layout_for_path(p, entries)
            if lay is None:
                continue
            cloned = replace(lay)
            self._layouts[i] = cloned
            self._layout_by_path[p] = cloned
            filled += 1
        if layout_valid(self._layouts[0] if self._layouts else None):
            self._anchor_user_touched = True
        if filled:
            _burst_gui_log(f"已从项目文件恢复 {filled} 帧定位：{self._project_path}")
            self._refresh_lock_status()
            self._show_current_source_frame()
        if layout_valid(self._layouts[0] if self._layouts else None) and any(
            i > 0 and self._layouts[i] is None for i in range(len(self._layouts))
        ):
            self._schedule_propagate()

    def _stop_propagate_if_running(self) -> None:
        w = self._prop_worker
        if w is not None and w.isRunning():
            self._prop_job_id += 1
            w.requestInterruption()
            w.wait(1500)

    def _on_open_project(self) -> None:
        _burst_gui_log("操作：打开项目…")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开动图项目",
            self._file_dialog_start_dir(),
            f"动图项目 (*{PROJECT_SUFFIX});;JSON (*.json);;所有文件 (*.*)",
        )
        if not path:
            return
        self._load_project_from_path(Path(path), user_set=True)

    def _ensure_project_suffix(self, path: Path) -> Path:
        name = path.name
        lower = name.lower()
        if lower.endswith(PROJECT_SUFFIX):
            return path
        if lower.endswith(".json"):
            return path.with_name(path.stem + PROJECT_SUFFIX)
        return path.with_name(name + PROJECT_SUFFIX)

    def _on_save_project_as(self) -> None:
        paths = self._collect_paths()
        if not paths:
            QMessageBox.information(self, "提示", "请先添加图片，再另存项目。")
            return
        start = self._project_path
        if start is None:
            start = default_project_path_for_images(paths)
        start_str = str(start) if start is not None else self._file_dialog_start_dir()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "项目另存为",
            start_str,
            f"动图项目 (*{PROJECT_SUFFIX})",
        )
        if not path:
            return
        dest = self._ensure_project_suffix(Path(path))
        self._project_path = dest
        self._project_path_user_set = True
        self._update_project_label()
        self._save_project_now()
        if dest.is_file():
            _burst_gui_log(f"操作：项目已另存为 {dest}")
        else:
            QMessageBox.warning(self, "另存失败", f"无法写入项目文件：\n{dest}")

    def _load_project_from_path(
        self,
        path: Path,
        *,
        user_set: bool,
        warn_missing: bool = True,
        apply_options: bool = True,
    ) -> None:
        path = Path(path)
        try:
            data = load_project_file(path)
        except Exception as ex:
            _burst_gui_log(f"打开动图项目失败：{ex}")
            if warn_missing:
                QMessageBox.critical(self, "打开项目失败", str(ex))
            return
        self._stop_propagate_if_running()
        self._loading_project = True
        self._begin_list_bulk()
        try:
            self.list_w.clear()
            for p in data.paths:
                self.list_w.addItem(p)
            self._layouts = [replace(x) if x is not None else None for x in data.layouts]
            self._layout_paths = list(data.paths)
            self._layout_by_path = {
                p: lay
                for p, lay in zip(self._layout_paths, self._layouts)
                if p and lay is not None
            }
            self._frame_idx = int(data.frame_idx)
            self._first_wh = None
            self._later_anchor_dirty = False
            self._anchor_user_touched = layout_valid(
                self._layouts[0] if self._layouts else None
            )
            self._project_path = path
            self._project_path_user_set = bool(user_set)
            self._last_project_path_hint = str(path)
            self._default_dir = str(path.parent)
            self._bgr_cache.clear()
            self._ae_corr_cache.clear()
            if apply_options and data.options:
                self._block_option_signals(True)
                try:
                    self._apply_burst_dialog_state(data.options)
                finally:
                    self._block_option_signals(False)
            self._update_project_label()
        finally:
            self._end_list_bulk()
            self._loading_project = False
        miss_n = len(data.missing)
        _burst_gui_log(
            f"已打开动图项目：{path}，恢复 {len(data.paths)} 张"
            + (f"，缺少 {miss_n} 个文件" if miss_n else "")
        )
        if warn_missing and data.missing:
            shown = "\n".join(data.missing[:8])
            extra = f"\n…共 {len(data.missing)} 个" if len(data.missing) > 8 else ""
            QMessageBox.warning(
                self,
                "部分图片缺失",
                f"项目中有文件找不到，已跳过：\n{shown}{extra}",
            )
        self._show_current_source_frame()
        self._refresh_lock_status()
        if layout_valid(self._layouts[0] if self._layouts else None) and any(
            i > 0 and self._layouts[i] is None for i in range(len(self._layouts))
        ):
            self._schedule_propagate()

    def _save_project_now(self) -> None:
        if not getattr(self, "_project_io_ready", False):
            return
        if getattr(self, "_loading_project", False) or getattr(
            self, "_project_bulk_update", False
        ):
            return
        if self._project_path is None:
            self._ensure_default_project_path()
        if self._project_path is None:
            return
        paths = self._collect_paths()
        if not paths:
            return
        self._sync_layouts_to_list()
        try:
            payload = build_project_dict(
                paths,
                self._layouts,
                project_path=self._project_path,
                frame_idx=self._frame_idx,
                options=self._project_options_snapshot(),
            )
            save_project_file(self._project_path, payload)
            self._last_project_path_hint = str(self._project_path)
            self._update_project_label()
            _burst_gui_log(f"已保存动图项目：{self._project_path}")
        except OSError as ex:
            _burst_gui_log(f"保存动图项目失败：{ex}")

    def _on_list_item_clicked(self, item) -> None:
        row = self.list_w.row(item)
        if row >= 0:
            self._goto_frame(row)

    def _frame_image_wh(self, idx: int) -> Optional[Tuple[int, int]]:
        if idx == int(self._frame_idx):
            w, h = self._pv_canvas.image_wh()
            if w > 1 and h > 1:
                return w, h
        paths = self._collect_paths()
        if idx < 0 or idx >= len(paths):
            return None
        bgr = self._load_processed_bgr(paths[idx])
        if bgr is None:
            return None
        return int(bgr.shape[1]), int(bgr.shape[0])

    def _apply_later_anchor(self, idx: int, ax: float, ay: float) -> bool:
        """后续帧：按首图裁剪相对位置与大小，把绿框放到新标定点上。"""
        geom = self._geom_ready()
        if geom is None:
            return False
        wh = self._frame_image_wh(idx)
        if wh is None:
            return False
        w, h = wh
        self._layouts[idx] = layout_from_anchor(
            ax, ay, geom, w, h, auto=False, conf=1.0
        )
        self._later_anchor_dirty = True
        return True

    def _commit_later_frame_lock(self) -> None:
        """离开当前后续帧时：若刚指定了标定点，按首图几何确认裁剪区并锁定。"""
        idx = int(self._frame_idx)
        if idx <= 0 or idx >= len(self._layouts):
            return
        lay = self._layouts[idx]
        if lay is None:
            return
        if self._later_anchor_dirty:
            if self._apply_later_anchor(idx, float(lay.ax), float(lay.ay)):
                _burst_gui_log(
                    f"离开第 {idx + 1} 帧：已锁定，传播将跳过本页（不整段重跑）"
                )
            else:
                lay.auto = False
            return
        if not lay.auto:
            lay.auto = False

    def _goto_frame(self, idx: int) -> None:
        n = self.list_w.count()
        if n <= 0:
            return
        idx = int(np.clip(idx, 0, n - 1))
        if idx == int(self._frame_idx):
            return
        self._commit_later_frame_lock()
        self._later_anchor_dirty = False
        self._stop_playback_keep_edit()
        self._frame_idx = idx
        self._show_current_source_frame()

    def _step_frame(self, delta: int) -> None:
        n = self.list_w.count()
        if n <= 0:
            return
        self._goto_frame(int(self._frame_idx) + int(delta))

    def _stop_playback_keep_edit(self) -> None:
        self._timer.stop()
        self._preview_qimages = []
        self._pv_canvas.stop_playback()

    def _cache_key(self, path: str) -> Tuple[str, bool]:
        return (path, self.cb_wb.isChecked())

    def _load_processed_bgr(self, path: str) -> Optional[np.ndarray]:
        key = self._cache_key(path)
        hit = self._bgr_cache.get(key)
        if hit is not None:
            return hit
        bgr = imread_bgr(path, raw_half_size=True)
        if bgr is None or bgr.size == 0:
            return None
        x = bgr
        if self.cb_wb.isChecked():
            x = gray_world_white_balance(x)
        x = np.ascontiguousarray(x, dtype=np.uint8)
        self._bgr_cache[key] = x
        if len(self._bgr_cache) > 24:
            oldest = next(iter(self._bgr_cache))
            if oldest != key:
                self._bgr_cache.pop(oldest, None)
        return x

    def _current_ae_meter_box(self, bgr: np.ndarray) -> Optional[List[int]]:
        """当前帧裁剪区的测光框（像素 xyxy）；未画裁剪区则 None（全图测光）。"""
        idx = int(self._frame_idx)
        if idx < 0 or idx >= len(self._layouts):
            return None
        lay = self._layouts[idx]
        if lay is None or not layout_valid(lay):
            return None
        h, w = bgr.shape[:2]
        geom = None
        first = self._layouts[0] if self._layouts else None
        if first is not None and layout_valid(first) and self._first_wh is not None:
            w0, h0 = self._first_wh
            geom = geom_from_first(first, w0, h0)
        if geom is not None:
            return in_bounds_crop_xyxy(lay, geom, w, h)
        x0 = int(np.clip(round(min(lay.x0, lay.x1) * w), 0, w))
        x1 = int(np.clip(round(max(lay.x0, lay.x1) * w), 0, w))
        y0 = int(np.clip(round(min(lay.y0, lay.y1) * h), 0, h))
        y1 = int(np.clip(round(max(lay.y0, lay.y1) * h), 0, h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [x0, y0, x1, y1]

    def _apply_ae_for_display(self, path: str, wb_bgr: np.ndarray) -> np.ndarray:
        """在白平衡图上套自动曝光，供当前图预览；强度变化复用已算好的 strength=1 结果。"""
        if not self.cb_ae.isChecked():
            return wb_bgr
        strength = float(self.slider_ae.value()) / 100.0
        if strength <= 0.0:
            return wb_bgr
        box = self._current_ae_meter_box(wb_bgr)
        key = (path, bool(self.cb_wb.isChecked()), tuple(box) if box else None)
        corr = self._ae_corr_cache.get(key)
        if corr is None:
            from auto_exposure import auto_expose_bgr

            corr = auto_expose_bgr(
                wb_bgr, strength=1.0, detect=False, meter_box=box
            )
            self._ae_corr_cache[key] = corr
            if len(self._ae_corr_cache) > 16:
                oldest = next(iter(self._ae_corr_cache))
                if oldest != key:
                    self._ae_corr_cache.pop(oldest, None)
        from auto_exposure import apply_exposure_strength

        return apply_exposure_strength(wb_bgr, corr, strength)

    def _invalidate_ae_preview(self) -> None:
        self._ae_corr_cache.clear()
        if self.cb_ae.isChecked():
            self._schedule_ref_refresh()

    def _show_current_source_frame(self) -> None:
        if self._pv_worker is not None and self._pv_worker.isRunning():
            _burst_gui_log("当前帧刷新中止：预览线程仍占用。")
            return
        self._stop_playback_keep_edit()
        self._sync_layouts_to_list()
        paths = self._collect_paths()
        n = len(paths)
        if n <= 0:
            w0 = getattr(self, "_ref_bird_worker", None)
            if w0 is not None and w0.isRunning():
                w0.requestInterruption()
                w0.wait(2000)
            self._pv_canvas.set_reference_bgr(None)
            self._pv_canvas.set_layout(None)
            self._first_wh = None
            self._update_frame_nav_label()
            return
        self._frame_idx = int(np.clip(self._frame_idx, 0, n - 1))
        path = paths[self._frame_idx]
        if path != getattr(self, "_logged_show_path", None):
            _burst_gui_log(
                f"显示第 {self._frame_idx + 1}/{n} 张：{os.path.basename(path)}"
            )
            self._logged_show_path = path
        t0 = time.monotonic()
        wb = self._load_processed_bgr(path)
        if wb is None:
            self._pv_canvas.set_reference_bgr(None)
            self.lbl_pv_status.setText(f"无法读取：{path}")
            return
        h, w = wb.shape[:2]
        if self._frame_idx == 0:
            self._first_wh = (w, h)
        x = self._apply_ae_for_display(path, wb)
        self._pv_canvas.set_reference_bgr(x)
        lay = self._layouts[self._frame_idx] if self._frame_idx < len(self._layouts) else None
        if lay is not None and (layout_valid(lay) or not lay.auto):
            self._pv_canvas.set_layout(lay)
        else:
            ax, ay = 0.5, 0.5
            if lay is not None:
                ax, ay = float(lay.ax), float(lay.ay)
            elif self._frame_idx == 0:
                ax, ay = self._suggested_ax, self._suggested_ay
            self._pv_canvas.set_layout(None)
            self._pv_canvas.set_anchor_norm(ax, ay)
        self._update_frame_nav_label()
        elapsed = time.monotonic() - t0
        if elapsed >= 0.05:
            _burst_gui_log(
                f"当前帧解码完成 {w}×{h}，用时 {elapsed:.2f}s"
            )
        if self._frame_idx == 0 and not self._anchor_user_touched:
            first_lay = self._layouts[0] if self._layouts else None
            if first_lay is None or not layout_valid(first_lay):
                bird_key = (path, bool(self.cb_wb.isChecked()))
                if self._ref_bird_key != bird_key:
                    self._ref_bird_key = bird_key
                    self._start_ref_bird_worker(wb.copy())

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
            return
        self._suggested_ax = float(nx)
        self._suggested_ay = float(ny)
        if self._anchor_user_touched or self._frame_idx != 0:
            return
        first = self._layouts[0] if self._layouts else None
        if first is not None and layout_valid(first):
            return
        if first is not None:
            first.ax = float(nx)
            first.ay = float(ny)
        self._pv_canvas.set_anchor_norm(nx, ny)
        _burst_gui_log(f"首张参考：画布已应用鸟检标定点（{nx:.4f},{ny:.4f}）。")

    def _geom_ready(self) -> Optional[object]:
        if not self._layouts or not layout_valid(self._layouts[0]):
            return None
        if self._first_wh is None:
            paths = self._collect_paths()
            if not paths:
                return None
            bgr = self._load_processed_bgr(paths[0])
            if bgr is None:
                return None
            self._first_wh = (bgr.shape[1], bgr.shape[0])
        w0, h0 = self._first_wh
        return geom_from_first(self._layouts[0], w0, h0)

    def _on_anchor_from_canvas(self, ax: float, ay: float) -> None:
        self._stop_playback_keep_edit()
        self._sync_layouts_to_list()
        idx = self._frame_idx
        if idx < 0 or idx >= len(self._layouts):
            return
        self._anchor_user_touched = True
        geom = self._geom_ready()
        prev = self._layouts[idx]
        if idx == 0:
            if prev is None:
                prev = FrameLayout(
                    ax=ax,
                    ay=ay,
                    x0=0.0,
                    y0=0.0,
                    x1=0.0,
                    y1=0.0,
                    auto=False,
                )
            prev.ax = float(ax)
            prev.ay = float(ay)
            prev.auto = False
            self._layouts[0] = prev
            _burst_gui_log(f"操作：首图标定点 → ({ax:.4f},{ay:.4f})")
            if layout_valid(prev):
                self._schedule_propagate()
        else:
            if geom is None:
                QMessageBox.information(self, "提示", "请先在首图上设置标定点与裁剪区。")
                return
            if not self._apply_later_anchor(idx, ax, ay):
                QMessageBox.information(self, "提示", "无法读取当前帧，稍后再指定标定点。")
                return
            _burst_gui_log(
                f"操作：第 {idx + 1} 帧标定点 → ({ax:.4f},{ay:.4f})，"
                f"裁剪区已按首图相对位置更新（切换其它页后锁定）"
            )
        self._pv_canvas.set_layout(self._layouts[idx])
        self._refresh_lock_status()
        if idx > 0 and self._later_anchor_dirty:
            self.lbl_pv_status.setText(
                "裁剪区已按首图相对位置与大小更新。切换其它页后锁定；"
                "正在进行的传播会跳过锁定页，不会因此整段重跑。"
            )
        self._invalidate_ae_preview()
        self._schedule_burst_state_save()
        self._schedule_project_save()

    def _on_crop_from_canvas(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        self._stop_playback_keep_edit()
        self._sync_layouts_to_list()
        idx = self._frame_idx
        if idx < 0 or idx >= len(self._layouts):
            return
        prev = self._layouts[idx]
        ax = prev.ax if prev is not None else self._suggested_ax
        ay = prev.ay if prev is not None else self._suggested_ay
        lay = FrameLayout(
            ax=float(ax),
            ay=float(ay),
            x0=float(x0),
            y0=float(y0),
            x1=float(x1),
            y1=float(y1),
            auto=False,
            conf=1.0,
        )
        self._layouts[idx] = lay
        _burst_gui_log(
            f"操作：第 {idx + 1} 帧裁剪区 → ({x0:.3f},{y0:.3f})–({x1:.3f},{y1:.3f})"
        )
        self._pv_canvas.set_layout(lay)
        self._refresh_lock_status()
        if idx == 0 and layout_valid(lay):
            self._schedule_propagate()
        elif idx > 0:
            self._later_anchor_dirty = False
        self._invalidate_ae_preview()
        self._schedule_burst_state_save()
        self._schedule_project_save()

    def _on_recompute_later(self) -> None:
        if not self._layouts or not layout_valid(self._layouts[0]):
            QMessageBox.information(self, "提示", "请先在首图上设置标定点与裁剪区。")
            return
        for i in range(1, len(self._layouts)):
            if self._layouts[i] is not None:
                self._layouts[i].auto = True
        _burst_gui_log("操作：按首图重算后续（已解锁后续帧）。")
        self._later_anchor_dirty = False
        self._start_propagate()

    def _on_reset_initial(self) -> None:
        _burst_gui_log("操作：恢复初始设置（清除布局，回到首张）")
        self._stop_playback_keep_edit()
        self._anchor_user_touched = False
        self._frame_idx = 0
        self._layouts = [None] * self.list_w.count()
        self._layout_paths = self._collect_paths()
        self._layout_by_path = {}
        self._suggested_ax = 0.5
        self._suggested_ay = 0.5
        self._ae_corr_cache.clear()
        self._ref_bird_key = None
        self._logged_show_path = None
        self._later_anchor_dirty = False
        self._show_current_source_frame()
        self._schedule_project_save()

    def _schedule_propagate(self) -> None:
        self._prop_debounce.stop()
        self._prop_debounce.start(350)

    def _start_propagate(self) -> None:
        paths = self._collect_paths()
        if len(paths) < 2:
            return
        self._sync_layouts_to_list()
        if not layout_valid(self._layouts[0] if self._layouts else None):
            return
        w = self._prop_worker
        if w is not None and w.isRunning():
            self._prop_job_id += 1
            w.requestInterruption()
            w.wait(1500)
        self._prop_job_id += 1
        jid = int(self._prop_job_id)
        self.lbl_pv_status.setText("正在用鸟体检测跟踪后续标定点…")
        self._prop_worker = BurstAnchorPropagateWorker(
            paths,
            self._clone_layouts(),
            self._burst_mode(),
            self.cb_wb.isChecked(),
            jid,
            parent=self,
            get_detector=self._get_burst_detector,
        )
        self._prop_worker.progress.connect(
            self._on_preview_progress, Qt.QueuedConnection
        )
        self._prop_worker.done.connect(self._on_propagate_done, Qt.QueuedConnection)
        self._prop_worker.failed.connect(self._on_propagate_fail, Qt.QueuedConnection)
        self._prop_worker.start()
        _burst_gui_log(f"标定点传播已启动（job={jid}，模式={self._burst_mode()}）。")

    @pyqtSlot(object, int)
    def _on_propagate_done(self, layouts, job_id: int) -> None:
        if int(job_id) != int(self._prop_job_id):
            return
        incoming = list(layouts or [])
        self._sync_layouts_to_list()
        n = self.list_w.count()
        merged: List[Optional[FrameLayout]] = []
        for i in range(n):
            cur = self._layouts[i] if i < len(self._layouts) else None
            inc = incoming[i] if i < len(incoming) else None
            if i == 0:
                if cur is not None and layout_valid(cur):
                    merged.append(cur)
                else:
                    merged.append(inc)
                continue
            # 锁定页跳过：保留用户手改，不采用本次传播结果
            if cur is not None and not cur.auto:
                merged.append(cur)
            else:
                merged.append(inc)
        self._layouts = merged
        self._layout_paths = self._collect_paths()
        self._flush_layouts_to_sticky()
        n_auto, n_lock = self._layout_lock_counts()
        self._refresh_lock_status("后续帧已更新：")
        _burst_gui_log(
            f"标定点传播完成：自动={n_auto} 锁定={n_lock}（锁定页已跳过，未整段重跑）"
        )
        self._ae_corr_cache.clear()
        if not self._pv_canvas.is_playing_back():
            self._show_current_source_frame()
        self._schedule_project_save()

    @pyqtSlot(str, int)
    def _on_propagate_fail(self, msg: str, job_id: int) -> None:
        if int(job_id) != int(self._prop_job_id):
            return
        self.lbl_pv_status.setText(f"自动传播失败：{msg}")

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

    def _export_layouts_or_none(self) -> Optional[List[FrameLayout]]:
        self._sync_layouts_to_list()
        n = len(self._layouts)
        if n < 2 or not layout_valid(self._layouts[0]):
            return None
        if any(self._layouts[i] is None for i in range(n)):
            return None
        return list(self._layouts)

    def _opts_from_ui(self) -> BurstWebpBuildOptions:
        wo, wf = self._watermark_opts_and_folder()
        lays = self._export_layouts_or_none()
        return BurstWebpBuildOptions(
            enable_white_balance=self.cb_wb.isChecked(),
            enable_auto_exposure=self.cb_ae.isChecked(),
            auto_exposure_strength=float(self.slider_ae.value()) / 100.0,
            mode=self._burst_mode(),
            fps=float(self.spn_fps.value()),
            frame_layouts=lays,
            max_long_edge=_int_safe_combo_data(self.cmb_max, 1600),
            webp_quality=int(self.spn_q.value()),
            watermark_options=wo,
            watermark_source_folder=wf,
            prefer_folder_name_as_species=True,
            watermark_species_or_theme=(
                self.ed_wm_theme.text().strip() if self.cb_wm.isChecked() else ""
            ),
        )

    def _get_burst_detector(self):
        par = self.parent()
        if par is not None and hasattr(par, "get_burst_webp_bird_detector"):
            try:
                return par.get_burst_webp_bird_detector()
            except Exception:
                return None
        return None

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
        if self._export_layouts_or_none() is None:
            QMessageBox.information(
                self,
                "提示",
                "请先在首图上单击标定点、拖拽裁剪区，并等待后续帧自动传播完成。",
            )
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
        self.lbl_pv_status.setText(
            "已开始…\n解码阶段请查看运行 Birdy 的终端窗口（逐张打印文件名）。"
        )
        _burst_gui_log(
            f"预览参数：WB={opts.enable_white_balance}, "
            f"自动曝光={opts.enable_auto_exposure}×{opts.auto_exposure_strength:.2f}, "
            f"模式={opts.mode}, fps={opts.fps:g}, 水印={'开' if opts.watermark_options else '关'}"
        )
        self._pv_worker = BurstWebpPreviewWorker(paths, opts, parent=self)
        self._pv_worker.progress.connect(
            self._on_preview_progress, Qt.QueuedConnection
        )
        self._pv_worker.done.connect(self._on_preview_done, Qt.QueuedConnection)
        self._pv_worker.failed.connect(self._on_preview_fail, Qt.QueuedConnection)
        self._pv_worker.start()

    def _on_preview_done(self, qimgs, dur_ms: float, _note: str) -> None:
        self.btn_prev.setEnabled(True)
        self.btn_prev.setText("更新预览")
        self._preview_qimages = list(qimgs or [])
        self._preview_dur_ms = max(1.0, float(dur_ms))
        self._preview_idx = 0
        if not self._preview_qimages:
            _burst_gui_log("预览完成回调：无有效帧，将刷新当前参考。")
            self.lbl_pv_status.setText("预览失败：无有效帧")
            self._pv_canvas.stop_playback()
            self._schedule_ref_refresh()
            return
        _burst_gui_log(
            f"预览完成：{len(self._preview_qimages)} 帧，每帧 {self._preview_dur_ms:.0f} ms；"
            "右侧进入动效播放。"
        )
        self._pv_canvas.set_playback_frames(self._preview_qimages)
        self._pv_canvas.set_playback_index(0)
        self._timer.start(max(1, int(round(self._preview_dur_ms))))
        self._refresh_fps_hint()

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
        if self._export_layouts_or_none() is None:
            QMessageBox.warning(
                self,
                "未设置标定点/裁剪区",
                "请在首图上单击设置标定点、拖拽设置裁剪区，"
                "并等待后续帧自动传播（或点「按首图重算后续」）后再导出。",
            )
            return
        if self._prop_worker is not None and self._prop_worker.isRunning():
            QMessageBox.information(self, "提示", "正在自动查找后续标定点，请稍候再生成。")
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
            f"最长边={opts.max_long_edge}, WebP质量={opts.webp_quality}, "
            f"模式={opts.mode}, fps={opts.fps:g}"
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
        if r.get("fps") is not None:
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

    def hideEvent(self, e) -> None:
        if getattr(self, "_save_project_timer", None) is not None:
            self._save_project_timer.stop()
        if getattr(self, "_project_io_ready", False):
            self._save_project_now()
        if getattr(self, "_state_io_ready", False):
            self._save_state_timer.stop()
            self._save_burst_dialog_state()
        super().hideEvent(e)

    def closeEvent(self, e) -> None:
        if getattr(self, "_save_project_timer", None) is not None:
            self._save_project_timer.stop()
        if getattr(self, "_project_io_ready", False):
            self._save_project_now()
        self._save_state_timer.stop()
        if getattr(self, "_state_io_ready", False):
            self._save_burst_dialog_state()
        _burst_gui_log("对话框 closeEvent：停止定时器并等待后台线程…")
        self._timer.stop()
        self._ref_debounce.stop()
        self._prop_debounce.stop()
        if self._pv_worker and self._pv_worker.isRunning():
            _burst_gui_log("等待预览线程结束（最多 3s）…")
            self._pv_worker.wait(3000)
        if self._build_worker and self._build_worker.isRunning():
            _burst_gui_log("等待导出线程结束（最多 3s）…")
            self._build_worker.wait(3000)
        if self._ref_bird_worker and self._ref_bird_worker.isRunning():
            _burst_gui_log("等待首张鸟检后台线程结束（最多 5s）…")
            self._ref_bird_worker.wait(5000)
        if self._prop_worker and self._prop_worker.isRunning():
            self._prop_worker.requestInterruption()
            self._prop_worker.wait(3000)
        _burst_gui_log("动图对话框关闭流程结束。")
        super().closeEvent(e)

    def done(self, result: int) -> None:
        if getattr(self, "_save_project_timer", None) is not None:
            self._save_project_timer.stop()
        if getattr(self, "_project_io_ready", False):
            self._save_project_now()
        self._save_state_timer.stop()
        if getattr(self, "_state_io_ready", False):
            self._save_burst_dialog_state()
        super().done(result)

    def _option_state_widgets(self):
        return (
            self.cb_wb,
            self.cb_ae,
            self.slider_ae,
            self.cb_wm,
            self.ed_wm_theme,
            self.rb_mode_fixed,
            self.rb_mode_track,
            self.spn_fps,
            self.cmb_max,
            self.spn_q,
            self.rb_export_webp,
            self.rb_export_mp4,
            self.ed_out,
            self._bg_mode,
            self._bg_export_fmt,
        )

    def _block_option_signals(self, blocked: bool) -> None:
        for w in self._option_state_widgets():
            w.blockSignals(blocked)

    def _collect_burst_dialog_state(self) -> dict:
        """当前左侧选项与导出格式（不含待合成图片列表）。"""
        try:
            maxed = self.isMaximized()
            rg = self.normalGeometry() if maxed else self.geometry()
            geom = [int(rg.x()), int(rg.y()), int(rg.width()), int(rg.height())]
        except Exception:
            maxed, geom = True, [100, 100, 1240, 820]
        return {
            "version": 6,
            "enable_wb": self.cb_wb.isChecked(),
            "enable_auto_exposure": self.cb_ae.isChecked(),
            "auto_exposure_strength": float(self.slider_ae.value()) / 100.0,
            "enable_wm": self.cb_wm.isChecked(),
            "wm_theme": self.ed_wm_theme.text(),
            "burst_mode": self._burst_mode(),
            "fps": float(self.spn_fps.value()),
            "max_long_edge": _int_safe_combo_data(self.cmb_max, 1600),
            "webp_quality": int(self.spn_q.value()),
            "out_path": self.ed_out.text().strip(),
            "export_format": ("mp4" if self.rb_export_mp4.isChecked() else "webp"),
            "window_maximized": bool(maxed),
            "window_geometry": geom,
            "last_project_path": (
                str(self._project_path)
                if self._project_path is not None
                else str(self._last_project_path_hint or "")
            ),
        }

    def _apply_burst_dialog_state(self, raw: dict) -> None:
        """把已解析的 JSON 应用到控件；单字段失败不影响其余项。"""

        def _b(key: str, default: bool = True) -> bool:
            if key not in raw:
                return default
            return bool(raw[key])

        try:
            self.cb_wb.setChecked(_b("enable_wb", True))
        except Exception:
            pass
        try:
            self.cb_ae.setChecked(_b("enable_auto_exposure", _b("enable_eco", True)))
        except Exception:
            pass
        if "auto_exposure_strength" in raw:
            try:
                st = float(raw["auto_exposure_strength"])
                self.slider_ae.setValue(int(np.clip(round(st * 100.0), 0, 300)))
            except (TypeError, ValueError):
                pass
        self.slider_ae.setEnabled(self.cb_ae.isChecked())
        self.lbl_ae_strength.setText(f"{self.slider_ae.value() / 100:.2f}")
        try:
            self.cb_wm.setChecked(_b("enable_wm", True))
        except Exception:
            pass
        if "wm_theme" in raw and isinstance(raw["wm_theme"], str):
            try:
                self.ed_wm_theme.setText(raw["wm_theme"])
            except Exception:
                pass
        fps_v = raw.get("fps", raw.get("speed"))
        if fps_v is not None:
            try:
                fv = float(fps_v)
                # 旧版 speed 是倍率（约 0.25–8），新版是张/秒；过大则当旧数据忽略
                if 0.2 <= fv <= 30.0 and "fps" in raw:
                    self.spn_fps.setValue(fv)
                elif "fps" in raw:
                    self.spn_fps.setValue(float(np.clip(fv, 0.25, 30.0)))
                else:
                    self.spn_fps.setValue(2.0)
            except (TypeError, ValueError):
                pass
        try:
            mode = str(raw.get("burst_mode", "fixed")).strip().lower()
            if mode == "track":
                self.rb_mode_track.setChecked(True)
            else:
                self.rb_mode_fixed.setChecked(True)
        except Exception:
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
            try:
                self.ed_out.setText(raw["out_path"])
            except Exception:
                pass
        try:
            if str(raw.get("export_format", "webp")).lower() == "mp4":
                self.rb_export_mp4.setChecked(True)
            else:
                self.rb_export_webp.setChecked(True)
        except Exception:
            pass
        self._apply_export_path_placeholder()
        self._sync_out_path_extension_to_format()
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
        lp = raw.get("last_project_path")
        if isinstance(lp, str) and lp.strip():
            self._last_project_path_hint = lp.strip()

    def _load_burst_dialog_state(self) -> None:
        """恢复上次关闭时的参数（不含待合成图片列表）。"""
        primary = _burst_webp_dialog_state_path()
        legacy = _burst_webp_dialog_state_legacy_path()
        path = primary if primary.is_file() else legacy
        self._state_window_maximized = True
        self._state_window_geometry = None
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as ex:
            _burst_gui_log(f"加载上次动图窗口参数失败（将用默认）：{ex}")
            return
        if not isinstance(raw, dict):
            return
        try:
            ver = int(raw.get("version", 1))
        except (TypeError, ValueError):
            ver = 1
        if ver < 1:
            return

        self._block_option_signals(True)
        try:
            self._apply_burst_dialog_state(raw)
        except Exception as ex:
            _burst_gui_log(f"应用上次动图窗口参数失败（将用默认）：{ex}")
        finally:
            self._block_option_signals(False)

        _burst_gui_log(f"已加载上次的动图窗口参数：{path}")
        if path == legacy and not primary.is_file():
            try:
                shutil.copy2(legacy, primary)
                _burst_gui_log(f"已将动图设置复制到首选路径：{primary}")
            except OSError as ex:
                _burst_gui_log(f"复制到首选路径失败（不影响本次使用）：{ex}")
        self._log_wm_toggle()
        self._refresh_fps_hint()

    def _save_burst_dialog_state(self) -> None:
        """写入参数（不含待合成图片列表）；首选本机 AppData，失败则写 src 旁旧路径。"""
        if not getattr(self, "_state_io_ready", False):
            return
        primary = _burst_webp_dialog_state_path()
        legacy = _burst_webp_dialog_state_legacy_path()
        data = self._collect_burst_dialog_state()
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        try:
            _atomic_write_text(primary, text)
            _burst_gui_log(f"已保存动图窗口参数（不含图片列表）：{primary}")
            return
        except OSError as ex1:
            if primary != legacy:
                try:
                    _atomic_write_text(legacy, text)
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
