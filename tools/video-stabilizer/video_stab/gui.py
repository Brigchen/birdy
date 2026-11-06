# -*- coding: utf-8 -*-
"""Video Stabilizer 独立 GUI。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .media import list_videos, read_frame_at, video_duration
from .options import StabilizeOptions
from .paths import default_output_dir, find_window_icon, setup_import_paths
from .pipeline import suggest_output_dir, suggest_output_path
from .preview import BoxCanvas
from .shortcut import create_desktop_shortcut
from .worker import StabilizeWorker

setup_import_paths()

APP_TITLE = "Video Stabilizer"
APP_SUB = "观鸟录像 · 三脚架固定 / 鸟类追踪稳定"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(860, 780)
        icon = find_window_icon()
        if icon:
            self.setWindowIcon(QIcon(str(icon)))

        self._duration = 0.0
        self._preview_path = ""
        self._folder_files: list[str] = []

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.addWidget(QLabel(f"<b>{APP_TITLE}</b> — {APP_SUB}"))

        form = QFormLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("视频文件，或含 MP4 的目录")
        btn_file = QPushButton("文件…")
        btn_file.clicked.connect(self._pick_file)
        btn_dir = QPushButton("目录…")
        btn_dir.clicked.connect(self._pick_dir)
        row_in = QHBoxLayout()
        row_in.addWidget(self.input_edit, 1)
        row_in.addWidget(btn_file)
        row_in.addWidget(btn_dir)
        form.addRow("输入", row_in)

        self.recursive = QCheckBox("包含子目录")
        self.recursive.stateChanged.connect(self._reload_folder)
        form.addRow("", self.recursive)

        self.output_edit = QLineEdit(str(default_output_dir()))
        self.output_edit.setToolTip("稳定后的视频固定保存到这个文件夹")
        btn_out = QPushButton("另存…")
        btn_out.clicked.connect(self._pick_output)
        row_out = QHBoxLayout()
        row_out.addWidget(self.output_edit, 1)
        row_out.addWidget(btn_out)
        form.addRow("输出", row_out)
        layout.addLayout(form)

        mode_box = QGroupBox("稳定模式")
        mode_layout = QFormLayout(mode_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("三脚架固定视角（背景钉死）", "tripod")
        self.mode_combo.addItem("追踪稳定器（鸟在画面中心）", "track")
        self.mode_combo.currentIndexChanged.connect(self._sync_mode_ui)
        mode_layout.addRow("模式", self.mode_combo)

        self.tripod_engine = QComboBox()
        self.tripod_engine.addItem("光流锁定首帧（推荐）", "optical")
        self.tripod_engine.addItem("FFmpeg libvidstab opttripod", "ffmpeg")
        self.tripod_engine.setToolTip(
            "光流：每帧对齐第一帧，再磨掉高频抖动，并按行把岸线收到同一视角，"
            "按全程晃动裁掉多余画面（可以裁掉一半以上），"
            "避免校正被裁边卡住后乱跳、闪黑边。\n"
            "FFmpeg：libvidstab 的 opttripod。"
        )
        mode_layout.addRow("三脚架引擎", self.tripod_engine)

        self.track_engine = QComboBox()
        self.track_engine.addItem("YOLO 鸟检 + ByteTrack", "yolo")
        self.track_engine.addItem("框跟踪（CSRT / MIL，无模型）", "csrt")
        mode_layout.addRow("追踪引擎", self.track_engine)

        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(3, 15)
        self.smooth_spin.setValue(7)
        mode_layout.addRow("追踪平滑（帧）", self.smooth_spin)

        self.keep_audio = QCheckBox("保留原音轨（需 ffmpeg）")
        self.keep_audio.setChecked(True)
        mode_layout.addRow("", self.keep_audio)
        layout.addWidget(mode_box)

        self.preview_box = QGroupBox("框选")
        preview_layout = QVBoxLayout(self.preview_box)
        preview_layout.addWidget(
            QLabel("青色框选鸟，橙色框选输出画面。目录批处理时，相对位置套用到每个视频。")
        )
        draw_row = QHBoxLayout()
        self.draw_bird = QRadioButton("框选鸟体")
        self.draw_view = QRadioButton("框选画面")
        self.draw_bird.setChecked(True)
        self._draw_group = QButtonGroup(self)
        self._draw_group.addButton(self.draw_bird)
        self._draw_group.addButton(self.draw_view)
        self.draw_bird.toggled.connect(self._sync_draw_target)
        draw_row.addWidget(self.draw_bird)
        draw_row.addWidget(self.draw_view)
        draw_row.addStretch(1)
        preview_layout.addLayout(draw_row)
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("预览"))
        self.preview_combo = QComboBox()
        self.preview_combo.currentIndexChanged.connect(self._preview_combo_changed)
        pick_row.addWidget(self.preview_combo, 1)
        preview_layout.addLayout(pick_row)

        self.canvas = BoxCanvas()
        self.canvas.box_changed.connect(self._on_box_changed)
        preview_layout.addWidget(self.canvas, 1)

        seek_row = QHBoxLayout()
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.sliderReleased.connect(self._seek_released)
        self.seek_label = QLabel("0.0s")
        seek_row.addWidget(self.seek, 1)
        seek_row.addWidget(self.seek_label)
        preview_layout.addLayout(seek_row)

        box_row = QHBoxLayout()
        self.box_label = QLabel("鸟未框    画面未框")
        clear_btn = QPushButton("清除当前框")
        clear_btn.clicked.connect(lambda: self.canvas.clear_box())
        box_row.addWidget(self.box_label, 1)
        box_row.addWidget(clear_btn)
        preview_layout.addLayout(box_row)
        layout.addWidget(self.preview_box, 1)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.shortcut_btn = QPushButton("创建桌面快捷方式")
        self.shortcut_btn.clicked.connect(self._create_shortcut)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.shortcut_btn)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        layout.addWidget(self.log)

        self._worker: StabilizeWorker | None = None
        self._sync_mode_ui()

    def _is_folder_input(self) -> bool:
        text = self.input_edit.text().strip()
        return bool(text) and Path(text).is_dir()

    def _sync_mode_ui(self) -> None:
        track = self.mode_combo.currentData() == "track"
        self.track_engine.setEnabled(track)
        self.smooth_spin.setEnabled(track)
        self.tripod_engine.setEnabled(not track)
        if track:
            self.draw_bird.setChecked(True)
        else:
            self.draw_view.setChecked(True)

    def _set_preview_files(self, files: list[str]) -> None:
        self._folder_files = list(files)
        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        for path in files:
            self.preview_combo.addItem(Path(path).name, path)
        self.preview_combo.blockSignals(False)
        if files:
            self._load_preview(files[0], 0.0)
        else:
            self._preview_path = ""
            self.canvas.clear_frame()

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video (*.mp4 *.mov *.mkv *.m4v);;All (*.*)"
        )
        if not path:
            return
        self.input_edit.setText(path)
        self.output_edit.setText(suggest_output_path(path, self.mode_combo.currentData()))
        self._set_preview_files([path])

    def _pick_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择视频目录")
        if not path:
            return
        self.input_edit.setText(path)
        self.output_edit.setText(suggest_output_dir(path))
        self._reload_folder()

    def _reload_folder(self) -> None:
        path = self.input_edit.text().strip()
        if not path or not Path(path).is_dir():
            return
        files = list_videos(path, recursive=self.recursive.isChecked())
        self._set_preview_files(files)
        self._append_log(f"目录内视频 {len(files)} 个")

    def _pick_output(self) -> None:
        if self._is_folder_input():
            path = QFileDialog.getExistingDirectory(self, "输出目录", self.output_edit.text())
            if path:
                self.output_edit.setText(path)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "输出路径", self.output_edit.text(), "MP4 (*.mp4)"
        )
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_edit.setText(path)

    def _preview_combo_changed(self) -> None:
        path = self.preview_combo.currentData()
        if path:
            frac = self.seek.value() / 1000.0
            self._load_preview(str(path), self._duration * frac if self._preview_path == path else 0.0)

    def _load_preview(self, path: str, time_sec: float) -> None:
        self._preview_path = path
        self._duration = video_duration(path)
        frame = read_frame_at(path, time_sec)
        if frame is None:
            self.canvas.clear_frame()
            self.seek_label.setText("无法读取")
            return
        h, w = frame.shape[:2]
        max_w = 1280
        if w > max_w:
            import cv2

            frame = cv2.resize(frame, (max_w, max(1, int(h * max_w / w))))
        self.canvas.set_bgr(frame)
        self.seek_label.setText(f"{time_sec:.1f}s / {self._duration:.1f}s")

    def _seek_released(self) -> None:
        if not self._preview_path or self._duration <= 0:
            return
        t = self._duration * (self.seek.value() / 1000.0)
        self._load_preview(self._preview_path, t)

    def _sync_draw_target(self) -> None:
        self.canvas.set_active("bird" if self.draw_bird.isChecked() else "view")
        self._on_box_changed()

    def _fmt_box(self, name: str, title: str) -> str:
        box = self.canvas.norm_box(name)
        if not box:
            return f"{title}未框"
        x, y, w, h = box
        return f"{title} {w:.0%}×{h:.0%}"

    def _on_box_changed(self) -> None:
        self.box_label.setText(
            f"{self._fmt_box('bird', '鸟')}    {self._fmt_box('view', '画面')}"
        )

    def _append_log(self, msg: str) -> None:
        self.log.append(msg)

    def _options(self) -> StabilizeOptions:
        track = self.mode_combo.currentData() == "track"
        return StabilizeOptions(
            mode=self.mode_combo.currentData(),
            tripod_engine=self.tripod_engine.currentData(),
            track_engine=self.track_engine.currentData(),
            smoothing_window=self.smooth_spin.value(),
            keep_audio=self.keep_audio.isChecked(),
            init_bbox_norm=self.canvas.norm_box("bird") if track else None,
            view_norm=self.canvas.norm_box("view"),
        )

    def _start(self) -> None:
        src = self.input_edit.text().strip()
        out = self.output_edit.text().strip()
        path = Path(src) if src else None
        if path is None or not path.exists():
            QMessageBox.warning(self, APP_TITLE, "请选择有效的视频文件或目录")
            return
        opts = self._options()
        self.log.clear()
        self.progress.setValue(0)
        if path.is_dir():
            files = list_videos(str(path), recursive=self.recursive.isChecked())
            if not files:
                QMessageBox.warning(self, APP_TITLE, "目录里没有 mp4/mov/mkv/m4v")
                return
            if not out:
                out = suggest_output_dir(str(path))
                self.output_edit.setText(out)
            self._worker = StabilizeWorker(opts, folder_jobs=files, output_dir=out)
        else:
            if not out:
                out = suggest_output_path(str(path), opts.mode)
                self.output_edit.setText(out)
            self._worker = StabilizeWorker(opts, file_job=(str(path), out))

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._worker.log_line.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self._append_log("正在请求停止…")

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress.setValue(min(100, int(100 * done / total)))

    def _on_ok(self, path: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        QMessageBox.information(self, APP_TITLE, path)

    def _on_fail(self, err: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._append_log(err)
        QMessageBox.warning(self, APP_TITLE, err)

    def _create_shortcut(self) -> None:
        try:
            path = create_desktop_shortcut()
        except Exception as e:
            QMessageBox.warning(self, APP_TITLE, f"创建失败:\n{e}")
            return
        QMessageBox.information(self, APP_TITLE, f"已放到桌面:\n{path}")


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
