# -*- coding: utf-8 -*-
"""BIRDY-观鸟地图 主界面。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QListWidget,
)

from .config_io import load_config, save_config
from .paths import (
    default_output_dir,
    find_window_icon,
    runtime_dir,
    setup_import_paths,
)
from .worker import TrackMapWorker

setup_import_paths()

from gpx_track import TrackMapPreviewPanel, resolve_gpx_path_list  # noqa: E402
from gpx_track.track_map import iter_skipped_photo_log_lines  # noqa: E402
from gpx_track.gpx_match import DEFAULT_EXIF_TZ, DEFAULT_GPX_TZ  # noqa: E402
from gpx_track.timezone_util import (  # noqa: E402
    normalize_tz_name,
    read_combo_timezone,
    set_combo_timezone,
    timezone_combo_entries,
)


APP_TITLE = "BIRDY-观鸟地图"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self._config = load_config()
        self._worker: Optional[TrackMapWorker] = None
        self._progress: Optional[QProgressDialog] = None
        self._pending_preview = False
        icon_path = find_window_icon()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        self._build_ui()
        self._load_config_to_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        split = QHBoxLayout(root)
        split.setContentsMargins(8, 8, 8, 8)
        split.setSpacing(10)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(280)
        left_inner = QWidget()
        layout = QVBoxLayout(left_inner)
        layout.setContentsMargins(4, 4, 8, 4)

        header = QHBoxLayout()
        logo_path = find_window_icon()
        if logo_path is not None:
            pm = QPixmap(str(logo_path))
            if not pm.isNull():
                logo = QLabel()
                logo.setPixmap(
                    pm.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                header.addWidget(logo)
        title = QLabel(
            f"<b>{APP_TITLE}</b><br>"
            "<span style='color:#666;font-size:11px'>观鸟行迹 PNG · GPX + 鸟图</span>"
        )
        header.addWidget(title, 1)
        layout.addLayout(header)

        inp = QGroupBox("输入")
        form = QFormLayout(inp)
        self.photo_folder_input = QLineEdit()
        self.photo_folder_input.setPlaceholderText("含鸟图的文件夹（分类归档或任意目录）")
        photo_row = QHBoxLayout()
        photo_row.addWidget(self.photo_folder_input, 1)
        pb = QPushButton("浏览…")
        pb.clicked.connect(lambda: self._pick_dir(self.photo_folder_input))
        photo_row.addWidget(pb)
        form.addRow("鸟图目录:", photo_row)

        self.gpx_list = QListWidget()
        self.gpx_list.setMaximumHeight(72)
        self.gpx_list.setToolTip("可添加多个 GPX（分段记录），按时间合并匹配")
        gpx_btn_row = QHBoxLayout()
        gb = QPushButton("添加 GPX...")
        gb.clicked.connect(self._add_gpx_files)
        gr = QPushButton("移除")
        gr.clicked.connect(self._remove_selected_gpx)
        gpx_btn_row.addWidget(gb)
        gpx_btn_row.addWidget(gr)
        gpx_btn_row.addStretch(1)
        form.addRow("GPX 文件:", self.gpx_list)
        form.addRow("", gpx_btn_row)

        self.use_gpx_checkbox = QCheckBox("使用 GPX 轨迹匹配拍摄时间")
        self.use_exif_checkbox = QCheckBox("补充使用照片 EXIF 中的 GPS")
        form.addRow("", self.use_gpx_checkbox)
        form.addRow("", self.use_exif_checkbox)
        layout.addWidget(inp)

        title_grp = QGroupBox("地图标题（可选）")
        tform = QFormLayout(title_grp)
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("如：厦门大学翔安校区（留空则标题为「观鸟地图」）")
        tform.addRow("地点名称:", self.location_input)
        layout.addWidget(title_grp)

        opt = QGroupBox("地图选项")
        oform = QFormLayout(opt)
        self.radius_input = QDoubleSpinBox()
        self.radius_input.setRange(0.1, 100.0)
        self.radius_input.setDecimals(1)
        self.radius_input.setSuffix(" km")
        oform.addRow("物种去重半径:", self.radius_input)
        self.elevation_checkbox = QCheckBox("叠加海拔剖面（内嵌于地图底部）")
        oform.addRow("", self.elevation_checkbox)
        self.exif_tz_combo = self._make_tz_combo()
        self.gpx_tz_combo = self._make_tz_combo()
        oform.addRow("EXIF 时区:", self.exif_tz_combo)
        oform.addRow("GPX 时区:", self.gpx_tz_combo)
        self.logo_input = QLineEdit()
        self.logo_input.setPlaceholderText("可选：签名 Logo（与水印相同）")
        logo_row = QHBoxLayout()
        logo_row.addWidget(self.logo_input, 1)
        lb = QPushButton("浏览…")
        lb.clicked.connect(self._pick_logo)
        logo_row.addWidget(lb)
        oform.addRow("签名 Logo:", logo_row)
        self.logo_ratio_input = QDoubleSpinBox()
        self.logo_ratio_input.setRange(0.05, 0.80)
        self.logo_ratio_input.setDecimals(2)
        self.logo_ratio_input.setSingleStep(0.05)
        oform.addRow("Logo 宽度比例:", self.logo_ratio_input)
        layout.addWidget(opt)

        out = QGroupBox("输出")
        oform2 = QFormLayout(out)
        self.output_folder_input = QLineEdit()
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_folder_input, 1)
        ob = QPushButton("浏览…")
        ob.clicked.connect(lambda: self._pick_dir(self.output_folder_input))
        out_row.addWidget(ob)
        oform2.addRow("保存目录:", out_row)
        layout.addWidget(out)

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("预览")
        self.preview_btn.clicked.connect(lambda: self._run(preview=True))
        self.save_btn = QPushButton("生成并保存 PNG")
        self.save_btn.clicked.connect(lambda: self._run(preview=False))
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)
        layout.addStretch(1)

        left_scroll.setWidget(left_inner)
        split.addWidget(left_scroll, 1)

        preview_wrap = QGroupBox("地图预览")
        preview_layout = QVBoxLayout(preview_wrap)
        self.preview_panel = TrackMapPreviewPanel()
        preview_layout.addWidget(self.preview_panel)
        split.addWidget(preview_wrap, 3)

    def _make_tz_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setMinimumContentsLength(28)
        for label, tzid in timezone_combo_entries():
            combo.addItem(label, tzid)
        names = [combo.itemText(i) for i in range(combo.count())]
        completer = QCompleter(names, combo)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        combo.setCompleter(completer)
        return combo

    def _load_config_to_ui(self) -> None:
        c = self._config
        self.photo_folder_input.setText(c.get("photo_folder", ""))
        self.gpx_list.clear()
        gpx_paths = resolve_gpx_path_list(
            c.get("gpx_file_path"),
            c.get("gpx_file_paths"),
        )
        for p in gpx_paths:
            self.gpx_list.addItem(p)
        self.output_folder_input.setText(
            c.get("output_folder") or str(default_output_dir())
        )
        self.location_input.setText(c.get("location_name", ""))
        self.use_gpx_checkbox.setChecked(bool(c.get("use_gpx_track", True)))
        self.use_exif_checkbox.setChecked(bool(c.get("use_exif_gps", True)))
        self.radius_input.setValue(float(c.get("radius_km", 1.0)))
        self.elevation_checkbox.setChecked(bool(c.get("include_elevation", True)))
        self.logo_input.setText(c.get("wm_logo_path", ""))
        self.logo_ratio_input.setValue(float(c.get("wm_logo_width_ratio", 0.30)))
        set_combo_timezone(
            self.exif_tz_combo,
            normalize_tz_name(c.get("gpx_match_exif_tz", DEFAULT_EXIF_TZ)),
        )
        set_combo_timezone(
            self.gpx_tz_combo,
            normalize_tz_name(c.get("gpx_match_gpx_tz", DEFAULT_GPX_TZ)),
        )

    def _sync_config_from_ui(self) -> None:
        c = self._config
        c["photo_folder"] = self.photo_folder_input.text().strip()
        gpx_paths = self._gpx_paths_from_ui()
        c["gpx_file_paths"] = gpx_paths
        c["gpx_file_path"] = gpx_paths[0] if gpx_paths else ""
        c["output_folder"] = self.output_folder_input.text().strip()
        c["location_name"] = self.location_input.text().strip()
        c["use_gpx_track"] = self.use_gpx_checkbox.isChecked()
        c["use_exif_gps"] = self.use_exif_checkbox.isChecked()
        c["radius_km"] = float(self.radius_input.value())
        c["include_elevation"] = self.elevation_checkbox.isChecked()
        c["wm_logo_path"] = self.logo_input.text().strip()
        c["wm_logo_width_ratio"] = float(self.logo_ratio_input.value())
        c["gpx_match_exif_tz"] = read_combo_timezone(self.exif_tz_combo)
        c["gpx_match_gpx_tz"] = read_combo_timezone(self.gpx_tz_combo)

    def _append_log(self, msg: str) -> None:
        self.log.append(msg)

    def _pick_dir(self, field: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择文件夹", field.text())
        if d:
            field.setText(d)

    def _gpx_paths_from_ui(self) -> List[str]:
        paths: List[str] = []
        for i in range(self.gpx_list.count()):
            p = self.gpx_list.item(i).text().strip()
            if p and p not in paths:
                paths.append(p)
        return resolve_gpx_path_list(gpx_paths=paths)

    def _add_gpx_files(self) -> None:
        start = self.gpx_list.item(0).text() if self.gpx_list.count() else ""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 GPX 文件（可多选）",
            start,
            "GPX (*.gpx);;All (*.*)",
        )
        if not paths:
            return
        existing = {
            self.gpx_list.item(i).text()
            for i in range(self.gpx_list.count())
        }
        for p in paths:
            if p not in existing:
                self.gpx_list.addItem(p)

    def _remove_selected_gpx(self) -> None:
        for item in self.gpx_list.selectedItems():
            self.gpx_list.takeItem(self.gpx_list.row(item))

    def _pick_logo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Logo 图片",
            self.logo_input.text(),
            "Images (*.png *.jpg *.jpeg *.webp);;All (*.*)",
        )
        if path:
            self.logo_input.setText(path)

    def _build_kwargs(self, *, preview: bool) -> Dict[str, Any]:
        c = self._config
        gpx_paths = resolve_gpx_path_list(
            c.get("gpx_file_path"),
            c.get("gpx_file_paths"),
        )
        use_gpx = bool(c.get("use_gpx_track"))
        out_dir = (c.get("output_folder") or "").strip() or str(default_output_dir())
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return dict(
            reports_dir=out_dir,
            gpx_paths=gpx_paths if use_gpx else None,
            photo_folder=c.get("photo_folder", "").strip(),
            use_gpx_track=use_gpx,
            use_exif_gps=bool(c.get("use_exif_gps")),
            radius_km=float(c.get("radius_km", 1.0)),
            include_elevation=bool(c.get("include_elevation")),
            basemap_style="none",
            preview_only=preview,
            preview_max_photos=40,
            location_name=c.get("location_name", "").strip(),
            province="",
            city="",
            logo_path=str(c.get("wm_logo_path") or ""),
            logo_width_ratio=float(c.get("wm_logo_width_ratio", 0.30)),
            exif_tz=normalize_tz_name(c.get("gpx_match_exif_tz", DEFAULT_EXIF_TZ)),
            gpx_tz=normalize_tz_name(c.get("gpx_match_gpx_tz", DEFAULT_GPX_TZ)),
        )

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _run(self, preview: bool = False) -> None:
        if self._busy():
            QMessageBox.information(self, "请稍候", "正在生成中…")
            return
        self._sync_config_from_ui()
        save_config(self._config)
        photo = self._config.get("photo_folder", "").strip()
        if not photo or not os.path.isdir(photo):
            QMessageBox.warning(self, "提示", "请选择有效的鸟图目录。")
            return
        use_gpx = self.use_gpx_checkbox.isChecked()
        gpx_paths = self._gpx_paths_from_ui()
        if use_gpx and not gpx_paths:
            QMessageBox.warning(self, "提示", "已勾选 GPX，请添加至少一个有效的 GPX 文件。")
            return

        self._pending_preview = preview
        label = "预览" if preview else "保存"
        msg = f"观鸟地图{label}：正在生成（约 30–90 秒）…"
        self._append_log(msg)
        self.preview_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        dlg = QProgressDialog(msg, None, 0, 0, self)
        dlg.setWindowTitle("观鸟地图")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.show()
        self._progress = dlg

        kwargs = self._build_kwargs(preview=preview)
        self._worker = TrackMapWorker(kwargs, self)
        self._worker.log_line.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _finish_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None
        self.preview_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def _show_result_in_preview(self, png_path: str) -> None:
        if png_path and os.path.isfile(png_path):
            self.preview_panel.set_image(png_path)

    def _on_ok(self, written: Dict[str, str]) -> None:
        self._finish_progress()
        main_png = written.get("track_png", "")
        lines = [f"已保存：{main_png}"]
        title = written.get("map_title")
        if title:
            lines.append(f"标题：{title}")
        lines.extend(iter_skipped_photo_log_lines(written))
        self._append_log("\n".join(lines))
        self._show_result_in_preview(main_png)
        if not self._pending_preview:
            QMessageBox.information(self, "完成", "\n".join(lines))

    def _on_fail(self, msg: str) -> None:
        self._finish_progress()
        self._append_log(f"失败：{msg}")
        QMessageBox.critical(self, "生成失败", msg[:4000])

    def closeEvent(self, event) -> None:
        self._sync_config_from_ui()
        save_config(self._config)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    setup_import_paths()
    if not (runtime_dir() / "gpx_track").is_dir():
        QMessageBox.critical(
            None,
            APP_TITLE,
            "缺少运行库 birdy_runtime/。\n\n"
            "请在工具目录执行：\n  python sync_runtime.py\n\n"
            "（维护者打包分享前亦需执行一次）",
        )
        return 1
    win = MainWindow()
    win.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
