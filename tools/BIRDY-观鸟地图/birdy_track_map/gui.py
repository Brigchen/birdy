# -*- coding: utf-8 -*-
"""BIRDY-观鸟地图 主界面。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
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
APP_NAME_CN = "观鸟地图"
APP_NAME_EN = "BIRDY Track Map · 独立工具"

APP_GLOBAL_STYLE = """
    QMainWindow {
        background-color: #F5F5F5;
    }
    QWidget {
        font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Arial', sans-serif;
        font-size: 10pt;
    }
    QLabel {
        color: #333333;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 5px 10px;
        color: #333333;
        font-size: 10pt;
        min-height: 1.1em;
    }
    QComboBox::drop-down {
        border: none;
        width: 22px;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border: 1px solid #2E8B57;
    }
    QListWidget {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 4px;
        font-size: 10pt;
    }
    QListWidget::item {
        padding: 3px 6px;
        border-radius: 4px;
    }
    QListWidget::item:selected {
        background-color: #E8F5E9;
        color: #1B5E20;
    }
    QPushButton {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 6px 14px;
        color: #333333;
        font-weight: 500;
        font-size: 10pt;
        min-height: 1.2em;
    }
    QPushButton:hover:enabled {
        background-color: #F0F0F0;
        border: 1px solid #1E90FF;
    }
    QPushButton:pressed:enabled {
        background-color: #E0E0E0;
    }
    QPushButton:disabled {
        background-color: #F5F5F5;
        color: #999999;
        border: 1px solid #E0E0E0;
    }
    QCheckBox {
        spacing: 6px;
        font-size: 10pt;
    }
    QCheckBox::indicator {
        width: 20px;
        height: 20px;
        border: 2px solid #E0E0E0;
        border-radius: 5px;
        background-color: #FFFFFF;
    }
    QCheckBox::indicator:checked {
        background-color: #2E8B57;
        border: 2px solid #2E8B57;
    }
    QCheckBox::indicator:hover {
        border: 2px solid #1E90FF;
    }
    QTextEdit {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 6px;
        font-family: 'Consolas', 'Courier New', 'Microsoft YaHei UI', monospace;
        font-size: 9pt;
    }
    QScrollArea {
        border: none;
        background: transparent;
    }
    QScrollArea > QWidget > QWidget {
        background: transparent;
    }
"""

BTN_PRIMARY_STYLE = """
    QPushButton {
        background-color: #2E8B57;
        color: white;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 10pt;
        border: none;
    }
    QPushButton:hover:enabled {
        background-color: #277A4B;
    }
    QPushButton:pressed:enabled {
        background-color: #226A3F;
    }
    QPushButton:disabled {
        background-color: #BDC3C7;
        color: #7F8C8D;
    }
"""

BTN_SECONDARY_STYLE = """
    QPushButton {
        background-color: #FFFFFF;
        color: #2E6B4A;
        font-weight: bold;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 10pt;
        border: 1px solid #2E8B57;
    }
    QPushButton:hover:enabled {
        background-color: #E8F5E9;
    }
    QPushButton:pressed:enabled {
        background-color: #D5EDDA;
    }
    QPushButton:disabled {
        background-color: #F5F5F5;
        color: #999999;
        border: 1px solid #E0E0E0;
    }
"""


def _hint_label(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(True)
    lb.setStyleSheet("color: #555555; font-size: 9pt;")
    return lb


def _create_card(title: str) -> Tuple[QWidget, QWidget]:
    """与主程序 birdy_gui 一致的白色圆角卡片。"""
    card = QWidget()
    card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    card.setStyleSheet(
        "QWidget#birdyCard { background-color: #FFFFFF; border-radius: 8px; }"
    )
    card.setObjectName("birdyCard")
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(0, 0, 0, 0)
    card_layout.setSpacing(0)

    title_label = QLabel(title)
    title_label.setStyleSheet(
        "QLabel { background-color: #FFFFFF; color: #333333; font-weight: bold; "
        "font-size: 11pt; padding: 6px 12px; border-top-left-radius: 8px; "
        "border-top-right-radius: 8px; border-bottom: 1px solid #F0F0F0; }"
    )
    card_layout.addWidget(title_label)

    content = QWidget()
    content.setStyleSheet("background-color: #FFFFFF;")
    card_layout.addWidget(content)
    return card, content


def _form_layout(content: QWidget) -> QFormLayout:
    form = QFormLayout(content)
    form.setSpacing(8)
    form.setContentsMargins(12, 10, 12, 12)
    return form


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setStyleSheet(APP_GLOBAL_STYLE)
        self._config = load_config()
        self._worker: Optional[TrackMapWorker] = None
        self._progress: Optional[QProgressDialog] = None
        self._pending_preview = False
        icon_path = find_window_icon()
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        self._build_ui()
        self._load_config_to_ui()

    def _create_top_banner(self) -> QWidget:
        banner = QWidget()
        banner.setObjectName("birdyTopBanner")
        banner.setStyleSheet(
            "#birdyTopBanner { background-color: #FFFFFF; "
            "border-bottom: 1px solid #E0E0E0; }"
        )
        row = QHBoxLayout(banner)
        row.setContentsMargins(14, 8, 18, 8)
        row.setSpacing(10)

        logo_h = 56
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = find_window_icon()
        if logo_path is not None:
            pm = QPixmap(str(logo_path))
            if not pm.isNull():
                logo_label.setPixmap(
                    pm.scaled(
                        logo_h,
                        logo_h,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        logo_label.setFixedHeight(logo_h)
        logo_label.setMinimumWidth(logo_h)
        row.addWidget(logo_label, 0, Qt.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        cn = QLabel(APP_NAME_CN)
        cn.setStyleSheet(
            "color: #2E3A3F; font-size: 14pt; font-weight: bold;"
        )
        text_col.addWidget(cn)
        en = QLabel(APP_NAME_EN)
        en.setStyleSheet("color: #5A6B73; font-size: 10pt;")
        text_col.addWidget(en)
        sub = QLabel("GPX + 鸟图 → 观鸟行迹 PNG · 经纬度网格底图")
        sub.setStyleSheet("color: #7A8A92; font-size: 9pt;")
        text_col.addWidget(sub)
        row.addLayout(text_col)
        row.addStretch(1)
        return banner

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._create_top_banner())

        body = QWidget()
        body.setStyleSheet("background-color: #F5F5F5;")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(12, 10, 12, 12)
        body_layout.setSpacing(12)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)

        left_panel = QWidget()
        layout = QVBoxLayout(left_panel)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 8, 0)

        inp_card, inp_body = _create_card("📁 输入数据")
        form = _form_layout(inp_body)
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
        gb = QPushButton("添加 GPX…")
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
        layout.addWidget(inp_card)

        title_card, title_body = _create_card("🏷 地图标题")
        tform = _form_layout(title_body)
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText(
            "如：厦门大学翔安校区（留空则标题为「观鸟记录」）"
        )
        tform.addRow("地点名称:", self.location_input)
        tform.addRow("", _hint_label("留空地点时仍显示日期与「观鸟记录」及签名 Logo。"))
        layout.addWidget(title_card)

        opt_card, opt_body = _create_card("🗺 地图选项")
        oform = _form_layout(opt_body)
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
        layout.addWidget(opt_card)

        out_card, out_body = _create_card("💾 输出")
        oform2 = _form_layout(out_body)
        self.output_folder_input = QLineEdit()
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_folder_input, 1)
        ob = QPushButton("浏览…")
        ob.clicked.connect(lambda: self._pick_dir(self.output_folder_input))
        out_row.addWidget(ob)
        oform2.addRow("保存目录:", out_row)
        layout.addWidget(out_card)

        btn_card, btn_body = _create_card("▶ 操作")
        btn_layout = QHBoxLayout(btn_body)
        btn_layout.setSpacing(12)
        btn_layout.setContentsMargins(12, 10, 12, 12)
        self.preview_btn = QPushButton("预览")
        self.preview_btn.setStyleSheet(BTN_SECONDARY_STYLE)
        self.preview_btn.clicked.connect(lambda: self._run(preview=True))
        self.save_btn = QPushButton("生成并保存 PNG")
        self.save_btn.setStyleSheet(BTN_PRIMARY_STYLE)
        self.save_btn.clicked.connect(lambda: self._run(preview=False))
        btn_layout.addWidget(self.preview_btn, 1)
        btn_layout.addWidget(self.save_btn, 2)
        layout.addWidget(btn_card)

        log_card, log_body = _create_card("📋 运行日志")
        log_layout = QVBoxLayout(log_body)
        log_layout.setContentsMargins(12, 8, 12, 12)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(100)
        self.log.setMaximumHeight(140)
        log_layout.addWidget(self.log)
        layout.addWidget(log_card)

        hint = QLabel(
            "💡 本工具使用经纬度网格底图，无需高德 API Key；"
            "请自备 GPX 与鸟图目录。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666666; font-size: 10pt; margin-top: 2px;")
        layout.addWidget(hint)

        watermark = QLabel("Birdy · 鸟图智慧仓储")
        watermark.setStyleSheet("color: #E0E0E0; font-size: 9pt;")
        watermark.setAlignment(Qt.AlignRight)
        layout.addWidget(watermark)

        layout.addStretch(1)
        left_scroll.setWidget(left_panel)
        body_layout.addWidget(left_scroll, 0)

        preview_card, preview_body = _create_card("🗺 地图预览")
        preview_layout = QVBoxLayout(preview_body)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        self.preview_panel = TrackMapPreviewPanel()
        preview_layout.addWidget(self.preview_panel)
        body_layout.addWidget(preview_card, 1)

        outer.addWidget(body, 1)
        self.setCentralWidget(central)

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
        dlg.setWindowTitle(APP_TITLE)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.setStyleSheet(
            "QProgressDialog { font-size: 10pt; }"
            "QLabel { font-size: 10pt; color: #333333; }"
        )
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
    from PyQt5.QtCore import Qt as _Qt

    QApplication.setAttribute(_Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(_Qt.AA_UseHighDpiPixmaps, True)
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
