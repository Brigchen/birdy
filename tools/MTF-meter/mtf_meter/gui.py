# -*- coding: utf-8 -*-
"""MTF-meter 独立 GUI。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config_io import load_config, save_config
from .export import (
    cjk_font_prop,
    draw_group_bars,
    export_iso_xlsx,
    export_scene_xlsx,
    suggested_xlsx_path,
)
from .grouping import (
    attach_group,
    cluster_label,
    cluster_scenes_by_folder,
    cluster_scenes_by_lens,
    cluster_scenes_by_time,
    selected_group_keys,
    summarize_groups,
)
from .image_load import read_bgr
from .paths import default_output_dir, find_window_icon, setup_import_paths, tool_dir
from .preview import overlay_rois
from .relative import rank_scene
from .worker import MeasureWorker

setup_import_paths()

try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    FigureCanvasQTAgg = None  # type: ignore
    Figure = None  # type: ignore

APP_TITLE = "MTF-meter"
APP_SUB = "镜头成像比较 · ISO 12233 / 同场景相对比较"

ISO_FILE_HEADERS = [
    ("文件", "文件名"),
    ("镜头", "EXIF 镜头型号"),
    ("焦距", "焦距（mm）"),
    ("ISO", "感光度"),
    ("MTF50", "对比度降至 50% 的空间频率（cycles/pixel），越高越锐"),
    ("MTF30", "对比度降至 30% 的空间频率（cy/px）"),
    ("MTF10", "对比度降至 10% 的空间频率（cy/px），接近极限分辨"),
    ("过锐峰", "SFR 低频峰值；明显大于 1 表示锐化过强"),
    ("中心", "中心区 MTF50（cy/px）"),
    ("边角", "四周区 MTF50 中位（cy/px）"),
    ("中心/边", "中心 MTF50 ÷ 四周 MTF50，越接近 1 越均匀"),
    ("边数", "检出的有效斜边条数"),
]
ISO_SUM_HEADERS = [
    ("分组", "按勾选字段汇总"),
    ("张", "该组照片张数"),
    ("有效", "测到有效 MTF50 的张数"),
    ("MTF50", "组内 MTF50 中位数（括号内为均值）"),
    ("MTF30", "组内 MTF30 中位数"),
    ("过锐峰", "组内过锐峰值中位数"),
    ("中心/边", "组内中心/边比中位数"),
]
SCENE_FILE_HEADERS = [
    ("场景", "场景分组名称"),
    ("名次", "该场景内相对名次"),
    ("相对", "组内最优为 1.00"),
    ("镜头", "EXIF 镜头型号"),
    ("焦距", "焦距（mm）"),
    ("MTF50", "对比度降至 50% 的空间频率（cy/px）"),
    ("MTF30", "对比度降至 30% 的空间频率（cy/px）"),
    ("MTF10", "对比度降至 10% 的空间频率（cy/px）"),
    ("过锐峰", "SFR 低频峰值；明显大于 1 表示锐化过强"),
    ("中心/边", "中心 MTF50 ÷ 四周 MTF50"),
    ("文件", "文件名"),
]
SCENE_LENS_HEADERS = [
    ("镜头", "EXIF 镜头型号"),
    ("出场", "参与比较的场景次数"),
    ("相对中位", "跨场景相对分中位数（1.00=场景最优）"),
    ("均次", "跨场景平均名次，越小越好"),
]
_PEAK_WARN = 1.08
_RATIO_WARN = 1.45

APP_STYLE = """
    QMainWindow { background-color: #F5F5F5; }
    QWidget {
        font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Arial', sans-serif;
        font-size: 9pt;
    }
    QLabel { color: #333333; }
    QLineEdit, QSpinBox, QComboBox {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 5px 10px;
        min-height: 1.1em;
    }
    QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border: 1px solid #2E8B57; }
    QPushButton {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 6px 14px;
        font-weight: 500;
        min-height: 1.2em;
    }
    QPushButton:hover:enabled { border: 1px solid #1E90FF; background: #F0F0F0; }
    QPushButton#primary {
        background-color: #2E8B57;
        color: #FFFFFF;
        border: 1px solid #2E8B57;
    }
    QPushButton#primary:hover:enabled { background-color: #3CB371; }
    QPushButton#danger {
        color: #C62828;
        border: 1px solid #EF9A9A;
    }
    QPushButton#danger:hover:enabled { background: #FFEBEE; border: 1px solid #E57373; }
    QPushButton#danger:disabled { color: #BDBDBD; }
    QCheckBox { spacing: 6px; }
    QTableWidget {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        gridline-color: #EEEEEE;
        font-size: 8pt;
        selection-background-color: #C8E6C9;
        selection-color: #1B5E20;
        alternate-background-color: #F7FBF7;
    }
    QTableWidget::item { padding: 1px 4px; }
    QHeaderView::section {
        background: #E8F5E9;
        padding: 3px 4px;
        border: none;
        border-right: 1px solid #D7EBD9;
        font-weight: 600;
        font-size: 8pt;
        color: #1B5E20;
    }
    QTextEdit {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        font-family: Consolas, 'Courier New', 'Microsoft YaHei UI', monospace;
        font-size: 9pt;
    }
    QTabWidget::pane { border: 1px solid #E0E0E0; background: #F5F5F5; }
    QTabBar::tab {
        padding: 8px 16px;
        background: #EEEEEE;
        border: 1px solid #E0E0E0;
        border-bottom: none;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }
    QTabBar::tab:selected { background: #FFFFFF; color: #1B5E20; font-weight: 600; }
"""


def _item(text: str, numeric: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    if numeric:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def _name_item(text: str) -> QTableWidgetItem:
    raw = str(text or "")
    it = QTableWidgetItem(raw)
    it.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    it.setToolTip(raw)
    return it


def _finite(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if fv != fv:
        return None
    return fv


def _metric_item(text: str, value: Any = None, kind: str = "") -> QTableWidgetItem:
    it = _item(text, numeric=True)
    fv = _finite(value)
    if kind == "peak":
        it.setToolTip("SFR 低频峰值；明显大于 1 表示锐化过强")
        if fv is not None and fv >= _PEAK_WARN:
            it.setForeground(QBrush(QColor("#C62828")))
            it.setBackground(QBrush(QColor("#FFF3E0")))
            it.setToolTip(f"过锐峰值 {fv:.3f}，可能锐化过强")
    elif kind == "ratio":
        it.setToolTip("中心 MTF50 ÷ 四周 MTF50，越接近 1 越均匀")
        if fv is not None and fv >= _RATIO_WARN:
            it.setForeground(QBrush(QColor("#E65100")))
            it.setToolTip(f"中心/边比 {fv:.2f}，边缘下降较明显")
        elif fv is not None and fv < 0.85:
            it.setForeground(QBrush(QColor("#1565C0")))
            it.setToolTip(f"中心/边比 {fv:.2f}，四周高于中心（少见）")
    return it


def _set_headers(table: QTableWidget, headers: Sequence[Tuple[str, str]]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels([h[0] for h in headers])
    for i, (_, tip) in enumerate(headers):
        item = table.horizontalHeaderItem(i)
        if item is not None:
            item.setToolTip(tip)


def _enable_wrap_table(table: QTableWidget) -> None:
    """小字号、单行省略，超出范围用滚动条，列宽可拖动。"""
    table.setWordWrap(False)
    table.setTextElideMode(Qt.ElideRight)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(20)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    hdr = table.horizontalHeader()
    hdr.setStretchLastSection(False)
    hdr.setSectionsMovable(False)
    hdr.setMinimumSectionSize(36)
    hdr.setDefaultAlignment(Qt.AlignCenter | Qt.AlignVCenter)
    hdr.setFixedHeight(24)
    for i in range(table.columnCount()):
        hdr.setSectionResizeMode(i, QHeaderView.Interactive)


def _relayout_table(
    table: QTableWidget,
    name_cols: Sequence[int] = (0,),
) -> None:
    """固定偏窄列宽，不把列挤进视口，以便出现横向滚动条。"""
    n = table.columnCount()
    names = {int(i) for i in name_cols if 0 <= int(i) < n}
    for i in range(n):
        if i in names:
            table.setColumnWidth(i, 118)
        else:
            table.setColumnWidth(i, 52)


def _bgr_to_pixmap(bgr) -> Optional[QPixmap]:
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class MeasurePreview(QWidget):
    """左下角：原图 + 自动斜边测量框。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pix: Optional[QPixmap] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        title = QLabel("测量预览")
        title.setStyleSheet("font-weight: 600;")
        self.caption = QLabel("点击右侧表格中的文件，查看测量框（绿=中下斜方块，橙=上/左/右）。")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color:#666;")
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setMinimumHeight(200)
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setStyleSheet(
            "background:#1e1e1e; border:1px solid #E0E0E0; border-radius:6px; color:#aaa;"
        )
        self.img.setText("尚无预览")
        lay.addWidget(title)
        lay.addWidget(self.caption)
        lay.addWidget(self.img, 1)

    def clear_preview(self) -> None:
        self._pix = None
        self.img.setPixmap(QPixmap())
        self.img.setText("尚无预览")
        self.caption.setText("点击右侧表格中的文件，查看测量框（绿=中下斜方块，橙=上/左/右）。")

    def show_record(self, rec: Optional[Dict[str, Any]]) -> None:
        if not rec:
            self.clear_preview()
            return
        path = str((rec.get("meta") or {}).get("path") or rec.get("path") or "")
        name = str((rec.get("meta") or {}).get("file") or rec.get("file") or Path(path).name)
        edges = rec.get("edges") or []
        n = len(edges)
        err = str(rec.get("error") or "")
        if not rec.get("ok", True) and err:
            self.caption.setText(f"{name}  ·  {err}")
        elif n == 0:
            self.caption.setText(f"{name}  ·  未检出有效斜边（框为空）")
        else:
            self.caption.setText(
                f"{name}  ·  {n} 条斜边（中下 / 上 / 左 / 右）  ·  绿=中下  橙=上左右"
            )
        bgr = read_bgr(path) if path else None
        if bgr is None:
            self._pix = None
            self.img.setPixmap(QPixmap())
            self.img.setText("无法读取图片")
            return
        vis = overlay_rois(bgr, edges)
        self._pix = _bgr_to_pixmap(vis)
        self._fit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._pix is None or self._pix.isNull():
            return
        box = self.img.size()
        if box.width() < 8 or box.height() < 8:
            return
        self.img.setText("")
        self.img.setPixmap(
            self._pix.scaled(box, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv != fv:
        return "—"
    return f"{fv:.{nd}f}"


class MtfMeterWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE}  ·  {APP_SUB}")
        self.resize(1380, 820)
        icon = find_window_icon()
        if icon:
            self.setWindowIcon(QIcon(str(icon)))
        self.cfg = load_config()
        self._iso_rows: List[Dict[str, Any]] = []
        self._scene_rows: List[Dict[str, Any]] = []
        self._scene_pack: List[Dict[str, Any]] = []
        self._scene_table_recs: List[Dict[str, Any]] = []
        self._filling = False
        self._worker: Optional[MeasureWorker] = None
        self._busy_mode = ""
        self._build()

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size: 18pt; font-weight: 700; color: #1B5E20;")
        sub = QLabel(
            "独立工具，不接入 Birdy 主界面。"
            "拍 ISO 12233 标板可测 MTF50 等；同场景多镜头则做相对比较。"
        )
        sub.setStyleSheet("color: #555;")
        layout.addWidget(title)
        layout.addWidget(sub)

        tabs = QTabWidget()
        tabs.addTab(self._iso_tab(), "ISO 12233 标准测量")
        tabs.addTab(self._scene_tab(), "同场景镜头比较")
        layout.addWidget(tabs, 1)
        self.tabs = tabs

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

    def _folder_row(self, edit: QLineEdit, on_browse) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        btn = QPushButton("浏览…")
        btn.clicked.connect(on_browse)
        h.addWidget(btn)
        return row

    def _group_checks(self) -> Tuple[QCheckBox, QCheckBox, QCheckBox, QCheckBox, QCheckBox]:
        lens = QCheckBox("镜头")
        focal = QCheckBox("焦距")
        iso = QCheckBox("ISO")
        ap = QCheckBox("光圈")
        cam = QCheckBox("机身")
        lens.setChecked(bool(self.cfg.get("group_by_lens", True)))
        focal.setChecked(bool(self.cfg.get("group_by_focal", True)))
        iso.setChecked(bool(self.cfg.get("group_by_iso", False)))
        ap.setChecked(bool(self.cfg.get("group_by_aperture", False)))
        cam.setChecked(bool(self.cfg.get("group_by_camera", False)))
        return lens, focal, iso, ap, cam

    def _iso_tab(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        form = QFormLayout()
        self.iso_folder = QLineEdit(self.cfg.get("iso_folder", ""))
        form.addRow(
            "标板照片目录:",
            self._folder_row(self.iso_folder, lambda: self._browse(self.iso_folder)),
        )
        self.iso_recursive = QCheckBox("包含子目录")
        self.iso_recursive.setChecked(True)
        form.addRow("", self.iso_recursive)
        self.g_lens, self.g_focal, self.g_iso, self.g_ap, self.g_cam = self._group_checks()
        gwrap = QWidget()
        gh = QHBoxLayout(gwrap)
        gh.setContentsMargins(0, 0, 0, 0)
        for c in (self.g_lens, self.g_focal, self.g_iso, self.g_ap, self.g_cam):
            gh.addWidget(c)
        gh.addStretch()
        form.addRow("分组比较:", gwrap)
        left.addLayout(form)
        hint = QLabel(
            "请拍摄 ISO 12233 / 综合分辨率标板（斜边约 5°），构图尽量充满画面。"
            "只测左/右黑斜块、上方斜条和中心偏下斜方块，避开圆环与星形图。"
            "文件表给出 MTF50/30/10、过锐峰值与中心/边比；过锐峰明显大于 1 会标橙。"
            "每测完一张即写入表格，可随时点「中止」保留已测结果。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;")
        left.addWidget(hint)
        btns = QHBoxLayout()
        self.iso_run = QPushButton("开始测量")
        self.iso_run.setObjectName("primary")
        self.iso_run.clicked.connect(lambda: self._start("iso"))
        self.iso_export = QPushButton("导出 Excel")
        self.iso_export.clicked.connect(self._export_iso)
        btns.addWidget(self.iso_run)
        self.iso_stop = QPushButton("中止")
        self.iso_stop.setObjectName("danger")
        self.iso_stop.setEnabled(False)
        self.iso_stop.clicked.connect(self._stop)
        btns.addWidget(self.iso_stop)
        btns.addWidget(self.iso_export)
        btns.addStretch()
        left.addLayout(btns)
        self.iso_sum = QTableWidget()
        _set_headers(self.iso_sum, ISO_SUM_HEADERS)
        _enable_wrap_table(self.iso_sum)
        self.iso_sum.setMinimumHeight(120)
        left.addWidget(QLabel("分组汇总"))
        left.addWidget(self.iso_sum, 1)
        self.iso_preview = MeasurePreview()
        left.addWidget(self.iso_preview, 1)
        h.addLayout(left, 2)

        right = QVBoxLayout()
        self.iso_table = QTableWidget()
        _set_headers(self.iso_table, ISO_FILE_HEADERS)
        _enable_wrap_table(self.iso_table)
        self.iso_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.iso_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.iso_table.itemSelectionChanged.connect(self._on_iso_table_sel)
        self.iso_sum.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.iso_sum.setSelectionMode(QAbstractItemView.SingleSelection)
        self.iso_sum.itemSelectionChanged.connect(self._on_iso_sum_sel)
        right.addWidget(self.iso_table, 2)
        self.iso_canvas = self._make_canvas()
        if self.iso_canvas:
            right.addWidget(self.iso_canvas, 1)
        self.iso_log = QTextEdit()
        self.iso_log.setReadOnly(True)
        self.iso_log.setMaximumHeight(72)
        right.addWidget(self.iso_log)
        h.addLayout(right, 3)
        return w

    def _scene_tab(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        left = QVBoxLayout()
        form = QFormLayout()
        self.scene_folder = QLineEdit(self.cfg.get("scene_folder", ""))
        form.addRow(
            "同场景照片目录:",
            self._folder_row(self.scene_folder, lambda: self._browse(self.scene_folder)),
        )
        self.scene_mode = QComboBox()
        self.scene_mode.addItem("按拍摄时间聚类", "time")
        self.scene_mode.addItem("按子文件夹分组", "folder")
        self.scene_mode.addItem("按镜头类型分组", "lens")
        mode = self.cfg.get("scene_mode", "time")
        i = self.scene_mode.findData(mode)
        self.scene_mode.setCurrentIndex(i if i >= 0 else 0)
        self.scene_mode.currentIndexChanged.connect(self._sync_scene_gap_enabled)
        form.addRow("场景划分:", self.scene_mode)
        self.scene_gap = QSpinBox()
        self.scene_gap.setRange(15, 3600)
        self.scene_gap.setValue(int(self.cfg.get("scene_gap_sec", 180)))
        self.scene_gap.setSuffix(" 秒")
        form.addRow("时间间隔阈值:", self.scene_gap)
        left.addLayout(form)
        hint = QLabel(
            "同一目标换镜头拍的一组照片：可按拍摄时间、子文件夹，或 EXIF 镜头类型划分。"
            "有斜边时比较 MTF50/30/10 与过锐峰、中心/边比；否则用梯度能量。"
            "组内最优为 1.00。点击文件行查看测量框。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;")
        left.addWidget(hint)
        btns = QHBoxLayout()
        self.scene_run = QPushButton("开始比较")
        self.scene_run.setObjectName("primary")
        self.scene_run.clicked.connect(lambda: self._start("scene"))
        self.scene_export = QPushButton("导出 Excel")
        self.scene_export.clicked.connect(self._export_scene)
        btns.addWidget(self.scene_run)
        self.scene_stop = QPushButton("中止")
        self.scene_stop.setObjectName("danger")
        self.scene_stop.setEnabled(False)
        self.scene_stop.clicked.connect(self._stop)
        btns.addWidget(self.scene_stop)
        btns.addWidget(self.scene_export)
        btns.addStretch()
        left.addLayout(btns)
        self.scene_lens = QTableWidget()
        _set_headers(self.scene_lens, SCENE_LENS_HEADERS)
        _enable_wrap_table(self.scene_lens)
        left.addWidget(QLabel("镜头总评（跨场景）"))
        left.addWidget(self.scene_lens, 1)
        self.scene_preview = MeasurePreview()
        left.addWidget(self.scene_preview, 1)
        h.addLayout(left, 2)

        right = QVBoxLayout()
        self.scene_table = QTableWidget()
        _set_headers(self.scene_table, SCENE_FILE_HEADERS)
        _enable_wrap_table(self.scene_table)
        self.scene_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scene_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.scene_table.itemSelectionChanged.connect(self._on_scene_table_sel)
        right.addWidget(self.scene_table, 2)
        self.scene_canvas = self._make_canvas()
        if self.scene_canvas:
            right.addWidget(self.scene_canvas, 1)
        self.scene_log = QTextEdit()
        self.scene_log.setReadOnly(True)
        self.scene_log.setMaximumHeight(72)
        right.addWidget(self.scene_log)
        h.addLayout(right, 3)
        self._sync_scene_gap_enabled()
        return w

    def _make_canvas(self):
        if Figure is None or FigureCanvasQTAgg is None:
            return None
        fig = Figure(figsize=(5.8, 2.8))
        fig.subplots_adjust(left=0.36, right=0.96, top=0.88, bottom=0.18)
        cjk_font_prop()
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(168)
        canvas._ax = fig.add_subplot(111)
        return canvas

    def _sync_scene_gap_enabled(self) -> None:
        self.scene_gap.setEnabled(self.scene_mode.currentData() == "time")

    def _browse(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择目录", edit.text().strip() or "")
        if d:
            edit.setText(d)

    def _group_keys(self) -> Tuple[str, ...]:
        return selected_group_keys(
            lens=self.g_lens.isChecked(),
            focal=self.g_focal.isChecked(),
            iso=self.g_iso.isChecked(),
            aperture=self.g_ap.isChecked(),
            camera=self.g_cam.isChecked(),
        )

    def _set_busy(self, busy: bool) -> None:
        self.iso_run.setEnabled(not busy)
        self.scene_run.setEnabled(not busy)
        iso_run = busy and self._busy_mode == "iso"
        scene_run = busy and self._busy_mode == "scene"
        self.iso_stop.setEnabled(iso_run)
        self.scene_stop.setEnabled(scene_run)

    def _stop(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.cancel()
        self.iso_stop.setEnabled(False)
        self.scene_stop.setEnabled(False)
        self._log(self._busy_mode, "正在中止，当前这张测完后停止…")

    def _log(self, mode: str, msg: str) -> None:
        box = self.iso_log if mode == "iso" else self.scene_log
        box.append(msg)

    def _start(self, mode: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "提示", "正在测量。若要停止，请点「中止」。")
            return
        folder = (
            self.iso_folder.text().strip()
            if mode == "iso"
            else self.scene_folder.text().strip()
        )
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "提示", "请选择有效的照片目录。")
            return
        self._persist()
        self._busy_mode = mode
        self.progress.setValue(0)
        if mode == "iso":
            self._iso_rows = []
            self.iso_table.setRowCount(0)
            self.iso_sum.setRowCount(0)
            self.iso_preview.clear_preview()
        else:
            self._scene_rows = []
            self._scene_pack = []
            self._scene_table_recs = []
            self.scene_table.setRowCount(0)
            self.scene_lens.setRowCount(0)
            self.scene_preview.clear_preview()
        self._set_busy(True)
        self._log(mode, f"扫描 {folder} …")
        rec = mode == "iso" and self.iso_recursive.isChecked()
        self._worker = MeasureWorker(folder, recursive=True if mode == "scene" else rec)
        self._worker.progress.connect(self._on_progress)
        self._worker.row_ready.connect(self._on_row)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_progress(self, done: int, total: int, path: str) -> None:
        total = max(1, total)
        self.progress.setValue(int(100 * done / total))
        name = Path(path).name if path else ""
        if name:
            self._log(self._busy_mode, f"{done}/{total}  {name}")

    def _on_row(self, rec: object, done: int, total: int) -> None:
        if not isinstance(rec, dict):
            return
        if self._busy_mode == "iso":
            self._append_iso_row(rec)
        else:
            keep = None
            if self.scene_table.currentRow() >= 0 and self._scene_table_recs:
                i = self.scene_table.currentRow()
                if 0 <= i < len(self._scene_table_recs):
                    keep = self._scene_table_recs[i].get("path")
            self._scene_rows.append(rec)
            self._fill_scene(self._scene_rows, keep_path=str(keep) if keep else None)

    def _on_fail(self, msg: str) -> None:
        self._set_busy(False)
        self._log(self._busy_mode, f"失败：{msg}")
        QMessageBox.warning(self, "测量失败", msg)

    def _on_done(self, rows: List[Dict[str, Any]]) -> None:
        cancelled = bool(self._worker and self._worker.was_cancelled())
        self._set_busy(False)
        self.progress.setValue(100 if rows else 0)
        ok_n = sum(1 for r in rows if r.get("ok"))
        if cancelled:
            self._log(self._busy_mode, f"已中止：保留 {ok_n}/{len(rows)} 张结果")
        else:
            self._log(self._busy_mode, f"完成：{ok_n}/{len(rows)} 张可读")
        if self._busy_mode == "iso":
            if rows and len(self._iso_rows) != len(rows):
                self._fill_iso(rows)
        else:
            keep = None
            if self.scene_table.currentRow() >= 0 and self._scene_table_recs:
                i = self.scene_table.currentRow()
                if 0 <= i < len(self._scene_table_recs):
                    keep = self._scene_table_recs[i].get("path")
            self._fill_scene(rows, keep_path=str(keep) if keep else None)

    def _fill_iso(self, rows: List[Dict[str, Any]], *, select_first: bool = True) -> None:
        self._filling = True
        attach_group(rows, self._group_keys())
        self._iso_rows = list(rows)
        self.iso_table.setRowCount(0)
        for r in rows:
            self._put_iso_file_row(r)
        _relayout_table(self.iso_table, (0, 1))
        self._refresh_iso_sum(rows)
        self._filling = False
        if select_first and self.iso_table.rowCount():
            self.iso_table.selectRow(0)
            self._on_iso_table_sel()
        elif not self.iso_table.rowCount():
            self.iso_preview.clear_preview()

    def _put_iso_file_row(self, r: Dict[str, Any]) -> None:
        meta = r.get("meta") or {}
        i = self.iso_table.rowCount()
        self.iso_table.insertRow(i)
        peak = r.get("mtf_peak")
        ratio = r.get("center_edge_ratio")
        cells = [
            _name_item(str(meta.get("file") or "")),
            _name_item(str(meta.get("lens") or "")),
            _item(_fmt(meta.get("focal_mm"), 1), True),
            _item(str(meta.get("iso") or "—"), True),
            _item(_fmt(r.get("mtf50_cy_px")), True),
            _item(_fmt(r.get("mtf30_cy_px")), True),
            _item(_fmt(r.get("mtf10_cy_px")), True),
            _metric_item(_fmt(peak), peak, "peak"),
            _item(_fmt(r.get("mtf50_center")), True),
            _item(_fmt(r.get("mtf50_corner")), True),
            _metric_item(_fmt(ratio, 2), ratio, "ratio"),
            _item(
                str(r.get("n_edges") if r.get("ok") else r.get("error") or "0"),
                True,
            ),
        ]
        for c, it in enumerate(cells):
            self.iso_table.setItem(i, c, it)

    def _append_iso_row(self, rec: Dict[str, Any]) -> None:
        first = not self._iso_rows
        self._iso_rows.append(rec)
        attach_group(self._iso_rows, self._group_keys())
        self._put_iso_file_row(rec)
        if first:
            _relayout_table(self.iso_table, (0, 1))
        self._refresh_iso_sum(self._iso_rows)
        self.iso_table.scrollToBottom()
        if first:
            self.iso_table.selectRow(0)
            self._on_iso_table_sel()

    def _refresh_iso_sum(self, rows: List[Dict[str, Any]]) -> None:
        summary = summarize_groups(
            rows,
            "mtf50_cy_px",
            extra=("mtf30_cy_px", "mtf_peak", "center_edge_ratio"),
        )
        self.iso_sum.setRowCount(0)
        for s in summary:
            i = self.iso_sum.rowCount()
            self.iso_sum.insertRow(i)
            gitem = _name_item(str(s["group"]))
            gitem.setData(Qt.UserRole, str(s["group"]))
            m50 = _item(_fmt(s["median"]), True)
            if s.get("mean") is not None:
                m50.setToolTip(f"中位 {_fmt(s['median'])}  ·  均值 {_fmt(s['mean'])}")
            peak = s.get("mtf_peak")
            ratio = s.get("center_edge_ratio")
            self.iso_sum.setItem(i, 0, gitem)
            self.iso_sum.setItem(i, 1, _item(str(s["n"]), True))
            self.iso_sum.setItem(i, 2, _item(str(s["n_valid"]), True))
            self.iso_sum.setItem(i, 3, m50)
            self.iso_sum.setItem(i, 4, _item(_fmt(s.get("mtf30_cy_px")), True))
            self.iso_sum.setItem(i, 5, _metric_item(_fmt(peak), peak, "peak"))
            self.iso_sum.setItem(i, 6, _metric_item(_fmt(ratio, 2), ratio, "ratio"))
        _relayout_table(self.iso_sum, (0,))
        if self.iso_canvas is not None:
            draw_group_bars(self.iso_canvas._ax, summary)
            self.iso_canvas.draw()

    def _fill_scene(
        self,
        rows: List[Dict[str, Any]],
        *,
        keep_path: Optional[str] = None,
    ) -> None:
        self._filling = True
        self._scene_rows = rows
        mode = self.scene_mode.currentData()
        if mode == "folder":
            clusters = cluster_scenes_by_folder(rows)
        elif mode == "lens":
            clusters = cluster_scenes_by_lens(rows)
        else:
            clusters = cluster_scenes_by_time(rows, float(self.scene_gap.value()))
        pack: List[Dict[str, Any]] = []
        self._scene_table_recs = []
        self.scene_table.setRowCount(0)
        for si, cl in enumerate(clusters, start=1):
            ranked = rank_scene(cl)
            name = cluster_label(cl, str(mode or "time"), si)
            pack.append({"name": name, "ranked": ranked})
            for r in ranked:
                i = self.scene_table.rowCount()
                self.scene_table.insertRow(i)
                self._scene_table_recs.append(r)
                peak = r.get("mtf_peak")
                ratio = r.get("center_edge_ratio")
                cells = [
                    _name_item(name),
                    _item(str(r.get("rank")), True),
                    _item(_fmt(r.get("relative"), 3), True),
                    _name_item(str(r.get("lens") or "")),
                    _item(_fmt(r.get("focal_mm"), 1), True),
                    _item(_fmt(r.get("mtf50_cy_px")), True),
                    _item(_fmt(r.get("mtf30_cy_px")), True),
                    _item(_fmt(r.get("mtf10_cy_px")), True),
                    _metric_item(_fmt(peak), peak, "peak"),
                    _metric_item(_fmt(ratio, 2), ratio, "ratio"),
                    _name_item(str(r.get("file") or "")),
                ]
                for c, it in enumerate(cells):
                    self.scene_table.setItem(i, c, it)
        _relayout_table(self.scene_table, (0, 3, 10))
        self._scene_pack = pack
        lens_stats: Dict[str, List[float]] = {}
        lens_rank: Dict[str, List[int]] = {}
        for sc in pack:
            for r in sc["ranked"]:
                lens = str(r.get("lens") or "未知镜头")
                lens_stats.setdefault(lens, []).append(float(r.get("relative") or 0))
                lens_rank.setdefault(lens, []).append(int(r.get("rank") or 0))
        summary = []
        for lens, rels in lens_stats.items():
            arr = np.array(rels, dtype=np.float64)
            ranks = lens_rank[lens]
            summary.append(
                {
                    "group": lens,
                    "n": len(rels),
                    "median": float(np.median(arr)),
                    "mean_rank": float(sum(ranks) / len(ranks)),
                }
            )
        summary.sort(key=lambda d: -(d["median"] or 0))
        self.scene_lens.setRowCount(0)
        for s in summary:
            i = self.scene_lens.rowCount()
            self.scene_lens.insertRow(i)
            self.scene_lens.setItem(i, 0, _name_item(str(s["group"])))
            self.scene_lens.setItem(i, 1, _item(str(s["n"]), True))
            self.scene_lens.setItem(i, 2, _item(_fmt(s["median"], 3), True))
            self.scene_lens.setItem(i, 3, _item(_fmt(s["mean_rank"], 2), True))
        _relayout_table(self.scene_lens, (0,))
        if self.scene_canvas is not None:
            draw_group_bars(
                self.scene_canvas._ax,
                summary,
                xlabel="相对分中位（1.00=该场景最优）",
            )
            self.scene_canvas.draw()
        self._filling = False
        if keep_path:
            for i, r in enumerate(self._scene_table_recs):
                if str(r.get("path") or "") == keep_path:
                    self.scene_table.selectRow(i)
                    self._on_scene_table_sel()
                    return
        if self.scene_table.rowCount():
            self.scene_table.selectRow(self.scene_table.rowCount() - 1)
            self._on_scene_table_sel()
        else:
            self.scene_preview.clear_preview()

    def _on_iso_table_sel(self) -> None:
        if self._filling:
            return
        row = self.iso_table.currentRow()
        if 0 <= row < len(self._iso_rows):
            self.iso_preview.show_record(self._iso_rows[row])

    def _on_iso_sum_sel(self) -> None:
        if self._filling:
            return
        row = self.iso_sum.currentRow()
        item = self.iso_sum.item(row, 0) if row >= 0 else None
        group = str(item.data(Qt.UserRole) or "") if item else ""
        if not group:
            return
        for i, rec in enumerate(self._iso_rows):
            if str(rec.get("group") or "") == group:
                self.iso_table.selectRow(i)
                return

    def _on_scene_table_sel(self) -> None:
        if self._filling:
            return
        row = self.scene_table.currentRow()
        if 0 <= row < len(self._scene_table_recs):
            self.scene_preview.show_record(self._scene_table_recs[row])

    def _export_iso(self) -> None:
        if not self._iso_rows:
            QMessageBox.information(self, "提示", "请先测量。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 ISO 测量 Excel",
            str(suggested_xlsx_path(self.iso_folder.text().strip())),
            "Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            export_iso_xlsx(path, self._iso_rows)
        except ImportError:
            QMessageBox.warning(
                self, "缺少组件", "导出 Excel 需要 openpyxl：\npython -m pip install openpyxl"
            )
            return
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        self._log("iso", f"已导出 {path}")

    def _export_scene(self) -> None:
        if not self._scene_pack:
            QMessageBox.information(self, "提示", "请先比较。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出同场景比较 Excel",
            str(suggested_xlsx_path(self.scene_folder.text().strip())),
            "Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        try:
            export_scene_xlsx(path, self._scene_pack)
        except ImportError:
            QMessageBox.warning(
                self, "缺少组件", "导出 Excel 需要 openpyxl：\npython -m pip install openpyxl"
            )
            return
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        self._log("scene", f"已导出 {path}")

    def _persist(self) -> None:
        self.cfg["iso_folder"] = self.iso_folder.text().strip()
        self.cfg["scene_folder"] = self.scene_folder.text().strip()
        self.cfg["group_by_lens"] = self.g_lens.isChecked()
        self.cfg["group_by_focal"] = self.g_focal.isChecked()
        self.cfg["group_by_iso"] = self.g_iso.isChecked()
        self.cfg["group_by_aperture"] = self.g_ap.isChecked()
        self.cfg["group_by_camera"] = self.g_cam.isChecked()
        self.cfg["scene_mode"] = self.scene_mode.currentData()
        self.cfg["scene_gap_sec"] = int(self.scene_gap.value())
        if not (self.cfg.get("output_folder") or "").strip():
            self.cfg["output_folder"] = str(default_output_dir())
        save_config(self.cfg)

    def closeEvent(self, event) -> None:
        try:
            self._persist()
        except Exception:
            pass
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)


def main() -> int:
    os.chdir(str(tool_dir()))
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    app.setApplicationName(APP_TITLE)
    win = MtfMeterWindow()
    win.show()
    return app.exec_()
