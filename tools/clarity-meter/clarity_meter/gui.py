# -*- coding: utf-8 -*-
"""清晰度打分 GUI：调用 Birdy 算法，生成 HTML 报告，不删图。"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .paths import setup_import_paths
from .report import default_report_path, write_html_report
from .worker import ScoreWorker

setup_import_paths()

APP_TITLE = "清晰度打分"
APP_SUB = "调用 Birdy 认种前模糊评分 · 只读 · HTML 报告"

APP_STYLE = """
    QMainWindow { background-color: #F5F5F5; }
    QWidget {
        font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Arial', sans-serif;
        font-size: 10pt;
    }
    QLineEdit, QSpinBox {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 5px 10px;
        min-height: 1.1em;
    }
    QPushButton {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 8px 16px;
    }
    QPushButton#primary {
        background-color: #2E7D32;
        color: white;
        border: none;
        font-weight: 600;
    }
    QTextEdit {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
    }
"""


class ClarityMeterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(720, 480)
        self._worker: Optional[ScoreWorker] = None
        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        title = QLabel(f"<b>{APP_TITLE}</b><br/><span style='color:#666'>{APP_SUB}</span>")
        title.setTextFormat(Qt.RichText)
        lay.addWidget(title)

        form = QFormLayout()
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("选择要打分的图片目录（递归）")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        form.addRow("图片目录:", row)

        self.th_spin = QSpinBox()
        self.th_spin.setRange(0, 100)
        self.th_spin.setValue(35)
        self.th_spin.setToolTip("与 GUI「模糊阈值」相同：鸟体掩膜分低于此值判定为模糊")
        form.addRow("模糊阈值:", self.th_spin)

        self.full_cb = QCheckBox("目录已是切割图（整图套掩膜）")
        self.full_cb.setToolTip(
            "对应主流程切割后再清洗。掩膜计分仍按鸟体裁到统一尺度，"
            "避免整张切割图里小鸟过小、糊边被当成锐度。"
        )
        form.addRow("", self.full_cb)
        lay.addLayout(form)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("开始打分")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch()
        lay.addLayout(btns)

        self.prog = QProgressBar()
        self.prog.setValue(0)
        lay.addWidget(self.prog)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)
        hint = QLabel("不会删除或改写源图片。完成后用浏览器打开 HTML 表格报告。")
        hint.setStyleSheet("color:#666;")
        lay.addWidget(hint)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择图片目录", self.folder_edit.text())
        if d:
            self.folder_edit.setText(d)

    def _append(self, msg: str) -> None:
        self.log.append(msg)

    def _start(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, APP_TITLE, "请选择有效的图片目录。")
            return
        self.prog.setValue(0)
        self.log.clear()
        self._append(f"目录: {folder}")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._worker = ScoreWorker(
            folder,
            min_clarity=float(self.th_spin.value()),
            use_full_frame=self.full_cb.isChecked(),
            parent=self,
        )
        self._worker.progress.connect(self._on_prog)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._append("正在取消…")

    def _on_prog(self, done: int, total: int, name: str) -> None:
        self.prog.setMaximum(max(1, total))
        self.prog.setValue(done)
        if name:
            self._append(f"[{done}/{total}] {name}")

    def _on_ok(self, result: dict) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        out = default_report_path(str(result.get("root") or ""))
        path = write_html_report(result, str(out))
        self._append(
            f"完成：共 {result.get('total')}，通过 {result.get('n_pass')}，"
            f"模糊 {result.get('n_blur')}，未检出 {result.get('n_nobird')}"
        )
        self._append(f"报告: {path}")
        webbrowser.open(Path(path).resolve().as_uri())

    def _on_fail(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._append(f"失败: {msg}")
        QMessageBox.warning(self, APP_TITLE, msg)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    w = ClarityMeterWindow()
    w.show()
    return app.exec_()
