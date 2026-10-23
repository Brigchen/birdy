# -*- coding: utf-8 -*-
"""左栏按内容分配宽度：浏览等按键应完整可见，不能被视口锁掉右侧。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PyQt5.QtCore import QRect  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QScrollArea  # noqa: E402

from birdy_gui import BirdDetectionGUI, soft_break_long_text  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_soft_break_long_text_inserts_zwsp():
    raw = r"C:/Users/brigc/Pictures/Photo_Output/classification_20260927_塔礁湿地公园"
    wrapped = soft_break_long_text(raw)
    assert "\u200b" in wrapped
    assert wrapped.replace("\u200b", "") == raw


def _row_browse_button(line_edit: QLineEdit) -> QPushButton:
    w = line_edit.parent()
    while w is not None:
        for b in w.findChildren(QPushButton):
            if str(b.text()).startswith("浏览"):
                return b
        w = w.parent()
    raise AssertionError("未找到浏览按钮")


def test_real_gui_left_panel_keeps_right_side_controls(qapp, monkeypatch):
    monkeypatch.setattr(BirdDetectionGUI, "showMaximized", lambda self: None)

    win = BirdDetectionGUI()
    long_path = (
        "C:/Users/brigc/Pictures/Photo_Output/"
        "classification_20260927_塔礁湿地公园"
    )
    win.image_folder_input.setText(long_path)
    win.output_root_input.setText("C:/Users/brigc/Pictures/Photo_Output")
    win._refresh_derived_paths_display()
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()

    btn = _row_browse_button(win.image_folder_input)
    assert btn.isVisible()
    assert btn.width() >= 24

    # 左栏不得被锁成只剩标签列：浏览按钮必须落在窗口客户区内
    central = win.centralWidget()
    top_left = btn.mapTo(central, btn.rect().topLeft())
    bottom_right = btn.mapTo(central, btn.rect().bottomRight())
    assert central.rect().contains(QRect(top_left, bottom_right), True)

    # 左栏滚动区应占到接近半窗，而不是被锁成窄条
    scroll = win.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.width() >= 400

    win.close()
