# -*- coding: utf-8 -*-
"""地理位置地址历史：保存、去重、以及界面选用时不再联网。"""

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

from location_history import (  # noqa: E402
    load_location_history,
    lookup_location,
    remember_location,
)
from geo_encoder import geocode_location  # noqa: E402


def test_remember_dedupes_and_keeps_newest_first(tmp_path):
    path = tmp_path / "location_history.json"
    remember_location("甲地", 24.1, 118.1, "福建", "厦门", path=path)
    remember_location("乙地", 25.2, 119.2, "福建", "福州", path=path)
    remember_location("甲地", 24.5, 118.5, path=path)

    history = load_location_history(path)
    assert [item["name"] for item in history] == ["甲地", "乙地"]
    assert history[0]["lat"] == pytest.approx(24.5)
    assert history[0]["province"] == "福建"
    assert history[0]["city"] == "厦门"
    assert lookup_location("乙地", path=path)["lon"] == pytest.approx(119.2)


def test_remember_caps_history(tmp_path):
    path = tmp_path / "location_history.json"
    for i in range(45):
        remember_location(f"地点{i}", 10.0 + i * 0.01, 100.0, path=path)
    history = load_location_history(path)
    assert len(history) == 40
    assert history[0]["name"] == "地点44"
    assert lookup_location("地点0", path=path) is None


def test_geocode_reuses_saved_address_without_network(tmp_path, monkeypatch):
    path = tmp_path / "location_history.json"
    remember_location("历史湿地", 24.25, 118.75, "福建", "厦门", path=path)
    monkeypatch.setattr(
        "location_history.default_history_path", lambda: path
    )

    def _boom(_name):
        raise AssertionError("不应联网查询")

    monkeypatch.setattr("geo_encoder._geocode_with_amap", _boom)
    monkeypatch.setattr("geo_encoder._geocode_with_tencent", _boom)
    monkeypatch.setattr("geo_encoder._geocode_with_baidu", _boom)
    monkeypatch.setattr("geo_encoder._geocode_with_free_api", _boom)
    monkeypatch.setattr("geo_encoder._geocode_with_nominatim", _boom)

    assert geocode_location("历史湿地") == (24.25, 118.75)


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_gui_history_pick_skips_geocode(qapp, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QComboBox

    path = tmp_path / "location_history.json"
    remember_location("下拉甲", 24.11, 118.11, "福建", "厦门", path=path)
    remember_location("下拉乙", 25.22, 119.22, "福建", "福州", path=path)
    monkeypatch.setattr("location_history.default_history_path", lambda: path)

    from birdy_gui import BirdDetectionGUI

    monkeypatch.setattr(BirdDetectionGUI, "showMaximized", lambda self: None)
    monkeypatch.setattr(BirdDetectionGUI, "_save_config", lambda self: None)

    def _boom(_name):
        raise AssertionError("选用历史位置不应查询")

    monkeypatch.setattr("birdy_gui.geocode_location", _boom)

    win = BirdDetectionGUI()
    try:
        assert isinstance(win.location_input, QComboBox)
        assert win.location_input.findText("下拉乙") >= 0
        win.lat_input.setText("0")
        win.lon_input.setText("0")
        win.config["location_name"] = ""
        index = win.location_input.findText("下拉乙")
        win._on_location_history_picked(index)
        assert win.lat_input.text() == "25.220000"
        assert win.lon_input.text() == "119.220000"
        assert win.province_city_display.text() == "福建 福州"
        assert win._location_text() == "下拉乙"

        win.location_input.setEditText("下拉甲")
        win._on_location_edit_finished()
        assert win.lat_input.text() == "24.110000"
        assert "下拉甲" == load_location_history(path)[0]["name"]
    finally:
        win.close()
