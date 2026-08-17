# -*- coding: utf-8 -*-
"""子进程助手：先创建 QApplication 再导入动图弹窗（Windows 上顺序反了会直接 abort）。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtWidgets import QApplication

_QAPP = QApplication.instance() or QApplication([])

import burst_webp_dialog as m  # noqa: E402


def _patch_paths(tmp: Path):
    primary = tmp / "burst_webp_dialog_state.json"
    legacy = tmp / "legacy_burst_webp_dialog_state.json"
    m._burst_webp_dialog_state_path = lambda: primary
    m._burst_webp_dialog_state_legacy_path = lambda: legacy
    return primary, legacy


def _set_non_default(dlg) -> None:
    dlg.cb_wb.setChecked(False)
    dlg.cb_ae.setChecked(True)
    dlg.slider_ae.setValue(220)
    dlg.cb_wm.setChecked(False)
    dlg.ed_wm_theme.setText("家燕啄新泥")
    dlg.rb_mode_track.setChecked(True)
    dlg.spn_fps.setValue(3.5)
    dlg.spn_q.setValue(72)
    dlg.rb_export_mp4.setChecked(True)
    for i in range(dlg.cmb_max.count()):
        try:
            if int(dlg.cmb_max.itemData(i) or -1) == 1920:
                dlg.cmb_max.setCurrentIndex(i)
                break
        except (TypeError, ValueError):
            continue


def _assert_non_default(dlg) -> None:
    assert dlg.cb_wb.isChecked() is False
    assert dlg.cb_ae.isChecked() is True
    assert dlg.slider_ae.value() == 220
    assert dlg.cb_wm.isChecked() is False
    assert dlg.ed_wm_theme.text() == "家燕啄新泥"
    assert dlg.rb_mode_track.isChecked() is True
    assert abs(dlg.spn_fps.value() - 3.5) < 1e-6
    assert dlg.spn_q.value() == 72
    assert dlg.rb_export_mp4.isChecked() is True
    assert int(dlg.cmb_max.currentData()) == 1920


def cmd_roundtrip(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    primary, _legacy = _patch_paths(tmp)
    d1 = m.BurstWebpDialog()
    _set_non_default(d1)
    d1._save_burst_dialog_state()
    d1.close()

    raw = json.loads(primary.read_text(encoding="utf-8"))
    assert raw["enable_wb"] is False
    assert abs(float(raw["auto_exposure_strength"]) - 2.2) < 1e-6
    assert raw["burst_mode"] == "track"
    assert raw["export_format"] == "mp4"
    assert raw["max_long_edge"] == 1920
    assert "paths" not in raw
    assert "images" not in raw

    d2 = m.BurstWebpDialog()
    _assert_non_default(d2)
    d2.close()


def _write_jpg(path: Path) -> None:
    import cv2
    import numpy as np

    img = np.zeros((16, 24, 3), dtype=np.uint8)
    img[:] = (30, 90, 160)
    assert cv2.imwrite(str(path), img)


def cmd_project(tmp: Path) -> None:
    from burst_anchor import FrameLayout
    from burst_project import PROJECT_SUFFIX

    tmp.mkdir(parents=True, exist_ok=True)
    _patch_paths(tmp)
    folder = tmp / "swallow"
    folder.mkdir()
    a = folder / "a.jpg"
    b = folder / "b.jpg"
    _write_jpg(a)
    _write_jpg(b)

    d1 = m.BurstWebpDialog()
    d1._import_image_paths([str(a), str(b)])
    proj = folder / f"swallow{PROJECT_SUFFIX}"
    assert d1._project_path is not None
    assert d1._project_path.resolve() == proj.resolve()
    assert proj.is_file()
    assert d1.list_w.count() == 2
    d1._layouts[0] = FrameLayout(
        ax=0.33, ay=0.44, x0=0.10, y0=0.15, x1=0.70, y1=0.80, auto=False, conf=1.0
    )
    d1._layouts[1] = FrameLayout(
        ax=0.35, ay=0.46, x0=0.12, y0=0.16, x1=0.72, y1=0.82, auto=False, conf=1.0
    )
    d1._flush_layouts_to_sticky()
    d1._save_project_now()
    d1.close()

    d2 = m.BurstWebpDialog(default_dir=str(folder))
    assert d2.list_w.count() == 2
    assert d2._layouts[0] is not None
    assert d2._layouts[1] is not None
    assert abs(d2._layouts[0].ax - 0.33) < 1e-6
    assert d2._layouts[0].auto is False
    assert abs(d2._layouts[1].ay - 0.46) < 1e-6
    assert d2._project_path is not None
    assert d2._project_path.resolve() == proj.resolve()
    d2.close()


def cmd_noclobber(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    primary, _legacy = _patch_paths(tmp)
    payload = {
        "version": 5,
        "enable_wb": False,
        "enable_auto_exposure": True,
        "auto_exposure_strength": 2.2,
        "enable_wm": False,
        "wm_theme": "家燕啄新泥",
        "burst_mode": "track",
        "fps": 4.0,
        "max_long_edge": 1920,
        "webp_quality": 70,
        "out_path": "",
        "export_format": "mp4",
        "window_maximized": False,
        "window_geometry": [10, 10, 800, 600],
    }
    primary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dlg = m.BurstWebpDialog()
    on_disk = json.loads(primary.read_text(encoding="utf-8"))
    assert on_disk["burst_mode"] == "track"
    assert on_disk["enable_wb"] is False
    assert dlg.rb_mode_track.isChecked()
    assert dlg.cb_wb.isChecked() is False
    assert dlg.slider_ae.value() == 220
    assert abs(dlg.spn_fps.value() - 4.0) < 1e-6
    assert dlg.rb_export_mp4.isChecked()
    assert int(dlg.cmb_max.currentData()) == 1920
    dlg.close()
    after = json.loads(primary.read_text(encoding="utf-8"))
    assert after["burst_mode"] == "track"
    assert after["enable_wb"] is False
    assert after["wm_theme"] == "家燕啄新泥"
    assert after["export_format"] == "mp4"
    assert after["max_long_edge"] == 1920
    assert after["webp_quality"] == 70


def main() -> int:
    cmd = sys.argv[1]
    tmp = Path(sys.argv[2])
    if cmd == "roundtrip":
        cmd_roundtrip(tmp)
    elif cmd == "noclobber":
        cmd_noclobber(tmp)
    elif cmd == "project":
        cmd_project(tmp)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
