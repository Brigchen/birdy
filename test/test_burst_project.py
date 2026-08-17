# -*- coding: utf-8 -*-
"""连拍动图项目文件：默认路径、相对路径往返、缺文件与按文件名匹配布局。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from burst_anchor import FrameLayout  # noqa: E402
from burst_project import (  # noqa: E402
    PROJECT_KIND,
    PROJECT_SUFFIX,
    build_project_dict,
    default_project_path_for_images,
    is_burst_project_path,
    load_project_file,
    match_layout_for_path,
    parse_project_dict,
    path_for_project,
    resolve_project_image_path,
    save_project_file,
)


def _write_jpg(path: Path) -> None:
    img = np.zeros((8, 12, 3), dtype=np.uint8)
    img[:] = (40, 80, 120)
    assert cv2.imwrite(str(path), img)


def test_default_project_path_uses_folder_name(tmp_path: Path):
    folder = tmp_path / "swallow"
    folder.mkdir()
    a = folder / "a.jpg"
    b = folder / "b.jpg"
    _write_jpg(a)
    _write_jpg(b)
    got = default_project_path_for_images([str(a), str(b)])
    assert got == folder / f"swallow{PROJECT_SUFFIX}"


def test_is_burst_project_path():
    assert is_burst_project_path(r"D:\shots\swallow.birdy-burst.json")
    assert is_burst_project_path("foo.BIRDY-BURST.JSON")
    assert not is_burst_project_path("photo.jpg")
    assert not is_burst_project_path("notes.json")


def test_relative_path_roundtrip(tmp_path: Path):
    folder = tmp_path / "nest"
    folder.mkdir()
    img = folder / "one.jpg"
    _write_jpg(img)
    proj = folder / f"nest{PROJECT_SUFFIX}"
    stored = path_for_project(str(img), folder)
    assert stored == "one.jpg"
    resolved = resolve_project_image_path(stored, folder)
    assert resolved.resolve() == img.resolve()
    outside = tmp_path / "other.jpg"
    _write_jpg(outside)
    abs_stored = path_for_project(str(outside), folder)
    assert Path(abs_stored).is_absolute()


def test_save_load_skips_missing_and_keeps_layout(tmp_path: Path):
    folder = tmp_path / "trip"
    folder.mkdir()
    a = folder / "keep.jpg"
    missing = folder / "gone.jpg"
    _write_jpg(a)
    lay = FrameLayout(ax=0.31, ay=0.42, x0=0.1, y0=0.2, x1=0.8, y1=0.9, auto=False, conf=0.7)
    proj = folder / f"trip{PROJECT_SUFFIX}"
    payload = build_project_dict(
        [str(a), str(missing)],
        [lay, FrameLayout(ax=0.5, ay=0.5, x0=0.2, y0=0.2, x1=0.7, y1=0.7)],
        project_path=proj,
        frame_idx=1,
        options={"burst_mode": "track", "fps": 3.0},
    )
    assert payload["kind"] == PROJECT_KIND
    save_project_file(proj, payload)
    raw = json.loads(proj.read_text(encoding="utf-8"))
    assert raw["frames"][0]["path"] == "keep.jpg"
    data = load_project_file(proj)
    assert data.paths == [str(a.resolve())]
    assert len(data.layouts) == 1
    assert data.layouts[0] is not None
    assert abs(data.layouts[0].ax - 0.31) < 1e-9
    assert data.layouts[0].auto is False
    assert len(data.missing) == 1
    assert "gone.jpg" in data.missing[0]
    assert data.frame_idx == 0
    assert data.options["burst_mode"] == "track"


def test_match_layout_prefers_abspath_then_unique_basename(tmp_path: Path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    p1 = d1 / "bird.jpg"
    p2 = d2 / "bird.jpg"
    p3 = d1 / "other.jpg"
    _write_jpg(p1)
    _write_jpg(p2)
    _write_jpg(p3)
    lay1 = FrameLayout(ax=0.11, ay=0.12, x0=0.1, y0=0.1, x1=0.5, y1=0.5, auto=False)
    lay2 = FrameLayout(ax=0.21, ay=0.22, x0=0.2, y0=0.2, x1=0.6, y1=0.6, auto=False)
    lay3 = FrameLayout(ax=0.91, ay=0.92, x0=0.3, y0=0.3, x1=0.7, y1=0.7, auto=True)
    entries = [
        (str(p1), lay1),
        (str(p2), lay2),
        (str(p3), lay3),
    ]
    assert match_layout_for_path(str(p2), entries) is lay2
    uniq = tmp_path / "c" / "other.jpg"
    uniq.parent.mkdir()
    _write_jpg(uniq)
    assert match_layout_for_path(str(uniq), entries) is lay3
    amb = tmp_path / "d" / "bird.jpg"
    amb.parent.mkdir()
    _write_jpg(amb)
    assert match_layout_for_path(str(amb), entries) is None


def test_parse_ignores_wrong_kind_via_load(tmp_path: Path):
    folder = tmp_path / "x"
    folder.mkdir()
    img = folder / "i.jpg"
    _write_jpg(img)
    proj = folder / f"x{PROJECT_SUFFIX}"
    payload = build_project_dict([str(img)], [None], project_path=proj)
    save_project_file(proj, payload)
    data = parse_project_dict(
        json.loads(proj.read_text(encoding="utf-8")),
        proj,
    )
    assert data.paths == [str(img.resolve())]
    assert data.layouts == [None]
