# -*- coding: utf-8 -*-
"""DxO PureRAW batch 接口：输出命名预测与映射（不启动 PureRAW）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dxo_pureraw import (  # noqa: E402
    dxo_output_candidates,
    find_pureraw_exe,
    map_sources_to_outputs,
)


def test_run_pureraw_batch_raises_when_missing(tmp_path: Path, monkeypatch):
    import dxo_pureraw

    monkeypatch.setattr(dxo_pureraw, "find_pureraw_exe", lambda: None)
    import pytest

    src = tmp_path / "noout.ARW"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        dxo_pureraw.run_pureraw_batch([str(src)])


def test_run_pureraw_batch_skips_when_all_outputs_exist(tmp_path: Path, monkeypatch):
    """全部源已有 -DxO 输出时不启动 PureRAW，直接返回成功。"""
    import dxo_pureraw

    called = {"n": 0}
    monkeypatch.setattr(dxo_pureraw, "find_pureraw_exe", lambda: None)

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("Popen 不应被调用")

    monkeypatch.setattr(dxo_pureraw.subprocess, "Popen", _boom)
    src = tmp_path / "a.ARW"
    src.write_bytes(b"x")
    (tmp_path / "a-DxO.jpg").write_bytes(b"j")
    r = dxo_pureraw.run_pureraw_batch([str(src)])
    assert called["n"] == 0
    assert r["ok"] == 1 and r["fail"] == 0


def test_find_pureraw_exe_env_var_overrides(tmp_path: Path, monkeypatch):
    """环境变量 DXO_PURERAW_EXE 指向的文件优先于自动探测。"""
    import os

    import dxo_pureraw

    fake = tmp_path / "PureRAWv9.exe"
    fake.write_bytes(b"x")
    monkeypatch.setenv("DXO_PURERAW_EXE", str(fake))
    assert dxo_pureraw.find_pureraw_exe() == str(fake)
    assert str(fake) in [c for c in dxo_pureraw._candidate_exe_paths()][:1]


def test_find_pureraw_exe_on_this_machine():
    exe = find_pureraw_exe()
    assert exe and Path(exe).is_file()


def test_output_candidates_cover_dxo_suffix(tmp_path: Path):
    """输出发现：同目录下存在的 <stem>-DxO.* 都应被列出。"""
    src = tmp_path / "f27601408.ARW"
    src.write_bytes(b"x")
    jpg = tmp_path / "f27601408-DxO.jpg"
    dng = tmp_path / "f27601408-DxO.dng"
    jpg.write_bytes(b"j")
    dng.write_bytes(b"d")
    cands = dxo_output_candidates(str(src))
    names = {c.name for c in cands}
    assert "f27601408-DxO.jpg" in names
    assert "f27601408-DxO.dng" in names
    # jpg 优先
    assert cands[0] == jpg


def test_map_sources_to_outputs_prefers_jpeg(tmp_path: Path):
    src = tmp_path / "a.ARW"
    src.write_bytes(b"x")
    jpg = tmp_path / "a-DxO.jpg"
    dng = tmp_path / "a-DxO.dng"
    jpg.write_bytes(b"j")
    dng.write_bytes(b"d")
    m = map_sources_to_outputs([str(src)])
    assert m[str(src)].lower().endswith(".jpg")


def test_map_and_filter_replaces_raw_and_drops_dxo_files(tmp_path: Path):
    """RAW → DxO 输出替换；目录里的 -DxO.jpg 本身不应再作为识别来源。"""
    from dxo_pureraw import map_and_filter

    raw = tmp_path / "a.ARW"
    raw.write_bytes(b"x")
    (tmp_path / "a-DxO.jpg").write_bytes(b"j")
    files = [str(raw), str(tmp_path / "a-DxO.jpg")]
    new_list, m = map_and_filter(files)
    assert m
    assert len(new_list) == 1
    assert new_list[0].lower().endswith(".jpg")


def test_output_found_in_subfolder(tmp_path: Path):
    """预设自定义子目录：输出在源的子目录下也能被发现并映射。"""
    from dxo_pureraw import dxo_output_candidates, map_and_filter

    sub = tmp_path / "DxO输出"
    sub.mkdir()
    raw = tmp_path / "b.ARW"
    raw.write_bytes(b"x")
    out = sub / "b-DxO_DeepPRIME XD3.jpg"
    out.write_bytes(b"j")
    cands = dxo_output_candidates(str(raw))
    assert out in cands
    new_list, m = map_and_filter([str(raw)])
    assert m[str(raw)] == str(out.resolve())
    assert new_list == [str(out.resolve())]


def test_existing_outputs_skip_with_subfolder(tmp_path: Path):
    """全部输出在子目录时，批量函数应跳过而不启动 PureRAW。"""
    import dxo_pureraw

    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("Popen 不应被调用")

    monkey_stub = dxo_pureraw.subprocess
    real_popen = monkey_stub.Popen
    monkey_stub.Popen = _boom
    try:
        sub = tmp_path / "out"
        sub.mkdir()
        raw = tmp_path / "c.ARW"
        raw.write_bytes(b"x")
        (sub / "c-DxO.jpg").write_bytes(b"j")
        r = dxo_pureraw.run_pureraw_batch([str(raw)])
        assert called["n"] == 0
        assert r["ok"] == 1 and r["fail"] == 0
    finally:
        monkey_stub.Popen = real_popen


def test_ensure_jpeg_outputs_converts_and_removes_dng(tmp_path, monkeypatch):
    """DNG 输出兜底：渲染为 100% JPEG 并删除 DNG。"""
    import cv2
    import numpy as np

    import dxo_pureraw
    import image_io

    dng = tmp_path / "x-DxO.dng"
    dng.write_bytes(b"fake-dng")
    src = tmp_path / "x.ARW"
    src.write_bytes(b"raw")

    def _fake_render(path, half_size=False):
        return np.full((4, 4, 3), 128, dtype=np.uint8)

    monkeypatch.setattr(image_io, "read_raw_bgr_full", _fake_render)
    monkeypatch.setattr(
        image_io, "copy_exif_from_raw_to_jpeg", lambda a, b: True
    )
    monkeypatch.setattr(cv2, "imwrite", lambda p, a, params=None: True)

    out = dxo_pureraw._ensure_jpeg_outputs([str(dng)], [str(src)], print)
    assert out == [str((tmp_path / "x-DxO.jpg").resolve())]
    assert not dng.exists()


def test_map_sources_missing_output(tmp_path: Path):
    src = tmp_path / "b.ARW"
    src.write_bytes(b"x")
    assert map_sources_to_outputs([str(src)]) == {}
