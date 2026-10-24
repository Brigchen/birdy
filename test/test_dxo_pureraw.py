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
    src = tmp_path / "f27601408.ARW"
    src.write_bytes(b"x")
    cands = dxo_output_candidates(str(src))
    names = {c.name for c in cands}
    assert "f27601408-DxO.jpg" in names
    assert "f27601408-DxO.dng" in names


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


def test_map_sources_missing_output(tmp_path: Path):
    src = tmp_path / "b.ARW"
    src.write_bytes(b"x")
    assert map_sources_to_outputs([str(src)]) == {}
