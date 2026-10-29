# -*- coding: utf-8 -*-
"""RAW 写 GPS 后同步到 DxO 生成图（exiftool 集成测试，需本机装有 exiftool）。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geo_encoder import _sync_gps_to_dxo_outputs  # noqa: E402


def _exiftool():
    p = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Programs", "exiftool", "ExifTool.exe"
    )
    if os.path.isfile(p):
        return p
    return shutil.which("exiftool")


@pytest.mark.skipif(_exiftool() is None, reason="需要 exiftool")
def test_sync_gps_to_dxo_jpg(tmp_path: Path):
    import cv2
    import numpy as np

    raw = tmp_path / "DSC03277.ARW"
    raw.write_bytes(b"dummy")  # 仅用于词干匹配
    jpg = tmp_path / "DSC03277-DxO_DeepPRIME 3.jpg"
    cv2.imwrite(
        str(jpg), np.full((8, 8, 3), 200, dtype=np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    )
    n = _sync_gps_to_dxo_outputs(str(raw), 29.715, 121.987, 12.0, verbose=False)
    assert n == 1
    r = subprocess.run(
        [_exiftool(), "-GPSLatitude", "-n", str(jpg)],
        capture_output=True, text=True, timeout=30,
    )
    assert "29" in r.stdout


@pytest.mark.skipif(_exiftool() is None, reason="需要 exiftool")
def test_sync_skips_nonexistent_outputs(tmp_path: Path):
    raw = tmp_path / "lonely.ARW"
    raw.write_bytes(b"dummy")
    assert _sync_gps_to_dxo_outputs(str(raw), 1.0, 2.0, None) == 0
