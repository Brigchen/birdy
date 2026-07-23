#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_format import (  # noqa: E402
    DUAL_FORMAT_JPG_COPY_RAW,
    DUAL_FORMAT_JPG_ONLY,
    extensions_for_dual_mode,
    find_raw_companion,
    normalize_dual_format_mode,
    screened_raw_dir_for,
)


def test_normalize_modes():
    assert normalize_dual_format_mode("jpg_only") == DUAL_FORMAT_JPG_ONLY
    assert normalize_dual_format_mode("jpg_copy_raw") == DUAL_FORMAT_JPG_COPY_RAW
    assert normalize_dual_format_mode("") == "off"


def test_jpg_only_extensions():
    exts = extensions_for_dual_mode(DUAL_FORMAT_JPG_ONLY)
    assert ".jpg" in exts and ".cr2" not in exts


def test_screened_raw_dir():
    assert screened_raw_dir_for("/out/screened_x/Screened_images").endswith(
        "Screened_raw_images"
    )


def test_find_raw_companion(tmp_path):
    jpg = tmp_path / "DSC_0001.JPG"
    raw = tmp_path / "DSC_0001.CR2"
    jpg.write_bytes(b"x")
    raw.write_bytes(b"y")
    assert find_raw_companion(str(jpg)) == str(raw.resolve())


if __name__ == "__main__":
    test_normalize_modes()
    test_jpg_only_extensions()
    test_screened_raw_dir()
    print("ok")
