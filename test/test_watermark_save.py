# -*- coding: utf-8 -*-
"""水印输出：接近无损保存（JPEG 100/4:4:4 或 PNG）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watermark_generator import (  # noqa: E402
    generate_watermarks,
    save_watermarked_image,
    unique_watermark_dest,
    watermark_output_suffix,
    WatermarkOptions,
)


def _jpeg_chroma_sampling(path: Path):
    data = path.read_bytes()
    i = 0
    n = len(data)
    while i < n - 10:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            ncomp = data[i + 9]
            factors = []
            p = i + 10
            for _ in range(ncomp):
                samp = data[p + 1]
                factors.append((samp >> 4, samp & 0x0F))
                p += 3
            return factors
        if marker in (0xD8, 0xD9, 0x01) or (0xD0 <= marker <= 0xD7):
            i += 2
            continue
        if i + 3 >= n:
            break
        seglen = (data[i + 2] << 8) + data[i + 3]
        i += 2 + seglen
    return None


def test_watermark_output_suffix_png_and_jpeg():
    assert watermark_output_suffix("a.png") == ".png"
    assert watermark_output_suffix("a.TIFF") == ".png"
    assert watermark_output_suffix("a.jpg") == ".jpg"
    assert watermark_output_suffix("a.ARW") == ".jpg"


def test_unique_watermark_dest_adds_index(tmp_path: Path):
    src = tmp_path / "src" / "bird.jpg"
    src.parent.mkdir()
    src.write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    first = unique_watermark_dest(str(out_dir), str(src))
    Path(first).write_bytes(b"y")
    second = unique_watermark_dest(str(out_dir), str(src))
    assert Path(first).name == "bird.jpg"
    assert Path(second).name == "bird_1.jpg"


def test_save_png_is_lossless(tmp_path: Path):
    img = Image.new("RGB", (24, 18))
    px = img.load()
    for y in range(18):
        for x in range(24):
            px[x, y] = (x * 10, y * 12, (x + y) * 7)
    src = tmp_path / "src.png"
    img.save(src)
    dst = tmp_path / "out.png"
    save_watermarked_image(img, str(dst), str(src))
    got = np.asarray(Image.open(dst).convert("RGB"))
    assert np.array_equal(got, np.asarray(img))


def test_save_jpeg_quality100_444_roundtrip(tmp_path: Path):
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (48, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    src = tmp_path / "src.jpg"
    img.save(src, "JPEG", quality=95)
    dst = tmp_path / "out.jpg"
    save_watermarked_image(img, str(dst), str(src))
    factors = _jpeg_chroma_sampling(dst)
    assert factors is not None
    assert all(h == 1 and v == 1 for h, v in factors)
    q95 = tmp_path / "q95.jpg"
    img.save(q95, "JPEG", quality=95)
    assert dst.stat().st_size >= q95.stat().st_size
    got = np.asarray(Image.open(dst).convert("RGB"), dtype=np.int16)
    mae = np.abs(got - arr.astype(np.int16)).mean()
    assert mae < 1.0


def test_generate_watermarks_writes_near_lossless_jpeg(tmp_path: Path):
    src_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    img = Image.new("RGB", (80, 60), (40, 80, 120))
    img.save(src_dir / "one.jpg", "JPEG", quality=90)
    r = generate_watermarks(
        str(src_dir),
        str(out_dir),
        WatermarkOptions(
            enable_location=False,
            enable_date=False,
            enable_species=False,
            enable_camera_params=False,
            enable_auto_enhance=False,
            watermark_style="inline",
        ),
        prefer_folder_name_as_species=False,
    )
    assert r["ok"] == 1
    dest = out_dir / "one.jpg"
    assert dest.is_file()
    factors = _jpeg_chroma_sampling(dest)
    assert factors is not None
    assert all(h == 1 and v == 1 for h, v in factors)
