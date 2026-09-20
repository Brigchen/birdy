# -*- coding: utf-8 -*-
"""清晰度打分工具：HTML 报告与只读评分（不加载 YOLO）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "clarity-meter"
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))

from clarity_meter.report import default_report_path, write_html_report  # noqa: E402
from clarity_meter.score import (  # noqa: E402
    preview_with_mask,
    score_folder,
    score_one,
)


class _EmptyDet:
    def detect(self, bgr):
        return []


def test_score_one_does_not_delete(tmp_path: Path):
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[:, ::4] = 200
    p = tmp_path / "a.jpg"
    cv2.imwrite(str(p), img)
    row = score_one(str(p), _EmptyDet(), min_clarity=35)
    assert p.is_file()
    assert row["width"] == 120 and row["height"] == 80
    assert row["n_birds"] == 0
    assert row["mask_score"] is not None
    assert row["verdict"] == "未检出鸟体"
    assert str(row.get("preview_uri") or "").startswith("data:image/jpeg")


def test_preview_draws_red_mask_outline():
    img = np.full((80, 80, 3), 40, dtype=np.uint8)
    birds = [
        {
            "bbox": [20, 20, 60, 60],
            "mask_xy": [[20, 20], [60, 20], [60, 60], [20, 60]],
        }
    ]
    vis = preview_with_mask(img, birds, max_side=80)
    assert vis[:, :, 2].max() > vis[:, :, 0].max()


def test_write_html_report_table(tmp_path: Path):
    result = {
        "root": str(tmp_path),
        "min_clarity": 35,
        "use_full_frame": False,
        "total": 2,
        "n_pass": 1,
        "n_blur": 1,
        "n_nobird": 0,
        "n_fail": 0,
        "rows": [
            {
                "name": "sharp.jpg",
                "width": 60,
                "height": 40,
                "n_birds": 1,
                "bbox_score": 50.0,
                "mask_score": 48.0,
                "verdict": "通过",
                "preview_uri": "data:image/jpeg;base64,AAAA",
            },
            {
                "name": "blur.jpg",
                "width": 60,
                "height": 40,
                "n_birds": 1,
                "bbox_score": 20.0,
                "mask_score": 18.0,
                "verdict": "模糊",
                "preview_uri": "data:image/jpeg;base64,BBBB",
            },
        ],
    }
    out = tmp_path / "clarity_report.html"
    write_html_report(result, str(out))
    html = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64,AAAA" in html
    assert "href=\"sharp.jpg\"" in html and "href=\"blur.jpg\"" in html
    assert "48.00" in html and "18.00" in html
    assert "通过" in html and "模糊" in html
    assert "未删除任何文件" in html


def test_score_folder_keeps_files(tmp_path: Path):
    img = np.full((48, 64, 3), 80, dtype=np.uint8)
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), img)
    res = score_folder(str(tmp_path), min_clarity=35, detector=_EmptyDet())
    assert p.is_file()
    assert res["total"] == 1
    assert res["rows"][0]["verdict"] == "未检出鸟体"
    html_path = default_report_path(str(tmp_path))
    write_html_report(res, str(html_path))
    assert html_path.parent == tmp_path
    text = html_path.read_text(encoding="utf-8")
    assert "x.jpg" in text
    assert "data:image/jpeg" in text
