# -*- coding: utf-8 -*-
"""鸟体检测：imgsz=1280 + 中心复检的框合并与触发条件。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detect_bird_and_eye import (  # noqa: E402
    box_iou_xyxy,
    center_crop_origin_size,
    merge_detections_nms,
    run_yolo_birds_full_and_center,
    should_run_center_pass,
)


def test_box_iou_and_nms_keeps_higher_conf():
    a = {"bbox": [0, 0, 100, 100], "conf": 0.9}
    b = {"bbox": [10, 10, 110, 110], "conf": 0.4}
    c = {"bbox": [300, 300, 340, 340], "conf": 0.6}
    assert box_iou_xyxy(a["bbox"], b["bbox"]) > 0.5
    kept = merge_detections_nms([a, b, c], iou_thr=0.5)
    assert len(kept) == 2
    assert kept[0]["conf"] == 0.9
    assert any(k["conf"] == 0.6 for k in kept)


def test_center_crop_is_middle_half():
    x0, y0, cw, ch = center_crop_origin_size(1000, 800, 0.5)
    assert (cw, ch) == (500, 400)
    assert (x0, y0) == (250, 200)


def test_center_pass_only_on_large_images():
    assert should_run_center_pass(9504, 6336) is True
    assert should_run_center_pass(5001, 3000) is True
    assert should_run_center_pass(5000, 4000) is False
    assert should_run_center_pass(800, 600) is False


class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.conf = [np.array([conf], dtype=np.float32)]
        self.cls = [np.array([cls], dtype=np.float32)]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes
        self.masks = None
        self.names = {0: "bird"}


class _FakeYolo:
    def __init__(self):
        self.shapes = []

    def predict(self, source, conf, imgsz, verbose, half):
        self.shapes.append(tuple(source.shape[:2]))
        h, w = source.shape[:2]
        box = _Box([w * 0.4, h * 0.4, w * 0.6, h * 0.6], 0.8, 0)
        return [_Result([box])]


def test_large_image_runs_full_then_center(monkeypatch):
    model = _FakeYolo()
    img = np.zeros((4000, 6000, 3), dtype=np.uint8)
    birds = run_yolo_birds_full_and_center(model, img, conf=0.25)
    assert model.shapes[0] == (4000, 6000)
    assert model.shapes[1] == (2000, 3000)
    assert len(birds) >= 1
    x1, y1, x2, y2 = birds[0]["bbox"]
    assert 0 <= x1 < x2 <= 6000
    assert 0 <= y1 < y2 <= 4000


def test_small_image_skips_center_pass():
    model = _FakeYolo()
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    run_yolo_birds_full_and_center(model, img, conf=0.25)
    assert model.shapes == [(600, 800)]
    model2 = _FakeYolo()
    img2 = np.zeros((3000, 5000, 3), dtype=np.uint8)
    run_yolo_birds_full_and_center(model2, img2, conf=0.25)
    assert model2.shapes == [(3000, 5000)]
