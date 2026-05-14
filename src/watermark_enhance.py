# -*- coding: utf-8 -*-
"""
水印前自动生态显影 + 逐张微调参数（临时 JSON）。

按「水印输入目录」绝对路径哈希划分临时文件，避免不同工程互相覆盖。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from ecology_jpeg_develop import develop_bgr_ecology_wildlife

_MAX_DEVELOP_EDGE = 3200


def rel_key(src_abs: str, source_folder: str) -> str:
    """相对 source_folder 的稳定键（POSIX）。"""
    sf = os.path.abspath(source_folder)
    ap = os.path.abspath(src_abs)
    try:
        return os.path.relpath(ap, sf).replace("\\", "/")
    except ValueError:
        return os.path.basename(ap)


def override_store_path(source_folder: str) -> str:
    """本目录专用的逐张微调 JSON 路径（位于系统临时目录）。"""
    root = os.path.normcase(os.path.abspath(source_folder))
    h = hashlib.sha256(root.encode("utf-8", errors="ignore")).hexdigest()[:24]
    d = Path(tempfile.gettempdir()) / "birdy_wm_enhance" / h
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "overrides.json")


def load_overrides_json(path: str) -> Dict[str, dict]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        br = data.get("by_relpath")
        if not isinstance(br, dict):
            return {}
        out: Dict[str, dict] = {}
        for k, v in br.items():
            if isinstance(v, dict):
                out[str(k)] = {
                    "strength": float(v.get("strength", 1.0)),
                    "exposure_fine": float(v.get("exposure_fine", 0.0)),
                }
        return out
    except Exception:
        return {}


def save_overrides_json(path: str, source_root: str, by_rel: Dict[str, dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        "version": 1,
        "source_root": os.path.abspath(source_root),
        "by_relpath": {
            k: {
                "strength": float(max(0.0, min(1.0, v.get("strength", 1.0)))),
                "exposure_fine": float(
                    max(-0.22, min(0.22, v.get("exposure_fine", 0.0)))
                ),
            }
            for k, v in by_rel.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def apply_auto_enhance_pil(img: Image.Image, ov: Optional[dict] = None) -> Image.Image:
    """
    对 RGB PIL 图做与 RAW 生态显影同一套逻辑的自动增强，再按 ov 混合原图与曝光微调。
    ov: strength 0~1（自动结果占比），exposure_fine 约 -0.22~0.22（V 通道缩放）。
    """
    if img is None:
        return img
    ov = ov or {}
    s = float(np.clip(float(ov.get("strength", 1.0)), 0.0, 1.0))
    ef = float(np.clip(float(ov.get("exposure_fine", 0.0)), -0.22, 0.22))

    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m > _MAX_DEVELOP_EDGE:
        sc = _MAX_DEVELOP_EDGE / float(m)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        auto = develop_bgr_ecology_wildlife(small)
        auto = cv2.resize(auto, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        auto = develop_bgr_ecology_wildlife(bgr)

    blended = (
        s * auto.astype(np.float32) + (1.0 - s) * bgr.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    if abs(ef) > 1e-5:
        hsv = cv2.cvtColor(blended, cv2.COLOR_BGR2HSV)
        hh, ss, vv = cv2.split(hsv)
        vv = np.clip(vv.astype(np.float32) * (1.0 + ef), 0, 255).astype(np.uint8)
        blended = cv2.cvtColor(cv2.merge((hh, ss, vv)), cv2.COLOR_HSV2BGR)
    rgb2 = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb2)
