# -*- coding: utf-8 -*-
"""从分类归档或筛选目录收集鸟图及物种、时间、坐标。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from record_submit.exif_read import is_image_path, read_datetime_original, read_gps_from_image

_INST_IN_NAME = re.compile(r"_inst\d+", re.I)


@dataclass
class BirdPhoto:
    path: str
    species_cn: str
    when: Optional[datetime]  # EXIF 原始拍摄时间（本地）
    lat: Optional[float]
    lon: Optional[float]
    ele: Optional[float] = None
    when_track: Optional[datetime] = None  # 与 GPX 对齐后的时间轴时刻


def _species_from_classification_path(root: Path, file_path: Path) -> str:
    rel = file_path.parent.relative_to(root)
    parts = [x for x in rel.parts if x not in (".",)]
    if len(parts) >= 4:
        return parts[3]
    if len(parts) >= 2:
        return parts[-1]
    if parts:
        return parts[-1]
    return "未知"


def _read_time(path: str) -> Optional[datetime]:
    t = read_datetime_original(path)
    if t is not None:
        return t
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def collect_bird_photos(
    folder: str,
    *,
    require_gps: bool = False,
) -> List[BirdPhoto]:
    """递归收集目录下图片；物种名来自 classification 相对路径层级。"""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        return []
    out: List[BirdPhoto] = []
    for dirpath, _dirs, files in os.walk(str(root)):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if not is_image_path(p):
                continue
            gps = read_gps_from_image(p)
            lat = lon = None
            if gps:
                lat, lon = gps
            if require_gps and (lat is None or lon is None):
                continue
            sp = _species_from_classification_path(root, Path(p))
            if sp in ("未知", "未知种", "未知属") or sp.startswith("未知"):
                continue
            out.append(
                BirdPhoto(
                    path=p,
                    species_cn=sp,
                    when=_read_time(p),
                    lat=lat,
                    lon=lon,
                )
            )
    out.sort(key=lambda x: x.when or datetime.min)
    return out
