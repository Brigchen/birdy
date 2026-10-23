# -*- coding: utf-8 -*-
"""从分类归档或筛选目录收集鸟图及物种、时间、坐标。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    order_cn: str = ""  # 目（来自分类目录层级或名录）
    family_cn: str = ""  # 科


def _taxonomy_from_classification_path(
    root: Path, file_path: Path
) -> Tuple[str, str, str]:
    """从 classification 相对路径取 (目, 科, 种)；目录为 目/科/属/种 层级。"""
    try:
        rel = file_path.parent.relative_to(root)
    except ValueError:
        return "", "", "未知"
    parts = [x for x in rel.parts if x not in (".",)]
    if len(parts) >= 4:
        return parts[0], parts[1], parts[3]
    sp = parts[-1] if parts else "未知"
    return "", "", sp


def _species_from_classification_path(root: Path, file_path: Path) -> str:
    return _taxonomy_from_classification_path(root, file_path)[2]


# ---------------------------------------------------------------------------
# 分类学排序：目 → 科 → 种（按 data/species/bird_classification.json 名录序号），
# 同级按拍摄时间。名录缺失时退化为按目录/名称分组，仍保证同目相邻。
# ---------------------------------------------------------------------------

_TAXO_TABLES: Optional[
    Tuple[Dict[str, int], Dict[str, int], Dict[str, Tuple[str, str, int]]]
] = None


def _load_taxo_tables() -> Tuple[
    Dict[str, int], Dict[str, int], Dict[str, Tuple[str, str, int]]
]:
    """目→序号、科→序号、种→(目, 科, 序号)；读名录失败时返回空表。"""
    global _TAXO_TABLES
    if _TAXO_TABLES is not None:
        return _TAXO_TABLES
    orders: Dict[str, int] = {}
    families: Dict[str, int] = {}
    species: Dict[str, Tuple[str, str, int]] = {}
    try:
        root = Path(__file__).resolve().parents[2]
        path = root / "data" / "species" / "bird_classification.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for i, (name_cn, info) in enumerate(data.items()):
                    if not isinstance(info, dict):
                        continue
                    o = str(info.get("order_cn") or "")
                    f = str(info.get("family_cn") or "")
                    species[str(name_cn)] = (o, f, i)
                    if o:
                        orders.setdefault(o, i)
                    if f:
                        families.setdefault(f, i)
    except Exception:
        orders, families, species = {}, {}, {}
    _TAXO_TABLES = (orders, families, species)
    return _TAXO_TABLES


def photo_taxo_sort_key(ph: BirdPhoto) -> Tuple:
    """分类学排序键：目 → 科 → 种（名录序号），再按拍摄时间、路径。

    用于轨迹图同一记录点矩阵内鸟图的阅读顺序：同目/同科物种相邻。
    """
    orders, families, species = _load_taxo_tables()
    meta = species.get(ph.species_cn)
    order_name = ph.order_cn or (meta[0] if meta else "")
    family_name = ph.family_cn or (meta[1] if meta else "")
    o_rank = orders.get(order_name)
    f_rank = families.get(family_name)
    return (
        (o_rank if o_rank is not None else 10**6, order_name),
        (f_rank if f_rank is not None else 10**6, family_name),
        (meta[2] if meta is not None else 10**6, ph.species_cn),
        ph.when or datetime.min,
        ph.path,
    )


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
    stats: Optional[Dict[str, object]] = None,
) -> List[BirdPhoto]:
    """递归收集目录下图片；物种名来自 classification 相对路径层级。"""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        if stats is not None:
            stats["scanned"] = 0
            stats["excluded"] = []
        return []
    out: List[BirdPhoto] = []
    excluded: List[Tuple[str, str]] = []
    scanned = 0
    for dirpath, _dirs, files in os.walk(str(root)):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if not is_image_path(p):
                continue
            scanned += 1
            gps = read_gps_from_image(p)
            lat = lon = None
            if gps:
                lat, lon = gps
            if require_gps and (lat is None or lon is None):
                excluded.append((p, "缺少 EXIF GPS"))
                continue
            order_cn, family_cn, sp = _taxonomy_from_classification_path(
                root, Path(p)
            )
            if sp in ("未知", "未知种", "未知属") or sp.startswith("未知"):
                excluded.append(
                    (p, f"物种目录为「{sp}」，未知物种不纳入地图")
                )
                continue
            out.append(
                BirdPhoto(
                    path=p,
                    species_cn=sp,
                    when=_read_time(p),
                    lat=lat,
                    lon=lon,
                    order_cn=order_cn,
                    family_cn=family_cn,
                )
            )
    out.sort(key=lambda x: x.when or datetime.min)
    if stats is not None:
        stats["scanned"] = scanned
        stats["excluded"] = excluded
    return out
