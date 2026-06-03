# -*- coding: utf-8 -*-
"""扫描 Birdy「分类归档」目录（目/科/属/种/文件 或 两级非鸟结构）。"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dedupe_counts import (
    ImageObservation,
    aggregate_species_count,
    apply_first_photo_coords_per_cluster,
    should_use_spatial_clustering,
)
from .exif_read import is_image_path, read_datetime_original, read_gps_from_image

_INST_IN_NAME = re.compile(r"_inst\d+", re.I)
_INST_TAG = re.compile(r"_inst(\d+)", re.I)
_ALL_IN_STEM = re.compile(r"_all$", re.I)
_SEQ_IN_NAME = re.compile(r"_\d{5}$")

_DEFAULT_SPATIAL_KM = 0.1
_DEFAULT_LOCATION_TIME_MIN = 30.0
_DEFAULT_INDIVIDUAL_TIME_MIN = 120.0
# 合并为单文件导出时，跨日同点仍视为同一次观鸟活动（个体去重）
_MERGED_EXPORT_INDIVIDUAL_TIME_MIN = 7 * 24 * 60.0


@dataclass
class TaxonPath:
    """从相对 classification 根的路径解析出的分类（与 Birdy 归档目录一致）。"""

    order_cn: str
    family_cn: str
    genus_cn: str
    species_cn: str
    rel_dir: str
    image_paths: List[str] = field(default_factory=list)

    @property
    def is_four_level(self) -> bool:
        return bool(self.genus_cn) and bool(self.species_cn)


@dataclass
class ChecklistBucket:
    """按 eBird 规则：单日 + 单点（经纬度四舍五入）聚合；合并导出时可跨多日。"""

    day: date
    lat: Optional[float]
    lon: Optional[float]
    species_counts: Dict[str, int] = field(default_factory=dict)
    sample_files: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    day_end: Optional[date] = None

    def key(self) -> Tuple[date, str, str]:
        la = f"{self.lat:.4f}" if self.lat is not None else ""
        lo = f"{self.lon:.4f}" if self.lon is not None else ""
        return (self.day, la, lo)

    @property
    def last_day(self) -> date:
        return self.day_end or self.day


@dataclass
class _ScanLine:
    """扫描到的单条归档图记录（写入 checklist 前）。"""

    path: str
    dt: datetime
    lat: Optional[float]
    lon: Optional[float]
    species_key: str
    img_key: str


def _image_observation_key(image_path: str) -> str:
    """
    同一原图键：EXIF 拍摄时刻（秒级）；无时刻时用去掉 inst/全局序号的文件名 stem。
    同窗内多张归档若时刻不同则视为不同原图，由上层按时间/空间批取 max 再累加。
    """
    dt = read_datetime_original(image_path)
    if dt is not None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    stem = Path(image_path).stem
    stem = _INST_IN_NAME.sub("", stem)
    stem = _SEQ_IN_NAME.sub("", stem)
    return stem or Path(image_path).name


def _register_archive_on_source_photo(inst_tags: set[int], image_path: str) -> None:
    """把单张归档图计入「该原图+该物种」的个体集合（不按文件数累加）。"""
    stem = Path(image_path).stem
    m = _INST_TAG.search(stem)
    if m:
        inst_tags.add(int(m.group(1)))
        return
    if _ALL_IN_STEM.search(stem):
        return


def _count_from_source_photo(inst_tags: set[int]) -> int:
    """原图上的物种只数：有 inst 标签时取不同 inst 个数，否则计 1。"""
    return len(inst_tags) if inst_tags else 1


def _parse_taxon_from_rel(rel: Path) -> Optional[TaxonPath]:
    parts = [x for x in rel.parts if x not in (".",)]
    if len(parts) == 2:
        order, leaf = parts[0], parts[1]
        return TaxonPath(order, leaf, "", leaf, str(Path(*parts[:2])))
    if len(parts) < 3:
        return None
    if len(parts) >= 4:
        order, family, genus, species = parts[0], parts[1], parts[2], parts[3]
        return TaxonPath(order, family, genus, species, str(Path(*parts[:4])))
    if len(parts) == 3:
        order, family, leaf = parts[0], parts[1], parts[2]
        return TaxonPath(order, family, "", leaf, str(Path(*parts[:3])))
    return None


def _assign_checklist_coords_by_day(
    lines: List[_ScanLine],
    *,
    prefer_spatial_gps: bool,
    spatial_threshold_km: float,
    time_threshold_minutes: float,
) -> List[_ScanLine]:
    """
    同一自然日内：按 0.1 km（精确 GPS/GPX）或 30 分钟合并为同一 checklist 地点，
    经纬度取该批次**首张**（最早拍摄时刻）相片上的值。
    """
    if not lines:
        return []
    by_day: Dict[date, List[int]] = defaultdict(list)
    for i, ln in enumerate(lines):
        by_day[ln.dt.date()].append(i)

    out = list(lines)
    for _day, idxs in by_day.items():
        obs = [
            ImageObservation(
                img_key=out[i].img_key,
                path=out[i].path,
                dt=out[i].dt,
                lat=out[i].lat,
                lon=out[i].lon,
                count=1,
            )
            for i in idxs
        ]
        normed = apply_first_photo_coords_per_cluster(
            obs,
            prefer_spatial_gps=prefer_spatial_gps,
            spatial_threshold_km=spatial_threshold_km,
            time_threshold_minutes=time_threshold_minutes,
        )
        for j, i in enumerate(idxs):
            ln = out[i]
            o = normed[j]
            out[i] = _ScanLine(
                path=ln.path,
                dt=ln.dt,
                lat=o.lat,
                lon=o.lon,
                species_key=ln.species_key,
                img_key=ln.img_key,
            )
    return out


def _collapse_buckets_for_export(
    buckets: Dict[Tuple[date, str, str], ChecklistBucket],
    per_bucket_species_obs: Dict[
        Tuple[date, str, str], Dict[str, Dict[str, ImageObservation]]
    ],
    *,
    count_individuals: bool,
    prefer_spatial_gps: bool,
    spatial_threshold_km: float,
    individual_time_threshold_minutes: float,
) -> Dict[Tuple[date, str, str], ChecklistBucket]:
    """
    一次导出合并为单个 checklist：连日、同点（给定地点）观鸟只生成一份 eBird/记录中心文件。
    物种只数在合并后的全量观测上重新按个体去重规则累计（避免分桶简单相加）。
    """
    if len(buckets) <= 1:
        return buckets

    merged_day = min(b.day for b in buckets.values())
    merged_day_end = max(b.last_day for b in buckets.values())
    merged_start: Optional[datetime] = None
    merged_end: Optional[datetime] = None
    merged_lat: Optional[float] = None
    merged_lon: Optional[float] = None
    samples: List[str] = []

    combined: Dict[str, List[ImageObservation]] = defaultdict(list)
    for bkey in buckets:
        for species_key, obs_by_key in per_bucket_species_obs.get(bkey, {}).items():
            for o in obs_by_key.values():
                combined[species_key].append(
                    ImageObservation(
                        img_key=o.img_key,
                        path=o.path,
                        dt=o.dt,
                        lat=o.lat,
                        lon=o.lon,
                        count=max(1, o.count),
                    )
                )

    for b in buckets.values():
        if merged_start is None or (
            b.start_time is not None and b.start_time < merged_start
        ):
            merged_start = b.start_time
            if b.lat is not None:
                merged_lat = b.lat
            if b.lon is not None:
                merged_lon = b.lon
        if b.end_time is not None and (
            merged_end is None or b.end_time > merged_end
        ):
            merged_end = b.end_time
        for p in b.sample_files:
            if len(samples) < 8 and p not in samples:
                samples.append(p)

    merge_time_min = max(
        individual_time_threshold_minutes, _MERGED_EXPORT_INDIVIDUAL_TIME_MIN
    )
    species_counts: Dict[str, int] = {}
    for species_key, observations in combined.items():
        use_spatial = should_use_spatial_clustering(
            observations,
            prefer_spatial_gps=prefer_spatial_gps,
        )
        species_counts[species_key] = aggregate_species_count(
            observations,
            count_individuals=count_individuals,
            use_spatial=use_spatial,
            spatial_threshold_km=spatial_threshold_km,
            time_threshold_minutes=merge_time_min,
        )

    la = round(merged_lat, 4) if merged_lat is not None else None
    lo = round(merged_lon, 4) if merged_lon is not None else None
    la_s = f"{la:.4f}" if la is not None else ""
    lo_s = f"{lo:.4f}" if lo is not None else ""
    bkey = (merged_day, la_s, lo_s)
    return {
        bkey: ChecklistBucket(
            day=merged_day,
            day_end=merged_day_end if merged_day_end != merged_day else None,
            lat=la,
            lon=lo,
            species_counts=species_counts,
            sample_files=samples,
            start_time=merged_start,
            end_time=merged_end,
        )
    }


def scan_classification_tree(
    classification_root: str,
    *,
    count_individuals: bool = True,
    prefer_spatial_gps: bool = False,
    spatial_threshold_km: float = _DEFAULT_SPATIAL_KM,
    time_threshold_minutes: float = _DEFAULT_LOCATION_TIME_MIN,
    individual_time_threshold_minutes: Optional[float] = None,
    merge_single_checklist: bool = True,
) -> Tuple[List[TaxonPath], Dict[Tuple[date, str, str], ChecklistBucket]]:
    """
    返回 (物种叶节点列表, 按日+坐标聚合的桶)。

    Checklist 地点：同日内在 0.1 km（GPX/精确 GPS）或 30 分钟内合并，
    导出坐标取该批次**首张**相片的经纬度。

    物种只数：每张原图按 inst 标签计个体（同拍摄时刻合并为一张原图）；
    默认 120 分钟内（或空间邻近）为一批，每批取各原图只数的最大值再累加。
    ``merge_single_checklist=True`` 时整次扫描合并为一份 eBird/记录中心导出文件。
    """
    ind_time = (
        individual_time_threshold_minutes
        if individual_time_threshold_minutes is not None
        else _DEFAULT_INDIVIDUAL_TIME_MIN
    )
    root = Path(classification_root).expanduser().resolve()
    if not root.is_dir():
        return [], {}

    leaves: List[TaxonPath] = []
    scan_lines: List[_ScanLine] = []

    for dirpath, _dirs, files in os.walk(str(root)):
        imgs = [
            os.path.join(dirpath, f)
            for f in files
            if is_image_path(os.path.join(dirpath, f))
        ]
        if not imgs:
            continue
        rel_dir = os.path.relpath(dirpath, str(root))
        if rel_dir == ".":
            continue
        tax = _parse_taxon_from_rel(Path(rel_dir))
        if tax is None:
            continue
        tax.image_paths = sorted(imgs)
        leaves.append(tax)

        species_key = tax.species_cn or tax.genus_cn or "未知"
        for p in imgs:
            dt = read_datetime_original(p)
            if dt is None:
                dt = datetime.now().replace(
                    hour=12, minute=0, second=0, microsecond=0
                )
            gps = read_gps_from_image(p)
            scan_lines.append(
                _ScanLine(
                    path=p,
                    dt=dt,
                    lat=gps[0] if gps else None,
                    lon=gps[1] if gps else None,
                    species_key=species_key,
                    img_key=_image_observation_key(p),
                )
            )

    scan_lines = _assign_checklist_coords_by_day(
        scan_lines,
        prefer_spatial_gps=prefer_spatial_gps,
        spatial_threshold_km=spatial_threshold_km,
        time_threshold_minutes=time_threshold_minutes,
    )

    buckets: Dict[Tuple[date, str, str], ChecklistBucket] = {}
    per_bucket_species_obs: Dict[
        Tuple[date, str, str], Dict[str, Dict[str, ImageObservation]]
    ] = defaultdict(lambda: defaultdict(dict))
    per_photo_inst: Dict[
        Tuple[date, str, str], Dict[str, Dict[str, set[int]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for ln in scan_lines:
        dday = ln.dt.date()
        la = round(ln.lat, 4) if ln.lat is not None else None
        lo = round(ln.lon, 4) if ln.lon is not None else None
        la_s = f"{la:.4f}" if la is not None else ""
        lo_s = f"{lo:.4f}" if lo is not None else ""
        bkey = (dday, la_s, lo_s)

        if bkey not in buckets:
            buckets[bkey] = ChecklistBucket(
                day=dday, lat=la, lon=lo, start_time=ln.dt
            )
        b = buckets[bkey]
        if b.start_time is None or ln.dt < b.start_time:
            b.start_time = ln.dt
        if b.end_time is None or ln.dt > b.end_time:
            b.end_time = ln.dt
        if len(b.sample_files) < 8:
            b.sample_files.append(ln.path)

        sp_obs = per_bucket_species_obs[bkey][ln.species_key]
        inst_tags = per_photo_inst[bkey][ln.species_key][ln.img_key]
        _register_archive_on_source_photo(inst_tags, ln.path)
        if ln.img_key not in sp_obs:
            sp_obs[ln.img_key] = ImageObservation(
                img_key=ln.img_key,
                path=ln.path,
                dt=ln.dt,
                lat=ln.lat,
                lon=ln.lon,
                count=1,
            )

    for bkey, b in buckets.items():
        sp_map = per_bucket_species_obs.get(bkey, {})
        inst_map = per_photo_inst.get(bkey, {})
        for species_key, obs_by_key in sp_map.items():
            species_inst = inst_map.get(species_key, {})
            observations = []
            for o in obs_by_key.values():
                cnt = _count_from_source_photo(
                    species_inst.get(o.img_key, set())
                )
                o.count = cnt
                observations.append(
                    ImageObservation(
                        img_key=o.img_key,
                        path=o.path,
                        dt=o.dt,
                        lat=o.lat,
                        lon=o.lon,
                        count=cnt,
                    )
                )
            use_spatial = should_use_spatial_clustering(
                observations,
                prefer_spatial_gps=prefer_spatial_gps,
            )
            b.species_counts[species_key] = aggregate_species_count(
                observations,
                count_individuals=count_individuals,
                use_spatial=use_spatial,
                spatial_threshold_km=spatial_threshold_km,
                time_threshold_minutes=ind_time,
            )

    if merge_single_checklist:
        buckets = _collapse_buckets_for_export(
            buckets,
            per_bucket_species_obs,
            count_individuals=count_individuals,
            prefer_spatial_gps=prefer_spatial_gps,
            spatial_threshold_km=spatial_threshold_km,
            individual_time_threshold_minutes=ind_time,
        )

    return leaves, buckets
