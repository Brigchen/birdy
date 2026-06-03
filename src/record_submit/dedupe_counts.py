# -*- coding: utf-8 -*-
"""观鸟记录导出：同批次个体去重后再累加只数。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ImageObservation:
    """单张原图（或同一观测键）在该物种下的归档代表。"""

    img_key: str
    path: str
    dt: datetime
    lat: Optional[float]
    lon: Optional[float]
    count: int  # 该图内该物种归档文件数（≥1）


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _cluster_indices(n: int, can_merge) -> List[List[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if can_merge(i, j):
                union(i, j)
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def gps_points_are_precise(observations: Sequence[ImageObservation]) -> bool:
    """多点 GPS 不全相同 → 视为轨迹/精确坐标（非地名统写）。"""
    with_gps = [
        o for o in observations if o.lat is not None and o.lon is not None
    ]
    if len(with_gps) < 2:
        return len(with_gps) == 1
    uniq = {(round(o.lat, 5), round(o.lon, 5)) for o in with_gps}
    return len(uniq) > 1


def should_use_spatial_clustering(
    observations: Sequence[ImageObservation],
    *,
    prefer_spatial_gps: bool = False,
) -> bool:
    """
    有 GPS 且（主流程已 GPX 匹配 / 坐标存在差异）时用 0.1 km 空间聚类；
    否则用时间聚类（地名统写、无 GPS）。
    """
    with_gps = [
        o for o in observations if o.lat is not None and o.lon is not None
    ]
    if not with_gps:
        return False
    if prefer_spatial_gps:
        return True
    return gps_points_are_precise(observations)


def _first_photo_index(
    observations: Sequence[ImageObservation],
    indices: List[int],
) -> int:
    """同批次内按拍摄时间取首张（用于统一经纬度）。"""
    return min(indices, key=lambda i: (observations[i].dt, i))


def observation_location_clusters(
    observations: Sequence[ImageObservation],
    *,
    prefer_spatial_gps: bool = False,
    spatial_threshold_km: float = 0.1,
    time_threshold_minutes: float = 30.0,
) -> List[List[int]]:
    """按空间或时间返回观测下标分组（与 aggregate_species_count 规则一致）。"""
    n = len(observations)
    if n == 0:
        return []
    use_spatial = should_use_spatial_clustering(
        observations, prefer_spatial_gps=prefer_spatial_gps
    )
    time_sec = time_threshold_minutes * 60.0

    def can_merge(i: int, j: int) -> bool:
        a, b = observations[i], observations[j]
        dt_close = abs((a.dt - b.dt).total_seconds()) <= time_sec
        if use_spatial:
            if (
                a.lat is not None
                and a.lon is not None
                and b.lat is not None
                and b.lon is not None
                and haversine_km(a.lat, a.lon, b.lat, b.lon)
                <= spatial_threshold_km
            ):
                return True
            # 定点观鸟：GPS 沿轨迹微移仍视为同一个体，用时间窗合并
            return dt_close
        return dt_close

    return _cluster_indices(n, can_merge)


def apply_first_photo_coords_per_cluster(
    observations: Sequence[ImageObservation],
    *,
    prefer_spatial_gps: bool = False,
    spatial_threshold_km: float = 0.1,
    time_threshold_minutes: float = 30.0,
) -> List[ImageObservation]:
    """
    GPX 匹配后坐标会有微小差异；同一地理/时间批次统一为**首张**（最早拍摄时刻）的经纬度。
    """
    if not observations:
        return []
    out: List[ImageObservation] = [
        ImageObservation(
            img_key=o.img_key,
            path=o.path,
            dt=o.dt,
            lat=o.lat,
            lon=o.lon,
            count=o.count,
        )
        for o in observations
    ]
    for grp in observation_location_clusters(
        observations,
        prefer_spatial_gps=prefer_spatial_gps,
        spatial_threshold_km=spatial_threshold_km,
        time_threshold_minutes=time_threshold_minutes,
    ):
        lead = _first_photo_index(observations, grp)
        la, lo = observations[lead].lat, observations[lead].lon
        for i in grp:
            o = out[i]
            out[i] = ImageObservation(
                img_key=o.img_key,
                path=o.path,
                dt=o.dt,
                lat=la,
                lon=lo,
                count=o.count,
            )
    return out


def aggregate_species_count(
    observations: Sequence[ImageObservation],
    *,
    count_individuals: bool = True,
    use_spatial: bool = False,
    spatial_threshold_km: float = 0.1,
    time_threshold_minutes: float = 30.0,
) -> int:
    """
  同 checklist 内单物种：
    - 先按空间或时间合并为「同一批个体」；
    - 每批只保留该批内 count 最大的一张图；
    - 各批 count 相加。若 count_individuals 为 False，有观测则计 1。
    """
    if not observations:
        return 0
    if not count_individuals:
        return 1

    counts = [max(1, o.count) for o in observations]
    groups = observation_location_clusters(
        observations,
        prefer_spatial_gps=use_spatial,
        spatial_threshold_km=spatial_threshold_km,
        time_threshold_minutes=time_threshold_minutes,
    )
    return sum(max(counts[i] for i in grp) for grp in groups)
