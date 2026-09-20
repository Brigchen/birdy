# -*- coding: utf-8 -*-
"""按镜头/焦距/ISO 分组，以及同场景时间/子目录聚类。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .exif_meta import group_label


def selected_group_keys(
    *,
    lens: bool = True,
    focal: bool = True,
    iso: bool = False,
    aperture: bool = False,
    camera: bool = False,
) -> Tuple[str, ...]:
    keys: List[str] = []
    if lens:
        keys.append("lens")
    if focal:
        keys.append("focal")
    if iso:
        keys.append("iso")
    if aperture:
        keys.append("aperture")
    if camera:
        keys.append("camera")
    return tuple(keys) if keys else ("lens",)


def attach_group(rows: List[Dict[str, Any]], keys: Sequence[str]) -> None:
    k = tuple(keys)
    for r in rows:
        meta = r.get("meta") or r
        r["group"] = group_label(meta, k)


def summarize_groups(
    rows: List[Dict[str, Any]],
    metric: str = "mtf50_cy_px",
    extra: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    extra_buckets: Dict[str, Dict[str, List[float]]] = {
        k: defaultdict(list) for k in extra
    }
    counts: Dict[str, int] = defaultdict(int)

    def _take(row: Dict[str, Any], key: str) -> Optional[float]:
        v = row.get(key)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return fv if np.isfinite(fv) else None

    for r in rows:
        g = str(r.get("group") or "全部")
        counts[g] += 1
        fv = _take(r, metric)
        if fv is not None:
            buckets[g].append(fv)
        for k in extra:
            ev = _take(r, k)
            if ev is not None:
                extra_buckets[k][g].append(ev)
    out = []
    for g, n in counts.items():
        vals = buckets.get(g) or []
        arr = np.array(vals, dtype=np.float64) if vals else np.array([])
        rec: Dict[str, Any] = {
            "group": g,
            "n": n,
            "n_valid": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "std": float(arr.std(ddof=0)) if arr.size else None,
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
        }
        for k in extra:
            earr = extra_buckets[k].get(g) or []
            rec[k] = float(np.median(earr)) if earr else None
        out.append(rec)
    out.sort(key=lambda d: (-(d["median"] or -1.0), d["group"]))
    return out


def cluster_scenes_by_time(
    rows: List[Dict[str, Any]], gap_sec: float = 180.0
) -> List[List[Dict[str, Any]]]:
    dated = []
    undated = []
    for r in rows:
        ts = (r.get("meta") or {}).get("datetime_ts")
        if ts is None:
            undated.append(r)
        else:
            dated.append((float(ts), r))
    dated.sort(key=lambda x: x[0])
    clusters: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    last_ts = None
    for ts, r in dated:
        if last_ts is None or (ts - last_ts) <= float(gap_sec):
            cur.append(r)
        else:
            if cur:
                clusters.append(cur)
            cur = [r]
        last_ts = ts
    if cur:
        clusters.append(cur)
    for r in undated:
        clusters.append([r])
    return clusters


def cluster_scenes_by_folder(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    from pathlib import Path

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        p = str((r.get("meta") or {}).get("path") or r.get("path") or "")
        parent = str(Path(p).parent) if p else ""
        buckets[parent].append(r)
    return [buckets[k] for k in sorted(buckets.keys())]


def cluster_scenes_by_lens(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """按 EXIF 镜头型号划分（同型号为一组）。"""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        lens = str((r.get("meta") or {}).get("lens") or "").strip() or "未知镜头"
        buckets[lens].append(r)
    return [buckets[k] for k in sorted(buckets.keys())]


def cluster_label(cluster: List[Dict[str, Any]], mode: str, index: int) -> str:
    n = len(cluster)
    if mode == "lens":
        lens = "未知镜头"
        if cluster:
            lens = str((cluster[0].get("meta") or {}).get("lens") or "").strip() or "未知镜头"
        return f"{lens}（{n}张）"
    if mode == "folder":
        from pathlib import Path

        parent = ""
        if cluster:
            p = str((cluster[0].get("meta") or {}).get("path") or "")
            parent = Path(p).parent.name if p else ""
        return f"{parent or f'场景{index:02d}'}（{n}张）"
    return f"场景{index:02d}（{n}张）"
