# -*- coding: utf-8 -*-
"""GPX 轨迹解析、照片时间匹配、观鸟行迹/物种分布图生成。"""

from .gpx_io import GpxPoint, load_gpx, load_gpx_many, merge_gpx_files, resolve_gpx_path_list
from .gpx_match import (
    batch_write_gps_from_gpx,
    interpolate_track_at,
    match_photos_to_track,
)
from .photo_collect import collect_bird_photos
from .preview_dialog import show_track_map_preview, TrackMapPreviewPanel
from .track_map import generate_track_maps

__all__ = [
    "GpxPoint",
    "load_gpx",
    "load_gpx_many",
    "merge_gpx_files",
    "resolve_gpx_path_list",
    "interpolate_track_at",
    "match_photos_to_track",
    "batch_write_gps_from_gpx",
    "collect_bird_photos",
    "generate_track_maps",
    "show_track_map_preview",
    "TrackMapPreviewPanel",
]
