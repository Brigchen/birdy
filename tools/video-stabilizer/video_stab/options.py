# -*- coding: utf-8 -*-
"""稳定化任务参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StabilizeOptions:
    """两种主模式：tripod（三脚架固定视角）、track（追踪稳定器）。"""

    mode: str = "tripod"  # tripod | track
    # tripod 子引擎: optical（OpenCV 光流钉首帧，默认）| ffmpeg（libvidstab opttripod）
    tripod_engine: str = "optical"
    # track 子引擎: yolo（鸟检+跟踪，推荐）| csrt（中心框 CSRT，无模型时回退）
    track_engine: str = "yolo"

    start_time: float = 0.0
    end_time: float = -1.0  # -1 = 到结尾

    smoothing_window: int = 7  # 追踪平滑（帧）。过长会让飞得快的鸟滞后晃动
    margin_ratio: float = 0.12  # 兼容旧参数；三脚架裁切改由全程晃动决定
    track_margin_ratio: float = 0.15  # 追踪模式裁边余量

    keep_audio: bool = True
    bird_model: Optional[str] = None  # 默认自动找仓库 models/bird-seg.pt
    init_bbox: Optional[Tuple[int, int, int, int]] = None  # 像素框 x,y,w,h
    init_bbox_norm: Optional[Tuple[float, float, float, float]] = None  # 鸟体相对框 0–1
    view_norm: Optional[Tuple[float, float, float, float]] = None  # 输出画面相对框 0–1

    output_fps: float = -1.0
