# -*- coding: utf-8 -*-
"""单文件/批量稳定化流水线。"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2

from . import ffmpeg_util
from .media import crop_video_rect, fit_even_rect, norm_to_pixels
from .options import StabilizeOptions
from .paths import default_bird_seg_model, default_output_dir
from .track import run_track_stabilizer
from .tripod import run_virtual_tripod

LogFn = Optional[Callable[[str], None]]
ProgressFn = Optional[Callable[[int, int], None]]
CancelFn = Optional[Callable[[], bool]]


def _log(msg: str, log: LogFn) -> None:
    if log:
        log(msg)
    else:
        print(msg, flush=True)


def mode_tag(mode: str) -> str:
    return "tripod" if mode == "tripod" else "track"


def suggest_output_path(input_path: str, mode: str) -> str:
    p = Path(input_path)
    suffix = p.suffix if p.suffix else ".mp4"
    return str(default_output_dir() / f"{p.stem}_{mode_tag(mode)}{suffix}")


def suggest_output_dir(input_dir: str) -> str:
    _ = input_dir
    return str(default_output_dir())


def output_path_for(input_path: str, output_dir: str, mode: str) -> str:
    p = Path(input_path)
    suffix = ".mp4" if p.suffix.lower() not in {".mp4", ".mov", ".mkv", ".m4v"} else p.suffix
    return str(Path(output_dir) / f"{p.stem}_{mode_tag(mode)}{suffix}")


def stabilize_file(
    input_path: str,
    output_path: str,
    options: Optional[StabilizeOptions] = None,
    *,
    log: LogFn = None,
    progress: ProgressFn = None,
    should_cancel: CancelFn = None,
) -> str:
    opts = options or StabilizeOptions()
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    if opts.mode == "track" and opts.init_bbox is None and opts.init_bbox_norm:
        opts = replace(
            opts,
            init_bbox=norm_to_pixels(opts.init_bbox_norm, width, height),
        )

    _log(f"输入: {Path(input_path).name}  {duration:.1f}s", log)

    temp_dir = tempfile.mkdtemp(prefix="video_stab_")
    temp_files: list[str] = []
    try:
        source = input_path
        need_trim = opts.start_time > 0.05 or (
            opts.end_time > 0 and duration > 0 and opts.end_time < duration - 0.05
        )
        if need_trim and ffmpeg_util.find_ffmpeg():
            trimmed = os.path.join(temp_dir, "trimmed.mp4")
            end = opts.end_time if opts.end_time > 0 else duration + 1.0
            if ffmpeg_util.trim_segment(
                input_path, trimmed, opts.start_time, end, log=log
            ):
                source = trimmed
                temp_files.append(trimmed)
            else:
                _log("时间截取失败，处理全片", log)

        work_video = os.path.join(temp_dir, "stabilized_no_audio.mp4")
        temp_files.append(work_video)

        if should_cancel and should_cancel():
            raise InterruptedError("已取消")

        if opts.mode == "tripod":
            _log(f"模式: 三脚架固定视角 ({opts.tripod_engine})", log)
            tripod_source = source
            if opts.view_norm:
                cap_sz = cv2.VideoCapture(source)
                vw = int(cap_sz.get(cv2.CAP_PROP_FRAME_WIDTH))
                vh = int(cap_sz.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_sz.release()
                rect = fit_even_rect(opts.view_norm, vw, vh)
                cropped_path = os.path.join(temp_dir, "view_crop.mp4")
                crop_video_rect(source, cropped_path, rect, log=log)
                temp_files.append(cropped_path)
                tripod_source = cropped_path
            wrote_mpeg4 = True
            if opts.tripod_engine == "ffmpeg" and ffmpeg_util.find_ffmpeg():
                ok = ffmpeg_util.stabilize_ffmpeg_tripod(
                    tripod_source, work_video, smoothing=0, log=log, work_dir=temp_dir
                )
                if ok:
                    wrote_mpeg4 = False
                else:
                    _log("ffmpeg 三脚架失败，回退光流算法", log)
                    run_virtual_tripod(
                        tripod_source,
                        work_video,
                        margin_ratio=opts.margin_ratio,
                        progress=progress,
                        log=log,
                    )
            else:
                run_virtual_tripod(
                    tripod_source,
                    work_video,
                    margin_ratio=opts.margin_ratio,
                    progress=progress,
                    log=log,
                )
            if wrote_mpeg4:
                h264 = os.path.join(temp_dir, "stabilized_h264.mp4")
                if ffmpeg_util.reencode_h264(work_video, h264, log=log):
                    temp_files.append(h264)
                    work_video = h264
        elif opts.mode == "track":
            model = opts.bird_model
            if not model:
                dm = default_bird_seg_model()
                model = str(dm) if dm.is_file() else None
            engine = opts.track_engine
            if engine == "yolo" and not (model and os.path.isfile(model)):
                _log("未找到 bird-seg 模型，回退框跟踪", log)
                engine = "csrt"
            _log(f"模式: 追踪稳定器 ({engine})", log)
            run_track_stabilizer(
                source,
                work_video,
                engine=engine,
                bird_model=model,
                init_bbox=opts.init_bbox,
                smoothing_window=opts.smoothing_window,
                margin_ratio=opts.track_margin_ratio,
                view_norm=opts.view_norm,
                progress=progress,
                log=log,
            )
        else:
            raise ValueError(f"未知模式: {opts.mode}")

        if should_cancel and should_cancel():
            raise InterruptedError("已取消")

        final_no_audio = work_video
        if opts.keep_audio and ffmpeg_util.find_ffmpeg():
            merged = os.path.join(temp_dir, "with_audio.mp4")
            audio_src = source if os.path.isfile(source) else input_path
            if ffmpeg_util.merge_audio(work_video, audio_src, merged, log=log):
                final_no_audio = merged
                temp_files.append(merged)
            else:
                _log("音轨合并失败，输出无声视频", log)

        shutil.copy2(final_no_audio, output_path)
        _log(f"完成: {output_path}", log)
        return output_path
    finally:
        for f in temp_files:
            try:
                if os.path.isfile(f):
                    os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def stabilize_paths(
    input_paths: Sequence[str],
    output_dir: str,
    options: Optional[StabilizeOptions] = None,
    *,
    log: LogFn = None,
    progress: ProgressFn = None,
    should_cancel: CancelFn = None,
) -> Tuple[List[str], List[str]]:
    """按顺序处理多个视频。返回 (成功路径, 失败说明)。"""
    opts = options or StabilizeOptions()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(Path(p).resolve()) for p in input_paths]
    if not paths:
        raise FileNotFoundError("没有可处理的视频")

    done: List[str] = []
    errors: List[str] = []
    n = len(paths)
    for i, src in enumerate(paths):
        if should_cancel and should_cancel():
            raise InterruptedError("已取消")
        name = Path(src).name
        _log(f"[{i + 1}/{n}] {name}", log)
        dest = output_path_for(src, str(out_dir), opts.mode)

        def _prog(d: int, t: int, index: int = i) -> None:
            if not progress:
                return
            inner = (d / t) if t else 0.0
            progress(int((index + inner) * 1000), n * 1000)

        try:
            stabilize_file(
                src,
                dest,
                opts,
                log=log,
                progress=_prog,
                should_cancel=should_cancel,
            )
            done.append(dest)
        except InterruptedError:
            raise
        except Exception as e:
            msg = f"{name}: {e}"
            errors.append(msg)
            _log(f"失败 {msg}", log)
    _log(f"目录完成: 成功 {len(done)}，失败 {len(errors)}", log)
    return done, errors
