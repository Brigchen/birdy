# -*- coding: utf-8 -*-
"""ffmpeg 查找、截取、音视频合并、libvidstab 两 pass。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .paths import bundled_ffmpeg

LogFn = Optional[Callable[[str], None]]


def _log(msg: str, log: LogFn) -> None:
    if log:
        log(msg)
    else:
        print(msg, flush=True)


def find_ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    bf = bundled_ffmpeg()
    if bf.is_file():
        return str(bf)
    for path in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ):
        if os.path.isfile(path):
            return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_ffmpeg(cmd: List[str], timeout: int = 7200) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        return result.returncode, stdout, stderr
    except Exception as e:
        return -1, "", str(e)


def trim_segment(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    *,
    log: LogFn = None,
) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        _log("未找到 ffmpeg，无法截取片段", log)
        return False
    cmd = [ffmpeg, "-y", "-i", input_path]
    if start_time > 0:
        cmd.extend(["-ss", f"{start_time:.3f}"])
    if end_time > start_time:
        cmd.extend(["-t", f"{end_time - start_time:.3f}"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )
    _log(f"ffmpeg 截取: {Path(input_path).name}", log)
    code, _, stderr = run_ffmpeg(cmd)
    if code != 0:
        _log(f"ffmpeg 截取失败: {stderr[-400:] if stderr else code}", log)
    return os.path.isfile(output_path) and os.path.getsize(output_path) > 0


def reencode_h264(
    src: str,
    dst: str,
    *,
    crf: int = 16,
    log: LogFn = None,
) -> bool:
    """把 OpenCV 的 mpeg4 转成 H.264，避免块效应看起来像高频抖动。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(int(crf)),
        "-movflags",
        "+faststart",
        dst,
    ]
    code, _, stderr = run_ffmpeg(cmd)
    ok = os.path.isfile(dst) and os.path.getsize(dst) > 0
    if not ok:
        _log(f"H.264 转码失败: {stderr[-300:] if stderr else code}", log)
    return ok


def merge_audio(
    video_path: str,
    audio_source: str,
    output_path: str,
    *,
    log: LogFn = None,
) -> bool:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        _log("未找到 ffmpeg，跳过音频合并", log)
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_source,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]
    _log("ffmpeg 合并音轨…", log)
    code, _, stderr = run_ffmpeg(cmd)
    ok = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    if not ok:
        _log(f"ffmpeg 合并失败: {stderr[-400:] if stderr else code}", log)
    return ok


def stabilize_ffmpeg_tripod(
    input_path: str,
    output_path: str,
    *,
    shakiness: int = 5,
    accuracy: int = 15,
    smoothing: int = 0,
    log: LogFn = None,
    work_dir: Optional[str] = None,
) -> bool:
    """FFmpeg libvidstab 两 pass；smoothing=0 + opttripod 近似三脚架锁定。"""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    import tempfile

    td = work_dir or tempfile.mkdtemp(prefix="vidstab_ff_")
    trf = os.path.join(td, "transforms.trf")
    detect = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-vf",
        f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={trf}",
        "-f",
        "null",
        "-",
    ]
    _log("ffmpeg vidstabdetect (pass 1)…", log)
    code, _, stderr = run_ffmpeg(detect)
    if code != 0 or not os.path.isfile(trf):
        _log(f"vidstabdetect 失败: {stderr[-400:] if stderr else code}", log)
        return False
    transform = [
        ffmpeg,
        "-y",
        "-i",
        input_path,
        "-vf",
        (
            f"vidstabtransform=input={trf}:zoom=0:smoothing={smoothing}:"
            f"opttripod=1:interpol=bilinear"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        output_path,
    ]
    _log("ffmpeg vidstabtransform (pass 2)…", log)
    code, _, stderr = run_ffmpeg(transform)
    ok = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    if not ok:
        _log(f"vidstabtransform 失败: {stderr[-400:] if stderr else code}", log)
    return ok
