# -*- coding: utf-8 -*-
"""CLI：python -m video_stab [选项] input.mp4"""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

from .paths import setup_import_paths

setup_import_paths()

from .media import list_videos, parse_norm_box  # noqa: E402
from .options import StabilizeOptions  # noqa: E402
from .pipeline import (  # noqa: E402
    stabilize_file,
    stabilize_paths,
    suggest_output_dir,
    suggest_output_path,
)


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="观鸟录像稳定：三脚架固定视角 / 鸟类追踪居中"
    )
    p.add_argument("input", nargs="?", help="输入视频或目录")
    p.add_argument("-o", "--output", help="输出文件或目录")
    p.add_argument("--recursive", action="store_true", help="目录模式包含子目录")
    p.add_argument(
        "--bbox",
        help="鸟体框，相对画面 x,y,w,h（0–1），例如 0.40,0.30,0.20,0.25",
    )
    p.add_argument(
        "--view",
        help="输出画面范围，相对画面 x,y,w,h（0–1），例如 0.20,0.15,0.60,0.70",
    )
    p.add_argument(
        "--mode",
        choices=("tripod", "track"),
        default="tripod",
        help="tripod=三脚架固定; track=追踪稳定器",
    )
    p.add_argument(
        "--tripod-engine",
        choices=("optical", "ffmpeg"),
        default="optical",
    )
    p.add_argument(
        "--track-engine",
        choices=("yolo", "csrt"),
        default="yolo",
    )
    p.add_argument("--smooth", type=int, default=30, help="追踪轨迹平滑窗口（帧）")
    p.add_argument("--start", type=float, default=0.0, help="起始时间（秒）")
    p.add_argument("--end", type=float, default=-1.0, help="结束时间（秒，-1=结尾）")
    p.add_argument("--no-audio", action="store_true", help="不合并音轨")
    p.add_argument("--bird-model", help="YOLO 鸟检模型路径")
    p.add_argument("--gui", action="store_true", help="打开 GUI")
    args = p.parse_args(argv)

    if args.gui or not args.input:
        from .gui import main as gui_main

        return gui_main()

    inp = str(Path(args.input).resolve())
    src = Path(inp)
    if not src.exists():
        print(f"路径不存在: {inp}", file=sys.stderr)
        return 2
    try:
        bbox = parse_norm_box(args.bbox) if args.bbox else None
        view = parse_norm_box(args.view) if args.view else None
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    opts = StabilizeOptions(
        mode=args.mode,
        tripod_engine=args.tripod_engine,
        track_engine=args.track_engine,
        smoothing_window=args.smooth,
        start_time=args.start,
        end_time=args.end,
        keep_audio=not args.no_audio,
        bird_model=args.bird_model,
        init_bbox_norm=bbox,
        view_norm=view,
    )

    def prog(d, t):
        if t > 0 and d % 30 == 0:
            print(f"进度 {d}/{t}", flush=True)

    try:
        if src.is_dir():
            files = list_videos(inp, recursive=args.recursive)
            if not files:
                print(f"目录里没有视频: {inp}", file=sys.stderr)
                return 2
            out_dir = args.output or suggest_output_dir(inp)
            done, errors = stabilize_paths(
                files, out_dir, opts, log=print, progress=prog
            )
            if errors and not done:
                return 1
            return 0
        out = args.output or suggest_output_path(inp, args.mode)
        stabilize_file(inp, out, opts, log=print, progress=prog)
    except Exception as e:
        print(f"失败: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
