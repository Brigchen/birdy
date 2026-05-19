#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并多个 GPX 为一条轨迹。用法见 README.md"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpx_track import merge_gpx_files  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="合并 GPX 轨迹文件")
    ap.add_argument("gpx_files", nargs="+", help="输入 GPX 路径")
    ap.add_argument("-o", "--output", required=True, help="输出 GPX 路径")
    args = ap.parse_args()
    out = merge_gpx_files(args.gpx_files, args.output)
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()
