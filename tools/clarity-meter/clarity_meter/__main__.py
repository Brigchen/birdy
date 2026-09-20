# -*- coding: utf-8 -*-
"""清晰度打分入口：无参数开 GUI；指定 --folder 则直接出报告。"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from .paths import setup_import_paths

setup_import_paths()


def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Birdy 清晰度打分（只读，生成 HTML 报告）")
    p.add_argument("--folder", "-f", help="图片目录（递归）")
    p.add_argument("--threshold", "-t", type=float, default=35.0, help="模糊阈值，默认 35")
    p.add_argument(
        "--full-frame",
        action="store_true",
        help="目录已是切割图，整图套掩膜计分",
    )
    p.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    p.add_argument("--out", help="报告 HTML 路径")
    args = p.parse_args(argv)
    if not args.folder:
        from .gui import main as gui_main

        return gui_main()
    from .report import default_report_path, write_html_report
    from .score import score_folder

    folder = args.folder
    if not Path(folder).is_dir():
        print(f"目录不存在: {folder}", file=sys.stderr)
        return 2

    def prog(d):
        if d.get("kind") == "tick":
            print(f"[{d.get('done')}/{d.get('total')}] {d.get('name') or ''}", flush=True)

    result = score_folder(
        folder,
        min_clarity=float(args.threshold),
        use_full_frame=bool(args.full_frame),
        progress=prog,
    )
    out = args.out or str(default_report_path(str(result.get("root") or folder)))
    path = write_html_report(result, out)
    print(f"报告: {path}")
    print(
        f"合计 {result['total']}  通过 {result['n_pass']}  "
        f"模糊 {result['n_blur']}  未检出 {result['n_nobird']}"
    )
    if not args.no_open:
        webbrowser.open(Path(path).resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
