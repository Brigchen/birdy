#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 GPX 轨迹导出为 Folium 交互地图 HTML。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpx_track import load_gpx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gpx", help="GPX 文件路径")
    ap.add_argument("-o", "--output", default="track_map.html", help="输出 HTML")
    args = ap.parse_args()
    try:
        import folium
    except ImportError:
        print("请安装 folium: pip install folium")
        sys.exit(1)

    pts = load_gpx(args.gpx)
    if not pts:
        print("GPX 无轨迹点")
        sys.exit(1)
    m = folium.Map(location=[pts[0].lat, pts[0].lon], zoom_start=12)
    folium.PolyLine(
        [(p.lat, p.lon) for p in pts],
        color="#2980B9",
        weight=4,
        opacity=0.8,
    ).add_to(m)
    folium.Marker([pts[0].lat, pts[0].lon], popup="起点", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([pts[-1].lat, pts[-1].lon], popup="终点", icon=folium.Icon(color="red")).add_to(m)
    out = Path(args.output).expanduser().resolve()
    m.save(str(out))
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
