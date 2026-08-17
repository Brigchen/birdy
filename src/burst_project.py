# -*- coding: utf-8 -*-
"""连拍动图项目文件：图片列表与每帧标定点/裁剪区，保存在相片目录旁。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from burst_anchor import FrameLayout

PROJECT_KIND = "birdy-burst-project"
PROJECT_VERSION = 1
PROJECT_SUFFIX = ".birdy-burst.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


def is_burst_project_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name.endswith(PROJECT_SUFFIX) or (
        name.endswith(".json") and "birdy-burst" in name
    )


def default_project_path_for_images(image_paths: Sequence[str]) -> Optional[Path]:
    """相片所在目录下、与目录同名的项目文件。多目录时用第一张图所在目录。"""
    abs_paths = [os.path.abspath(p) for p in image_paths if p]
    if not abs_paths:
        return None
    dirs = [Path(p).resolve().parent for p in abs_paths]
    folder = dirs[0]
    if folder.name:
        return folder / f"{folder.name}{PROJECT_SUFFIX}"
    return folder / f"burst{PROJECT_SUFFIX}"


def path_for_project(path: str, project_dir: Path) -> str:
    """尽量写成相对项目文件目录的路径，便于整夹搬迁。"""
    src = Path(os.path.abspath(path))
    base = project_dir.resolve()
    try:
        rel = src.resolve().relative_to(base)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(src)


def resolve_project_image_path(stored: str, project_dir: Path) -> Path:
    p = Path(stored)
    if p.is_absolute():
        return p
    return (project_dir / p).resolve()


def layout_to_json(lay: Optional[FrameLayout]) -> Optional[dict]:
    if lay is None:
        return None
    return lay.to_dict()


def layout_from_json(raw: Any) -> Optional[FrameLayout]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return FrameLayout.from_dict(raw)
    except (TypeError, ValueError):
        return None


@dataclass
class BurstProjectData:
    paths: List[str]
    layouts: List[Optional[FrameLayout]]
    frame_idx: int
    options: Dict[str, Any]
    missing: List[str]


def build_project_dict(
    image_paths: Sequence[str],
    layouts: Sequence[Optional[FrameLayout]],
    *,
    project_path: Path,
    frame_idx: int = 0,
    options: Optional[dict] = None,
) -> dict:
    base = project_path.parent
    frames: List[dict] = []
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        lay = layouts[i] if i < len(layouts) else None
        frames.append(
            {
                "path": path_for_project(p, base),
                "name": Path(p).name,
                "layout": layout_to_json(lay),
            }
        )
    return {
        "kind": PROJECT_KIND,
        "version": PROJECT_VERSION,
        "frame_idx": int(max(0, min(frame_idx, max(0, n - 1)))),
        "options": dict(options or {}),
        "frames": frames,
    }


def parse_project_dict(raw: dict, project_path: Path) -> BurstProjectData:
    base = project_path.parent
    frames = raw.get("frames")
    if not isinstance(frames, list):
        frames = []
    paths: List[str] = []
    layouts: List[Optional[FrameLayout]] = []
    missing: List[str] = []
    for item in frames:
        if not isinstance(item, dict):
            continue
        stored = item.get("path") or item.get("file") or ""
        if not isinstance(stored, str) or not stored.strip():
            continue
        resolved = resolve_project_image_path(stored.strip(), base)
        if not resolved.is_file():
            missing.append(str(resolved))
            continue
        paths.append(str(resolved))
        layouts.append(layout_from_json(item.get("layout")))
    try:
        frame_idx = int(raw.get("frame_idx", 0))
    except (TypeError, ValueError):
        frame_idx = 0
    if paths:
        frame_idx = int(max(0, min(frame_idx, len(paths) - 1)))
    else:
        frame_idx = 0
    opts = raw.get("options")
    if not isinstance(opts, dict):
        opts = {}
    return BurstProjectData(
        paths=paths,
        layouts=layouts,
        frame_idx=frame_idx,
        options=opts,
        missing=missing,
    )


def load_project_file(path: Path) -> BurstProjectData:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("项目文件格式无效")
    kind = str(raw.get("kind") or "")
    if kind and kind != PROJECT_KIND:
        raise ValueError(f"不是动图项目文件（kind={kind}）")
    return parse_project_dict(raw, path)


def save_project_file(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, text)


def match_layout_for_path(
    image_path: str,
    entries: Sequence[Tuple[str, Optional[FrameLayout]]],
) -> Optional[FrameLayout]:
    """entries: (resolved_path, layout)。先绝对路径，再唯一文件名。"""
    want = Path(os.path.abspath(image_path))
    for p, lay in entries:
        try:
            if Path(os.path.abspath(p)) == want:
                return lay
        except OSError:
            continue
    name = want.name.lower()
    hits = [
        lay
        for p, lay in entries
        if Path(p).name.lower() == name
    ]
    if len(hits) == 1:
        return hits[0]
    return None
