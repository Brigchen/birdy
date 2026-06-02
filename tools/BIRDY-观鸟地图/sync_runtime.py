#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Birdy 主项目运行时代码同步到本工具 birdy_runtime/，便于打包独立分发。

用法（在 tools/BIRDY-观鸟地图 目录）:
    python sync_runtime.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
REPO_SRC = TOOL_DIR.parent.parent / "src"
RUNTIME = TOOL_DIR / "birdy_runtime"

# 独立工具不使用高德底图/地理编码，仅同步轨迹图与 EXIF 读取
LEGACY_RUNTIME_FILES = (
    "geo_encoder.py",
    "api_config_defaults.py",
    "geocoding_config.py",
)

RECORD_SUBMIT_INIT = '''# -*- coding: utf-8 -*-
"""观鸟地图工具最小 record_submit 包（仅 exif_read）。"""
'''

RUNTIME_README = """# birdy_runtime

本目录为 **BIRDY-观鸟地图** 独立运行库，**纳入 Git 版本管理**。

克隆仓库后可直接运行 `start.bat`，无需再执行 `sync_runtime.py`。

## 内容

由 `../sync_runtime.py` 从 Birdy 主项目 `src/` 同步：

- `gpx_track/` — 轨迹图生成（工具固定使用经纬度网格底图，无高德）
- `record_submit/exif_read.py` — 照片 EXIF 读取

## 何时重新 sync

修改主项目 `src/gpx_track/` 或 `record_submit/exif_read.py` 后，在工具目录执行：

```bat
python sync_runtime.py
```

然后将本目录变更一并提交，以便分发包与主程序保持一致。
"""


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    if not (REPO_SRC / "gpx_track").is_dir():
        print(f"错误: 找不到 {REPO_SRC / 'gpx_track'}，请在 Birdy 仓库内运行。")
        return 1

    RUNTIME.mkdir(parents=True, exist_ok=True)

    _copy_tree(REPO_SRC / "gpx_track", RUNTIME / "gpx_track")
    for name in LEGACY_RUNTIME_FILES:
        legacy = RUNTIME / name
        if legacy.is_file():
            legacy.unlink()

    rs = RUNTIME / "record_submit"
    rs.mkdir(parents=True, exist_ok=True)
    (rs / "__init__.py").write_text(RECORD_SUBMIT_INIT, encoding="utf-8")
    _copy_file(REPO_SRC / "record_submit" / "exif_read.py", rs / "exif_read.py")

    assets = TOOL_DIR / "assets"
    assets.mkdir(exist_ok=True)
    repo_res = TOOL_DIR.parent.parent / "resources"
    for name in ("birdy_logo_128.png", "birdy_logo_640.png", "logo.png"):
        src = repo_res / name
        if src.is_file():
            _copy_file(src, assets / name)

    print(f"已同步到 {RUNTIME}")
    print("  gpx_track/, record_submit/exif_read.py")
    if any((assets / n).is_file() for n in ("birdy_logo_128.png", "logo.png")):
        print(f"  图标 -> {assets}")

    (RUNTIME / "README.md").write_text(RUNTIME_README, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
