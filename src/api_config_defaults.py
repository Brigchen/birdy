# -*- coding: utf-8 -*-
"""
API 配置文件缺省结构：仓库可不包含 doubao_api_config.json / amap_api_config.json
（见根目录 .gitignore）。首次需要配置时由 ensure_* 自动生成合乎格式的空模板。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

# 与 doubao_bird_api 中默认 Endpoint 保持一致（避免循环 import）
DEFAULT_DOUBAO_API_CONFIG = {
    "api_key": "",
    "api_base": "https://ark.cn-beijing.volces.com/api/v3",
    "model": "doubao-seed-2-0-lite-260215",
    "models": [
        "doubao-seed-2-0-lite-260215",
        "doubao-1-5-vision-pro-32k-250115",
        "doubao-seed-2-0-mini-260215",
    ],
    "usage_stats_path": "",
    "daily_token_limit_per_model": 2_000_000,
    "token_switch_ratio": 0.75,
    "enable_token_rotation": True,
    "min_interval_seconds": 1.0,
    "retry_count": 6,
    "retry_backoff_base": 2.0,
    "max_retry_wait_seconds": 120,
    "DOUBAO_MAX_NAME_LEN": 8,
}

DEFAULT_AMAP_API_CONFIG = {
    "api_key": "",
}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(copy.deepcopy(data), f, ensure_ascii=False, indent=2)


def ensure_doubao_api_config_file(base_dir: Optional[Path] = None) -> Path:
    """若不存在则写入默认结构的 doubao_api_config.json。返回该文件路径。"""
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    path = base / "doubao_api_config.json"
    if not path.is_file():
        _write_json(path, DEFAULT_DOUBAO_API_CONFIG)
    return path


def ensure_amap_api_config_file(base_dir: Optional[Path] = None) -> Path:
    """若不存在则写入默认结构的 amap_api_config.json。返回该文件路径。"""
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    path = base / "amap_api_config.json"
    if not path.is_file():
        _write_json(path, DEFAULT_AMAP_API_CONFIG)
    return path
