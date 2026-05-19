# -*- coding: utf-8 -*-
"""观鸟记录中心（birdreport 系）侧：导出可人工补录或经自有中继转发的 JSON 草案。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .scan import ChecklistBucket


def build_birdreport_batch(
    buckets: Dict[Tuple, ChecklistBucket],
    species_en_sci: Dict[str, Tuple[str, str]],
) -> dict:
    checklists: List[dict] = []
    for _k, b in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        observations = []
        for sp_cn, cnt in sorted(b.species_counts.items()):
            en, sci = species_en_sci.get(sp_cn, ("", ""))
            observations.append(
                {
                    "species_name_cn": sp_cn,
                    "species_name_en": en,
                    "scientific_name": sci,
                    "count": int(cnt),
                }
            )
        checklists.append(
            {
                "observation_date": b.day.isoformat(),
                "start_time_local": (
                    b.start_time.strftime("%H:%M:%S") if b.start_time else None
                ),
                "latitude": b.lat,
                "longitude": b.lon,
                "observations": observations,
                "sample_media_paths": list(dict.fromkeys(b.sample_files))[:8],
            }
        )
    return {
        "export_version": 1,
        "source": "birdy-record_submit",
        "notes": (
            "字段为草案，与官方 App/接口未必一一对应；count 按批次去重后累加"
            "（0.1 km 或 30 分钟内视为同一批，取该批只数最多的一张），"
            "请在提交前自行修改数量与地点。"
        ),
        "checklists": checklists,
    }


def write_birdreport_batch_json(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
