# -*- coding: utf-8 -*-
"""中文种名 → 英文名 / 学名（读 data/species/bird_species_list.csv）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple


def default_species_csv_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "bird_species_list.csv"


def load_cn_to_en_sci(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    返回 dict: 中文名称 -> (英文名称, 学名)。
    若重名后者覆盖前者（罕见）。
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not csv_path.is_file():
        return out
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cn = (row.get("中文名称") or "").strip()
            if not cn:
                continue
            en = (row.get("英文名称") or "").strip()
            sci = (row.get("学名") or "").strip()
            out[cn] = (en, sci)
    return out


def lookup_species(
    species_cn: str, table: Dict[str, Tuple[str, str]]
) -> Tuple[str, str]:
    """(english, scientific) 查不到则 ('','')。"""
    t = table.get((species_cn or "").strip())
    if t:
        return t[0], t[1]
    return "", ""
