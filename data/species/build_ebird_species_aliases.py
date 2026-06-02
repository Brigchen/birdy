#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照 eBird/Clements checklist CSV，生成 ebird_species_aliases.json。

用法（需先下载 Clements 整合清单 CSV）::

  python data/species/build_ebird_species_aliases.py \\
    --clements path/to/eBird-Clements-....csv

默认读取同目录下的 ``eBird_taxonomy_v2025-4.xlsx``。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_DIR = Path(__file__).resolve().parent
OUR_CSV = _DIR / "bird_species_list.csv"
OUT_JSON = _DIR / "ebird_species_aliases.json"
DEFAULT_TAXONOMY = _DIR / "eBird_taxonomy_v2025-4.xlsx"

MANUAL = [
    {
        "english": "Rock Dove",
        "chinese": "原鸽",
        "scientific": "Columba livia",
        "ebird": "Rock Pigeon (Feral Pigeon)",
        "note": "eBird 导入默认选 Feral Pigeon；野型需在网页逐条改",
    },
]


def _load_taxonomy_species(path: Path) -> dict[str, str]:
    import openpyxl

    sci_map: dict[str, str] = {}
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or (row[1] or "").strip() != "species":
                continue
            sci = (row[5] or "").strip()
            en = (row[4] or "").strip()
            if sci and en:
                sci_map[sci] = en
    finally:
        wb.close()
    return sci_map


def build_aliases(taxonomy_path: Path) -> list[dict]:
    sci_map = _load_taxonomy_species(taxonomy_path)
    aliases: list[dict] = []
    seen: set[tuple] = set()

    def add(**kw: str) -> None:
        ebird = kw.pop("ebird")
        key = (ebird, tuple(sorted((k, v) for k, v in kw.items() if v)))
        if key in seen:
            return
        seen.add(key)
        item = {k: v for k, v in kw.items() if v}
        item["ebird"] = ebird
        aliases.append(item)

    for item in MANUAL:
        add(
            english=item.get("english", ""),
            chinese=item.get("chinese", ""),
            scientific=item.get("scientific", ""),
            ebird=item["ebird"],
        )

    with OUR_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cn = (row.get("中文名称") or "").strip()
            en = (row.get("英文名称") or "").strip()
            sci = (row.get("学名") or "").strip()
            eb = sci_map.get(sci)
            if not eb or eb == en:
                continue
            add(english=en, ebird=eb)
            if cn:
                add(chinese=cn, ebird=eb)
            if sci:
                add(scientific=sci, ebird=eb)
    return aliases


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
        help="eBird Taxonomy xlsx (e.g. eBird_taxonomy_v2025-4.xlsx)",
    )
    args = p.parse_args()
    if not args.taxonomy.is_file():
        raise SystemExit(f"Taxonomy xlsx not found: {args.taxonomy}")
    aliases = build_aliases(args.taxonomy)
    OUT_JSON.write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(aliases)} entries -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
