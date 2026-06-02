# -*- coding: utf-8 -*-
"""中文种名 → 英文名 / 学名（读 data/species/bird_species_list.csv）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from .ebird_taxonomy import resolve_ebird_species

_EBIRD_ALIAS_INDEX: Optional[Dict[str, str]] = None


def default_species_csv_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "bird_species_list.csv"


def default_species_aliases_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "species_text_aliases.json"


def default_ebird_species_aliases_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "ebird_species_aliases.json"


def load_species_aliases(aliases_path: Optional[Path] = None) -> Dict[str, str]:
    """归档中文名异写 → 规范中文名（用于查 bird_species_list）。"""
    path = aliases_path or default_species_aliases_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, str] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = (item.get("from") or "").strip()
        dst = (item.get("to") or "").strip()
        if src and dst:
            out[src] = dst
    return out


def load_ebird_species_alias_index(
    aliases_path: Optional[Path] = None,
) -> Dict[str, str]:
    """手动覆盖：中/英/学名 → eBird 俗名（如 Rock Pigeon (Feral Pigeon)）。"""
    global _EBIRD_ALIAS_INDEX
    if _EBIRD_ALIAS_INDEX is not None and aliases_path is None:
        return _EBIRD_ALIAS_INDEX

    path = aliases_path or default_ebird_species_aliases_path()
    index: Dict[str, str] = {}
    if not path.is_file():
        if aliases_path is None:
            _EBIRD_ALIAS_INDEX = index
        return index
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        if aliases_path is None:
            _EBIRD_ALIAS_INDEX = index
        return index
    if not isinstance(raw, list):
        if aliases_path is None:
            _EBIRD_ALIAS_INDEX = index
        return index
    for item in raw:
        if not isinstance(item, dict):
            continue
        ebird = (item.get("ebird") or item.get("to") or "").strip()
        if not ebird:
            continue
        for field in ("english", "chinese", "scientific"):
            key = (item.get(field) or "").strip()
            if key and key not in index:
                index[key] = ebird
    if aliases_path is None:
        _EBIRD_ALIAS_INDEX = index
    return index


def load_cn_to_en_sci(
    csv_path: Path,
    *,
    aliases_path: Optional[Path] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    返回 dict: 中文名称 -> (英文名称, 学名)。
    会合并 species_text_aliases.json 中的异写映射。
    """
    out: Dict[str, Tuple[str, str]] = {}
    if csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                cn = (row.get("中文名称") or "").strip()
                if not cn:
                    continue
                en = (row.get("英文名称") or "").strip()
                sci = (row.get("学名") or "").strip()
                out[cn] = (en, sci)
    aliases = load_species_aliases(aliases_path)
    for src, dst in aliases.items():
        if dst in out and src not in out:
            out[src] = out[dst]
    return out


def lookup_species(
    species_cn: str, table: Dict[str, Tuple[str, str]]
) -> Tuple[str, str]:
    """(english, scientific) 查不到则 ('','')。"""
    key = (species_cn or "").strip()
    t = table.get(key)
    if t:
        return t[0], t[1]
    return "", ""


def _sanitize_ebird_field(text: str) -> str:
    from .ebird_taxonomy import normalize_taxon_text

    return normalize_taxon_text(text).replace(",", ";")


def ebird_species_name(
    english: str,
    scientific: str,
    chinese: str,
    *,
    ebird_aliases: Optional[Dict[str, str]] = None,
) -> str:
    """仅返回 A 列英文俗名（兼容旧调用）。"""
    common, _sci = ebird_species_cells(
        english, scientific, chinese, ebird_aliases=ebird_aliases
    )
    return common


def ebird_species_cells(
    english: str,
    scientific: str,
    chinese: str,
    *,
    ebird_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    eBird Checklist 物种行：A=PRIMARY_COM_NAME，B=学名（与 Taxonomy xlsx 一致）。

    解析顺序：手动 aliases → eBird_taxonomy_v2025-4.xlsx（按学名）→ 本地英文名。
    """
    en = _sanitize_ebird_field(english)
    sci = _sanitize_ebird_field(scientific)
    cn = _sanitize_ebird_field(chinese)
    aliases = ebird_aliases if ebird_aliases is not None else load_ebird_species_alias_index()

    for key in (en, cn, sci):
        if key and key in aliases:
            name = aliases[key].replace(",", ";")
            return name, sci

    common, sci_out = resolve_ebird_species(en, sci)
    common = common.replace(",", ";")
    sci_out = sci_out.replace(",", ";")
    return common, sci_out
