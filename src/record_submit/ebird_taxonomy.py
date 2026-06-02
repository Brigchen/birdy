# -*- coding: utf-8 -*-
"""eBird Taxonomy（eBird_taxonomy_v2025-4.xlsx）学名 → 官方英文名。"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

_TAXONOMY_CACHE: Optional[Dict[str, Tuple[str, str]]] = None


def default_ebird_taxonomy_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "eBird_taxonomy_v2025-4.xlsx"


def normalize_taxon_text(text: str) -> str:
    """去除不可见字符，统一撇号与空白。"""
    s = unicodedata.normalize("NFKC", (text or "").strip())
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_key(text: str) -> str:
    return normalize_taxon_text(text).lower()


def load_ebird_taxonomy_by_sci(
    taxonomy_path: Optional[Path] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    学名（小写）→ (PRIMARY_COM_NAME, SCI_NAME)。

    仅收录 category=species；同名学名以清单中首次出现为准。
    """
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None and taxonomy_path is None:
        return _TAXONOMY_CACHE

    path = taxonomy_path or default_ebird_taxonomy_path()
    out: Dict[str, Tuple[str, str]] = {}
    if not path.is_file():
        if taxonomy_path is None:
            _TAXONOMY_CACHE = out
        return out

    try:
        import openpyxl
    except ImportError as e:
        raise ImportError(
            "读取 eBird Taxonomy 需要 openpyxl：python -m pip install openpyxl"
        ) from e

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or len(row) < 6:
                continue
            if (row[1] or "").strip() != "species":
                continue
            primary = normalize_taxon_text(str(row[4] or ""))
            sci = normalize_taxon_text(str(row[5] or ""))
            if not primary or not sci:
                continue
            key = _norm_key(sci)
            if key not in out:
                out[key] = (primary, sci)
    finally:
        wb.close()

    if taxonomy_path is None:
        _TAXONOMY_CACHE = out
    return out


def resolve_ebird_species(
    english: str,
    scientific: str,
    *,
    taxonomy_by_sci: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Tuple[str, str]:
    """
    返回 (A 列英文俗名, B 列学名)，与 eBird Taxonomy v2025 一致。

    优先按学名在 taxonomy xlsx 中查找 PRIMARY_COM_NAME。
    """
    en = normalize_taxon_text(english)
    sci = normalize_taxon_text(scientific)
    table = taxonomy_by_sci if taxonomy_by_sci is not None else load_ebird_taxonomy_by_sci()

    if sci:
        hit = table.get(_norm_key(sci))
        if hit:
            return hit[0], hit[1]

    return en, sci
