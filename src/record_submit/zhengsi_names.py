# -*- coding: utf-8 -*-
"""观鸟记录中心导出：把中文名改成《郑四鸟种名录》中的写法。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ebird_taxonomy import normalize_taxon_text
from .taxonomy_cn import (
    default_species_csv_path,
    load_cn_to_en_sci,
    species_cn_taxo_sort_key,
)

_CATALOG_CACHE: Optional["ZhengsiCatalog"] = None
_CN_TABLE_CACHE: Optional[Dict[str, Tuple[str, str]]] = None
_SCI_TOKEN = re.compile(r"[a-z]+")


class ZhengsiCatalog:
    """郑四名录：中文名、学名、名录序号。"""

    def __init__(self) -> None:
        self.by_cn: Dict[str, str] = {}
        self.seq: Dict[str, int] = {}
        self.by_sci: Dict[str, str] = {}
        self.by_binomial: Dict[str, str] = {}


def default_zhengsi_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "郑四鸟种名录.xlsx"


def _sci_tokens(text: str) -> List[str]:
    s = normalize_taxon_text(text).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    return _SCI_TOKEN.findall(s)


def load_zhengsi_catalog(path: Optional[Path] = None) -> ZhengsiCatalog:
    """读取郑四名录。默认路径结果会缓存。"""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and path is None:
        return _CATALOG_CACHE

    catalog = ZhengsiCatalog()
    src = path or default_zhengsi_path()
    if not src.is_file():
        if path is None:
            _CATALOG_CACHE = catalog
        return catalog

    try:
        import openpyxl
    except ImportError as e:
        raise ImportError(
            "读取郑四鸟种名录需要 openpyxl：python -m pip install openpyxl"
        ) from e

    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    binomial_hits: Dict[str, List[str]] = {}
    try:
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or len(row) < 3:
                continue
            cn = normalize_taxon_text(str(row[1] or ""))
            sci = normalize_taxon_text(str(row[2] or ""))
            if not cn:
                continue
            catalog.by_cn.setdefault(cn, cn)
            if cn not in catalog.seq:
                try:
                    catalog.seq[cn] = int(row[0])
                except (TypeError, ValueError):
                    catalog.seq[cn] = 10**6
            tokens = _sci_tokens(sci)
            if len(tokens) >= 2:
                full = " ".join(tokens[:3] if len(tokens) >= 3 else tokens[:2])
                catalog.by_sci.setdefault(full, cn)
                binomial = " ".join(tokens[:2])
                binomial_hits.setdefault(binomial, [])
                if cn not in binomial_hits[binomial]:
                    binomial_hits[binomial].append(cn)
    finally:
        wb.close()

    for binomial, names in binomial_hits.items():
        if len(names) == 1:
            catalog.by_binomial[binomial] = names[0]

    if path is None:
        _CATALOG_CACHE = catalog
    return catalog


def _cn_table() -> Dict[str, Tuple[str, str]]:
    global _CN_TABLE_CACHE
    if _CN_TABLE_CACHE is None:
        _CN_TABLE_CACHE = load_cn_to_en_sci(default_species_csv_path())
    return _CN_TABLE_CACHE


def resolve_zhengsi_name(
    species_cn: str,
    catalog: Optional[ZhengsiCatalog] = None,
    cn_table: Optional[Dict[str, Tuple[str, str]]] = None,
) -> str:
    """
    先用 bird_species_list 的学名（含异写）对郑四中文名。
    这样「大山雀 / Parus major」会写成郑四的「欧亚大山雀」，而不是名录里另一种「大山雀 / Parus minor」。
    学名对不上、但中文名已在名录中时保留该中文名；都对不上则保留原名。
    """
    name = normalize_taxon_text(species_cn)
    if not name:
        return name
    book = catalog if catalog is not None else load_zhengsi_catalog()
    table = cn_table if cn_table is not None else _cn_table()
    _en, sci = table.get(name, ("", ""))
    tokens = _sci_tokens(sci)
    if len(tokens) >= 2:
        full = " ".join(tokens[:3] if len(tokens) >= 3 else tokens[:2])
        hit = book.by_sci.get(full)
        if hit:
            return hit
        binomial = " ".join(tokens[:2])
        hit = book.by_binomial.get(binomial)
        if hit:
            return hit
    if name in book.by_cn:
        return book.by_cn[name]
    return name


def remap_species_counts_zhengsi(
    species_counts: Dict[str, int],
    catalog: Optional[ZhengsiCatalog] = None,
) -> Dict[str, int]:
    """把计数键改成郑四中文名；多名对应同一郑四名时合并只数。"""
    book = catalog if catalog is not None else load_zhengsi_catalog()
    table = _cn_table()
    merged: Dict[str, int] = {}
    for raw, cnt in species_counts.items():
        n = int(cnt)
        if n < 1:
            continue
        official = resolve_zhengsi_name(str(raw), book, table)
        merged[official] = merged.get(official, 0) + n
    return merged


def iter_species_counts_zhengsi(
    species_counts: Dict[str, int],
) -> List[Tuple[str, int]]:
    """郑四序号排序；(名录外的种仍按现有分类顺序排在后面)。"""
    book = load_zhengsi_catalog()
    merged = remap_species_counts_zhengsi(species_counts, book)

    def sort_key(item: Tuple[str, int]):
        name = item[0]
        seq = book.seq.get(name)
        if seq is not None:
            return (0, seq, name)
        return (1, species_cn_taxo_sort_key(name), name)

    return sorted(merged.items(), key=sort_key)
