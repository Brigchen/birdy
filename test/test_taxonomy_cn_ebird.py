#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from record_submit.taxonomy_cn import (  # noqa: E402
    default_species_csv_path,
    ebird_species_name,
    iter_species_counts_taxo,
    load_cn_to_en_sci,
    load_ebird_species_alias_index,
    species_cn_taxo_sort_key,
)


def test_ebird_prefers_english_common_name():
    assert (
        ebird_species_name("Eurasian Tree Sparrow", "Passer montanus", "麻雀")
        == "Eurasian Tree Sparrow"
    )


def test_ebird_rock_dove_alias():
    aliases = load_ebird_species_alias_index()
    assert (
        ebird_species_name("Rock Dove", "Columba livia", "", ebird_aliases=aliases)
        == "Rock Pigeon (Feral Pigeon)"
    )


def test_ebird_english_when_no_scientific():
    assert ebird_species_name("Eurasian Tree Sparrow", "", "麻雀") == (
        "Eurasian Tree Sparrow"
    )


def test_load_includes_sparrow():
    table = load_cn_to_en_sci(default_species_csv_path())
    en, sci = table.get("麻雀", ("", ""))
    assert en
    assert sci


def test_species_cn_taxo_sort_key_orders_by_order():
    """鹈形目苍鹭应排在鸻形目反嘴鹬之前（中文名则是「反」在「苍」前）。"""
    heron = species_cn_taxo_sort_key("苍鹭")
    avocet = species_cn_taxo_sort_key("反嘴鹬")
    sparrow = species_cn_taxo_sort_key("麻雀")
    assert heron < avocet < sparrow
    assert "反嘴鹬" < "苍鹭"
    names = [n for n, _ in iter_species_counts_taxo({"麻雀": 1, "反嘴鹬": 2, "苍鹭": 3})]
    assert names == ["苍鹭", "反嘴鹬", "麻雀"]


if __name__ == "__main__":
    test_ebird_prefers_english_common_name()
    test_ebird_rock_dove_alias()
    test_ebird_english_when_no_scientific()
    test_load_includes_sparrow()
    print("ok")
