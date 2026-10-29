# -*- coding: utf-8 -*-
"""版本元数据一致性：version / release_date / README / 一页说明 / CHANGELOG 同步。"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_meta() -> dict:
    return json.loads(
        (ROOT / "version-info.json").read_text(encoding="utf-8")
    )


def _first_changelog_heading() -> tuple:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(r"^## v([\d.]+) \((\d{4}-\d{2}-\d{2})\)", text, re.M)
    assert m, "CHANGELOG.md 缺少 '## vX.Y.Z (YYYY-MM-DD)' 标题"
    return m.group(1), m.group(2)


def test_version_matches_changelog_head():
    meta = _load_meta()
    ver, _date = _first_changelog_heading()
    assert meta["version"] == ver


def test_release_date_matches_changelog_date():
    meta = _load_meta()
    _ver, date = _first_changelog_heading()
    assert meta["release_date"] == date, (
        f"version-info.json release_date={meta['release_date']} "
        f"但 CHANGELOG 最新条目日期为 {date}"
    )


def test_readme_shows_current_version():
    meta = _load_meta()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**{meta['version']}**" in readme


def test_one_page_doc_shows_version_and_date():
    meta = _load_meta()
    doc = (ROOT / "docs" / "Birdy-一页使用说明.html").read_text(encoding="utf-8")
    assert f"版本 {meta['version']}（{meta['release_date']}）" in doc


def test_changelog_has_entry_for_current_version():
    meta = _load_meta()
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## v{meta['version']} " in text
