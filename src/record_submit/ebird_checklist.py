# -*- coding: utf-8 -*-
"""
eBird Checklist Format：按官方模版布局导出 .csv（每文件一个 checklist，列 C）。

布局与 ``data/species/ebird_checklist_format_template.xls`` 一致：
A1 为空；第 1–14 行为努力信息；自第 15 行起 A 列为英文名、C 列为只数。
导入时在 eBird 选择「Checklist Format」并上传 CSV。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .export_filenames import unique_checklist_slug
from .scan import ChecklistBucket

_FIRST_CHECKLIST_COL = 2  # Excel 列 C
_SPECIES_ROW_START = 14  # Excel 第 15 行
_EFFORT_LABELS = (
    "Location Name",
    "Latitude",
    "Longitude",
    "Date",
    "Start Time",
    "State",
    "Country",
    "Protocol",
    "Num Observers",
    "Duration (min)",
    "All Obs Reported (Y/N)",
    "Dist Traveled (Miles)",
    "Area Covered (Acres)",
    "Notes",
)


def default_ebird_checklist_template_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "ebird_checklist_format_template.xls"


def _fmt_date_us(d) -> str:
    return f"{d.month}/{d.day}/{d.year}"


def _fmt_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "8:00 AM"
    h24 = dt.hour
    h = h24 % 12
    if h == 0:
        h = 12
    ampm = "AM" if h24 < 12 else "PM"
    return f"{h}:{dt.minute:02d} {ampm}"


def _validate_checklist_template(rb) -> None:
    sh = rb.sheet_by_index(0)
    if sh.nrows < _SPECIES_ROW_START + 1:
        raise ValueError("eBird Checklist 模版行数不足")
    if str(sh.cell_value(0, 0)).strip():
        raise ValueError("eBird Checklist 模版 A1 须为空")
    sp_label = str(sh.cell_value(_SPECIES_ROW_START, 0)).strip().upper()
    if "SPECIES" not in sp_label or "COMMON" not in sp_label:
        raise ValueError(
            f"模版第 15 行 A 列应为物种英文名标签，当前为: {sh.cell_value(_SPECIES_ROW_START, 0)!r}"
        )
    if sh.ncols <= _FIRST_CHECKLIST_COL:
        raise ValueError("eBird Checklist 模版须至少包含列 C")
    for r, expected in enumerate(_EFFORT_LABELS):
        label = str(sh.cell_value(r, _FIRST_CHECKLIST_COL)).strip()
        if label != expected:
            raise ValueError(
                f"模版第 {r + 1} 行列 C 应为 {expected!r}，当前为 {label!r}"
            )


def _ensure_template_valid(template_path: Optional[str] = None) -> Path:
    tpl = Path(
        template_path or default_ebird_checklist_template_path()
    ).expanduser().resolve()
    if not tpl.is_file():
        raise FileNotFoundError(f"找不到 eBird Checklist 模版: {tpl}")
    try:
        import xlrd
    except ImportError as e:
        raise ImportError(
            "校验 eBird 模版需要 xlrd：python -m pip install xlrd"
        ) from e
    _validate_checklist_template(xlrd.open_workbook(str(tpl)))
    return tpl


def _effort_values_for_bucket(
    bucket: ChecklistBucket,
    *,
    country: str,
    state_province: str,
    protocol: str,
    duration_min: int,
    num_observers: int,
    locality_prefix: str,
) -> List[str]:
    loc = locality_prefix
    if bucket.lat is not None and bucket.lon is not None:
        loc += f" {bucket.lat:.4f},{bucket.lon:.4f}"
    lat_s = f"{bucket.lat:.6f}" if bucket.lat is not None else ""
    lon_s = f"{bucket.lon:.6f}" if bucket.lon is not None else ""
    notes = "Exported from Birdy; verify before upload."
    return [
        loc.replace(",", ";"),
        lat_s,
        lon_s,
        _fmt_date_us(bucket.day),
        _fmt_time(bucket.start_time),
        state_province,
        country,
        protocol,
        str(int(num_observers)),
        str(int(duration_min)),
        "Y",
        "",
        "",
        notes.replace(",", ";"),
    ]


def _species_rows_for_bucket(
    bucket: ChecklistBucket,
    species_en_sci: Dict[str, Tuple[str, str]],
) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for sp_cn, cnt in sorted(bucket.species_counts.items()):
        en, _sci = species_en_sci.get(sp_cn, ("", ""))
        common = (en or sp_cn).replace(",", ";")
        count_cell = str(int(cnt))
        comment = f"source_cn={sp_cn}".replace(",", ";").replace("|", " ")
        if comment:
            count_cell = f"{count_cell}|{comment}"
        rows.append((common, count_cell))
    return rows


def build_checklist_grid(
    bucket: ChecklistBucket,
    species_en_sci: Dict[str, Tuple[str, str]],
    *,
    country: str = "CN",
    state_province: str = "CN-FJ",
    protocol: str = "Traveling",
    duration_min: int = 60,
    num_observers: int = 1,
    locality_prefix: str = "Birdy archive",
) -> List[List[str]]:
    """三列（A/B/C）Checklist Format 网格，可直接写入 CSV。"""
    species = _species_rows_for_bucket(bucket, species_en_sci)
    nrows = _SPECIES_ROW_START + len(species)
    ncols = _FIRST_CHECKLIST_COL + 1
    grid: List[List[str]] = [[""] * ncols for _ in range(nrows)]

    effort = _effort_values_for_bucket(
        bucket,
        country=country,
        state_province=state_province,
        protocol=protocol,
        duration_min=duration_min,
        num_observers=num_observers,
        locality_prefix=locality_prefix,
    )
    for r, val in enumerate(effort):
        grid[r][_FIRST_CHECKLIST_COL] = val

    for i, (common, count_cell) in enumerate(species):
        row = _SPECIES_ROW_START + i
        grid[row][0] = common
        grid[row][1] = ""
        grid[row][_FIRST_CHECKLIST_COL] = count_cell

    return grid


def write_ebird_checklist_csv(
    output_path: str,
    bucket: ChecklistBucket,
    species_en_sci: Dict[str, Tuple[str, str]],
    *,
    template_path: Optional[str] = None,
    country: str = "CN",
    state_province: str = "CN-FJ",
    protocol: str = "Traveling",
    duration_min: int = 60,
    num_observers: int = 1,
    locality_prefix: str = "Birdy archive",
) -> None:
    _ensure_template_valid(template_path)
    grid = build_checklist_grid(
        bucket,
        species_en_sci,
        country=country,
        state_province=state_province,
        protocol=protocol,
        duration_min=duration_min,
        num_observers=num_observers,
        locality_prefix=locality_prefix,
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerows(grid)


def export_ebird_checklist_files(
    buckets: Dict[Tuple, ChecklistBucket],
    species_en_sci: Dict[str, Tuple[str, str]],
    out_dir: str,
    *,
    template_path: Optional[str] = None,
    country: str = "CN",
    state_province: str = "CN-FJ",
    protocol: str = "Traveling",
    duration_min: int = 60,
    num_observers: int = 1,
    locality_prefix: str = "Birdy archive",
) -> Dict[str, str]:
    """每个 checklist 桶写一个 ``ebird_checklist_{…}.csv``。"""
    _ensure_template_valid(template_path)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    export_moment = datetime.now()
    used_slugs: Set[str] = set()
    written: Dict[str, str] = {}
    paths: List[str] = []

    for _k, bucket in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if not bucket.species_counts:
            continue
        slug = unique_checklist_slug(
            bucket,
            used_slugs,
            region_code=state_province,
            export_moment=export_moment,
        )
        csv_path = out / f"ebird_checklist_{slug}.csv"
        write_ebird_checklist_csv(
            str(csv_path),
            bucket,
            species_en_sci,
            template_path=template_path,
            country=country,
            state_province=state_province,
            protocol=protocol,
            duration_min=duration_min,
            num_observers=num_observers,
            locality_prefix=locality_prefix,
        )
        paths.append(str(csv_path))

    if paths:
        written["ebird_checklist_format_csv"] = paths[0]
        if len(paths) > 1:
            written["ebird_checklist_format_csv_all"] = ";".join(paths)
    return written
