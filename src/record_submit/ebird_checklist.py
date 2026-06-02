# -*- coding: utf-8 -*-
"""
eBird Checklist Format：布局与 ``ebird_checklist_format_sample.xls`` 一致。

- 单 checklist 导出为 **3 列**（A/B/C）；勿导出 5 列，否则第 1 行尾 ``,,`` 导致 Missing location name；
- 第 1 行：A/B 空，C=地点名（非空）；
- 第 2–14 行：A=字段名（Latitude、Date…），C=对应值；Date/Time 用文本（M/D/YYYY、h:mm AM），勿用 Excel 序列号；
- 第 15 行起：A=英文俗名，B=学名（eBird Taxonomy），C=只数。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .export_filenames import unique_checklist_slug
from .gpx_travel_distance import GpxTravelDistanceResolver, try_create_gpx_resolver
from .location_pinyin import format_ebird_location_name
from .scan import ChecklistBucket
from .taxonomy_cn import ebird_species_cells, lookup_species

_NUM_COLS = 3  # 单 checklist 仅 A/B/C；sample.xls 的 5 列用于多列示例
_CHECKLIST_COL = 2  # 列 C（第 3 列）= checklist 数据
_SPECIES_ROW_START = 14  # 第 15 行起为物种
# 第 2–14 行 A 列字段名（第 1 行 A 列为空，地点写在 C 列）
_EFFORT_LABELS_COL_A: Tuple[Optional[str], ...] = (
    None,
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


def default_ebird_checklist_sample_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "ebird_checklist_format_sample.xls"


def default_ebird_checklist_template_path(project_root: Optional[Path] = None) -> Path:
    """兼容旧名：校验参照 sample。"""
    return default_ebird_checklist_sample_path(project_root)


def _fmt_date_us(d) -> str:
    return f"{d.month}/{d.day}/{d.year}"


def normalize_ebird_state_province(state_province: str) -> str:
    s = (state_province or "").strip()
    if not s:
        return ""
    if "-" in s:
        left, right = s.split("-", 1)
        if len(left) in (2, 3) and right.strip():
            return right.strip()[:3]
    return s[:3]


def _distance_miles_for_protocol(
    protocol: str,
    distance_mi: Optional[str] = None,
) -> str:
    if distance_mi is not None and str(distance_mi).strip() != "":
        return str(distance_mi).strip()
    p = (protocol or "").strip().lower()
    if p in ("traveling", "biking", "running", "motorized"):
        return "1"
    return ""


def _fmt_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return "8:00 AM"
    h24 = dt.hour
    h = h24 % 12
    if h == 0:
        h = 12
    ampm = "AM" if h24 < 12 else "PM"
    return f"{h}:{dt.minute:02d} {ampm}"


def _validate_checklist_sample(rb) -> None:
    """与 ebird_checklist_format_sample.xls 结构对照。"""
    sh = rb.sheet_by_index(0)
    if sh.ncols < _CHECKLIST_COL + 1:
        raise ValueError(
            f"eBird Checklist 样本须至少 {_CHECKLIST_COL + 1} 列，当前 {sh.ncols}"
        )
    if sh.nrows < _SPECIES_ROW_START + 1:
        raise ValueError("eBird Checklist 样本行数不足")
    if str(sh.cell_value(0, 0)).strip():
        raise ValueError("样本 A1 须为空")
    if str(sh.cell_value(1, 0)).strip() != "Latitude":
        raise ValueError("样本第 2 行 A 列应为 Latitude")
    sp = str(sh.cell_value(_SPECIES_ROW_START, 0)).strip()
    if not sp:
        raise ValueError("样本第 15 行 A 列应为物种名")


def _ensure_sample_valid(sample_path: Optional[str] = None) -> Path:
    path = Path(
        sample_path or default_ebird_checklist_sample_path()
    ).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 eBird Checklist 样本: {path}")
    try:
        import xlrd
    except ImportError as e:
        raise ImportError(
            "校验 eBird 样本需要 xlrd：python -m pip install xlrd"
        ) from e
    _validate_checklist_sample(xlrd.open_workbook(str(path)))
    return path


def _effort_values_for_bucket(
    bucket: ChecklistBucket,
    *,
    country: str,
    state_province: str,
    protocol: str,
    duration_min: int,
    num_observers: int,
    location_name: str,
    province_cn: str = "",
    city_cn: str = "",
    distance_mi: Optional[str] = None,
) -> List[str]:
    state = normalize_ebird_state_province(state_province)
    dist = _distance_miles_for_protocol(protocol, distance_mi)
    loc = format_ebird_location_name(
        location_name,
        country_code=country,
        state_province=state_province,
        province_cn=province_cn,
        city_cn=city_cn,
    )
    if not loc:
        loc = "China"
    lat_s = f"{bucket.lat:.6f}" if bucket.lat is not None else ""
    lon_s = f"{bucket.lon:.6f}" if bucket.lon is not None else ""
    notes = "Exported from Birdy; verify before upload."
    return [
        loc,
        lat_s,
        lon_s,
        _fmt_date_us(bucket.day),
        _fmt_time(bucket.start_time),
        state,
        country,
        protocol,
        str(int(num_observers)),
        str(int(duration_min)),
        "Y",
        dist,
        "",
        notes.replace(",", ";"),
    ]


def _species_rows_for_bucket(
    bucket: ChecklistBucket,
    species_en_sci: Dict[str, Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """(A 列俗名, B 列学名, C 列只数)。"""
    rows: List[Tuple[str, str, str]] = []
    for sp_cn, cnt in sorted(bucket.species_counts.items()):
        en, sci = lookup_species(sp_cn, species_en_sci)
        common, sci_name = ebird_species_cells(en, sci, sp_cn)
        rows.append((common, sci_name, str(int(cnt))))
    return rows


def build_checklist_grid(
    bucket: ChecklistBucket,
    species_en_sci: Dict[str, Tuple[str, str]],
    *,
    country: str = "CN",
    state_province: str = "FJ",
    protocol: str = "Traveling",
    duration_min: int = 60,
    num_observers: int = 1,
    location_name: str = "",
    province_cn: str = "",
    city_cn: str = "",
    distance_mi: Optional[str] = None,
) -> List[List[str]]:
    """3 列网格（A/B/C），与 sample 单 checklist 列布局一致。"""
    species = _species_rows_for_bucket(bucket, species_en_sci)
    nrows = _SPECIES_ROW_START + len(species)
    grid: List[List[str]] = [[""] * _NUM_COLS for _ in range(nrows)]

    effort = _effort_values_for_bucket(
        bucket,
        country=country,
        state_province=state_province,
        protocol=protocol,
        duration_min=duration_min,
        num_observers=num_observers,
        location_name=location_name,
        province_cn=province_cn,
        city_cn=city_cn,
        distance_mi=distance_mi,
    )
    for r, val in enumerate(effort):
        label = _EFFORT_LABELS_COL_A[r]
        if label:
            grid[r][0] = label
        grid[r][_CHECKLIST_COL] = val

    for i, (common, sci_name, count_cell) in enumerate(species):
        row = _SPECIES_ROW_START + i
        grid[row][0] = common
        grid[row][1] = sci_name
        grid[row][_CHECKLIST_COL] = count_cell

    return grid


def write_ebird_checklist_csv(
    output_path: str,
    bucket: ChecklistBucket,
    species_en_sci: Dict[str, Tuple[str, str]],
    *,
    sample_path: Optional[str] = None,
    country: str = "CN",
    state_province: str = "FJ",
    protocol: str = "Traveling",
    duration_min: int = 60,
    num_observers: int = 1,
    location_name: str = "",
    province_cn: str = "",
    city_cn: str = "",
    distance_mi: Optional[str] = None,
) -> None:
    _ensure_sample_valid(sample_path)
    grid = build_checklist_grid(
        bucket,
        species_en_sci,
        country=country,
        state_province=state_province,
        protocol=protocol,
        duration_min=duration_min,
        num_observers=num_observers,
        location_name=location_name,
        province_cn=province_cn,
        city_cn=city_cn,
        distance_mi=distance_mi,
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # 须为 UTF-8 无 BOM；勿用 utf-8-sig 或 Excel 另存（易带 BOM / 尾逗号）
    with open(out, "w", encoding="utf-8", newline="") as f:
        # 禁止字段引号（eBird：CSV 内勿含 "）；物种名中的逗号已替换为分号
        writer = csv.writer(
            f,
            lineterminator="\r\n",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writerows(grid)


def export_ebird_checklist_files(
    buckets: Dict[Tuple, ChecklistBucket],
    species_en_sci: Dict[str, Tuple[str, str]],
    out_dir: str,
    *,
    sample_path: Optional[str] = None,
    template_path: Optional[str] = None,
    country: str = "CN",
    state_province: str = "FJ",
    protocol: str = "Traveling",
    duration_min: int = 60,
    num_observers: int = 1,
    location_name: str = "",
    province_cn: str = "",
    city_cn: str = "",
    gpx_file_path: Optional[str] = None,
    gpx_file_paths: Optional[Sequence[str]] = None,
    gpx_exif_tz: str = "Asia/Shanghai",
    gpx_track_tz: str = "UTC",
    gpx_resolver: Optional[GpxTravelDistanceResolver] = None,
) -> Dict[str, str]:
    ref = sample_path or template_path
    _ensure_sample_valid(ref)
    resolver = gpx_resolver
    if resolver is None:
        resolver = try_create_gpx_resolver(
            gpx_file_path,
            gpx_file_paths=gpx_file_paths,
            exif_tz=gpx_exif_tz,
            gpx_tz=gpx_track_tz,
        )
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
        dist_mi: Optional[str] = None
        if resolver is not None:
            dist_mi = resolver.miles_for_bucket(bucket)
        write_ebird_checklist_csv(
            str(csv_path),
            bucket,
            species_en_sci,
            sample_path=ref,
            country=country,
            state_province=state_province,
            protocol=protocol,
            duration_min=duration_min,
            num_observers=num_observers,
            location_name=location_name,
            province_cn=province_cn,
            city_cn=city_cn,
            distance_mi=dist_mi,
        )
        paths.append(str(csv_path))

    if paths:
        written["ebird_checklist_format_csv"] = paths[0]
        if len(paths) > 1:
            written["ebird_checklist_format_csv_all"] = ";".join(paths)
    return written
