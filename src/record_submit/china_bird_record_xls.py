# -*- coding: utf-8 -*-
"""中国观鸟记录中心鸟种导入：严格两列 Excel（中文名、数量），与官方模版一致。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .export_filenames import unique_checklist_slug
from .scan import ChecklistBucket

# 模版列（不得增删）：与 data/species/鸟种导入模版.xls 一致
COL_CN_NAME = "中文名"
COL_COUNT = "数量"
_TEMPLATE_HEADERS = (COL_CN_NAME, COL_COUNT)


def default_import_template_path(project_root: Optional[Path] = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent.parent
    return root / "data" / "species" / "鸟种导入模版.xls"


def _validate_template_headers(rb) -> None:
    sh = rb.sheet_by_index(0)
    if sh.ncols != 2:
        raise ValueError(
            f"鸟种导入模版须为 2 列（{COL_CN_NAME}、{COL_COUNT}），当前为 {sh.ncols} 列"
        )
    h0 = str(sh.cell_value(0, 0)).strip()
    h1 = str(sh.cell_value(0, 1)).strip()
    if (h0, h1) != _TEMPLATE_HEADERS:
        raise ValueError(
            f"模版表头应为 {_TEMPLATE_HEADERS!r}，当前为 {h0!r}、{h1!r}"
        )


def write_china_bird_record_xls(
    output_path: str,
    species_counts: Dict[str, int],
    *,
    template_path: Optional[str] = None,
) -> None:
    """
    基于官方 ``鸟种导入模版.xls`` 复制格式，自第 2 行起写入物种与数量。
    仅两列，不增加列。
    """
    import xlrd
    from xlutils.copy import copy as copy_workbook

    tpl = Path(template_path or default_import_template_path()).expanduser().resolve()
    if not tpl.is_file():
        raise FileNotFoundError(f"找不到鸟种导入模版: {tpl}")

    rb = xlrd.open_workbook(str(tpl), formatting_info=True)
    _validate_template_headers(rb)
    wb = copy_workbook(rb)
    ws = wb.get_sheet(0)

    row = 1
    for sp_cn in sorted(species_counts.keys()):
        cnt = int(species_counts[sp_cn])
        if cnt < 1:
            continue
        ws.write(row, 0, sp_cn)
        ws.write(row, 1, cnt)
        row += 1

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out))


def export_china_bird_record_workbooks(
    buckets: Dict[Tuple, ChecklistBucket],
    out_dir: str,
    *,
    template_path: Optional[str] = None,
    region_code: str = "",
) -> List[str]:
    """
    每个 checklist 导出一个 .xls，文件名含英文日期、时间、坐标与导出时刻，避免覆盖。
    例如 ``china_bird_species_20260504_0800_lat24p5919_lon117p9492_exp153045.xls``。
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    tpl = template_path or str(default_import_template_path())
    export_moment = datetime.now()
    used_slugs: Set[str] = set()
    written: List[str] = []

    for _k, bucket in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if not bucket.species_counts:
            continue
        slug = unique_checklist_slug(
            bucket,
            used_slugs,
            region_code=region_code,
            export_moment=export_moment,
        )
        fname = f"china_bird_species_{slug}.xls"
        path = out / fname
        write_china_bird_record_xls(
            str(path),
            bucket.species_counts,
            template_path=tpl,
        )
        written.append(str(path))
    return written
