#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from record_submit.china_bird_record_xls import (  # noqa: E402
    default_import_template_path,
    write_china_bird_record_xls,
)


def test_write_two_columns_only():
    tpl = default_import_template_path()
    assert tpl.is_file(), tpl
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "鸟种导入.xls"
        write_china_bird_record_xls(
            str(out),
            {"白鹭": 3, "麻雀": 1},
            template_path=str(tpl),
        )
        import xlrd

        sh = xlrd.open_workbook(str(out)).sheet_by_index(0)
        assert sh.ncols == 2
        assert sh.cell_value(0, 0) == "中文名"
        assert sh.cell_value(0, 1) == "数量"
        rows = [
            [sh.cell_value(r, c) for c in range(2)] for r in range(1, sh.nrows)
        ]
        data = {r[0]: int(r[1]) for r in rows}
        assert data == {"白鹭": 3, "麻雀": 1}

        write_china_bird_record_xls(
            str(out),
            {"麻雀": 1, "反嘴鹬": 2, "苍鹭": 3},
            template_path=str(tpl),
        )
        sh = xlrd.open_workbook(str(out)).sheet_by_index(0)
        names = [sh.cell_value(r, 0) for r in range(1, sh.nrows)]
        assert names == ["苍鹭", "反嘴鹬", "麻雀"]


def test_zhengsi_names_replace_merge_and_order():
    """学名相同则改成郑四中文名；多名并到同一郑四名时只数相加，并按名录序号排列。"""
    tpl = default_import_template_path()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "鸟种导入.xls"
        write_china_bird_record_xls(
            str(out),
            {
                "黄苇鳽": 2,
                "黄斑苇鳽": 1,
                "西方松鸡": 1,
                "麻雀": 4,
                "不是一种鸟": 1,
                "大山雀": 2,
            },
            template_path=str(tpl),
        )
        import xlrd

        sh = xlrd.open_workbook(str(out)).sheet_by_index(0)
        rows = [
            (sh.cell_value(r, 0), int(sh.cell_value(r, 1)))
            for r in range(1, sh.nrows)
        ]
        assert rows == [
            ("松鸡", 1),
            ("黄斑苇鳽", 3),
            ("欧亚大山雀", 2),
            ("麻雀", 4),
            ("不是一种鸟", 1),
        ]


if __name__ == "__main__":
    test_write_two_columns_only()
    test_zhengsi_names_replace_merge_and_order()
    print("ok")
