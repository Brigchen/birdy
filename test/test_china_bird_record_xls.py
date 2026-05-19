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


if __name__ == "__main__":
    test_write_two_columns_only()
    print("ok")
