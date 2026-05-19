# -*- coding: utf-8 -*-
"""观鸟记录 .xls 导出依赖（xlrd / xlwt / xlutils）。"""

from __future__ import annotations

_INSTALL_HINT = (
    "观鸟记录 Excel 导出需要 xlrd、xlwt、xlutils。\n"
    "请在**启动 Birdy 所用的同一 Python** 下执行：\n"
    "  python -m pip install xlrd xlwt xlutils\n"
    "或在项目根目录：\n"
    "  python -m pip install -r requirements.txt"
)


def ensure_xls_dependencies() -> None:
    """导入失败时抛出带安装说明的 ImportError。"""
    missing: list[str] = []
    for mod in ("xlrd", "xlwt", "xlutils"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"{_INSTALL_HINT}\n缺少模块: {', '.join(missing)}"
        )
