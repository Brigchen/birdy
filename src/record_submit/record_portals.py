# -*- coding: utf-8 -*-
"""观鸟记录对外网站入口（eBird / 中国观鸟记录中心）。"""

from __future__ import annotations

# eBird 观测数据无公开「提交 checklist」API，仅支持网页导入 CSV（Record Format、无表头）。
EBIRD_IMPORT_URL = "https://ebird.org/import/upload.form?theme=ebird"

# 中国观鸟记录中心：鸟种导入为网页/小程序上传 Excel（鸟种导入模版.xls）
CHINA_BIRD_RECORD_HOME_URL = "http://www.birdreport.cn/"

PORTAL_LINKS = (
    ("ebird上传网页", EBIRD_IMPORT_URL),
    ("中国观鸟记录中心", CHINA_BIRD_RECORD_HOME_URL),
)
