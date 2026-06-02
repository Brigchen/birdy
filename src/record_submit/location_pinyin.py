# -*- coding: utf-8 -*-
"""观鸟记录导出：地点名格式化为 eBird 易匹配的英文地点描述。"""

from __future__ import annotations

import re
from typing import Optional

# eBird 省/州代码 → 英文省名（便于 Fix Locations 时匹配）
_EBIRD_STATE_EN = {
    "FJ": "Fujian",
    "JS": "Jiangsu",
    "ZJ": "Zhejiang",
    "GD": "Guangdong",
    "GX": "Guangxi",
    "HN": "Hainan",
    "SC": "Sichuan",
    "YN": "Yunnan",
    "XZ": "Tibet",
    "BJ": "Beijing",
    "SH": "Shanghai",
    "TJ": "Tianjin",
    "CQ": "Chongqing",
}

_PROVINCE_CN_EN = {
    "福建": "Fujian",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "海南": "Hainan",
    "四川": "Sichuan",
    "云南": "Yunnan",
    "西藏": "Tibet",
    "北京": "Beijing",
    "上海": "Shanghai",
    "天津": "Tianjin",
    "重庆": "Chongqing",
    "台湾": "Taiwan",
}

_COUNTRY_EN = {
    "CN": "China",
    "US": "United States",
    "UK": "United Kingdom",
    "TH": "Thailand",
}


def address_to_pinyin_for_ebird(address: str) -> str:
    """中文地址 → 拼音（仅作地点描述的一部分）。"""
    text = (address or "").strip()
    if not text:
        return ""
    text = text.replace(",", " ").replace(";", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars >= max(1, int(len(text) * 0.6)):
        return text

    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as e:
        raise ImportError(
            "eBird 地点拼音需要 pypinyin，请执行: python -m pip install pypinyin"
        ) from e

    syllables = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
    words = " ".join(s for s in syllables if s)
    if not words:
        return text
    return " ".join(part.capitalize() for part in words.split())


def _city_english(address: str, city_cn: str) -> str:
    city_cn = (city_cn or "").strip()
    if city_cn:
        try:
            from geo_encoder import CHINESE_TO_ENGLISH

            if city_cn in CHINESE_TO_ENGLISH:
                return CHINESE_TO_ENGLISH[city_cn]
        except ImportError:
            pass
        if city_cn in _PROVINCE_CN_EN:
            return _PROVINCE_CN_EN[city_cn]
    try:
        from geo_encoder import _extract_english_name

        en = _extract_english_name(address or city_cn)
        if en:
            return en
    except ImportError:
        pass
    if city_cn and ord(city_cn[0]) < 128:
        return city_cn
    return ""


def _province_english(state_code: str, province_cn: str) -> str:
    province_cn = (province_cn or "").strip()
    if province_cn and province_cn in _PROVINCE_CN_EN:
        return _PROVINCE_CN_EN[province_cn]
    code = (state_code or "").strip().upper()
    if "-" in code:
        code = code.split("-", 1)[1]
    return _EBIRD_STATE_EN.get(code[:3], code[:3] if code else "")


def _locality_detail(address: str, city_cn: str) -> str:
    """地址中除城市外的具体地点（拼音/英文）。"""
    addr = (address or "").strip()
    city_cn = (city_cn or "").strip()
    if city_cn and city_cn in addr:
        addr = addr.replace(city_cn, "", 1).strip(" ,、")
    if not addr:
        return ""
    return address_to_pinyin_for_ebird(addr)


def format_ebird_location_name(
    address: str,
    *,
    country_code: str = "CN",
    state_province: str = "FJ",
    province_cn: str = "",
    city_cn: str = "",
) -> str:
    """
    生成 eBird Location Name（第 1 行 C 列）：仅国家名（如 China）。

    不写具体地名或省州，便于在 eBird 中选择 entire country；
    精确位置靠第 2–3 行的 Latitude / Longitude。
    """
    _ = (address, state_province, province_cn, city_cn)
    code = (country_code or "CN").strip().upper()
    return _COUNTRY_EN.get(code, code if code else "China") or "China"
