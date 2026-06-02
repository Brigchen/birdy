#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from record_submit.location_pinyin import (  # noqa: E402
    format_ebird_location_name,
)


def test_country_only_location():
    loc = format_ebird_location_name(
        "厦门大学翔安校区",
        country_code="CN",
        state_province="FJ",
        province_cn="福建",
        city_cn="厦门",
    )
    assert loc == "China"


if __name__ == "__main__":
    test_country_only_location()
    print("ok")
