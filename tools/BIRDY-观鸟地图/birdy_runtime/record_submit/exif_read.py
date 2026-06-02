# -*- coding: utf-8 -*-
"""轻量 EXIF：日期与 GPS（仅依赖 Pillow / piexif，避免拉取 detect_bird 全栈）。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

_IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def _dms_to_decimal(dms: Tuple, ref: bytes | str) -> Optional[float]:
    try:
        if dms is None:
            return None
        d, m, s = dms[0], dms[1], dms[2]
        deg = float(d[0]) / float(d[1]) if d[1] else 0.0
        mn = float(m[0]) / float(m[1]) if m[1] else 0.0
        sc = float(s[0]) / float(s[1]) if s[1] else 0.0
        v = deg + mn / 60.0 + sc / 3600.0
        r = ref.decode("ascii") if isinstance(ref, bytes) else str(ref)
        if r in ("S", "W"):
            v = -abs(v)
        return float(v)
    except Exception:
        return None


def read_gps_from_image(path: str) -> Optional[Tuple[float, float]]:
    path = str(Path(path).expanduser().resolve(strict=False))
    try:
        import piexif

        exif = piexif.load(path)
        gps = exif.get("GPS")
        if not gps:
            return None
        lat = _dms_to_decimal(
            gps.get(piexif.GPSIFD.GPSLatitude),
            gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N"),
        )
        lon = _dms_to_decimal(
            gps.get(piexif.GPSIFD.GPSLongitude),
            gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E"),
        )
        if lat is None or lon is None:
            return None
        if abs(lat) > 90 or abs(lon) > 180:
            return None
        return (lat, lon)
    except Exception:
        pass
    try:
        from PIL import Image
        from PIL.ExifTags import IFD

        with Image.open(path) as im:
            ex = im.getexif()
            if ex is None:
                return None
            gps_ifd = ex.get_ifd(IFD.GPS)
            if not gps_ifd:
                return None
            lat = _dms_to_decimal(gps_ifd.get(2), gps_ifd.get(1, "N"))
            lon = _dms_to_decimal(gps_ifd.get(4), gps_ifd.get(3, "E"))
            if lat is None or lon is None:
                return None
            return (lat, lon)
    except Exception:
        return None


def read_datetime_original(path: str) -> Optional[datetime]:
    path = str(Path(path).expanduser().resolve(strict=False))
    try:
        import piexif

        exif = piexif.load(path)
        ex = exif.get("Exif") or {}
        raw = ex.get(piexif.ExifIFD.DateTimeOriginal) or ex.get(
            piexif.ExifIFD.DateTimeDigitized
        )
        if raw is None:
            z = exif.get("0th") or {}
            raw = z.get(piexif.ImageIFD.DateTime)
        if raw is None:
            return None
        s = raw.decode("ascii", errors="ignore") if isinstance(raw, bytes) else str(raw)
        s = s.strip()
        if len(s) >= 19:
            try:
                return datetime.strptime(s[:19], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        m = re.match(r"(\d{4})[:-](\d{2})[:-](\d{2})", s)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d, 12, 0, 0)
    except Exception:
        pass
    return None


def is_image_path(p: str) -> bool:
    return Path(p).suffix.lower() in _IMG_EXT
