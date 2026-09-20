# -*- coding: utf-8 -*-
"""从 EXIF 提取镜头、焦距、ISO、光圈、机身与传感器采样率。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ExifTags


def _rat_to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if hasattr(v, "numerator") and hasattr(v, "denominator"):
            den = float(v.denominator)
            return None if den == 0 else float(v.numerator) / den
        if isinstance(v, (tuple, list)) and len(v) == 2:
            den = float(v[1])
            return None if den == 0 else float(v[0]) / den
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _decode(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        for enc in ("utf-8", "latin-1"):
            try:
                return v.decode(enc, errors="ignore").strip("\x00").strip()
            except Exception:
                continue
        return ""
    return str(v).strip()


def _exif_dict(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        img = Image.open(path)
    except Exception:
        return out
    try:
        exif = img.getexif()
        if not exif:
            return out
        for tag_id, val in exif.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            out[name] = val
        ifd = getattr(exif, "get_ifd", None)
        if callable(ifd):
            try:
                extra = ifd(0x8769)  # ExifIFD
            except Exception:
                extra = {}
            if extra:
                for tag_id, val in extra.items():
                    name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    out.setdefault(name, val)
    finally:
        img.close()
    return out


def _pixels_per_mm(exif: Dict[str, Any]) -> Optional[float]:
    res = _rat_to_float(exif.get("FocalPlaneXResolution"))
    unit = exif.get("FocalPlaneResolutionUnit")
    try:
        unit_i = int(unit) if unit is not None else 0
    except (TypeError, ValueError):
        unit_i = 0
    if res is None or res <= 0:
        return None
    # 1=无, 2=英寸, 3=厘米, 4=毫米
    if unit_i == 2:
        return res / 25.4
    if unit_i == 3:
        return res / 10.0
    if unit_i == 4:
        return res
    return None


def _parse_datetime(raw: str) -> Optional[datetime]:
    s = (raw or "").strip()
    if not s:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def read_photo_meta(path: str) -> Dict[str, Any]:
    exif = _exif_dict(path)
    lens = (
        _decode(exif.get("LensModel"))
        or _decode(exif.get("LensMake"))
        or _decode(exif.get("Lens"))
        or ""
    )
    camera = " ".join(
        x for x in (_decode(exif.get("Make")), _decode(exif.get("Model"))) if x
    ).strip()
    focal = _rat_to_float(exif.get("FocalLength"))
    focal35 = _rat_to_float(exif.get("FocalLengthIn35mmFilm"))
    iso = _rat_to_float(exif.get("PhotographicSensitivity"))
    if iso is None:
        iso = _rat_to_float(exif.get("ISOSpeedRatings"))
        if iso is None and isinstance(exif.get("ISOSpeedRatings"), (list, tuple)):
            iso = _rat_to_float(exif.get("ISOSpeedRatings")[0])
    aperture = _rat_to_float(exif.get("FNumber"))
    if aperture is None:
        aperture = _rat_to_float(exif.get("ApertureValue"))
    dt = _parse_datetime(_decode(exif.get("DateTimeOriginal")) or _decode(exif.get("DateTime")))
    ppm = _pixels_per_mm(exif)
    w, h = None, None
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        pass
    return {
        "path": path,
        "file": Path(path).name,
        "lens": lens or "未知镜头",
        "camera": camera or "未知机身",
        "focal_mm": round(focal, 1) if focal else None,
        "focal_35mm": round(focal35, 1) if focal35 else None,
        "iso": int(round(iso)) if iso else None,
        "aperture": round(aperture, 1) if aperture else None,
        "datetime": dt.isoformat(sep=" ") if dt else "",
        "datetime_ts": dt.timestamp() if dt else None,
        "pixels_per_mm": ppm,
        "width": w,
        "height": h,
    }


def format_focal(mm: Optional[float]) -> str:
    if mm is None:
        return "—"
    if abs(mm - round(mm)) < 0.05:
        return f"{int(round(mm))}mm"
    return f"{mm:.1f}mm"


def group_label(meta: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    parts = []
    for k in keys:
        if k == "lens":
            parts.append(str(meta.get("lens") or "未知镜头"))
        elif k == "focal":
            parts.append(format_focal(meta.get("focal_mm")))
        elif k == "iso":
            iso = meta.get("iso")
            parts.append(f"ISO{iso}" if iso else "ISO—")
        elif k == "aperture":
            ap = meta.get("aperture")
            parts.append(f"f/{ap}" if ap else "f/—")
        elif k == "camera":
            parts.append(str(meta.get("camera") or "未知机身"))
    return " · ".join(parts) if parts else "全部"
