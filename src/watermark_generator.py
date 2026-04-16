# -*- coding: utf-8 -*-
"""
图片水印生成模块（Leica 风格边框 + 底栏文字 + 图内签名 Logo）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable

from PIL import Image, ImageDraw, ImageFont, ImageStat

try:
    import piexif
except Exception:  # pragma: no cover
    piexif = None

try:
    from geo_encoder import read_gps_exif
except Exception:  # pragma: no cover
    read_gps_exif = None  # type: ignore

try:
    from detect_bird_and_eye import locate_province, locate_city
except Exception:  # pragma: no cover
    locate_province = None  # type: ignore
    locate_city = None  # type: ignore


@dataclass
class WatermarkOptions:
    enable_location: bool = True
    location_text: str = ""
    use_gps_city: bool = True
    enable_date: bool = True
    enable_species: bool = True
    enable_camera_params: bool = True
    logo_path: str = ""
    logo_width_ratio: float = 0.30


def _safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _collect_images_recursive(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    out: List[str] = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p))
    return sorted(out)


def collect_images_recursive(root: str) -> List[str]:
    """公开：递归收集图片路径。"""
    return _collect_images_recursive(root)


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    probe = "中文Birdy"
    for fp in candidates:
        if os.path.isfile(fp):
            try:
                f = ImageFont.truetype(fp, size)
                # 仅接受可渲染中文的字体，避免出现方框/乱码。
                box = f.getbbox(probe)
                if box and (box[2] - box[0]) > 0:
                    return f
            except Exception:
                continue
    return ImageFont.load_default()


def _extract_exif_datetime(path: str) -> str:
    try:
        if piexif is None:
            return ""
        exif = piexif.load(path)
        dt = (
            exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            or exif.get("0th", {}).get(piexif.ImageIFD.DateTime)
        )
        if not dt:
            return ""
        if isinstance(dt, bytes):
            dt = dt.decode("utf-8", errors="ignore")
        s = str(dt).strip()
        # 2026:04:15 13:30:00 -> 2026-04-15
        if len(s) >= 10:
            return s[:10].replace(":", "-")
        return s
    except Exception:
        return ""


def _extract_exif_camera_params(path: str) -> str:
    try:
        if piexif is None:
            return ""
        exif = piexif.load(path)
        exif_ifd = exif.get("Exif", {})
        zeroth_ifd = exif.get("0th", {})
        model = zeroth_ifd.get(piexif.ImageIFD.Model)
        fnum = exif_ifd.get(piexif.ExifIFD.FNumber)
        expo = exif_ifd.get(piexif.ExifIFD.ExposureTime)
        iso = exif_ifd.get(piexif.ExifIFD.ISOSpeedRatings)
        focal = exif_ifd.get(piexif.ExifIFD.FocalLength)

        parts: List[str] = []
        # 仅显示机身型号（不含品牌 Make）
        if model:
            if isinstance(model, bytes):
                ms = model.decode("utf-8", errors="ignore").strip()
            else:
                ms = str(model).strip()
            if ms:
                parts.append(ms)
        if fnum and isinstance(fnum, tuple) and len(fnum) == 2 and fnum[1]:
            parts.append(f"f/{fnum[0] / fnum[1]:.1f}")
        if expo and isinstance(expo, tuple) and len(expo) == 2 and expo[1]:
            v = expo[0] / expo[1]
            if v < 1:
                parts.append(f"1/{int(round(1 / max(v, 1e-6)))}s")
            else:
                parts.append(f"{v:.1f}s")
        if focal and isinstance(focal, tuple) and len(focal) == 2 and focal[1]:
            parts.append(f"{int(round(focal[0] / focal[1]))}mm")
        if iso:
            if isinstance(iso, (tuple, list)):
                iso = iso[0]
            parts.append(f"ISO{int(iso)}")
        return "  ".join(parts)
    except Exception:
        return ""


def _city_from_gps(path: str) -> str:
    try:
        if read_gps_exif is None or locate_province is None or locate_city is None:
            return ""
        got = read_gps_exif(path, quiet=True)  # type: ignore
        if not got:
            return ""
        lat, lon = float(got[0]), float(got[1])
        prov = locate_province(lon, lat)  # type: ignore
        if not prov:
            return ""
        city = locate_city(lon, lat, prov)  # type: ignore
        return city or prov
    except Exception:
        return ""


def _species_from_path(img_path: str, source_root: str) -> str:
    p = Path(img_path).resolve()
    root = Path(source_root).resolve()
    try:
        rel = p.parent.relative_to(root)
    except Exception:
        rel = p.parent
    parts = rel.parts
    if not parts:
        return "未知"
    # 优先最后一级目录名
    return str(parts[-1]) or "未知"


def _fit_logo(logo: Image.Image, target_w: int, target_h: int) -> Image.Image:
    lw, lh = logo.size
    if lw <= 0 or lh <= 0:
        return logo
    scale = min(target_w / lw, target_h / lh)
    nw = max(1, int(lw * scale))
    nh = max(1, int(lh * scale))
    return logo.resize((nw, nh), Image.LANCZOS)


def _wrap_text_lines(
    draw: ImageDraw.ImageDraw, text: str, font, max_w: int
) -> List[str]:
    """按像素宽度自动换行（中文按字切分）。"""
    s = (text or "").strip()
    if not s or max_w <= 0:
        return []
    lines: List[str] = []
    cur = ""
    for ch in s:
        trial = cur + ch
        bbox = draw.textbbox((0, 0), trial, font=font)
        tw = bbox[2] - bbox[0]
        if tw <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _compose_leica_style(
    img: Image.Image,
    text_left: str,
    text_right: str,
    logo: Optional[Image.Image],
    logo_width_ratio: float = 0.30,
) -> Image.Image:
    w, h = img.size
    # 上、左、右白边：原 (宽+高)×5% 的 1/3（与灰色细线分开计量）
    side_white = max(2, int(0.01 * float(w + h)))
    gray_w = 1  # 界分白边与照片的灰色细框
    img_x = side_white + gray_w
    img_y = side_white + gray_w
    out_inner_w = gray_w + w + gray_w  # 左灰 + 图 + 右灰
    out_inner_h = gray_w + h + gray_w  # 上灰 + 图 + 下灰（图片下缘与底栏之间）
    canvas_w = side_white + out_inner_w + side_white
    y_bar_top = side_white + out_inner_h

    font_size = max(18, int(min(w, h) * 0.019))
    font = _get_font(font_size)
    pad_x = max(15, int(canvas_w * 0.018))
    pad_y_bar = max(12, int(font_size * 0.65))
    center_gap = max(16, int(canvas_w * 0.028))
    half_w = max(40, (canvas_w - 2 * pad_x - center_gap) // 2)
    draw_measure = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    left_lines = _wrap_text_lines(draw_measure, text_left, font, half_w)
    right_lines = _wrap_text_lines(draw_measure, text_right, font, half_w)
    n_left = len(left_lines) if left_lines else 0
    n_right = len(right_lines) if right_lines else 0
    n_lines = max(n_left, n_right, 1)
    bbox_ln = draw_measure.textbbox((0, 0), "国Ag", font=font)
    line_h = max(font_size + 2, bbox_ln[3] - bbox_ln[1])
    line_gap = max(2, int(line_h * 0.12))
    bar_h = max(44, pad_y_bar * 2 + n_lines * line_h + (n_lines - 1) * line_gap)

    out_h = y_bar_top + bar_h
    canvas = Image.new("RGB", (canvas_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    gray = (148, 148, 148)
    oiw = gray_w + w + gray_w
    oih = gray_w + h + gray_w
    # 上/左/右/下（图片与底栏之间）灰色细带：界分外侧白边与照片
    draw.rectangle(
        [side_white, side_white, side_white + oiw - 1, side_white + gray_w - 1],
        fill=gray,
    )
    draw.rectangle(
        [
            side_white,
            side_white + gray_w + h,
            side_white + oiw - 1,
            side_white + oih - 1,
        ],
        fill=gray,
    )
    draw.rectangle(
        [
            side_white,
            side_white + gray_w,
            side_white + gray_w - 1,
            side_white + gray_w + h - 1,
        ],
        fill=gray,
    )
    draw.rectangle(
        [
            side_white + gray_w + w,
            side_white + gray_w,
            side_white + oiw - 1,
            side_white + gray_w + h - 1,
        ],
        fill=gray,
    )

    canvas.paste(img, (img_x, img_y))

    # 底栏与照片之间的细分隔线
    draw.line([(0, y_bar_top), (canvas_w, y_bar_top)], fill=gray, width=1)

    text_fill = (28, 28, 28)
    y_text = y_bar_top + pad_y_bar
    for i, ln in enumerate(left_lines):
        yy = y_text + i * (line_h + line_gap)
        draw.text((pad_x, yy), ln, fill=text_fill, font=font)
    for i, ln in enumerate(right_lines):
        yy = y_text + i * (line_h + line_gap)
        bbox = draw.textbbox((0, 0), ln, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((canvas_w - pad_x - tw, yy), ln, fill=text_fill, font=font)

    # 图中 logo（底框上方中间）
    if logo is not None:
        ratio = min(0.8, max(0.05, float(logo_width_ratio)))
        area_w = max(40, int(w * ratio))
        area_h = max(28, int(h * 0.16))
        lg = _fit_logo(logo.convert("RGBA"), area_w, area_h)
        lx = img_x + max(0, (w - lg.size[0]) // 2)
        ly = img_y + h - lg.size[1] - max(8, int(h * 0.02))
        rx2 = min(img_x + w, lx + lg.size[0])
        ry2 = min(img_y + h, ly + lg.size[1])
        bg_luma = 0.0
        if rx2 > lx and ry2 > ly:
            bg_crop = canvas.crop((lx, ly, rx2, ry2)).convert("L")
            try:
                bg_luma = float(ImageStat.Stat(bg_crop).mean[0])
            except Exception:
                bg_luma = 0.0
        # 默认纯白/亮色剪影；仅当 logo 落点背景为白或近白时才用深色以保证可见性
        near_white = 200.0
        if bg_luma >= near_white:
            fg_rgb = (22, 22, 22)
        else:
            fg_rgb = (255, 255, 255)

        logo_rgba = lg.convert("RGBA")
        alpha = logo_rgba.split()[-1]
        alpha = alpha.point(lambda p: int(p * 0.92))
        fg_layer = Image.new("RGBA", lg.size, fg_rgb + (0,))
        fg_layer.putalpha(alpha)
        canvas.paste(fg_layer, (lx, ly), fg_layer)

    return canvas


def generate_watermarks(
    source_folder: str,
    output_folder: str,
    options: WatermarkOptions,
    prefer_folder_name_as_species: bool = True,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, int]:
    """
    批量生成水印图。
    """
    os.makedirs(output_folder, exist_ok=True)
    images = _collect_images_recursive(source_folder)
    logo_img = None
    if options.logo_path and os.path.isfile(options.logo_path):
        try:
            logo_img = Image.open(options.logo_path).convert("RGBA")
        except Exception:
            logo_img = None

    ok = 0
    fail = 0
    total = len(images)
    if progress_callback:
        try:
            progress_callback({"kind": "start", "done": 0, "total": max(1, total)})
        except Exception:
            pass
    for img_path in images:
        img = _safe_open_image(img_path)
        if img is None:
            fail += 1
            if progress_callback:
                try:
                    progress_callback(
                        {"kind": "tick", "done": ok + fail, "total": max(1, total)}
                    )
                except Exception:
                    pass
            continue
        try:
            loc = ""
            if options.enable_location:
                if options.location_text.strip():
                    loc = options.location_text.strip()
                elif options.use_gps_city:
                    loc = _city_from_gps(img_path)
            dt = _extract_exif_datetime(img_path) if options.enable_date else ""
            species = (
                _species_from_path(img_path, source_folder)
                if (options.enable_species and prefer_folder_name_as_species)
                else ""
            )
            cam = _extract_exif_camera_params(img_path) if options.enable_camera_params else ""

            left_fields = [x for x in (species, loc, dt) if x]
            right_fields = [x for x in (cam,) if x]
            left = "  |  ".join(left_fields) if left_fields else "Birdy"
            right = "  |  ".join(right_fields)

            out = _compose_leica_style(
                img, left, right, logo_img, options.logo_width_ratio
            )

            # 不再按原目录层级保存，统一直接输出到目标目录根下
            src = Path(img_path)
            dst = Path(output_folder) / src.name
            if dst.exists():
                # 文件重名时追加序号，避免覆盖
                stem = src.stem
                suf = src.suffix or ".jpg"
                i = 1
                while True:
                    cand = Path(output_folder) / f"{stem}_{i}{suf}"
                    if not cand.exists():
                        dst = cand
                        break
                    i += 1
            out.save(str(dst), quality=95)
            ok += 1
        except Exception:
            fail += 1
        if progress_callback:
            try:
                progress_callback(
                    {"kind": "tick", "done": ok + fail, "total": max(1, total)}
                )
            except Exception:
                pass

    if progress_callback:
        try:
            progress_callback({"kind": "done", "done": max(1, total), "total": max(1, total)})
        except Exception:
            pass
    return {"total": len(images), "ok": ok, "fail": fail}


def render_watermark_for_image(
    image_path: str,
    source_folder: str,
    options: WatermarkOptions,
    prefer_folder_name_as_species: bool = True,
) -> Optional[Image.Image]:
    """
    对单张图片渲染水印效果（用于 GUI 预览）。
    """
    img = _safe_open_image(image_path)
    if img is None:
        return None

    logo_img = None
    if options.logo_path and os.path.isfile(options.logo_path):
        try:
            logo_img = Image.open(options.logo_path).convert("RGBA")
        except Exception:
            logo_img = None

    loc = ""
    if options.enable_location:
        if options.location_text.strip():
            loc = options.location_text.strip()
        elif options.use_gps_city:
            loc = _city_from_gps(image_path)
    dt = _extract_exif_datetime(image_path) if options.enable_date else ""
    species = (
        _species_from_path(image_path, source_folder)
        if (options.enable_species and prefer_folder_name_as_species)
        else ""
    )
    cam = _extract_exif_camera_params(image_path) if options.enable_camera_params else ""

    left_fields = [x for x in (species, loc, dt) if x]
    right_fields = [x for x in (cam,) if x]
    left = "  |  ".join(left_fields) if left_fields else "Birdy"
    right = "  |  ".join(right_fields)
    return _compose_leica_style(img, left, right, logo_img, options.logo_width_ratio)


def choose_default_watermark_source(
    image_folder: str,
    crop_output_folder: str,
    output_folder: str,
) -> str:
    """
    不指定水印输入目录时，默认优先归档 ROI 图目录；其次 Screened_images；最后原图目录。
    """
    candidates = [
        crop_output_folder,
        os.path.join(output_folder, "Screened_images"),
        image_folder,
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            imgs = _collect_images_recursive(c)
            if imgs:
                return c
    return image_folder

