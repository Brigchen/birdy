# -*- coding: utf-8 -*-
"""
图片水印生成：外框模式（Leica 风格边框 + 底栏文字 + 图内签名），
或 inline 模式（无外框，图内中下方「签名 | 竖线 | 物种/城市地点」两行标签）。
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Literal, Optional, Callable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageStat

from image_io import all_supported_extensions, open_pil_rgb

from watermark_enhance import apply_auto_enhance_pil, load_overrides_json

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore

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


WatermarkStyle = Literal["frame", "inline"]


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
    # frame：外框 + 底栏文字 + 图内签名；inline：无外框，图内「签名 | 竖线 | 物种/城市地点」两行标签
    watermark_style: WatermarkStyle = "frame"
    # 水印前自动生态显影（与 RAW 入库显影同源逻辑）；逐张微调见 enhance_overrides_path
    enable_auto_enhance: bool = True
    enhance_overrides_path: str = ""  # 临时 JSON，含 by_relpath → strength / exposure_fine
    # AI 增强（水印前预处理，与 auto_enhance 独立）
    # 流水线顺序：自动曝光 → AI 降噪 → AI 锐化
    enable_ai_exposure: bool = False  # 基于鸟体测光的自动曝光（避免剪影）
    ai_exposure_strength: float = 1.0  # 0=原图，1=完全调整
    enable_ai_denoise: bool = False
    enable_ai_sharpen: bool = False
    ai_denoise_model: str = "realesrgan"  # "realesrgan" 或 "nafnet"
    ai_denoise_strength: float = 0.5  # 0=原图，1=完全降噪
    ai_sharpen_strength: float = 0.5  # 0=原图，1=完全锐化
    ai_tile_size: int = 512  # 分块大小，越大越快但越吃显存


def _safe_open_image(path: str) -> Optional[Image.Image]:
    return open_pil_rgb(path, raw_half_size=False)


def _collect_images_recursive(root: str) -> List[str]:
    exts = all_supported_extensions()
    out: List[str] = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(str(p))
    return sorted(out)


def collect_images_recursive(root: str) -> List[str]:
    """公开：递归收集图片路径。"""
    return _collect_images_recursive(root)


def sample_images_per_species_dir(
    images: Sequence[str],
    per_dir: int,
    *,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    按「物种目录」（图片所在父目录）分组，每组随机抽取至多 per_dir 张。
    per_dir <= 0 时返回原列表副本（不抽样）。
    """
    paths = [str(p) for p in images]
    if per_dir <= 0 or not paths:
        return list(paths)
    rnd = rng if rng is not None else random.Random()
    by_dir: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        by_dir[str(Path(p).parent)].append(p)
    out: List[str] = []
    for _dir in sorted(by_dir.keys()):
        group = by_dir[_dir]
        if len(group) <= per_dir:
            out.extend(group)
        else:
            out.extend(rnd.sample(group, per_dir))
    return out


def apply_watermark_photo_pipeline(
    img: Image.Image,
    options: WatermarkOptions,
    src_abs_path: str,
    source_folder: str,
    overrides_by_rel: Optional[Dict[str, dict]] = None,
) -> Image.Image:
    """
    水印前：可选自动生态显影 + 逐张 overrides（strength / exposure_fine）。
    """
    if not getattr(options, "enable_auto_enhance", True):
        return img
    try:
        from watermark_enhance import rel_key
    except Exception:
        return img

    rel = rel_key(src_abs_path, source_folder)
    ov: dict = {}
    if overrides_by_rel and rel in overrides_by_rel:
        ov = dict(overrides_by_rel[rel])
    try:
        return apply_auto_enhance_pil(img, ov)
    except Exception:
        return img


# ---- AI 增强（Real-ESRGAN 降噪 + OmniSR 锐化）----
_ai_enhancer_cache = None


def _ai_enhance_image(img: Image.Image, options: WatermarkOptions) -> Image.Image:
    """水印前 AI 增强，顺序：自动曝光 → AI 降噪 → AI 锐化。降级时返回原图。"""
    # 1) 自动曝光（基于鸟体测光）
    if getattr(options, "enable_ai_exposure", False):
        try:
            from auto_exposure import auto_expose_pil
            img = auto_expose_pil(
                img,
                strength=float(getattr(options, "ai_exposure_strength", 1.0)),
                detect=True,
            )
        except Exception as e:
            print(f"[watermark] 自动曝光失败，跳过: {e}", flush=True)

    # 2) AI 降噪 + 3) AI 锐化
    if not (options.enable_ai_denoise or options.enable_ai_sharpen):
        return img
    global _ai_enhancer_cache
    if _ai_enhancer_cache is not None:
        enhancer = _ai_enhancer_cache
    else:
        try:
            from ai_enhance import AIEnhancer

            enhancer = AIEnhancer(tile_size=getattr(options, "ai_tile_size", 512))
        except Exception as e:
            print(f"[watermark] ai_enhance 模块不可用: {e}", flush=True)
            enhancer = None
        _ai_enhancer_cache = enhancer
    if enhancer is None:
        return img
    try:
        return enhancer.enhance_pil(
            img,
            denoise=options.enable_ai_denoise,
            denoise_strength=options.ai_denoise_strength,
            sharpen=options.enable_ai_sharpen,
            sharpen_strength=options.ai_sharpen_strength,
            denoise_model=getattr(options, "ai_denoise_model", "realesrgan"),
        )
    except Exception as e:
        print(f"[watermark] AI 增强失败，使用原图: {e}", flush=True)
        return img


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
    # 优先 piexif，失败则回退 PIL getexif()
    try:
        if piexif is not None:
            exif = piexif.load(path)
            dt = (
                exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
                or exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeDigitized)
                or exif.get("0th", {}).get(piexif.ImageIFD.DateTime)
            )
            if dt:
                if isinstance(dt, bytes):
                    dt = dt.decode("utf-8", errors="ignore")
                s = str(dt).strip()
                # 2026:04:15 13:30:00 -> 2026-04-15
                if len(s) >= 10:
                    return s[:10].replace(":", "-")
                return s
    except Exception as e:
        print(f"[watermark exif] _extract_exif_datetime piexif 失败: {path} -> {e}", flush=True)

    # PIL fallback：getexif() 可读出部分 piexif 漏掉的 EXIF
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            if ex:
                # 36867=DateTimeOriginal, 36868=DateTimeDigitized, 306=DateTime
                dt = ex.get(36867) or ex.get(36868) or ex.get(306)
                if dt:
                    s = str(dt).strip()
                    if len(s) >= 10:
                        return s[:10].replace(":", "-")
                    return s
    except Exception as e:
        print(f"[watermark exif] _extract_exif_datetime PIL fallback 失败: {path} -> {e}", flush=True)
    return ""


def _extract_exif_camera_params(path: str) -> str:
    exif_ifd = {}
    zeroth_ifd = {}

    # 优先 piexif
    piexif_ok = False
    try:
        if piexif is not None:
            exif = piexif.load(path)
            exif_ifd = exif.get("Exif", {}) or {}
            zeroth_ifd = exif.get("0th", {}) or {}
            piexif_ok = True
    except Exception as e:
        print(f"[watermark exif] _extract_exif_camera_params piexif 失败: {path} -> {e}", flush=True)

    # PIL fallback
    if not piexif_ok or (not exif_ifd and not zeroth_ifd):
        try:
            with Image.open(path) as im:
                ex = im.getexif()
                if ex:
                    zeroth_ifd = dict(zeroth_ifd)
                    for k, v in ex.items():
                        zeroth_ifd.setdefault(k, v)
                    # Exif IFD tag id = 34665；get_ifd 在 Pillow 7.2+ 可用
                    try:
                        exif_sub = ex.get_ifd(34665)
                    except Exception:
                        exif_sub = None
                    if exif_sub:
                        for k, v in exif_sub.items():
                            exif_ifd.setdefault(k, v)
        except Exception as e:
            print(f"[watermark exif] _extract_exif_camera_params PIL fallback 失败: {path} -> {e}", flush=True)

    def _decode(v):
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore").strip()
        return str(v).strip() if v is not None else ""

    model = _decode(zeroth_ifd.get(piexif.ImageIFD.Model if piexif else 272)) if zeroth_ifd else ""
    fnum = exif_ifd.get(piexif.ExifIFD.FNumber if piexif else 33434)
    expo = exif_ifd.get(piexif.ExifIFD.ExposureTime if piexif else 33434)
    iso = exif_ifd.get(piexif.ExifIFD.ISOSpeedRatings if piexif else 34855)
    focal = exif_ifd.get(piexif.ExifIFD.FocalLength if piexif else 37386)

    parts: List[str] = []
    # 仅显示机身型号（不含品牌 Make）
    if model:
        parts.append(model)
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
        try:
            parts.append(f"ISO{int(iso)}")
        except (TypeError, ValueError):
            pass

    if not parts:
        print(
            f"[watermark exif] 相机参数为空: {path}  "
            f"piexif_ok={piexif_ok} zeroth_keys={list(zeroth_ifd.keys())[:8]} "
            f"exif_keys={list(exif_ifd.keys())[:8]}",
            flush=True,
        )
    return "  ".join(parts)


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


def _inline_place_line(img_path: str, options: WatermarkOptions) -> str:
    """inline 模式底行：城市（GPS）+ 地点（人工优先与 GPS 组合）。"""
    if not options.enable_location:
        return ""
    manual = (options.location_text or "").strip()
    city = ""
    if options.use_gps_city:
        city = (_city_from_gps(img_path) or "").strip()
    if manual and city:
        return f"{city} {manual}"
    if manual:
        return manual
    return city


def _text_pixel_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    b = draw.textbbox((0, 0), text, font=font)
    return max(0, b[2] - b[0])


def _truncate_line_to_width(
    draw: ImageDraw.ImageDraw, text: str, font, max_w: int
) -> str:
    s = (text or "").strip()
    if not s or max_w <= 1:
        return s
    ell = "…"
    if _text_pixel_width(draw, s, font) <= max_w:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        trial = s[:mid] + ell
        if _text_pixel_width(draw, trial, font) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    if lo <= 0:
        return ell
    return s[:lo] + ell


def _species_from_path(img_path: str, source_root: str) -> str:
    p = Path(img_path).resolve()
    root = Path(source_root).resolve()
    try:
        rel = p.parent.relative_to(root)
    except Exception:
        rel = p.parent
    parts = rel.parts
    # 直接选中物种级目录时，图片与 source_root 同级，relative 为空
    if not parts:
        return root.name or "未知"
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


def _norm_watermark_style(options: WatermarkOptions) -> WatermarkStyle:
    s = getattr(options, "watermark_style", "frame") or "frame"
    if s not in ("frame", "inline"):
        return "frame"
    return s  # type: ignore[return-value]


def _inline_text_line_height(draw: ImageDraw.ImageDraw, font) -> int:
    """与标签所用字体一致的单行视觉高度（中文+拉丁混排参考）。"""
    bbox = draw.textbbox((0, 0), "国AgyM", font=font)
    return max(1, bbox[3] - bbox[1])


def _inline_font_for_ref_height(
    ref_h: int, nrows: int
) -> tuple:
    """
    在竖条标签区按 ref_h（与签名 Logo 实际高度同量级）选最大可用字号。
    nrows 行时满足 nrows*line_h + (nrows-1)*gap <= ref_h。
    """
    ref_h = max(12, int(ref_h))
    gap = max(2, int(ref_h * 0.12))
    measure = ImageDraw.Draw(Image.new("RGB", (max(64, ref_h * 8), ref_h * 4)))

    def fits(fs: int) -> bool:
        font = _get_font(fs)
        lh = _inline_text_line_height(measure, font)
        return nrows * lh + (nrows - 1) * gap <= ref_h

    hi = min(320, max(24, ref_h * 5))
    best = 8
    for fs in range(hi, 7, -1):
        if fits(fs):
            best = fs
            break
    font = _get_font(best)
    line_h = _inline_text_line_height(measure, font)
    return font, line_h, gap


def _fg_rgb_for_bottom_overlay(img_rgb: Image.Image, y0: int, y1: int) -> tuple:
    """根据图像底部一带亮度选择前景色（与图内签名剪影规则一致）。"""
    w, h = img_rgb.size
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    crop = img_rgb.crop((0, y0, w, y1)).convert("L")
    try:
        luma = float(ImageStat.Stat(crop).mean[0])
    except Exception:
        luma = 128.0
    if luma >= 200.0:
        return (22, 22, 22)
    return (255, 255, 255)


def _compose_inline_signature_label(
    img: Image.Image,
    logo: Optional[Image.Image],
    species_line: str,
    place_line: str,
    logo_width_ratio: float,
    camera_line: str = "",
) -> Image.Image:
    """
    无外框：图内中下方 [签名 | 竖线 | 标签]。
    标签可有两行或三行（与左侧 Logo 等高合并）：
      - 物种名称
      - 地点（+日期）
      - 相机参数（如有）
    竖线与文字颜色与签名水印（剪影上色）一致。
    """
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    sp = (species_line or "").strip()
    pl = (place_line or "").strip()
    cam = (camera_line or "").strip()
    if logo is None and not sp and not pl and not cam:
        return img_rgb.copy()

    base = img_rgb.convert("RGBA")
    margin = max(8, int(h * 0.02))
    ratio = min(0.8, max(0.05, float(logo_width_ratio)))
    # 与外框模式图内签名同一套目标框
    area_w = max(40, int(w * ratio))
    area_h = max(28, int(h * 0.16))

    lg: Optional[Image.Image] = None
    lw, lh = 0, 0
    if logo is not None:
        lg = _fit_logo(logo.convert("RGBA"), area_w, area_h)
        lw, lh = lg.size

    # 收集所有非空文本行
    all_lines = [x for x in (sp, pl, cam) if x]
    will_draw_text = bool(all_lines)
    nrows = len(all_lines)

    # 标签字号：以签名实际高度（或同框外推高度）为标尺
    ref_h = lh if lg else min(area_h, max(28, int(round(area_w * 0.52))))
    if will_draw_text:
        font, line_h, line_gap = _inline_font_for_ref_height(ref_h, nrows)
    else:
        font = _get_font(12)
        measure_tmp = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        line_h = _inline_text_line_height(measure_tmp, font)
        line_gap = max(2, int(line_h * 0.14))

    measure = ImageDraw.Draw(Image.new("RGB", (max(128, w // 2), 256)))
    max_text_w = max(96, int(w * max(0.28, min(0.52, float(ratio) + 0.12))))
    truncated = [
        _truncate_line_to_width(measure, ln, font, max_text_w) for ln in all_lines
    ]
    tw = max(
        [_text_pixel_width(measure, ln, font) for ln in truncated] + [1 if truncated else 0]
    )

    has_text = bool(truncated)
    if has_text:
        text_h = nrows * line_h + (nrows - 1) * line_gap
    else:
        text_h = 0

    gap = max(6, int(w * 0.014))
    sep_draw_w = 2  # 竖分隔线线宽（与文字同色）

    if lg and has_text:
        total_w = lw + gap + sep_draw_w + gap + tw
    elif lg:
        total_w = lw
    elif has_text:
        total_w = tw
    else:
        total_w = 0

    block_h = max(lh, text_h, line_h)
    y_bottom = h - margin
    y_top = max(0, y_bottom - block_h)
    x0 = max(0, (w - total_w) // 2)

    fg = _fg_rgb_for_bottom_overlay(img_rgb, max(0, y_top - margin), h)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(overlay)
    rgba_text = fg + (255,)

    cx = x0
    ly = y_top + max(0, (block_h - lh) // 2) if lg else 0
    if lg:
        logo_rgba = lg.convert("RGBA")
        alpha = logo_rgba.split()[-1]
        alpha = alpha.point(lambda p: int(p * 0.92))
        fg_layer = Image.new("RGBA", lg.size, fg + (0,))
        fg_layer.putalpha(alpha)
        overlay.alpha_composite(fg_layer, (cx, ly))
        cx += lw + gap

    if lg and has_text:
        sep_x = cx + sep_draw_w // 2
        sep_y0 = y_top + max(2, int(block_h * 0.08))
        sep_y1 = y_bottom - max(2, int(block_h * 0.08))
        dr.line([(sep_x, sep_y0), (sep_x, sep_y1)], fill=rgba_text, width=sep_draw_w)
        cx += sep_draw_w + gap

    if has_text:
        tx = cx
        ty = y_top + max(0, (block_h - text_h) // 2)
        for i, ln in enumerate(truncated):
            dr.text((tx, ty), ln, fill=rgba_text, font=font)
            ty += line_h + (line_gap if i < len(truncated) - 1 else 0)

    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


def _finalize_watermarked_image(
    img: Image.Image,
    img_path: str,
    source_folder: str,
    options: WatermarkOptions,
    prefer_folder_name_as_species: bool,
    logo_img: Optional[Image.Image],
    species_or_theme_override: str = "",
) -> Image.Image:
    """按选项合成最终水印图（外框模式或图内签名+标签模式）。"""
    style = _norm_watermark_style(options)
    loc = ""
    if options.enable_location:
        if options.location_text.strip():
            loc = options.location_text.strip()
        elif options.use_gps_city:
            loc = _city_from_gps(img_path)
    dt = _extract_exif_datetime(img_path) if options.enable_date else ""
    ovr = (species_or_theme_override or "").strip()
    species = ""
    if options.enable_species:
        if ovr:
            species = ovr
        elif prefer_folder_name_as_species:
            species = _species_from_path(img_path, source_folder)
    cam = _extract_exif_camera_params(img_path) if options.enable_camera_params else ""

    if style == "inline":
        sp_line = species if options.enable_species else ""
        pl = _inline_place_line(img_path, options) if options.enable_location else ""
        # 日期追加到地点行末尾
        if dt and pl:
            pl = f"{pl}  {dt}"
        elif dt:
            pl = dt
        cam_line = cam if options.enable_camera_params else ""
        print(
            f"[watermark inline] path={img_path}\n"
            f"  sp={sp_line!r} pl={pl!r} cam={cam_line!r} dt={dt!r}\n"
            f"  enable_cam={options.enable_camera_params} enable_date={options.enable_date} "
            f"enable_species={options.enable_species} enable_location={options.enable_location}",
            flush=True,
        )
        return _compose_inline_signature_label(
            img, logo_img, sp_line, pl, options.logo_width_ratio, cam_line
        )

    left_fields = [x for x in (species, loc, dt) if x]
    right_fields = [x for x in (cam,) if x]
    left = "  |  ".join(left_fields) if left_fields else "Birdy"
    right = "  |  ".join(right_fields)
    return _compose_leica_style(
        img, left, right, logo_img, options.logo_width_ratio
    )


def generate_watermarks(
    source_folder: str,
    output_folder: str,
    options: WatermarkOptions,
    prefer_folder_name_as_species: bool = True,
    progress_callback: Optional[Callable[[Dict], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    random_per_species: Optional[int] = None,
) -> Dict[str, int]:
    """
    批量生成水印图。should_cancel 返回 True 时提前中断循环。

    random_per_species:
        None 或 <=0 → 处理全部图片；
        正整数 N → 每个物种目录（图片父目录）随机抽至多 N 张。
    """
    os.makedirs(output_folder, exist_ok=True)
    images = _collect_images_recursive(source_folder)
    if random_per_species is not None and int(random_per_species) > 0:
        images = sample_images_per_species_dir(images, int(random_per_species))
    logo_img = None
    if options.logo_path and os.path.isfile(options.logo_path):
        try:
            logo_img = Image.open(options.logo_path).convert("RGBA")
        except Exception:
            logo_img = None

    ov_map: Dict[str, dict] = {}
    if getattr(options, "enable_auto_enhance", True) and getattr(
        options, "enhance_overrides_path", ""
    ):
        ov_map = load_overrides_json(str(options.enhance_overrides_path))

    ok = 0
    fail = 0
    total = len(images)
    if progress_callback:
        try:
            progress_callback({"kind": "start", "done": 0, "total": max(1, total)})
        except Exception:
            pass
    for img_path in images:
        if should_cancel and should_cancel():
            break
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
        if options.enable_ai_exposure or options.enable_ai_denoise or options.enable_ai_sharpen:
            img = _ai_enhance_image(img, options)
        img = apply_watermark_photo_pipeline(
            img, options, img_path, source_folder, ov_map
        )
        try:
            out = _finalize_watermarked_image(
                img,
                img_path,
                source_folder,
                options,
                prefer_folder_name_as_species,
                logo_img,
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
    if options.enable_ai_exposure or options.enable_ai_denoise or options.enable_ai_sharpen:
        img = _ai_enhance_image(img, options)
    ov_map: Dict[str, dict] = {}
    if getattr(options, "enable_auto_enhance", True) and getattr(
        options, "enhance_overrides_path", ""
    ):
        ov_map = load_overrides_json(str(options.enhance_overrides_path))
    img = apply_watermark_photo_pipeline(
        img, options, image_path, source_folder, ov_map
    )

    logo_img = None
    if options.logo_path and os.path.isfile(options.logo_path):
        try:
            logo_img = Image.open(options.logo_path).convert("RGBA")
        except Exception:
            logo_img = None

    return _finalize_watermarked_image(
        img,
        image_path,
        source_folder,
        options,
        prefer_folder_name_as_species,
        logo_img,
    )


def render_watermark_on_pil_image(
    img: Image.Image,
    image_path: str,
    source_folder: str,
    options: WatermarkOptions,
    prefer_folder_name_as_species: bool = True,
    species_or_theme_override: str = "",
) -> Optional[Image.Image]:
    """
    对已解码的 RGB 图像叠加水印（如动图各帧），不再次读盘解码。
    不执行水印前自动显影（调用方若已做显影/调色，请通过本函数避免重复处理）。

    species_or_theme_override：非空时用作物种/左侧主题文案（动图等无法从目录推断物种时使用）。
    """
    if img is None:
        return None
    base = img.convert("RGB")
    opts = replace(
        options,
        enable_auto_enhance=False,
        enhance_overrides_path="",
    )
    logo_img = None
    if opts.logo_path and os.path.isfile(opts.logo_path):
        try:
            logo_img = Image.open(opts.logo_path).convert("RGBA")
        except Exception:
            logo_img = None
    try:
        return _finalize_watermarked_image(
            base,
            image_path,
            source_folder,
            opts,
            prefer_folder_name_as_species,
            logo_img,
            species_or_theme_override,
        )
    except Exception:
        return None


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

