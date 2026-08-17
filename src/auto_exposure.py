# -*- coding: utf-8 -*-
"""基于鸟体测光的自动曝光模块。

流程：
  1. 用 YOLO（BirdAndEyeDetector.detect_birds）检测鸟体框
  2. 取最大鸟体框作为主体（无检测时退化为全图平均测光）
  3. 计算主体区域加权平均亮度（RGB 灰度），与目标亮度（0.5）比较
  4. 用 gamma 调整全图：gamma<1 提亮（主体偏暗/剪影），gamma>1 压暗
  5. strength 控制调整幅度（0=原图，1=完全调整）

设计目标：避免逆光剪影，让鸟体细节可见；同时不过曝背景。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from PIL import Image

# 添加当前目录到 path（与 birdy_gui.py 同级）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==================== 鸟体检测器缓存（懒加载 + 单例） ====================

_bird_detector = None
_bird_detector_lock = None


def _get_bird_detector():
    """懒加载 BirdAndEyeDetector（关闭物种/鸟眼以加速），全局复用。"""
    global _bird_detector
    if _bird_detector is not None:
        return _bird_detector
    try:
        from detect_bird_and_eye import BirdAndEyeDetector
        _bird_detector = BirdAndEyeDetector(
            enable_species=False,
            enable_eye=False,
            bird_conf=0.5,
        )
        return _bird_detector
    except Exception as e:
        print(f"[auto_exposure] 鸟体检测器加载失败: {e}", flush=True)
        return None


def detect_bird_boxes(bgr: np.ndarray) -> List[Dict]:
    """检测鸟体，返回 xyxy 框列表（含 area 字段）。

    Returns:
        [{"bbox": [x1,y1,x2,y2], "conf": float, "area": int}, ...]
        失败时返回空列表。
    """
    det = _get_bird_detector()
    if det is None:
        return []
    try:
        birds = det.detect_birds(bgr)
    except Exception as e:
        print(f"[auto_exposure] 鸟体检测失败: {e}", flush=True)
        return []
    out: List[Dict] = []
    for b in birds:
        x1, y1, x2, y2 = b.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        out.append({"bbox": [x1, y1, x2, y2], "conf": float(b.get("conf", 0.0)), "area": area})
    return out


def _largest_bird_box(boxes: List[Dict]) -> Optional[List[int]]:
    """返回面积最大的鸟体框 xyxy，无则 None。"""
    if not boxes:
        return None
    best = max(boxes, key=lambda b: b.get("area", 0))
    if best.get("area", 0) <= 0:
        return None
    return best["bbox"]


# ==================== 测光 + Gamma 调整 ====================

# 目标亮度（0=纯黑, 1=纯白）。
# 0.6 让鸟体主体明显提亮（剪影场景下主体常落在 0.1~0.3，提到 0.6 才能看清细节）。
_TARGET_LUMA = 0.6
# 主体外周扩展比例（让测光区稍大于纯鸟体框，避免边缘锯齿影响）
_BOX_EXPAND_RATIO = 0.1
# 测光用低分位数（0.3 = 30% 分位），捕捉鸟体暗部，避免被框内高光/背景稀释
_LUMA_PERCENTILE = 30.0
# gamma 限制范围（放宽以支持强提亮）
_GAMMA_MIN = 0.2
_GAMMA_MAX = 5.0
# 诊断日志：仅首次调用打印一次，避免刷屏
_diag_logged = False
# 强度：0=原图，1=自动曝光结果，>1 在自动曝光上继续加档（最大 3 ≈ 再加 2 EV）
STRENGTH_MAX = 3.0


def _compute_subject_luma(bgr: np.ndarray, box: Optional[List[int]]) -> float:
    """计算主体区域亮度（0~1），使用低分位数捕捉暗部。

    box=None 时退化为全图测光（仍用低分位数，避免明亮背景稀释）。
    box 有效时取框内（稍扩展）Y 通道的 _LUMA_PERCENTILE 分位数。
    """
    h, w = bgr.shape[:2]
    if box is None:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y_flat = ycrcb[:, :, 0].flatten().astype(np.float32) / 255.0
        return float(np.percentile(y_flat, _LUMA_PERCENTILE))

    x1, y1, x2, y2 = box
    # 扩展框（向外扩 10%）
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - bw * _BOX_EXPAND_RATIO))
    y1 = max(0, int(y1 - bh * _BOX_EXPAND_RATIO))
    x2 = min(w, int(x2 + bw * _BOX_EXPAND_RATIO))
    y2 = min(h, int(y2 + bh * _BOX_EXPAND_RATIO))
    if x2 <= x1 or y2 <= y1:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y_flat = ycrcb[:, :, 0].flatten().astype(np.float32) / 255.0
        return float(np.percentile(y_flat, _LUMA_PERCENTILE))

    roi = bgr[y1:y2, x1:x2]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    y_flat = ycrcb[:, :, 0].flatten().astype(np.float32) / 255.0
    return float(np.percentile(y_flat, _LUMA_PERCENTILE))


def _gamma_correct(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """对 BGR 图像应用 gamma 校正。

    gamma < 1: 提亮（暗部提升更显著）
    gamma > 1: 压暗（亮部压缩）
    公式：output = 255 * (input/255) ^ gamma
    """
    if abs(gamma - 1.0) < 1e-3:
        return bgr
    g = max(gamma, 1e-3)
    table = np.array(
        [((i / 255.0) ** g) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(bgr, table)


def compute_gamma_for_subject(
    bgr: np.ndarray,
    box: Optional[List[int]],
    target_luma: float = _TARGET_LUMA,
) -> float:
    """根据主体亮度计算 gamma 值。

    策略：
      - 主体偏暗（luma < target）→ gamma<1 提亮
      - 主体偏亮（luma > target）→ gamma>1 压暗
      - 使用对数映射：gamma = log(target) / log(current)，限制在 [_GAMMA_MIN, _GAMMA_MAX]
    """
    luma = _compute_subject_luma(bgr, box)
    # 避免数值异常
    luma = max(min(luma, 0.98), 0.02)
    target = max(min(target_luma, 0.98), 0.02)
    gamma = np.log(target) / np.log(luma)
    # 限制范围，避免极端值
    return float(max(_GAMMA_MIN, min(_GAMMA_MAX, gamma)))


def _log_diag_once(bgr: np.ndarray, box: Optional[List[int]], luma: float, gamma: float, strength: float) -> None:
    """首次调用时打印一次诊断信息，便于排查测光/提亮问题。"""
    global _diag_logged
    if _diag_logged:
        return
    _diag_logged = True
    h, w = bgr.shape[:2]
    box_str = "无（全图测光）" if box is None else f"xyxy={box}, 面积={((box[2]-box[0])*(box[3]-box[1]))}"
    print(f"[auto_exposure] 诊断（仅首次）：图像 {w}x{h}, 鸟体框={box_str}", flush=True)
    print(f"[auto_exposure]   主体亮度(30%分位)={luma:.3f}, 目标={_TARGET_LUMA:.2f}, gamma={gamma:.3f}, strength={strength:.2f}", flush=True)
    print(f"[auto_exposure]   提示：若提亮仍弱，可增大曝光强度滑条或检查鸟体检测是否命中", flush=True)


def apply_exposure_strength(
    bgr: np.ndarray, corrected: np.ndarray, strength: float
) -> np.ndarray:
    """0=原图，1=自动曝光结果，>1 在自动曝光上按 EV 加曝光（2=+1EV，3=+2EV）。"""
    s = float(np.clip(strength, 0.0, STRENGTH_MAX))
    if s <= 0.0:
        return bgr
    if s < 1.0:
        orig = bgr.astype(np.float32)
        corr = corrected.astype(np.float32)
        mixed = orig + s * (corr - orig)
        return np.clip(mixed, 0, 255).astype(np.uint8)
    if abs(s - 1.0) < 1e-6:
        return corrected
    extra_ev = s - 1.0
    gain = 2.0 ** extra_ev
    out = corrected.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_expose_bgr(
    bgr: np.ndarray,
    strength: float = 1.0,
    detect: bool = True,
    meter_box: Optional[List[int]] = None,
) -> np.ndarray:
    """对 BGR 图像执行自动曝光。

    Args:
        bgr: BGR numpy 图像
        strength: 0=原图，1=按测光算出的自动曝光；>1 在该结果上继续加曝光（最大 3）
        detect: True=检测鸟体作为主体；False=全图测光（若未给 meter_box）
        meter_box: 可选测光框 xyxy（像素）。若提供则优先于鸟体检测（如动图裁剪区）。
    Returns:
        调整后的 BGR 图像
    """
    if strength <= 0.0:
        return bgr

    box = None
    if meter_box is not None and len(meter_box) == 4:
        box = [int(meter_box[0]), int(meter_box[1]), int(meter_box[2]), int(meter_box[3])]
    elif detect:
        boxes = detect_bird_boxes(bgr)
        box = _largest_bird_box(boxes)

    luma = _compute_subject_luma(bgr, box)
    gamma = compute_gamma_for_subject(bgr, box)
    _log_diag_once(bgr, box, luma, gamma, strength)
    corrected = _gamma_correct(bgr, gamma)
    return apply_exposure_strength(bgr, corrected, strength)


def auto_expose_pil(
    img: Image.Image,
    strength: float = 1.0,
    detect: bool = True,
) -> Image.Image:
    """对 PIL 图像执行自动曝光，返回 PIL 图像。"""
    if strength <= 0.0:
        return img
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out_bgr = auto_expose_bgr(bgr, strength=strength, detect=detect)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


# ==================== CLI 自测 ====================

if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="自动曝光测试（基于鸟体测光）")
    ap.add_argument("--image", required=True, help="输入图片路径")
    ap.add_argument("--output", default="", help="输出路径（默认 输入名_ae.jpg）")
    ap.add_argument("--strength", type=float, default=1.0, help="强度 0~1")
    ap.add_argument("--no-detect", action="store_true", help="禁用鸟体检测，用全图测光")
    args = ap.parse_args()

    t0 = time.time()
    img = Image.open(args.image).convert("RGB")
    out = auto_expose_pil(img, strength=args.strength, detect=not args.no_detect)
    dt = time.time() - t0

    out_path = args.output or (Path(args.image).stem + "_ae.jpg")
    out.save(out_path, quality=95)
    print(f"输入: {args.image}")
    print(f"输出: {out_path}")
    print(f"强度: {args.strength}, 检测: {not args.no_detect}, 耗时: {dt:.2f}s")
