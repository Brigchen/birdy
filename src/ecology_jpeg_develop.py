# -*- coding: utf-8 -*-
"""
RAW → JPEG 时的「鸟类 / 生态」向快速显影（OpenCV）。

在速度可接受的前提下做：整体明暗校正、轻度 L 通道局部对比（CLAHE，弱化背景拉花）、
暗部适度提亮、按分辨率自适应的降噪。不做锐化，避免主体被细碎纹理淹没。

连拍：首张 analyze_ecology_burst_params，后续帧 develop_bgr_ecology_wildlife_with_params
复用相同 CLAHE/伽马/降噪强度，显著加速。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# CLAHE：较大 tile 减轻草地/碎叶等背景的局部对比「乱跳」；再与原 L 混合，整体更稳。
_CLAHE_TILE: Tuple[int, int] = (16, 16)
_CLAHE_BLEND = 0.45  # 越大越接近纯 CLAHE 结果


def _clahe_clip_from_std_l(std_l: float) -> float:
    """随亮度分散度略调 clip，整体明显低于旧版，减轻背景被强化。"""
    return float(np.clip(1.05 + (0.14 - std_l) * 5.0, 0.75, 2.2))


def _apply_clahe_l_mild(L1: np.ndarray, clip: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=_CLAHE_TILE)
    Lc = clahe.apply(L1)
    m = _CLAHE_BLEND
    return np.clip(
        (1.0 - m) * L1.astype(np.float32) + m * Lc.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)


@dataclass(frozen=True)
class EcologyBurstParams:
    """由首张图分析得到的显影参数，供连拍后续帧复用。"""

    gamma: float
    clip: float
    noise_est: float
    use_edge_preserving: bool
    h_dn: float


def develop_bgr_ecology_wildlife(bgr: np.ndarray) -> np.ndarray:
    """
    BGR uint8 输入/输出。假设已为 sRGB 域的 8bit 渲染图（如 rawpy postprocess 结果）。
    """
    if bgr is None or bgr.size == 0:
        return bgr
    h, w = bgr.shape[:2]
    npx = float(h * w)

    lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    mean_l = float(np.mean(Lf))
    std_l = float(np.std(Lf))

    if mean_l < 0.38:
        gamma = 0.88
        Lf = np.clip(Lf**gamma, 0.0, 1.0)
    elif mean_l > 0.62:
        gamma = 1.10
        Lf = np.clip(Lf**gamma, 0.0, 1.0)

    L1 = np.clip(Lf * 255.0, 0, 255).astype(np.uint8)

    clip = _clahe_clip_from_std_l(std_l)
    L2 = _apply_clahe_l_mild(L1, clip)

    Lf2 = L2.astype(np.float32) / 255.0
    shadow_w = np.power(np.clip(1.0 - Lf2, 0.0, 1.0), 1.85)
    # 略减暗部提亮强度与范围，避免中灰背景被抬亮显得杂乱
    lift = 0.055 * shadow_w * np.clip(0.38 - Lf2, 0.0, None)
    Lf3 = np.clip(Lf2 + lift, 0.0, 1.0)
    L3 = (Lf3 * 255.0).astype(np.uint8)

    lab2 = cv2.merge((L3, A, B))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_est = float(np.sqrt(max(0.0, np.var(lap))))

    if npx > 22_000_000:
        bgr3 = cv2.edgePreservingFilter(bgr2, flags=1, sigma_s=50, sigma_r=0.30)
    else:
        h_dn = float(np.clip(2.2 + noise_est * 0.28, 2.2, 6.5))
        bgr3 = cv2.fastNlMeansDenoisingColored(bgr2, None, h_dn, h_dn, 7, 21)

    return np.clip(bgr3, 0, 255).astype(np.uint8)


def analyze_ecology_burst_params(bgr: np.ndarray) -> EcologyBurstParams:
    """
    仅对首张（或参考帧）做统计与中间结果，得到连拍复用的显影参数。
    与 develop_bgr_ecology_wildlife 首张统计逻辑一致。
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty bgr")
    h, w = bgr.shape[:2]
    npx = float(h * w)

    lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    mean_l = float(np.mean(Lf))
    std_l = float(np.std(Lf))

    if mean_l < 0.38:
        gamma = 0.88
        Lf = np.clip(Lf**gamma, 0.0, 1.0)
    elif mean_l > 0.62:
        gamma = 1.10
        Lf = np.clip(Lf**gamma, 0.0, 1.0)
    else:
        gamma = 1.0

    L1 = np.clip(Lf * 255.0, 0, 255).astype(np.uint8)

    clip = _clahe_clip_from_std_l(std_l)
    L2 = _apply_clahe_l_mild(L1, clip)

    Lf2 = L2.astype(np.float32) / 255.0
    shadow_w = np.power(np.clip(1.0 - Lf2, 0.0, 1.0), 1.85)
    lift = 0.055 * shadow_w * np.clip(0.38 - Lf2, 0.0, None)
    Lf3 = np.clip(Lf2 + lift, 0.0, 1.0)
    L3 = (Lf3 * 255.0).astype(np.uint8)

    lab2 = cv2.merge((L3, A, B))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_est = float(np.sqrt(max(0.0, np.var(lap))))

    use_edge_preserving = npx > 22_000_000
    if use_edge_preserving:
        h_dn = 0.0
    else:
        h_dn = float(np.clip(2.2 + noise_est * 0.28, 2.2, 6.5))

    return EcologyBurstParams(
        gamma=gamma,
        clip=clip,
        noise_est=noise_est,
        use_edge_preserving=use_edge_preserving,
        h_dn=h_dn,
    )


def develop_bgr_ecology_wildlife_with_params(
    bgr: np.ndarray, p: EcologyBurstParams
) -> np.ndarray:
    """按首张分析得到的参数做显影（连拍后续帧）。"""
    if bgr is None or bgr.size == 0:
        return bgr

    lab = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0

    if p.gamma != 1.0:
        Lf = np.clip(Lf**p.gamma, 0.0, 1.0)

    L1 = np.clip(Lf * 255.0, 0, 255).astype(np.uint8)

    L2 = _apply_clahe_l_mild(L1, p.clip)

    Lf2 = L2.astype(np.float32) / 255.0
    shadow_w = np.power(np.clip(1.0 - Lf2, 0.0, 1.0), 1.85)
    lift = 0.055 * shadow_w * np.clip(0.38 - Lf2, 0.0, None)
    Lf3 = np.clip(Lf2 + lift, 0.0, 1.0)
    L3 = (Lf3 * 255.0).astype(np.uint8)

    lab2 = cv2.merge((L3, A, B))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    if p.use_edge_preserving:
        bgr3 = cv2.edgePreservingFilter(bgr2, flags=1, sigma_s=50, sigma_r=0.30)
    else:
        bgr3 = cv2.fastNlMeansDenoisingColored(bgr2, None, p.h_dn, p.h_dn, 7, 21)

    return np.clip(bgr3, 0, 255).astype(np.uint8)


def develop_bgr_ecology_wildlife_burst_fast(frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
    """
    连拍加速：首张 analyze + develop，后续帧 develop_bgr_ecology_wildlife_with_params。
    输入为已白平衡（或原图）的 BGR 列表，长度 >= 1。
    """
    if not frames_bgr:
        return []
    if len(frames_bgr) == 1:
        return [develop_bgr_ecology_wildlife(frames_bgr[0])]
    p = analyze_ecology_burst_params(frames_bgr[0])
    out: List[np.ndarray] = [
        develop_bgr_ecology_wildlife_with_params(frames_bgr[0], p),
    ]
    for j in range(1, len(frames_bgr)):
        out.append(develop_bgr_ecology_wildlife_with_params(frames_bgr[j], p))
    return out
