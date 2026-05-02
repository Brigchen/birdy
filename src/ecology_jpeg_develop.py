# -*- coding: utf-8 -*-
"""
RAW → JPEG 时的「鸟类 / 生态」向快速显影（OpenCV）。

在速度可接受的前提下做：整体明暗校正、全局对比度（CLAHE）、
暗部局部提亮、按分辨率自适应的降噪与锐化。非科学级校色，偏观感增强。
"""

from __future__ import annotations

import numpy as np
import cv2


def develop_bgr_ecology_wildlife(bgr: np.ndarray) -> np.ndarray:
    """
    BGR uint8 输入/输出。假设已为 sRGB 域的 8bit 渲染图（如 rawpy postprocess 结果）。
    """
    if bgr is None or bgr.size == 0:
        return bgr
    work = bgr.astype(np.float32)
    h, w = work.shape[:2]
    npx = float(h * w)

    lab = cv2.cvtColor(work.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    Lf = L.astype(np.float32) / 255.0
    mean_l = float(np.mean(Lf))
    std_l = float(np.std(Lf))

    # 整体明暗：向中间调靠拢
    if mean_l < 0.38:
        gamma = 0.88
        Lf = np.clip(Lf ** gamma, 0.0, 1.0)
    elif mean_l > 0.62:
        gamma = 1.10
        Lf = np.clip(Lf ** gamma, 0.0, 1.0)

    L1 = np.clip(Lf * 255.0, 0, 255).astype(np.uint8)

    # 全局对比度：CLAHE clip 随场景对比度自适应
    clip = float(np.clip(2.2 + (0.18 - std_l) * 14.0, 1.6, 6.5))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    L2 = clahe.apply(L1)

    # 局部暗部：压高光、抬阴影（仅 L 通道）
    Lf2 = L2.astype(np.float32) / 255.0
    shadow_w = np.power(np.clip(1.0 - Lf2, 0.0, 1.0), 1.75)
    lift = 0.11 * shadow_w * np.clip(0.48 - Lf2, 0.0, None)
    Lf3 = np.clip(Lf2 + lift, 0.0, 1.0)
    L3 = (Lf3 * 255.0).astype(np.uint8)

    lab2 = cv2.merge((L3, A, B))
    bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 噪声估计 → 降噪强度
    gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_est = float(np.sqrt(max(0.0, np.var(lap))))

    if npx > 22_000_000:
        bgr3 = cv2.edgePreservingFilter(bgr2, flags=1, sigma_s=50, sigma_r=0.35)
    else:
        h_dn = float(np.clip(2.5 + noise_est * 0.35, 2.5, 8.0))
        bgr3 = cv2.fastNlMeansDenoisingColored(
            bgr2, None, h_dn, h_dn, 7, 21
        )

    # 自适应锐化：噪声高则减弱
    blur = cv2.GaussianBlur(bgr3, (0, 0), sigmaX=1.05)
    amt = 0.28 + 0.32 * (1.0 - min(std_l * 5.5, 1.0))
    amt = float(np.clip(amt, 0.12, 0.62))
    out = cv2.addWeighted(bgr3, 1.0 + amt, blur, -amt, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)
