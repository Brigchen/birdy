# -*- coding: utf-8 -*-
"""动图定点 / 跟踪：标定点模板匹配、相对裁剪几何与越界补边。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

BurstMode = str  # "fixed" | "track"


@dataclass
class FrameLayout:
    """单帧：标定点与裁剪区（归一化 0–1）。"""

    ax: float = 0.5
    ay: float = 0.5
    x0: float = 0.15
    y0: float = 0.15
    x1: float = 0.85
    y1: float = 0.85
    auto: bool = True
    conf: float = 1.0

    def crop_tuple(self) -> Tuple[float, float, float, float]:
        x0, x1 = sorted((float(self.x0), float(self.x1)))
        y0, y1 = sorted((float(self.y0), float(self.y1)))
        return (
            float(np.clip(x0, 0.0, 1.0)),
            float(np.clip(y0, 0.0, 1.0)),
            float(np.clip(x1, 0.0, 1.0)),
            float(np.clip(y1, 0.0, 1.0)),
        )

    def to_dict(self) -> dict:
        return {
            "ax": float(self.ax),
            "ay": float(self.ay),
            "x0": float(self.x0),
            "y0": float(self.y0),
            "x1": float(self.x1),
            "y1": float(self.y1),
            "auto": bool(self.auto),
            "conf": float(self.conf),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameLayout":
        return cls(
            ax=float(d.get("ax", 0.5)),
            ay=float(d.get("ay", 0.5)),
            x0=float(d.get("x0", 0.15)),
            y0=float(d.get("y0", 0.15)),
            x1=float(d.get("x1", 0.85)),
            y1=float(d.get("y1", 0.85)),
            auto=bool(d.get("auto", True)),
            conf=float(d.get("conf", 1.0)),
        )


@dataclass
class CropGeom:
    """由首图推出的常量：裁剪像素尺寸 + 标定点在裁剪区内的相对位置。"""

    crop_w: int
    crop_h: int
    rel_x: float
    rel_y: float
    ref_w: int
    ref_h: int


def clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def layout_valid(lay: Optional[FrameLayout], min_span: float = 0.02) -> bool:
    if lay is None:
        return False
    x0, y0, x1, y1 = lay.crop_tuple()
    return (x1 - x0) >= min_span and (y1 - y0) >= min_span


def geom_from_first(lay: FrameLayout, w: int, h: int) -> CropGeom:
    x0, y0, x1, y1 = lay.crop_tuple()
    cw = max(1, int(round((x1 - x0) * float(w))))
    ch = max(1, int(round((y1 - y0) * float(h))))
    cw = min(cw, int(w))
    ch = min(ch, int(h))
    span_x = max(1e-6, x1 - x0)
    span_y = max(1e-6, y1 - y0)
    rel_x = float(np.clip((float(lay.ax) - x0) / span_x, 0.0, 1.0))
    rel_y = float(np.clip((float(lay.ay) - y0) / span_y, 0.0, 1.0))
    return CropGeom(
        crop_w=cw,
        crop_h=ch,
        rel_x=rel_x,
        rel_y=rel_y,
        ref_w=int(w),
        ref_h=int(h),
    )


def layout_from_anchor(
    ax: float,
    ay: float,
    geom: CropGeom,
    w: int,
    h: int,
    *,
    auto: bool = True,
    conf: float = 1.0,
) -> FrameLayout:
    """由标定点 + 首图几何反推裁剪框（归一化；可越出 [0,1]，稍后裁剪时补边）。"""
    ax = clamp01(ax)
    ay = clamp01(ay)
    sx = float(w) / float(max(1, geom.ref_w))
    sy = float(h) / float(max(1, geom.ref_h))
    cw = max(1, int(round(geom.crop_w * sx)))
    ch = max(1, int(round(geom.crop_h * sy)))
    px = ax * float(w) - geom.rel_x * float(cw)
    py = ay * float(h) - geom.rel_y * float(ch)
    return FrameLayout(
        ax=ax,
        ay=ay,
        x0=px / float(w),
        y0=py / float(h),
        x1=(px + cw) / float(w),
        y1=(py + ch) / float(h),
        auto=auto,
        conf=float(conf),
    )


def crop_bgr_with_pad(bgr: np.ndarray, lay: FrameLayout, geom: CropGeom) -> np.ndarray:
    """按布局裁剪；越界补边（常数填充），不平移框。"""
    h, w = bgr.shape[:2]
    sx = float(w) / float(max(1, geom.ref_w))
    sy = float(h) / float(max(1, geom.ref_h))
    cw = max(1, int(round(geom.crop_w * sx)))
    ch = max(1, int(round(geom.crop_h * sy)))
    ax = clamp01(lay.ax) * float(w)
    ay = clamp01(lay.ay) * float(h)
    x0 = int(round(ax - geom.rel_x * float(cw)))
    y0 = int(round(ay - geom.rel_y * float(ch)))
    x1 = x0 + cw
    y1 = y0 + ch
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    src = bgr
    if pad_l or pad_t or pad_r or pad_b:
        src = cv2.copyMakeBorder(
            bgr,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        x0 += pad_l
        y0 += pad_t
        x1 += pad_l
        y1 += pad_t
    x0 = max(0, x0)
    y0 = max(0, y0)
    crop = src[y0:y1, x0:x1]
    if crop.shape[0] != ch or crop.shape[1] != cw:
        crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
    return crop.copy()


def meter_box_in_padded_crop(
    lay: FrameLayout, geom: CropGeom, w: int, h: int
) -> Optional[List[int]]:
    """
    裁剪（含越界补边）后，原图有效相交区在裁剪图中的像素 xyxy，供自动曝光测光。
    不含黑边，避免补边把测光拉暗。
    """
    sx = float(w) / float(max(1, geom.ref_w))
    sy = float(h) / float(max(1, geom.ref_h))
    cw = max(1, int(round(geom.crop_w * sx)))
    ch = max(1, int(round(geom.crop_h * sy)))
    ax = clamp01(lay.ax) * float(w)
    ay = clamp01(lay.ay) * float(h)
    x0 = int(round(ax - geom.rel_x * float(cw)))
    y0 = int(round(ay - geom.rel_y * float(ch)))
    box = in_bounds_crop_xyxy(lay, geom, w, h)
    if box is None:
        return None
    xa, ya, xb, yb = box
    mx0 = int(xa - x0)
    my0 = int(ya - y0)
    mx1 = int(xb - x0)
    my1 = int(yb - y0)
    mx0 = max(0, min(int(cw), mx0))
    mx1 = max(0, min(int(cw), mx1))
    my0 = max(0, min(int(ch), my0))
    my1 = max(0, min(int(ch), my1))
    if mx1 - mx0 < 2 or my1 - my0 < 2:
        return None
    return [mx0, my0, mx1, my1]


def in_bounds_crop_xyxy(
    lay: FrameLayout, geom: CropGeom, w: int, h: int
) -> Optional[List[int]]:
    """布局裁剪框与图像相交的像素 xyxy，供测光；无有效交集则 None。"""
    sx = float(w) / float(max(1, geom.ref_w))
    sy = float(h) / float(max(1, geom.ref_h))
    cw = max(1, int(round(geom.crop_w * sx)))
    ch = max(1, int(round(geom.crop_h * sy)))
    ax = clamp01(lay.ax) * float(w)
    ay = clamp01(lay.ay) * float(h)
    x0 = int(round(ax - geom.rel_x * float(cw)))
    y0 = int(round(ay - geom.rel_y * float(ch)))
    x1 = x0 + cw
    y1 = y0 + ch
    xa = max(0, min(int(w), x0))
    xb = max(0, min(int(w), x1))
    ya = max(0, min(int(h), y0))
    yb = max(0, min(int(h), y1))
    if xb - xa < 2 or yb - ya < 2:
        return None
    return [xa, ya, xb, yb]


def _gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def extract_template(
    bgr: np.ndarray,
    ax: float,
    ay: float,
    half: int = 48,
) -> Tuple[np.ndarray, int, int]:
    """从锚点邻域取灰度模板，返回 (tmpl, cx_px, cy_px)。"""
    g = _gray(bgr)
    h, w = g.shape[:2]
    cx = int(round(clamp01(ax) * float(w)))
    cy = int(round(clamp01(ay) * float(h)))
    half = max(8, int(half))
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    if x1 - x0 < 8 or y1 - y0 < 8:
        tmpl = g.copy()
    else:
        tmpl = g[y0:y1, x0:x1].copy()
    return tmpl, cx, cy


def match_template_anchor(
    bgr: np.ndarray,
    tmpl: np.ndarray,
    *,
    guess_xy: Optional[Tuple[float, float]] = None,
    search_frac: float = 0.45,
) -> Tuple[float, float, float]:
    """
    在图中匹配模板，返回 (ax_norm, ay_norm, score)。
    guess_xy 为归一化初值；search_frac 为搜索窗占短边比例（定点可用更大）。
    """
    g = _gray(bgr)
    h, w = g.shape[:2]
    th, tw = tmpl.shape[:2]
    if th < 4 or tw < 4 or h < th or w < tw:
        gx, gy = guess_xy if guess_xy is not None else (0.5, 0.5)
        return clamp01(gx), clamp01(gy), 0.0

    def _run(img: np.ndarray, tpl: np.ndarray) -> Tuple[float, int, int]:
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, loc = cv2.minMaxLoc(res)
        return float(mx), int(loc[0]), int(loc[1])

    # 金字塔：全图或局部
    best_score = -1.0
    best_ax, best_ay = 0.5, 0.5
    for scale in (1.0, 0.5):
        if scale < 1.0:
            iw = max(tw + 2, int(w * scale))
            ih = max(th + 2, int(h * scale))
            img = cv2.resize(g, (iw, ih), interpolation=cv2.INTER_AREA)
            tpl = cv2.resize(
                tmpl,
                (max(4, int(tw * scale)), max(4, int(th * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            img, tpl = g, tmpl
        ih, iw = img.shape[:2]
        tph, tpw = tpl.shape[:2]
        if ih < tph or iw < tpw:
            continue

        roi = img
        ox = oy = 0
        if guess_xy is not None and search_frac < 0.99:
            gx = clamp01(guess_xy[0]) * iw
            gy = clamp01(guess_xy[1]) * ih
            rad = max(tpw, tph, int(min(iw, ih) * search_frac))
            x0 = int(max(0, gx - rad))
            y0 = int(max(0, gy - rad))
            x1 = int(min(iw, gx + rad))
            y1 = int(min(ih, gy + rad))
            if x1 - x0 >= tpw and y1 - y0 >= tph:
                roi = img[y0:y1, x0:x1]
                ox, oy = x0, y0
        score, lx, ly = _run(roi, tpl)
        cx = (ox + lx + tpw * 0.5) / float(iw)
        cy = (oy + ly + tph * 0.5) / float(ih)
        if score > best_score:
            best_score = score
            best_ax, best_ay = cx, cy

    if best_score < 0:
        gx, gy = guess_xy if guess_xy is not None else (0.5, 0.5)
        return clamp01(gx), clamp01(gy), 0.0
    return clamp01(best_ax), clamp01(best_ay), float(best_score)


def lk_predict(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    ax: float,
    ay: float,
) -> Optional[Tuple[float, float]]:
    """Lucas-Kanade 光流预测下一帧锚点（归一化）。失败返回 None。"""
    g0 = _gray(prev_bgr)
    g1 = _gray(curr_bgr)
    h, w = g0.shape[:2]
    if h < 8 or w < 8:
        return None
    pt = np.array(
        [[[clamp01(ax) * w, clamp01(ay) * h]]],
        dtype=np.float32,
    )
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        g0,
        g1,
        pt,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if nxt is None or st is None or int(st.ravel()[0]) != 1:
        return None
    x, y = float(nxt.ravel()[0]), float(nxt.ravel()[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return clamp01(x / float(w)), clamp01(y / float(h))


DetectBirdsFn = Callable[[np.ndarray], Sequence]
PropagateProgressFn = Callable[[int, int, str], None]


def _bird_bbox(item) -> Optional[Tuple[int, int, int, int]]:
    if item is None:
        return None
    raw = item.get("bbox") if isinstance(item, dict) else item
    if raw is None or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])]
    except (TypeError, ValueError, IndexError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_area(bb: Tuple[int, int, int, int]) -> int:
    return max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])


def offset_in_bird_box(
    ax: float, ay: float, bbox: Tuple[int, int, int, int], w: int, h: int
) -> Tuple[float, float]:
    """标定点相对鸟框左上的比例（可略超出 [0,1]，表示点在框外沿）。"""
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    u = (clamp01(ax) * float(w) - float(x1)) / bw
    v = (clamp01(ay) * float(h) - float(y1)) / bh
    return float(np.clip(u, -0.35, 1.35)), float(np.clip(v, -0.35, 1.35))


def anchor_from_bird_offset(
    bbox: Tuple[int, int, int, int],
    uv: Tuple[float, float],
    w: int,
    h: int,
) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    u, v = uv
    ax = (float(x1) + u * float(x2 - x1)) / float(max(1, w))
    ay = (float(y1) + v * float(y2 - y1)) / float(max(1, h))
    return clamp01(ax), clamp01(ay)


def pick_bird_for_anchor(
    birds: Sequence,
    ax: float,
    ay: float,
    w: int,
    h: int,
    *,
    prev_bbox: Optional[Tuple[int, int, int, int]] = None,
    max_dist: float = 0.38,
) -> Optional[Tuple[int, int, int, int]]:
    """
    选与预测标定点最匹配的鸟框。优先包含该点的框，否则取对角线距离门限内最近的。
    max_dist 为到框中心的距离 / 图像对角线。
    """
    if w < 2 or h < 2 or not birds:
        return None
    px = clamp01(ax) * float(w)
    py = clamp01(ay) * float(h)
    diag = float(max(1.0, np.hypot(float(w), float(h))))
    best_bb: Optional[Tuple[int, int, int, int]] = None
    best_score = -1e9
    for item in birds:
        bb = _bird_bbox(item)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        dist = float(np.hypot(cx - px, cy - py)) / diag
        if dist > max_dist:
            continue
        contains = 1.0 if (x1 <= px <= x2 and y1 <= py <= y2) else 0.0
        conf = 0.0
        if isinstance(item, dict):
            try:
                conf = float(item.get("conf", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
        size_pen = 0.0
        if prev_bbox is not None:
            a0 = float(max(1, _bbox_area(prev_bbox)))
            a1 = float(max(1, _bbox_area(bb)))
            ratio = a1 / a0
            if ratio < 0.12 or ratio > 10.0:
                continue
            size_pen = -abs(float(np.log(ratio))) * 0.08
        score = contains * 3.0 + (1.0 - dist) + 0.25 * conf + size_pen
        if score > best_score:
            best_score = score
            best_bb = bb
    return best_bb


class AnchorKalman:
    """标定点常速卡尔曼：状态 [ax, ay, vx, vy]，观测为归一化坐标。"""

    def __init__(self, ax: float, ay: float, *, q: float = 4e-4, r: float = 6e-4):
        self.x = np.array(
            [clamp01(ax), clamp01(ay), 0.0, 0.0], dtype=np.float64
        )
        self.P = np.diag([2e-3, 2e-3, 4e-2, 4e-2]).astype(np.float64)
        self.q = float(q)
        self.r = float(r)

    def predict(self) -> Tuple[float, float]:
        f = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = np.diag([self.q, self.q, self.q * 6.0, self.q * 6.0]).astype(np.float64)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.x[1] = float(np.clip(self.x[1], 0.0, 1.0))
        return float(self.x[0]), float(self.x[1])

    def update(self, ax: float, ay: float) -> Tuple[float, float]:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        r = np.eye(2, dtype=np.float64) * self.r
        z = np.array([clamp01(ax), clamp01(ay)], dtype=np.float64)
        y = z - h @ self.x
        s = h @ self.P @ h.T + r
        try:
            k = self.P @ h.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            self.x[0], self.x[1] = float(z[0]), float(z[1])
            return float(self.x[0]), float(self.x[1])
        self.x = self.x + k @ y
        self.P = (np.eye(4, dtype=np.float64) - k @ h) @ self.P
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.x[1] = float(np.clip(self.x[1], 0.0, 1.0))
        return float(self.x[0]), float(self.x[1])

    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])


def _safe_detect(detect_fn: Optional[DetectBirdsFn], bgr: np.ndarray) -> list:
    if detect_fn is None or bgr is None or getattr(bgr, "size", 0) == 0:
        return []
    try:
        raw = detect_fn(bgr)
    except Exception:
        return []
    if not raw:
        return []
    return list(raw)


def propagate_layouts(
    frames_bgr: Sequence[np.ndarray],
    first: FrameLayout,
    mode: str = "fixed",
    *,
    min_score: float = 0.35,
    detect_birds_fn: Optional[DetectBirdsFn] = None,
    progress: Optional[PropagateProgressFn] = None,
) -> List[FrameLayout]:
    """
    以首图布局为基准，自动填写后续帧。
    有 detect_birds_fn 时优先 YOLO 鸟框 + 卡尔曼；否则（或未命中）用模板匹配。
    mode=fixed：模板始终取自首图；mode=track：模板随上一帧更新，LK 给初值。
    """
    if not frames_bgr:
        return []
    n = len(frames_bgr)
    seed: List[Optional[FrameLayout]] = [first] + [None] * (n - 1)
    return merge_propagate(
        frames_bgr,
        seed,
        mode,
        min_score=min_score,
        detect_birds_fn=detect_birds_fn,
        progress=progress,
    )


def merge_propagate(
    frames_bgr: Sequence[np.ndarray],
    layouts: Sequence[Optional[FrameLayout]],
    mode: str,
    *,
    min_score: float = 0.35,
    detect_birds_fn: Optional[DetectBirdsFn] = None,
    progress: Optional[PropagateProgressFn] = None,
) -> List[FrameLayout]:
    """
    首图必须有效。自动填写后续未锁定（auto=True 或空）的帧；
    auto=False 的帧原样跳过。跟踪模式穿过锁定页时仍可用其标定点作后续模板/初值。
    手动锁定不应触发整段重跑：调用方只需在合并结果时保留这些页。

    传播顺序：常速外推 + 卡尔曼 → YOLO 鸟框关联（保持首击在框内的相对位置）
    → 失败则模板匹配（定点全图/跟踪局部+光流）→ 再失败用外推位置。
    """
    if not frames_bgr:
        return []
    first = layouts[0] if layouts else None
    if first is None or not layout_valid(first):
        raise ValueError("请先在首图上设置标定点与裁剪区")
    n = len(frames_bgr)
    h0, w0 = frames_bgr[0].shape[:2]
    geom = geom_from_first(first, w0, h0)
    mode_l = (mode or "fixed").strip().lower()
    if mode_l not in ("fixed", "track"):
        mode_l = "fixed"

    out: List[FrameLayout] = [
        FrameLayout(
            ax=clamp01(first.ax),
            ay=clamp01(first.ay),
            x0=first.x0,
            y0=first.y0,
            x1=first.x1,
            y1=first.y1,
            auto=False,
            conf=1.0,
        )
    ]
    tmpl0, _cx, _cy = extract_template(frames_bgr[0], first.ax, first.ay)
    tmpl = tmpl0
    prev_ax, prev_ay = clamp01(first.ax), clamp01(first.ay)
    kf = AnchorKalman(prev_ax, prev_ay)
    uv_off: Optional[Tuple[float, float]] = None
    prev_bbox: Optional[Tuple[int, int, int, int]] = None
    vel_x, vel_y = 0.0, 0.0
    gate = 0.42 if mode_l == "track" else 0.36

    if progress is not None:
        progress(0, n, "传播：鸟体检测 首图")
    birds0 = _safe_detect(detect_birds_fn, frames_bgr[0])
    box0 = pick_bird_for_anchor(
        birds0, prev_ax, prev_ay, w0, h0, max_dist=gate
    )
    if box0 is not None:
        uv_off = offset_in_bird_box(prev_ax, prev_ay, box0, w0, h0)
        prev_bbox = box0

    for i in range(1, n):
        if progress is not None:
            progress(i, n, f"传播：鸟体追踪 {i}/{n - 1}")
        bgr = frames_bgr[i]
        hi, wi = bgr.shape[:2]
        kf.predict()
        cv_ax = clamp01(prev_ax + vel_x)
        cv_ay = clamp01(prev_ay + vel_y)
        pred_ax, pred_ay = kf.position()
        pred_ax = clamp01(0.35 * pred_ax + 0.65 * cv_ax)
        pred_ay = clamp01(0.35 * pred_ay + 0.65 * cv_ay)
        prev = layouts[i] if i < len(layouts) else None
        if prev is not None and not prev.auto:
            kept = replace(prev, auto=False)
            out.append(kept)
            vel_x = clamp01(kept.ax) - prev_ax
            vel_y = clamp01(kept.ay) - prev_ay
            prev_ax, prev_ay = clamp01(kept.ax), clamp01(kept.ay)
            kf.update(prev_ax, prev_ay)
            if mode_l == "track":
                tmpl, _, _ = extract_template(bgr, kept.ax, kept.ay)
            birds_i = _safe_detect(detect_birds_fn, bgr)
            hit = pick_bird_for_anchor(
                birds_i,
                prev_ax,
                prev_ay,
                wi,
                hi,
                prev_bbox=prev_bbox,
                max_dist=gate,
            )
            if hit is not None:
                prev_bbox = hit
                if uv_off is None:
                    uv_off = offset_in_bird_box(prev_ax, prev_ay, hit, wi, hi)
            continue

        ax, ay, score = pred_ax, pred_ay, 0.0
        used_yolo = False
        birds_i = _safe_detect(detect_birds_fn, bgr)
        hit = pick_bird_for_anchor(
            birds_i,
            pred_ax,
            pred_ay,
            wi,
            hi,
            prev_bbox=prev_bbox,
            max_dist=gate,
        )
        if hit is not None:
            if uv_off is None:
                uv_off = offset_in_bird_box(pred_ax, pred_ay, hit, wi, hi)
            ax, ay = anchor_from_bird_offset(hit, uv_off, wi, hi)
            conf_b = 0.85
            if isinstance(birds_i, list):
                for item in birds_i:
                    if _bird_bbox(item) == hit and isinstance(item, dict):
                        try:
                            conf_b = max(0.45, float(item.get("conf", 0.85) or 0.85))
                        except (TypeError, ValueError):
                            conf_b = 0.85
                        break
            score = float(conf_b)
            used_yolo = True
            prev_bbox = hit
            kf.update(ax, ay)
        elif prev_bbox is not None:
            ax, ay = pred_ax, pred_ay
            score = 0.22
        else:
            guess = (pred_ax, pred_ay)
            if mode_l == "track":
                pred_lk = lk_predict(frames_bgr[i - 1], bgr, prev_ax, prev_ay)
                if pred_lk is not None:
                    guess = pred_lk
            use_tmpl = tmpl0 if mode_l == "fixed" else tmpl
            tax, tay, tscore = match_template_anchor(
                bgr,
                use_tmpl,
                guess_xy=guess,
                search_frac=0.40 if mode_l == "track" else 0.85,
            )
            if tscore >= min_score:
                ax, ay, score = tax, tay, tscore
                kf.update(ax, ay)
                if mode_l == "track":
                    tmpl, _, _ = extract_template(bgr, ax, ay)
            else:
                ax, ay = pred_ax, pred_ay
                score = 0.20
        lay = layout_from_anchor(
            ax, ay, geom, wi, hi, auto=True, conf=score
        )
        out.append(lay)
        vel_x = ax - prev_ax
        vel_y = ay - prev_ay
        prev_ax, prev_ay = ax, ay
        if mode_l == "track" and (used_yolo or score >= min_score):
            tmpl, _, _ = extract_template(bgr, ax, ay)
    return out
