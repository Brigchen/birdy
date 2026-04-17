# -*- coding: utf-8 -*-
"""
连拍图片预分组模块：
1. 读取图片 EXIF 拍摄时间并按时间间隔分组
2. 启用鸟检时：先检鸟；对焦在最大鸟体框 ROI 上计算（`FOCUS_METRIC_MODE`：laplacian / hybrid / mask_hybrid）
3. 启用鸟检且无有效鸟体（面积低于阈值）的图一票否决；连拍组内不参与保留排序，非连拍单张直接丢弃
4. 每组保留张数 = min(组大小, max(最少保留, round(组大小×保留比例)))
5. 组内排序综合分 ≈ `FOCUS_WEIGHT * focus_score + BIRD_AREA_WEIGHT * (bird_area/10000)`

配置项在 burst_config.py 中设置

作者: brigchen@gmail.com
版权说明: 基于开源协议，仅限爱好者、公益、科研等非盈利用途，请勿用于商业用途
"""
import os
import cv2
import json
import time
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Set, Callable
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 尝试导入配置文件
try:
    from burst_config import (
        BURST_TIME_THRESHOLD,
        KEEP_TOP_N,
        BURST_KEEP_RATIO,
        BURST_KEEP_MIN,
        MIN_BIRD_AREA,
        ENABLE_FOCUS_SORT,
        FOCUS_WEIGHT,
        BIRD_AREA_WEIGHT,
        FOCUS_METRIC_MODE,
        FOCUS_ROI_MARGIN_FRAC,
        FOCUS_HYBRID_W_LAP,
        FOCUS_HYBRID_W_TEN,
        FOCUS_HYBRID_W_CENTER,
        FOCUS_HYBRID_RING_PX,
        FOCUS_HYBRID_BG_PENALTY,
        BIRD_CONF_THRESHOLD,
        ENABLE_EYE_DETECTION,
        EYE_CONF_THRESHOLD,
        EYE_ROI_MARGIN_FRAC,
        EYE_INSIDE_TOL_FRAC,
        EYE_MAX_PER_BIRD,
        EYE_BONUS_WEIGHT,
    )
except ImportError:
    BURST_TIME_THRESHOLD = 1.0
    KEEP_TOP_N = 2
    BURST_KEEP_RATIO = 0.2
    BURST_KEEP_MIN = 2
    MIN_BIRD_AREA = 1000
    ENABLE_FOCUS_SORT = True
    FOCUS_WEIGHT = 1.0
    BIRD_AREA_WEIGHT = 0.45
    FOCUS_METRIC_MODE = "hybrid"
    FOCUS_ROI_MARGIN_FRAC = 0.04
    FOCUS_HYBRID_W_LAP = 1.0
    FOCUS_HYBRID_W_TEN = 0.55
    FOCUS_HYBRID_W_CENTER = 0.35
    FOCUS_HYBRID_RING_PX = 10
    FOCUS_HYBRID_BG_PENALTY = 1.15
    BIRD_CONF_THRESHOLD = 0.3
    ENABLE_EYE_DETECTION = False
    EYE_CONF_THRESHOLD = 0.25
    EYE_ROI_MARGIN_FRAC = 0.12
    EYE_INSIDE_TOL_FRAC = 0.06
    EYE_MAX_PER_BIRD = 2
    EYE_BONUS_WEIGHT = 0.8

# 快速模式配置
FAST_MODE_ENABLED = False  # 是否启用快速模式（跳过某些计算）
BATCH_SIZE = 5  # 批处理大小（每批处理的连拍组数）
MODEL_CACHE = None  # 模型缓存，避免重复加载
EYE_MODEL_CACHE = None  # 鸟眼模型缓存


@dataclass
class ImageInfo:
    """单张图片的信息"""
    path: str
    time: datetime
    time_diff: float = 0.0  # 与上一张的时间差（秒）
    birds: List[Dict] = field(default_factory=list)  # 检测到的鸟
    focus_score: float = 0.0  # 对焦评分（见 burst_config.FOCUS_METRIC_MODE）
    bird_area: float = 0.0  # 最大鸟体面积
    eye_count: int = 0  # 真实鸟眼检出数量（眼中心落在任一鸟框内）
    has_eye: bool = False  # 是否有真实鸟眼检出
    keep: bool = True  # 是否保留


@dataclass
class BurstGroup:
    """一组连拍照片"""
    images: List[ImageInfo] = field(default_factory=list)
    group_id: int = 0
    
    @property
    def total(self) -> int:
        return len(self.images)
    
    @property
    def kept(self) -> int:
        return sum(1 for img in self.images if img.keep)


def read_exif_time(image_path: str) -> Optional[datetime]:
    """
    读取图片的EXIF拍摄时间
    
    Args:
        image_path: 图片路径
    
    Returns:
        datetime对象，读取失败返回None
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return None
            
            # 查找时间标签
            tag_map = {v: k for k, v in TAGS.items()}
            
            # 尝试多个时间标签
            time_tags = [306, 36867, 36868]  # DateTimeOriginal, DateTimeDigitized
            for tag_id in time_tags:
                if tag_id in exif:
                    dt_str = exif.get(tag_id)
                    if dt_str:
                        # 格式: "2024:03:15 14:30:45"
                        return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            
            return None
    except Exception as e:
        return None


def _expand_bbox_xyxy(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin_frac: float = None
) -> Tuple[int, int, int, int]:
    if margin_frac is None:
        try:
            from burst_config import FOCUS_ROI_MARGIN_FRAC as margin_frac
        except Exception:
            margin_frac = 0.04
    margin_frac = float(margin_frac)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    m = int(max(bw, bh) * margin_frac)
    nx1 = max(0, x1 - m)
    ny1 = max(0, y1 - m)
    nx2 = min(w, x2 + m)
    ny2 = min(h, y2 + m)
    if nx2 <= nx1 or ny2 <= ny1:
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return nx1, ny1, nx2, ny2


def _sobel_mean_magnitude(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))


def _center_weighted_laplacian_energy(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    yy, xx = np.indices((h, w))
    sigma = 0.28 * float(min(h, w)) + 1e-3
    wgt = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    wgt /= float(np.sum(wgt) + 1e-8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.sum(wgt * (lap * lap)))


def _edge_ring_vs_center_penalty(gray: np.ndarray, ring_px: int) -> float:
    """ROI 边缘环带 Sobel 能量相对中心区域过高时惩罚（树枝/背景清晰抢分）。"""
    try:
        from burst_config import FOCUS_HYBRID_BG_PENALTY as kpen
    except Exception:
        kpen = 1.15
    h, w = gray.shape[:2]
    rp = int(max(1, min(ring_px, min(h, w) // 5)))
    ring = np.zeros((h, w), np.uint8)
    ring[:rp, :] = 1
    ring[-rp:, :] = 1
    ring[:, :rp] = 1
    ring[:, -rp:] = 1
    cw = max(2, int(w * 0.34))
    ch = max(2, int(h * 0.34))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    center = np.zeros((h, w), np.uint8)
    center[y0 : y0 + ch, x0 : x0 + cw] = 1
    ring_m = (ring.astype(bool)) & (~center.astype(bool))
    if not np.any(ring_m) or not np.any(center):
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mr = float(np.mean(mag[ring_m]))
    mc = float(np.mean(mag[center.astype(bool)]))
    ratio = mr / (mc + 1e-6)
    return float(max(0.0, ratio - 1.35) * float(kpen))


def _mask_from_polygon_on_crop(
    poly_xy: np.ndarray, crop_w: int, crop_h: int, ox: int, oy: int
) -> Optional[np.ndarray]:
    if poly_xy is None or poly_xy.size < 6:
        return None
    m = np.zeros((crop_h, crop_w), np.uint8)
    pts = np.asarray(poly_xy, dtype=np.float32)
    pts[:, 0] -= float(ox)
    pts[:, 1] -= float(oy)
    pts_i = np.round(pts).astype(np.int32)
    cv2.fillPoly(m, [pts_i], 1)
    if int(np.sum(m)) < 50:
        return None
    return m


def _focus_score_hybrid(gray: np.ndarray) -> float:
    try:
        from burst_config import (
            FOCUS_HYBRID_W_LAP as wl,
            FOCUS_HYBRID_W_TEN as wt,
            FOCUS_HYBRID_W_CENTER as wc,
            FOCUS_HYBRID_RING_PX as wrp,
        )
    except Exception:
        wl, wt, wc, wrp = 1.0, 0.55, 0.35, 10
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    ten = _sobel_mean_magnitude(gray)
    cen = _center_weighted_laplacian_energy(gray)
    pen = _edge_ring_vs_center_penalty(gray, int(wrp))
    return float(
        float(wl) * np.log1p(max(0.0, lap_var))
        + float(wt) * np.log1p(max(0.0, ten))
        + float(wc) * np.log1p(max(0.0, cen))
        - pen
    )


def _focus_score_mask_hybrid(
    gray: np.ndarray, mask_u8: np.ndarray
) -> Optional[float]:
    """在分割掩膜内计算 hybrid；失败返回 None。"""
    if mask_u8 is None or mask_u8.shape[:2] != gray.shape[:2]:
        return None
    m = (mask_u8 > 0).astype(np.uint8)
    if int(np.sum(m)) < 80:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m_e = cv2.erode(m, k, iterations=1)
    m_d = cv2.dilate(m, k, iterations=2)
    ring = ((m_d - m) > 0).astype(bool)
    core = (m_e > 0).astype(bool)
    if not np.any(core):
        return None
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    lap_c = lap[core]
    lap_var = float(np.var(lap_c)) if lap_c.size > 1 else float(np.mean(lap_c * lap_c))
    ten = float(np.mean(mag[core]))
    cen = float(np.mean((lap[core] * lap[core])))
    mr = float(np.mean(mag[ring])) if np.any(ring) else 0.0
    mc = float(np.mean(mag[core])) + 1e-6
    try:
        from burst_config import FOCUS_HYBRID_BG_PENALTY as kpen
    except Exception:
        kpen = 1.15
    pen = float(max(0.0, mr / mc - 1.25) * float(kpen))
    try:
        from burst_config import (
            FOCUS_HYBRID_W_LAP as wl,
            FOCUS_HYBRID_W_TEN as wt,
            FOCUS_HYBRID_W_CENTER as wc,
        )
    except Exception:
        wl, wt, wc = 1.0, 0.55, 0.35
    return float(
        float(wl) * np.log1p(max(0.0, lap_var))
        + float(wt) * np.log1p(max(0.0, ten))
        + float(wc) * np.log1p(max(0.0, cen))
        - pen
    )


def calculate_focus_score(
    image_path: str,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    mask_xy: Optional[List[List[float]]] = None,
) -> float:
    """
    计算对焦评分。
    - laplacian：ROI 内 Laplacian 方差（旧版）
    - hybrid：Laplacian + Tenengrad + 中心加权 Laplacian 能量 − 边缘环背景惩罚
    - mask_hybrid：在 YOLO 分割多边形掩膜内计分；无掩膜则回退 hybrid
    """
    try:
        from burst_config import FOCUS_METRIC_MODE as _mode
    except Exception:
        _mode = "hybrid"
    mode = (str(_mode or "hybrid")).strip().lower()

    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.0
        h, w = img.shape[:2]
        ox = oy = 0
        if roi_xyxy is not None:
            x1, y1, x2, y2 = roi_xyxy
            x1, y1, x2, y2 = _expand_bbox_xyxy(
                int(x1), int(y1), int(x2), int(y2), w, h
            )
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return 0.0
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            ox, oy = x1, y1
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if mode == "laplacian":
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())

        if mode == "mask_hybrid" and mask_xy:
            poly = np.asarray(mask_xy, dtype=np.float32)
            if poly.ndim == 2 and poly.shape[0] >= 3:
                gh, gw = gray.shape[:2]
                mk = _mask_from_polygon_on_crop(poly, gw, gh, ox, oy)
                if mk is not None:
                    fs = _focus_score_mask_hybrid(gray, mk)
                    if fs is not None:
                        return float(fs)

        return float(_focus_score_hybrid(gray))
    except Exception:
        return 0.0


def compute_burst_keep_count(n: int, ratio: float, min_keep: int) -> int:
    """组内共 n 张时实际保留张数：max(最小保留, round(n×比例))，且不超过 n。"""
    if n <= 0:
        return 0
    r = max(0.01, min(1.0, float(ratio)))
    mk = max(1, int(min_keep))
    by_ratio = max(1, int(round(n * r)))
    return min(n, max(mk, by_ratio))


def detect_birds_in_image(image_path: str, conf: float = 0.25) -> List[Dict]:
    """
    使用YOLOv8检测图片中的鸟
    
    Args:
        image_path: 图片路径
        conf: 置信度阈值
    
    Returns:
        检测到的鸟列表，每个包含bbox、conf等
    """
    try:
        from ultralytics import YOLO
        
        # 使用默认的yolov8x模型
        model_path = PROJECT_ROOT / "models" / "bird-seg.pt"
        model = YOLO(str(model_path))
        
        results = model(image_path, conf=conf, verbose=False)
        
        birds = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # 只保留鸟类（class 0 = person, 需要检测鸟类的模型）
                # 使用专门的鸟检测模型
                pass
        
        return birds
    except Exception as e:
        print(f"  鸟体检测失败: {e}")
        return []


def calculate_bird_area(bbox: List[int]) -> float:
    """
    计算鸟的边界框面积
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        面积
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def group_images_by_time(
    image_folder: str,
    time_threshold: float = BURST_TIME_THRESHOLD
) -> Tuple[List[BurstGroup], List[ImageInfo]]:
    """
    按拍摄时间分组图片
    
    Args:
        image_folder: 图片文件夹路径
        time_threshold: 连拍时间阈值（秒）
    
    Returns:
        (连拍组列表, 非连拍图片列表)
    """
    folder = Path(image_folder)
    
    # 支持的图片格式
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    
    # 获取所有图片文件
    image_files = [
        f for f in folder.iterdir() 
        if f.suffix.lower() in extensions and f.is_file()
    ]
    
    if not image_files:
        print(f"警告: 文件夹中没有发现图片: {image_folder}")
        return [], []
    
    print(f"发现 {len(image_files)} 张图片")
    
    # 读取每张图片的拍摄时间
    image_infos: List[ImageInfo] = []
    
    for img_file in sorted(image_files):
        img_time = read_exif_time(str(img_file))
        
        if img_time is None:
            # 如果没有EXIF时间，尝试使用文件修改时间
            timestamp = img_file.stat().st_mtime
            img_time = datetime.fromtimestamp(timestamp)
            print(f"  [TIME] {img_file.name}: 使用文件时间")
        else:
            print(f"  [TIME] {img_file.name}: {img_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        image_infos.append(ImageInfo(
            path=str(img_file),
            time=img_time
        ))
    
    # 按时间排序
    image_infos.sort(key=lambda x: x.time)
    
    # 计算相邻图片的时间差
    for i in range(1, len(image_infos)):
        time_diff = (image_infos[i].time - image_infos[i-1].time).total_seconds()
        image_infos[i].time_diff = time_diff
    
    # 分组
    groups: List[BurstGroup] = []
    non_burst: List[ImageInfo] = []
    current_group = BurstGroup(group_id=0)
    
    for i, img_info in enumerate(image_infos):
        if i == 0:
            # 第一张图片
            current_group.images.append(img_info)
        elif img_info.time_diff <= time_threshold:
            # 时间差在阈值内，属于同一组连拍
            current_group.images.append(img_info)
        else:
            # 时间差超过阈值，保存当前组，开始新组
            if len(current_group.images) > 1:
                groups.append(current_group)
            else:
                non_burst.extend(current_group.images)
            
            current_group = BurstGroup(group_id=len(groups))
            current_group.images.append(img_info)
    
    # 处理最后一组
    if len(current_group.images) > 1:
        groups.append(current_group)
    else:
        non_burst.extend(current_group.images)
    
    return groups, non_burst


def get_or_load_model(force_reload: bool = False):
    """
    获取或加载YOLOv8模型（支持缓存以加速多组处理）
    
    Args:
        force_reload: 是否强制重新加载
    
    Returns:
        YOLO模型实例
    """
    global MODEL_CACHE
    
    from ultralytics import YOLO
    
    if MODEL_CACHE is not None and not force_reload:
        return MODEL_CACHE
    
    try:
        # model = YOLO("yolov8x.pt")
        model_path = PROJECT_ROOT / "models" / "bird-seg.pt"
        model = YOLO(str(model_path))
        MODEL_CACHE = model
        return model
    except Exception as e:
        print(f"  YOLOv8加载失败: {e}")
        return None


def get_or_load_eye_model(force_reload: bool = False):
    """获取或加载鸟眼检测模型（支持缓存）。"""
    global EYE_MODEL_CACHE
    from ultralytics import YOLO

    if EYE_MODEL_CACHE is not None and not force_reload:
        return EYE_MODEL_CACHE

    try:
        model_path = PROJECT_ROOT / "models" / "birdeye.pt"
        model = YOLO(str(model_path))
        EYE_MODEL_CACHE = model
        return model
    except Exception as e:
        print(f"  鸟眼模型加载失败: {e}")
        return None


def _point_in_bbox(x: float, y: float, bbox: List[int], tol_frac: float = 0.0) -> bool:
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    tx = bw * max(0.0, float(tol_frac))
    ty = bh * max(0.0, float(tol_frac))
    return (x1 - tx) <= x <= (x2 + tx) and (y1 - ty) <= y <= (y2 + ty)


def _attach_eyes_to_birds(img_info: ImageInfo, eye_model, conf: Optional[float] = None) -> None:
    """
    鸟眼检测后，仅当眼中心落在鸟框内才记为真实检出，并写回 birds[*]['eyes']。
    """
    if conf is None:
        conf = EYE_CONF_THRESHOLD
    conf = float(conf)

    for b in img_info.birds:
        b["eyes"] = []
        b["eye_count"] = 0
        b["has_eye"] = False
    img_info.eye_count = 0
    img_info.has_eye = False

    if eye_model is None or not img_info.birds:
        return

    try:
        img = cv2.imread(img_info.path)
        if img is None:
            return
        ih, iw = img.shape[:2]
        valid_eyes: List[Dict] = []
        for b in img_info.birds:
            bx1, by1, bx2, by2 = [int(v) for v in b["bbox"]]
            rx1, ry1, rx2, ry2 = _expand_bbox_xyxy(
                bx1, by1, bx2, by2, iw, ih, margin_frac=EYE_ROI_MARGIN_FRAC
            )
            crop = img[ry1:ry2, rx1:rx2]
            if crop is None or crop.size == 0:
                continue

            dets: List[Dict] = []
            results = eye_model(crop, conf=conf, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                for i in range(len(boxes)):
                    xy = boxes.xyxy[i].cpu().numpy()
                    x1 = float(xy[0]) + rx1
                    y1 = float(xy[1]) + ry1
                    x2 = float(xy[2]) + rx1
                    y2 = float(xy[3]) + ry1
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    if not _point_in_bbox(
                        cx, cy, b["bbox"], tol_frac=EYE_INSIDE_TOL_FRAC
                    ):
                        continue
                    c = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    dets.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "center": [float(cx), float(cy)],
                            "conf": c,
                            "class": cls,
                        }
                    )

            if dets:
                dets = sorted(dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True)[
                    : max(1, int(EYE_MAX_PER_BIRD))
                ]
                b["eyes"] = dets
                b["eye_count"] = len(dets)
                b["has_eye"] = True
                valid_eyes.extend(dets)

        img_info.eye_count = len(valid_eyes)
        img_info.has_eye = img_info.eye_count > 0
    except Exception as e:
        print(f"      鸟眼检测失败: {e}")


def _detect_birds_yolo(img_info: ImageInfo, model, conf: Optional[float] = None) -> None:
    """在 img_info 上写入 birds / bird_area（COCO class 14 = bird）；seg 模型可带 mask_xy。"""
    if conf is None:
        try:
            from burst_config import BIRD_CONF_THRESHOLD as conf
        except Exception:
            conf = BIRD_CONF_THRESHOLD
    conf = float(conf)
    img_info.birds.clear()
    img_info.bird_area = 0.0
    try:
        results = model(img_info.path, conf=conf, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            masks_xy = None
            if getattr(result, "masks", None) is not None:
                try:
                    masks_xy = result.masks.xy
                except Exception:
                    masks_xy = None
            n = len(boxes)
            for det_idx in range(n):
                cls = int(boxes.cls[det_idx].cpu().numpy())
                if cls != 14:
                    continue
                xy = boxes.xyxy[det_idx].cpu().numpy()
                x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
                area = calculate_bird_area([int(x1), int(y1), int(x2), int(y2)])
                c = float(boxes.conf[det_idx].cpu().numpy())
                mask_xy = None
                if masks_xy is not None and det_idx < len(masks_xy):
                    try:
                        t = masks_xy[det_idx]
                        arr = np.asarray(
                            t.cpu().numpy() if hasattr(t, "cpu") else t,
                            dtype=np.float32,
                        )
                        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                            mask_xy = arr.reshape(-1, 2).tolist()
                    except Exception:
                        mask_xy = None
                img_info.birds.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "conf": c,
                        "area": area,
                        "mask_xy": mask_xy,
                    }
                )
                if area > img_info.bird_area:
                    img_info.bird_area = area
    except Exception as e:
        print(f"      鸟体检测失败: {e}")


def evaluate_focus_for_group(
    group: BurstGroup,
    use_bird_detection: bool = True,
    model=None,
    use_eye_detection: bool = False,
    eye_model=None,
    fast_mode: bool = False,
    min_bird_area: int = MIN_BIRD_AREA,
) -> BurstGroup:
    """
    评估连拍组对焦。启用鸟检时：先检鸟；对焦在最大鸟体框 ROI 上按 FOCUS_METRIC_MODE 计分；
    无有效鸟体则对焦记 0（后续筛选一票否决）。
    快速模式与鸟检同时开启时不对焦采样（避免误用邻帧分数）。
    """
    print(f"\n  评估连拍组 {group.group_id}，共 {len(group.images)} 张图片")

    if use_bird_detection and model is None:
        model = get_or_load_model()
        if model is None:
            use_bird_detection = False
    if use_eye_detection and (not use_bird_detection):
        use_eye_detection = False
    if use_eye_detection and eye_model is None:
        eye_model = get_or_load_eye_model()
        if eye_model is None:
            use_eye_detection = False

    if use_bird_detection and model:
        for img_info in group.images:
            print(f"    鸟检: {Path(img_info.path).name}")
            _detect_birds_yolo(img_info, model)
            if use_eye_detection and eye_model is not None:
                _attach_eyes_to_birds(img_info, eye_model)
            print(
                f"      鸟 {len(img_info.birds)} 只，最大面积: {img_info.bird_area:.0f}"
            )
            if use_eye_detection:
                print(
                    f"      鸟眼有效检出: {img_info.eye_count}（仅统计落在鸟体框内）"
                )

    fast_sample = fast_mode and (not use_bird_detection) and len(group.images) > 3
    sample_indices: Optional[Set[int]] = None
    if fast_sample:
        total_images = len(group.images)
        sample_count = max(3, total_images // 3)
        step = total_images / sample_count
        sample_indices = {
            int(round(i * step)) for i in range(sample_count)
        }
        sample_indices = {i for i in sample_indices if 0 <= i < total_images}
        sample_indices = set(sorted(sample_indices))

    for i, img_info in enumerate(group.images):
        if fast_sample and sample_indices is not None and i not in sample_indices:
            continue
        print(f"    对焦: {Path(img_info.path).name}")
        if use_bird_detection:
            if img_info.bird_area < min_bird_area:
                img_info.focus_score = 0.0
                print("      无有效鸟体 ROI，跳过鸟体对焦（本张不入筛选）")
                continue
            best = max(img_info.birds, key=lambda b: b.get("area", 0.0))
            bx = best["bbox"]
            fs = calculate_focus_score(
                img_info.path,
                (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
                best.get("mask_xy"),
            )
            img_info.focus_score = fs
            print(f"      鸟体 ROI 对焦评分: {fs:.2f}")
        else:
            fs = calculate_focus_score(img_info.path)
            img_info.focus_score = fs
            print(f"      全图对焦评分: {fs:.2f}")

    if fast_sample and sample_indices is not None:
        for i, img_info in enumerate(group.images):
            if i in sample_indices:
                continue
            closest_idx = min(sample_indices, key=lambda x: abs(x - i))
            img_info.focus_score = group.images[closest_idx].focus_score * 0.95
            print(
                f"    对焦(近似): {Path(img_info.path).name} "
                f"← 邻帧 ×0.95 = {img_info.focus_score:.2f}"
            )

    return group


def keep_entire_burst_group_without_scoring(group: BurstGroup) -> BurstGroup:
    """当计算出的保留张数 >= 本组张数时，跳过对焦/鸟检，全部保留。"""
    n = len(group.images)
    print(
        f"\n  连拍组 {group.group_id} 共 {n} 张，"
        f"保留数≥组大小 → 跳过对焦与鸟体检测，全部保留"
    )
    for img in group.images:
        img.keep = True
    return group


def select_best_images(
    group: BurstGroup,
    keep_top_n: int = KEEP_TOP_N,
    min_bird_area: int = MIN_BIRD_AREA,
    use_bird_detection: bool = False,
) -> BurstGroup:
    """
    在连拍组内保留对焦最优的 keep_top_n 张。
    启用鸟检时：无有效鸟体（面积 < 阈值）的图不参与排序，一律丢弃（一票否决）。
    """
    if not group.images:
        return group

    def _eligible(im: ImageInfo) -> bool:
        if not use_bird_detection:
            return True
        return im.bird_area >= min_bird_area

    print(f"\n  选择最佳图片（本组至多保留 {keep_top_n} 张）")

    for im in group.images:
        im.keep = False

    eligible = [im for im in group.images if _eligible(im)]
    if not eligible:
        print("    本组无有效鸟体，全部丢弃（一票否决）")
        return group

    sorted_eligible = sorted(
        eligible,
        key=lambda x: (
            float(FOCUS_WEIGHT) * x.focus_score
            + float(BIRD_AREA_WEIGHT) * (x.bird_area / 10000.0)
            + (float(EYE_BONUS_WEIGHT) if bool(getattr(x, "has_eye", False)) else 0.0)
        ),
        reverse=True,
    )
    take = min(keep_top_n, len(sorted_eligible))
    for im in sorted_eligible[:take]:
        im.keep = True
        tag = "鸟体ROI对焦" if use_bird_detection else "全图对焦"
        print(
            f"    保留: {Path(im.path).name} - {tag} "
            f"(评分: {im.focus_score:.2f}, 鸟面积: {im.bird_area:.0f})"
        )
    for im in sorted_eligible[take:]:
        print(
            f"    丢弃: {Path(im.path).name} - 组内排名靠后 "
            f"(评分: {im.focus_score:.2f})"
        )
    for im in group.images:
        if not _eligible(im):
            print(
                f"    丢弃: {Path(im.path).name} - 无有效鸟体（一票否决）"
            )
    return group


def process_folder(
    image_folder: str,
    time_threshold: float = BURST_TIME_THRESHOLD,
    burst_keep_ratio: float = BURST_KEEP_RATIO,
    burst_keep_min: int = BURST_KEEP_MIN,
    use_bird_detection: bool = True,
    use_eye_detection: bool = ENABLE_EYE_DETECTION,
    output_report: str = None,
    fast_mode: bool = False,
    batch_size: int = BATCH_SIZE,
    screened_output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """
    处理整个文件夹的图片

    burst_keep_ratio / burst_keep_min：每组实际保留张数为
    min(组内张数, max(burst_keep_min, round(组内张数 * burst_keep_ratio)))。
    """
    print("=" * 60)
    print("连拍图片预分组处理")
    print("=" * 60)
    print(f"图片文件夹: {image_folder}")
    print(f"连拍阈值: {time_threshold}秒")
    print(
        f"保留策略: 比例 {burst_keep_ratio:.2f}，最少 {burst_keep_min} 张/组（按组分别计算）"
    )
    print(f"鸟体检测: {'启用' if use_bird_detection else '禁用'}")
    print(f"鸟眼检测: {'启用' if (use_bird_detection and use_eye_detection) else '禁用'}")
    print(f"快速模式: {'启用' if fast_mode else '禁用'}")
    if fast_mode:
        print(f"批处理大小: {batch_size}")
    print("=" * 60)
    
    # 1. 按时间分组
    groups, non_burst = group_images_by_time(image_folder, time_threshold)
    
    print(f"\n发现 {len(groups)} 组连拍，共 {sum(g.total for g in groups)} 张图片")
    print(f"非连拍图片: {len(non_burst)} 张")
    
    # 2. 处理每组连拍
    all_results = {
        "groups": [],
        "non_burst": [],
        "total_images": 0,
        "kept_images": 0,
        "discarded_images": 0,
        "processing_time": 0
    }
    
    # 预加载模型用于快速批处理
    model = None
    eye_model = None
    if use_bird_detection and fast_mode:
        model = get_or_load_model()
        if use_eye_detection:
            eye_model = get_or_load_eye_model()
    
    start_time = time.time()
    total_expected = int(sum(g.total for g in groups) + len(non_burst))
    processed_count = 0
    if progress_callback:
        try:
            progress_callback(
                {"kind": "start", "done": 0, "total": max(1, total_expected)}
            )
        except Exception:
            pass
    
    # 批处理：每batch_size组使用一个模型实例
    for batch_start in range(0, len(groups), batch_size):
        batch_end = min(batch_start + batch_size, len(groups))
        batch_groups = groups[batch_start:batch_end]
        
        if fast_mode and len(groups) > batch_size:
            print(f"\n[快速模式] 处理批次 {batch_start // batch_size + 1}/{(len(groups) + batch_size - 1) // batch_size}")
        
        for group in batch_groups:
            n_g = len(group.images)
            keep_count = compute_burst_keep_count(
                n_g, burst_keep_ratio, burst_keep_min
            )
            skip_scoring = (keep_count >= n_g) and (not use_bird_detection)
            if skip_scoring:
                group = keep_entire_burst_group_without_scoring(group)
            else:
                group = evaluate_focus_for_group(
                    group,
                    use_bird_detection,
                    model,
                    use_eye_detection=use_eye_detection,
                    eye_model=eye_model,
                    fast_mode=fast_mode,
                    min_bird_area=MIN_BIRD_AREA,
                )
                group = select_best_images(
                    group,
                    keep_top_n=keep_count,
                    min_bird_area=MIN_BIRD_AREA,
                    use_bird_detection=use_bird_detection,
                )
            
            # 保存结果
            group_result = {
                "group_id": group.group_id,
                "total": group.total,
                "kept": group.kept,
                "evaluation_skipped": skip_scoring,
                "images": []
            }
            
            for img in group.images:
                img_result = {
                    "path": img.path,
                    "name": Path(img.path).name,
                    "time": img.time.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_diff": img.time_diff,
                    "focus_score": round(img.focus_score, 2),
                    "bird_area": round(img.bird_area, 2),
                    "birds_detected": len(img.birds),
                    "eye_count": int(getattr(img, "eye_count", 0)),
                    "has_eye": bool(getattr(img, "has_eye", False)),
                    "kept": img.keep,
                    "birds": img.birds  # 保存详细的鸟检测信息
                }
                group_result["images"].append(img_result)
                
                all_results["total_images"] += 1
                if img.keep:
                    all_results["kept_images"] += 1
                else:
                    all_results["discarded_images"] += 1
                processed_count += 1
                if progress_callback:
                    try:
                        progress_callback(
                            {
                                "kind": "tick",
                                "done": processed_count,
                                "total": max(1, total_expected),
                            }
                        )
                    except Exception:
                        pass
            
            all_results["groups"].append(group_result)
    
    # 非连拍：启用鸟检且无有效鸟体则丢弃；有鸟则鸟体 ROI 对焦（与连拍一致）
    if non_burst:
        print(f"\n非连拍单张筛选: 共 {len(non_burst)} 张")
    nb_model = model
    nb_eye_model = eye_model
    if non_burst and use_bird_detection and nb_model is None:
        nb_model = get_or_load_model()
    if non_burst and use_bird_detection and use_eye_detection and nb_eye_model is None:
        nb_eye_model = get_or_load_eye_model()
    if non_burst and use_bird_detection and nb_model is None:
        print("  警告: 鸟体模型不可用，非连拍单张将按全图对焦并全部保留（未做无鸟丢弃）")

    for img in non_burst:
        kept = True
        if use_bird_detection and nb_model is not None:
            print(f"  单张: {Path(img.path).name}")
            _detect_birds_yolo(img, nb_model)
            if use_eye_detection and nb_eye_model is not None:
                _attach_eyes_to_birds(img, nb_eye_model)
            if img.bird_area < MIN_BIRD_AREA:
                img.focus_score = 0.0
                kept = False
                print("    丢弃: 无有效鸟体（一票否决）")
            else:
                best = max(img.birds, key=lambda b: b.get("area", 0.0))
                bx = best["bbox"]
                img.focus_score = calculate_focus_score(
                    img.path,
                    (int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
                    best.get("mask_xy"),
                )
                print(
                    f"    保留: 鸟体 ROI 对焦评分 {img.focus_score:.2f} "
                    f"(鸟面积 {img.bird_area:.0f})"
                )
        else:
            img.focus_score = calculate_focus_score(img.path)
            print(
                f"  单张: {Path(img.path).name} — 全图对焦 {img.focus_score:.2f}（保留）"
            )

        img.keep = kept
        entry = {
            "path": img.path,
            "name": Path(img.path).name,
            "time": img.time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_diff": img.time_diff,
            "focus_score": round(img.focus_score, 2),
            "bird_area": round(img.bird_area, 2),
            "birds_detected": len(img.birds),
            "eye_count": int(getattr(img, "eye_count", 0)),
            "has_eye": bool(getattr(img, "has_eye", False)),
            "kept": kept,
            "birds": list(img.birds),
        }
        all_results["non_burst"].append(entry)
        all_results["total_images"] += 1
        if kept:
            all_results["kept_images"] += 1
        else:
            all_results["discarded_images"] += 1
        processed_count += 1
        if progress_callback:
            try:
                progress_callback(
                    {
                        "kind": "tick",
                        "done": processed_count,
                        "total": max(1, total_expected),
                    }
                )
            except Exception:
                pass

    all_results["processing_time"] = round(time.time() - start_time, 2)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("处理结果摘要")
    print("=" * 60)
    print(f"总图片数: {all_results['total_images']}")
    print(f"保留图片: {all_results['kept_images']}")
    print(f"丢弃图片: {all_results['discarded_images']}")
    print(f"连拍组数: {len(groups)}")
    print(f"非连拍图片: {len(non_burst)}")
    print(f"处理时间: {all_results['processing_time']}秒")
    
    # 保存报告
    if output_report:
        with open(output_report, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {output_report}")

    if screened_output_dir:
        try:
            n_copied = copy_kept_images_to_screened(
                all_results, image_folder, screened_output_dir
            )
            print(f"\n已保留图片已复制到: {screened_output_dir} （共 {n_copied} 个文件）")
        except Exception as e:
            print(f"\n警告: 复制保留图片到 Screened_images 失败: {e}")
    if progress_callback:
        try:
            progress_callback(
                {"kind": "done", "done": max(1, total_expected), "total": max(1, total_expected)}
            )
        except Exception:
            pass
    
    return all_results


def get_kept_images(result: Dict) -> List[str]:
    """
    从处理结果中获取保留的图片路径
    
    Args:
        result: process_folder的返回结果
    
    Returns:
        保留的图片路径列表
    """
    kept = []
    
    # 连拍组中保留的图片
    for group in result.get("groups", []):
        for img in group.get("images", []):
            if img.get("kept"):
                kept.append(img["path"])
    
    # 非连拍图片全部保留
    for img in result.get("non_burst", []):
        if img.get("kept"):
            kept.append(img["path"])
    
    return kept


def copy_kept_images_to_screened(
    result: Dict,
    image_folder: str,
    screened_dir: str,
) -> int:
    """
    将连拍筛选保留的图片复制到 screened_dir（通常为 输出目录/Screened_images），
    尽量保持相对 image_folder 的子目录结构以避免重名覆盖。
    返回成功复制的文件数。
    """
    image_folder = os.path.abspath(image_folder)
    os.makedirs(screened_dir, exist_ok=True)
    n = 0
    for path in get_kept_images(result):
        abs_p = os.path.abspath(path)
        try:
            rel = os.path.relpath(abs_p, image_folder)
        except ValueError:
            rel = os.path.basename(abs_p)
        rel = rel.replace("\\", "/")
        dest = os.path.join(screened_dir, rel)
        dest_dir = os.path.dirname(dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(abs_p, dest)
        n += 1
    return n


# ─────────────────────────────────────────────────────────
# 配置文件（可在此修改默认参数）
# ─────────────────────────────────────────────────────────
BURST_CONFIG = """
# 连拍分组配置（实际生效请编辑 burst_config.py；此处为内嵌说明摘要）

BURST_TIME_THRESHOLD = 1.0
KEEP_TOP_N = 2
BURST_KEEP_RATIO = 0.2
BURST_KEEP_MIN = 2
MIN_BIRD_AREA = 1000
ENABLE_FOCUS_SORT = True

# 组内排序：FOCUS_WEIGHT * focus_score + BIRD_AREA_WEIGHT * (bird_area/10000)
FOCUS_WEIGHT = 1.0
BIRD_AREA_WEIGHT = 0.45

# laplacian | hybrid | mask_hybrid
FOCUS_METRIC_MODE = "mask_hybrid"
FOCUS_ROI_MARGIN_FRAC = 0.04
FOCUS_HYBRID_W_LAP = 1.0
FOCUS_HYBRID_W_TEN = 0.55
FOCUS_HYBRID_W_CENTER = 0.35
FOCUS_HYBRID_RING_PX = 10
FOCUS_HYBRID_BG_PENALTY = 1.15
BIRD_CONF_THRESHOLD = 0.3
ENABLE_EYE_DETECTION = False
EYE_CONF_THRESHOLD = 0.25
EYE_ROI_MARGIN_FRAC = 0.12
EYE_INSIDE_TOL_FRAC = 0.06
EYE_MAX_PER_BIRD = 2
EYE_BONUS_WEIGHT = 0.8
"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python burst_grouping.py <图片文件夹> [输出报告]")
        print("示例: python burst_grouping.py ./photos ./burst_report.json")
        sys.exit(1)
    
    folder = sys.argv[1]
    report = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_folder(folder, output_report=report)