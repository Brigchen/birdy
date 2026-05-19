#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地物种模型对比评测（ResNet34 vs EfficientNet-B0）

流程：
  1. 从已人工审核的 classification 目录读取 ground truth（物种 = 目录层级第 4 级中文名）
  2. bird-seg.pt 检测鸟体并裁剪（与主流程一致）
  3. 两套路权重分别推理 + 地理约束 + 可选「未知种类阈值」
  4. 汇总准确率、未知率，给出不同 geo / 阈值下的使用建议

用法（仓库根目录）:
  python test/benchmark_local_species_models.py ^
    --gt-dir "D:\\Birds_Transfer_Station_Level2\\classification_20260504-浦口"

  python test/benchmark_local_species_models.py --gt-dir ... --limit 50
  python test/benchmark_local_species_models.py --gt-dir ... --province 福建 --city 厦门
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from detect_bird_and_eye import (  # noqa: E402
    LOCAL_SPECIES_MODEL_EFFICIENTNET,
    LOCAL_SPECIES_MODEL_RESNET34,
    BirdSpeciesClassifier,
    _BIRD_INFO_PATH,
    _LOCAL_GEO_OUTSIDE_CONF,
    _local_model_geo_forced_unknown,
    geo_refine_species,
    gps_to_location_meta,
    lookup_classification,
    resolve_local_species_model_path,
)
from ultralytics import YOLO  # noqa: E402

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_DEFAULT_GT = r"D:\Birds_Transfer_Station_Level2\classification_20260504-浦口"
_DEFAULT_SEG = _REPO / "models" / "bird-seg.pt"


def _normalize_species_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip()
    s = re.sub(r"[（(].*?[）)]", "", s)
    s = s.replace(" ", "").replace("　", "").replace("·", "")
    return s


def _gt_accept_names(species_cn: str) -> Set[str]:
    """GT 物种名及 lookup 同义写法，用于判定是否识别正确。"""
    names: Set[str] = set()
    raw = (species_cn or "").strip()
    if raw:
        names.add(_normalize_species_name(raw))
    clf = lookup_classification(raw, "")
    for key in ("species_cn", "genus_cn", "chinese_name"):
        v = (clf.get(key) or "").strip()
        if v:
            names.add(_normalize_species_name(v))
    names.discard("")
    return names


def _pred_matches_gt(pred_cn: str, accept: Set[str]) -> bool:
    p = _normalize_species_name(pred_cn)
    if not p or not accept:
        return False
    if p in accept:
        return True
    for g in accept:
        if len(g) >= 2 and (p in g or g in p):
            return True
    return False


@dataclass
class GtSample:
    path: str
    species_cn: str
    order_cn: str
    family_cn: str
    genus_cn: str
    accept_names: Set[str]


def collect_gt_samples(root: str, *, limit: int = 0) -> List[GtSample]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"ground truth 目录不存在: {base}")
    out: List[GtSample] = []
    for dirpath, _dirs, files in os.walk(str(base)):
        rel_parts = Path(dirpath).relative_to(base).parts
        if len(rel_parts) < 4:
            continue
        species_cn = rel_parts[3]
        if species_cn in ("未知", "未知种", "未知属") or species_cn.startswith("未知"):
            continue
        order_cn, family_cn, genus_cn = rel_parts[0], rel_parts[1], rel_parts[2]
        accept = _gt_accept_names(species_cn)
        for fn in files:
            if Path(fn).suffix.lower() not in _IMAGE_EXTS:
                continue
            out.append(
                GtSample(
                    path=str(Path(dirpath) / fn),
                    species_cn=species_cn,
                    order_cn=order_cn,
                    family_cn=family_cn,
                    genus_cn=genus_cn,
                    accept_names=accept,
                )
            )
            if limit > 0 and len(out) >= limit:
                return out
    return out


def detect_largest_bird_crop(
    yolo: YOLO,
    image_bgr: np.ndarray,
    *,
    bird_conf: float,
    margin: int = 10,
) -> Tuple[Optional[np.ndarray], int, float]:
    """返回 (crop, num_birds, best_conf)。"""
    h, w = image_bgr.shape[:2]
    results = yolo(image_bgr, conf=bird_conf, verbose=False)
    birds: List[Tuple[float, List[int]]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            birds.append(
                (conf, [int(x1), int(y1), int(x2), int(y2)])
            )
    if not birds:
        return None, 0, 0.0
    birds.sort(key=lambda x: x[0], reverse=True)
    best_conf, (x1, y1, x2, y2) = birds[0]
    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)
    cx2 = min(w, x2 + margin)
    cy2 = min(h, y2 + margin)
    crop = image_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, len(birds), best_conf
    return crop.copy(), len(birds), best_conf


def classify_local_crop(
    clf: BirdSpeciesClassifier,
    crop_bgr: np.ndarray,
    *,
    province: Optional[str],
    city: Optional[str],
    geo_mode: str,
    min_accept_conf: Optional[float],
) -> Dict[str, Any]:
    """复现主流程本地模型 + 地理规则 + GUI 未知阈值。"""
    preds = clf.predict(crop_bgr, top_k=10)
    raw_top1 = preds[0] if preds else None
    unknown_reason = None

    if preds and _local_model_geo_forced_unknown(preds, province, geo_mode):
        unknown_reason = "local_top10_no_geo_species_top1_below_0_8"
        preds = []

    if preds:
        preds = geo_refine_species(
            preds,
            province,
            city,
            geo_mode=geo_mode,
            outside_list_conf=_LOCAL_GEO_OUTSIDE_CONF,
        )

    post_geo_top1 = preds[0] if preds else None
    if (
        preds
        and min_accept_conf is not None
        and float(preds[0].get("confidence") or 0) < float(min_accept_conf)
    ):
        unknown_reason = (
            f"top1_conf_below_min_accept "
            f"({float(preds[0].get('confidence') or 0):.3f} < {min_accept_conf:.3f})"
        )
        preds = []

    top = preds[0] if preds else None
    return {
        "pred_cn": (top or {}).get("chinese_name") or "",
        "pred_conf": float((top or {}).get("confidence") or 0),
        "pred_index": (top or {}).get("index"),
        "raw_top1_cn": (raw_top1 or {}).get("chinese_name") or "",
        "raw_top1_conf": float((raw_top1 or {}).get("confidence") or 0),
        "post_geo_top1_cn": (post_geo_top1 or {}).get("chinese_name") or "",
        "post_geo_top1_conf": float((post_geo_top1 or {}).get("confidence") or 0),
        "unknown_reason": unknown_reason,
        "is_unknown": top is None,
    }


@dataclass(frozen=True)
class EvalConfig:
    model: str
    geo_mode: str
    min_accept_conf: Optional[float]

    def key(self) -> str:
        mc = "none" if self.min_accept_conf is None else f"{self.min_accept_conf:.2f}"
        return f"{self.model}|geo={self.geo_mode}|thr={mc}"


def build_eval_grid(
    geo_modes: Sequence[str],
    thresholds: Sequence[Optional[float]],
) -> List[EvalConfig]:
    models = [LOCAL_SPECIES_MODEL_RESNET34, LOCAL_SPECIES_MODEL_EFFICIENTNET]
    grid: List[EvalConfig] = []
    for m in models:
        for g in geo_modes:
            for t in thresholds:
                grid.append(EvalConfig(model=m, geo_mode=g, min_accept_conf=t))
    return grid


def summarize_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r["config_key"]].append(r)

    summary: List[Dict[str, Any]] = []
    for key, items in sorted(buckets.items()):
        n = len(items)
        seg_ok = sum(1 for x in items if x.get("seg_ok"))
        correct = sum(1 for x in items if x.get("correct"))
        unknown = sum(1 for x in items if x.get("is_unknown"))
        confs = [float(x["pred_conf"]) for x in items if not x.get("is_unknown")]
        summary.append(
            {
                "config_key": key,
                "model": items[0]["model"],
                "geo_mode": items[0]["geo_mode"],
                "min_accept_conf": items[0]["min_accept_conf"],
                "n": n,
                "seg_detect_rate": round(seg_ok / n, 4) if n else 0,
                "top1_accuracy": round(correct / n, 4) if n else 0,
                "unknown_rate": round(unknown / n, 4) if n else 0,
                "mean_conf_when_labeled": round(sum(confs) / len(confs), 4) if confs else 0,
            }
        )
    summary.sort(
        key=lambda s: (s["top1_accuracy"], -s["unknown_rate"]),
        reverse=True,
    )
    return summary


def recommend_configs(summary: List[Dict[str, Any]]) -> str:
    lines = [
        "## 使用建议（自动生成）",
        "",
        "指标说明：",
        "- top1_accuracy：预测中文物种名与 GT 目录物种一致（含 lookup 别名）",
        "- unknown_rate：被判为未知的比例（地理规则或阈值导致）",
        "- seg_detect_rate：bird-seg 检出至少一只鸟的比例",
        "",
    ]
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for s in summary:
        by_model[s["model"]].append(s)

    for model, items in by_model.items():
        label = "ResNet34" if model == LOCAL_SPECIES_MODEL_RESNET34 else "EfficientNet-B0"
        lines.append(f"### {label} (`{model}`)")
        best = items[0]
        lines.append(
            f"- 综合最优（准确率优先）：`geo={best['geo_mode']}`, "
            f"阈值={best['min_accept_conf']}` → "
            f"准确率 {best['top1_accuracy']:.1%}, 未知率 {best['unknown_rate']:.1%}"
        )
        # 高准确率且未知率不太高
        balanced = sorted(
            items,
            key=lambda x: (x["top1_accuracy"] - 0.5 * x["unknown_rate"]),
            reverse=True,
        )[0]
        if balanced["config_key"] != best["config_key"]:
            lines.append(
                f"- 均衡（准确率 − 0.5×未知率）：`geo={balanced['geo_mode']}`, "
                f"阈值={balanced['min_accept_conf']}` → "
                f"准确率 {balanced['top1_accuracy']:.1%}, 未知率 {balanced['unknown_rate']:.1%}"
            )
        no_geo = [x for x in items if x["geo_mode"] == "none"]
        if no_geo:
            ng = no_geo[0]
            lines.append(
                f"- 无地理约束基线：`thr={ng['min_accept_conf']}` → "
                f"准确率 {ng['top1_accuracy']:.1%}"
            )
        lines.append("")

    if len(by_model) == 2:
        r34 = by_model.get(LOCAL_SPECIES_MODEL_RESNET34, [{}])[0]
        eff = by_model.get(LOCAL_SPECIES_MODEL_EFFICIENTNET, [{}])[0]
        if r34.get("top1_accuracy", 0) > eff.get("top1_accuracy", 0):
            lines.append(
                "**结论**：在当前 GT 集上 ResNet34 总体优于 EfficientNet-B0；"
                "建议在 GUI 默认选用 ResNet34，EfficientNet 仅作对比或特定阈值/geo 组合备选。"
            )
        elif eff.get("top1_accuracy", 0) > r34.get("top1_accuracy", 0):
            lines.append(
                "**结论**：EfficientNet-B0 总体更优；请检查权重文件是否与 bird_info 配对、"
                "以及 bird-seg 裁剪是否过小。"
            )
        else:
            lines.append("**结论**：两模型总体接近，请结合未知率与具体物种混淆表选择。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="本地物种模型 GT 对比评测")
    ap.add_argument("--gt-dir", default=_DEFAULT_GT, help="classification ground truth 根目录")
    ap.add_argument("--bird-seg", default=str(_DEFAULT_SEG), help="bird-seg.pt 路径")
    ap.add_argument("--bird-conf", type=float, default=0.5, help="鸟体检测置信度阈值")
    ap.add_argument("--limit", type=int, default=0, help="最多评测张数（0=全部）")
    ap.add_argument(
        "--province",
        default="福建",
        help="无 EXIF 省名时的默认省（浦口数据集对应厦门，默认福建）",
    )
    ap.add_argument(
        "--city",
        default="厦门",
        help="无 EXIF 市名时的默认市（浦口数据集默认厦门，非南京浦口）",
    )
    ap.add_argument(
        "--geo-modes",
        default="none,china,auto,province",
        help="逗号分隔：none|china|auto|province",
    )
    ap.add_argument(
        "--thresholds",
        default="none,0.5,0.6,0.7,0.75,0.8",
        help="未知种类阈值列表；none 表示不启用",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="报告输出目录（默认 <repo>/reports/species_benchmark_<时间戳>）",
    )
    args = ap.parse_args()

    geo_modes = [x.strip() for x in args.geo_modes.split(",") if x.strip()]
    thr_raw = [x.strip() for x in args.thresholds.split(",") if x.strip()]
    thresholds: List[Optional[float]] = []
    for t in thr_raw:
        if t.lower() in ("none", "null", "-"):
            thresholds.append(None)
        else:
            thresholds.append(float(t))

    print("收集 ground truth 样本…")
    samples = collect_gt_samples(args.gt_dir, limit=args.limit)
    if not samples:
        print(f"[错误] 未在 {args.gt_dir} 找到符合 目/科/属/种 层级的图片")
        sys.exit(1)
    print(f"  共 {len(samples)} 张，物种数 {len({s.species_cn for s in samples})}")

    seg_path = Path(args.bird_seg).expanduser().resolve()
    if not seg_path.is_file():
        print(f"[错误] 找不到 bird-seg: {seg_path}")
        sys.exit(1)

    print(f"加载 bird-seg: {seg_path}")
    yolo = YOLO(str(seg_path))

    classifiers: Dict[str, BirdSpeciesClassifier] = {}
    for kind in (LOCAL_SPECIES_MODEL_RESNET34, LOCAL_SPECIES_MODEL_EFFICIENTNET):
        mp = resolve_local_species_model_path(kind)
        print(f"加载 {kind}: {mp}")
        classifiers[kind] = BirdSpeciesClassifier(
            model_path=mp,
            bird_info_path=_BIRD_INFO_PATH,
            local_species_model=kind,
        )

    grid = build_eval_grid(geo_modes, thresholds)
    print(f"评测配置组合数: {len(grid)}（× {len(samples)} 张）")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else _REPO / "reports" / f"species_benchmark_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    for i, sample in enumerate(samples, 1):
        img = cv2.imread(sample.path)
        if img is None:
            continue

        province, city, gps_st, _gps_xy = gps_to_location_meta(sample.path)
        if not province:
            province = args.province.strip() or None
            city = args.city.strip() or None
            gps_st = "manual_default"

        crop, n_bird, seg_conf = detect_largest_bird_crop(
            yolo, img, bird_conf=args.bird_conf
        )
        seg_ok = crop is not None and crop.size > 0
        if not seg_ok:
            crop = img  # 仍跑分类便于对比 seg 失败影响

        for cfg in grid:
            clf = classifiers[cfg.model]
            eff_province = province
            eff_city = city
            if cfg.geo_mode == "province" and not eff_province:
                eff_province = args.province.strip() or None
                eff_city = args.city.strip() or None
            elif cfg.geo_mode == "none":
                eff_province, eff_city = None, None

            res = classify_local_crop(
                clf,
                crop,
                province=eff_province,
                city=eff_city,
                geo_mode=cfg.geo_mode,
                min_accept_conf=cfg.min_accept_conf,
            )
            correct = (not res["is_unknown"]) and _pred_matches_gt(
                res["pred_cn"], sample.accept_names
            )
            rows.append(
                {
                    "image": sample.path,
                    "gt_species": sample.species_cn,
                    "config_key": cfg.key(),
                    "model": cfg.model,
                    "geo_mode": cfg.geo_mode,
                    "min_accept_conf": cfg.min_accept_conf,
                    "province": eff_province or "",
                    "city": eff_city or "",
                    "gps_status": gps_st,
                    "seg_ok": seg_ok,
                    "seg_bird_count": n_bird,
                    "seg_best_conf": round(seg_conf, 4),
                    "correct": correct,
                    "is_unknown": res["is_unknown"],
                    "pred_cn": res["pred_cn"],
                    "pred_conf": res["pred_conf"],
                    "raw_top1_cn": res["raw_top1_cn"],
                    "raw_top1_conf": res["raw_top1_conf"],
                    "post_geo_top1_cn": res["post_geo_top1_cn"],
                    "post_geo_top1_conf": res["post_geo_top1_conf"],
                    "unknown_reason": res["unknown_reason"] or "",
                }
            )

        if i % 20 == 0 or i == len(samples):
            elapsed = time.monotonic() - t0
            print(f"  进度 {i}/{len(samples)} ({elapsed:.0f}s)")

    detail_csv = out_dir / "detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = summarize_rows(rows)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = recommend_configs(summary)
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# 本地物种模型对比评测",
                "",
                f"- GT: `{args.gt_dir}`",
                f"- 样本数: {len(samples)}",
                f"- bird-seg: `{seg_path}` (conf={args.bird_conf})",
                f"- bird_info: `{_BIRD_INFO_PATH}`",
                f"- 配置组合: {len(grid)}",
                "",
                md,
                "",
                "## 汇总表（按准确率降序）",
                "",
                "| model | geo | thr | n | seg率 | 准确率 | 未知率 | 均信度 |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
            + [
                f"| {s['model']} | {s['geo_mode']} | {s['min_accept_conf']} | {s['n']} | "
                f"{s['seg_detect_rate']:.2%} | {s['top1_accuracy']:.2%} | "
                f"{s['unknown_rate']:.2%} | {s['mean_conf_when_labeled']:.3f} |"
                for s in summary[:20]
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print(f"完成。明细: {detail_csv}")
    print(f"汇总:   {summary_path}")
    print(f"说明:   {readme}")
    print("=" * 60)
    if summary:
        top = summary[0]
        print(
            f"最佳组合: {top['config_key']} → 准确率 {top['top1_accuracy']:.1%}, "
            f"未知率 {top['unknown_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
