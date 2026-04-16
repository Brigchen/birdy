#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地物种分类模型自检：加载 ResNet34 + bird_info.json，对单张图运行 predict 并打印 top-k。

用法（在仓库根目录）:
  python test/test_local_species_model.py -i path/to/bird.jpg
  python test/test_local_species_model.py -i path/to/bird.jpg -k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2  # noqa: E402

from detect_bird_and_eye import (  # noqa: E402
    BirdSpeciesClassifier,
    _BIRD_INFO_PATH,
    _SPECIES_MODEL_PATH,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="本地物种模型 BirdSpeciesClassifier 自检")
    ap.add_argument(
        "-i",
        "--image",
        required=True,
        help="测试图片路径（整图或鸟区域裁剪图均可）",
    )
    ap.add_argument("-k", type=int, default=5, help="输出 top-k 个候选（默认 5）")
    ap.add_argument(
        "--model",
        default=_SPECIES_MODEL_PATH,
        help=f"权重路径（默认: {_SPECIES_MODEL_PATH}）",
    )
    ap.add_argument(
        "--bird-info",
        default=_BIRD_INFO_PATH,
        help=f"bird_info.json（默认: {_BIRD_INFO_PATH}）",
    )
    args = ap.parse_args()

    img_path = Path(args.image).expanduser().resolve()
    if not img_path.is_file():
        print(f"[错误] 找不到图片: {img_path}")
        sys.exit(1)

    mp = Path(args.model).expanduser().resolve()
    bp = Path(args.bird_info).expanduser().resolve()
    if not mp.is_file():
        print(f"[错误] 找不到权重: {mp}")
        sys.exit(1)
    if not bp.is_file():
        print(f"[错误] 找不到 bird_info: {bp}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[错误] OpenCV 无法读取图像: {img_path}")
        sys.exit(1)

    with open(bp, encoding="utf-8") as f:
        bird_n = len(json.load(f))

    print("=" * 60)
    print("本地物种模型自检")
    print("=" * 60)
    print(f"图片:     {img_path}")
    print(f"形状:     {img.shape} (H,W,C BGR)")
    print(f"权重:     {mp}")
    print(f"bird_info: {bp} （条目数 {bird_n}）")
    print("=" * 60)

    try:
        clf = BirdSpeciesClassifier(model_path=str(mp), bird_info_path=str(bp))
    except Exception as e:
        print(f"[错误] 分类器加载失败: {e}")
        sys.exit(1)

    if clf.num_classes != bird_n:
        print(
            f"[警告] 模型 fc 输出维数 {clf.num_classes} 与 bird_info 条目数 {bird_n} 不一致，"
            "可能导致 index 与中文名错位。"
        )

    preds = clf.predict(img, top_k=max(1, args.k))

    print(f"\n预测 top-{len(preds)}:")
    print("-" * 60)
    if not preds:
        print("  (空列表：图像为空、或推理异常)")
    else:
        for i, p in enumerate(preds, 1):
            print(
                f"  {i}. index={p['index']:<6} conf={p['confidence']:<8} "
                f"中文={p['chinese_name']} | 英文={p['english_name']} | 学名={p['scientific_name']}"
            )
    print("-" * 60)
    print("若 conf 普遍极低或中文名明显乱套，请检查权重与 bird_info 是否配对、BGR/RGB 预处理是否正常。")
    print("（本脚本与 BirdSpeciesClassifier.predict 使用相同预处理。）")


if __name__ == "__main__":
    main()
