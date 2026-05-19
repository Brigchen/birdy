#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperBirdID birdid2024 与 Birdy EfficientNet 管线对比（同一 GT 集、同一 JIT 权重）

目的：区分
  - 权重/解密是否一致（.enc 解密 vs bird_iden_efficient_b0.pt）
  - 预处理差异（BGR vs RGB、224 vs 256+crop、温度 T=0.6 vs 1.0）
  - 与 ResNet34 基线

用法（仓库根目录）:
  python test/benchmark_superbirdid_efficientnet.py
  python test/benchmark_superbirdid_efficientnet.py --limit 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from benchmark_local_species_models import (  # noqa: E402
    GtSample,
    _pred_matches_gt,
    collect_gt_samples,
    detect_largest_bird_crop,
)

_DEFAULT_GT = r"D:\Birds_Transfer_Station_Level2\classification_20260504-浦口"
_DEFAULT_SEG = _REPO / "models" / "bird-seg.pt"
_SB_ROOT = Path(r"C:\Users\brigc\WorkBuddy\python\SuperbirdID\superbirdid-master")
_SB_ENC = _SB_ROOT / "birdid2024.pt.enc"
_SB_PT = _SB_ROOT / "weights" / "birdid2024.pt"
_BIRDY_PT = _REPO / "models" / "bird_iden_efficient_b0.pt"
_SB_BIRDINFO = _SB_ROOT / "birdinfo.json"
_BIRDY_BIRDINFO = _REPO / "models" / "bird_info.json"
_SB_PASSWORD = "SuperBirdID_2024_AI_Model_Encryption_Key_v1"

# SuperBirdID BGR ImageNet
_BGR_MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
_BGR_STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)
_TEMPERATURE = 0.6


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def decrypt_superbirdid_enc(enc_path: Path, password: str) -> bytes:
    """与 SuperBirdId.py decrypt_model 一致。"""
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    data = enc_path.read_bytes()
    salt, iv, ciphertext = data[:16], data[16:32], data[32:]
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    key = kdf.derive(password.encode())
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    return padded[: -padded[-1]]


@dataclass(frozen=True)
class Pipeline:
    key: str
    desc: str


PIPELINES = [
    Pipeline("sb_api_224_bgr_t06", "SuperBirdID API: 224 LANCZOS, BGR norm, T=0.6"),
    Pipeline("sb_gui_256crop_bgr_t06", "SuperBirdID GUI: 256→center224, BGR, T=0.6"),
    Pipeline("birdy_rgb_256crop_t10", "Birdy 当前: RGB 256→CenterCrop224, ImageNet RGB, T=1.0"),
    Pipeline("birdy_bgr_256crop_t06", "修正: 256→center224 BGR norm, T=0.6"),
    Pipeline("birdy_bgr_224_t06", "修正: 直接 224 BGR norm, T=0.6"),
]


def load_birdinfo(path: Path) -> List[List]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tensor_from_bgr_norm(bgr: np.ndarray) -> torch.Tensor:
    arr = bgr.astype(np.float32) / 255.0
    arr = (arr - _BGR_MEAN) / _BGR_STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def _tensor_from_rgb_imagenet(rgb: np.ndarray) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


def preprocess(crop_bgr: np.ndarray, pipeline: str) -> torch.Tensor:
    if pipeline in ("sb_api_224_bgr_t06", "birdy_bgr_224_t06"):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((224, 224), Image.LANCZOS)
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return _tensor_from_bgr_norm(bgr)

    if pipeline in ("sb_gui_256crop_bgr_t06", "birdy_bgr_256crop_t06"):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((256, 256), Image.LANCZOS)
        cropped = pil.crop((16, 16, 240, 240))
        bgr = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        return _tensor_from_bgr_norm(bgr)

    if pipeline == "birdy_rgb_256crop_t10":
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((256, 256), Image.LANCZOS)
        cropped = pil.crop((16, 16, 240, 240))
        return _tensor_from_rgb_imagenet(np.array(cropped))

    raise ValueError(pipeline)


def infer_top1(
    model: torch.jit.ScriptModule,
    crop_bgr: np.ndarray,
    pipeline: str,
    birdinfo: List[List],
) -> Tuple[int, float, str, float]:
    """返回 (class_idx, conf, chinese_name, entropy_top100)。"""
    tensor = preprocess(crop_bgr, pipeline)
    with torch.no_grad():
        logits = model(tensor)
        if logits.dim() == 2:
            logits = logits[0]

    if pipeline.endswith("_t06"):
        probs = F.softmax(logits / _TEMPERATURE, dim=0)
    else:
        probs = F.softmax(logits, dim=0)

    conf, idx = torch.max(probs, dim=0)
    idx_i = int(idx.item())
    conf_f = float(conf.item())

    info = birdinfo[idx_i] if 0 <= idx_i < len(birdinfo) else []
    cn = (info[0] if info else "").strip()

    k = min(100, probs.numel())
    top_p, _ = torch.topk(probs, k)
    ent = float(-(top_p * torch.log(top_p + 1e-12)).sum().item())

    return idx_i, conf_f, cn, ent


def verify_weights() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["birdy_pt_sha256"] = _sha256(_BIRDY_PT)
    out["sb_pt_sha256"] = _sha256(_SB_PT)
    out["pt_files_identical"] = out["birdy_pt_sha256"] == out["sb_pt_sha256"]

    if _SB_ENC.is_file():
        plain = decrypt_superbirdid_enc(_SB_ENC, _SB_PASSWORD)
        out["enc_decrypt_sha256"] = hashlib.sha256(plain).hexdigest()
        out["enc_matches_birdy_pt"] = (
            out["enc_decrypt_sha256"] == out["birdy_pt_sha256"]
        )
        out["enc_size"] = _SB_ENC.stat().st_size
        out["decrypted_size"] = len(plain)
    else:
        out["enc_missing"] = str(_SB_ENC)

    sb_info = load_birdinfo(_SB_BIRDINFO)
    by_info = load_birdinfo(_BIRDY_BIRDINFO)
    sci_mis = 0
    for i, (a, b) in enumerate(zip(by_info, sb_info)):
        if (a[2] if len(a) > 2 else "") != (b[2] if len(b) > 2 else ""):
            sci_mis += 1
    out["birdinfo_len"] = len(sb_info)
    out["index_sci_mismatch"] = sci_mis
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default=_DEFAULT_GT)
    ap.add_argument("--bird-seg", default=str(_DEFAULT_SEG))
    ap.add_argument("--bird-conf", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    print("=== 权重与标签校验 ===")
    verify = verify_weights()
    for k, v in verify.items():
        print(f"  {k}: {v}")
    if not verify.get("pt_files_identical"):
        print("[警告] bird_iden_efficient_b0.pt 与 SuperBirdID 解密权重不一致")
    if verify.get("enc_matches_birdy_pt") is False:
        print("[警告] .enc 解密结果与 birdy 权重不一致 — 可能解密损坏")

    print("\n加载 JIT（bird_iden_efficient_b0.pt）…")
    model = torch.jit.load(str(_BIRDY_PT), map_location="cpu")
    model.eval()
    n_cls = model.state_dict()["head.fc.weight"].shape[0]
    print(f"  类别数: {n_cls}")

    sb_info = load_birdinfo(_SB_BIRDINFO)
    by_info = load_birdinfo(_BIRDY_BIRDINFO)

    samples = collect_gt_samples(args.gt_dir, limit=args.limit)
    if not samples:
        print(f"[错误] 无 GT 样本: {args.gt_dir}")
        sys.exit(1)
    print(f"\nGT 样本: {len(samples)} 张")

    yolo = YOLO(str(Path(args.bird_seg).resolve()))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else _REPO / "reports" / f"superbirdid_eff_benchmark_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    for i, sample in enumerate(samples, 1):
        img = cv2.imread(sample.path)
        if img is None:
            continue
        crop, n_bird, seg_conf = detect_largest_bird_crop(
            yolo, img, bird_conf=args.bird_conf
        )
        seg_ok = crop is not None and crop.size > 0
        if not seg_ok:
            crop = img

        for pl in PIPELINES:
            for label_src, birdinfo in (
                ("sb_birdinfo", sb_info),
                ("birdy_bird_info", by_info),
            ):
                if label_src == "birdy_bird_info" and pl.key != "birdy_rgb_256crop_t10":
                    continue  # 标签相同，只比一次
                idx, conf, cn, ent = infer_top1(
                    model, crop, pl.key, birdinfo
                )
                correct = _pred_matches_gt(cn, sample.accept_names)
                rows.append(
                    {
                        "image": sample.path,
                        "gt_species": sample.species_cn,
                        "pipeline": pl.key,
                        "pipeline_desc": pl.desc,
                        "label_src": label_src,
                        "seg_ok": seg_ok,
                        "pred_index": idx,
                        "pred_cn": cn,
                        "pred_conf": round(conf, 6),
                        "entropy_top100": round(ent, 4),
                        "correct": correct,
                    }
                )

        if i % 25 == 0 or i == len(samples):
            print(f"  进度 {i}/{len(samples)}")

    # 汇总
    summary: List[Dict[str, Any]] = []
    buckets: DefaultDict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        buckets[r["pipeline"]].append(r)

    for key, items in sorted(buckets.items()):
        n = len(items)
        correct = sum(1 for x in items if x["correct"])
        confs = [float(x["pred_conf"]) for x in items]
        ents = [float(x["entropy_top100"]) for x in items]
        summary.append(
            {
                "pipeline": key,
                "desc": items[0]["pipeline_desc"],
                "n": n,
                "top1_accuracy": round(correct / n, 4) if n else 0,
                "mean_conf": round(sum(confs) / len(confs), 4) if confs else 0,
                "median_conf": round(float(np.median(confs)), 4) if confs else 0,
                "mean_entropy_top100": round(sum(ents) / len(ents), 4) if ents else 0,
                "conf_ge_05": sum(1 for c in confs if c >= 0.5),
                "conf_ge_08": sum(1 for c in confs if c >= 0.8),
            }
        )
    summary.sort(key=lambda x: x["top1_accuracy"], reverse=True)

    detail_path = out_dir / "detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"verify": verify, "summary": summary, "n_samples": len(samples)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# SuperBirdID EfficientNet 对比报告",
        "",
        "## 权重与解密",
        "",
        f"- `bird_iden_efficient_b0.pt` 与 `weights/birdid2024.pt` 字节一致: **{verify.get('pt_files_identical')}**",
        f"- `.enc` 解密 SHA256 与 birdy 权重一致: **{verify.get('enc_matches_birdy_pt')}**",
        f"- `birdinfo.json` 与 `bird_info.json` 学名索引错位: **{verify.get('index_sci_mismatch')}** / {verify.get('birdinfo_len')}",
        "",
        "## 各推理管线 top1 准确率（无地理规则，bird-seg 裁剪）",
        "",
        "| 管线 | 准确率 | 平均 conf | 中位 conf | conf≥0.5 | conf≥0.8 | 平均熵( top100 ) |",
        "|------|--------|-----------|-----------|----------|----------|------------------|",
    ]
    for s in summary:
        lines.append(
            f"| `{s['pipeline']}` | {s['top1_accuracy']:.1%} | {s['mean_conf']:.3f} | "
            f"{s['median_conf']:.3f} | {s['conf_ge_05']} | {s['conf_ge_08']} | "
            f"{s['mean_entropy_top100']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 结论提示",
            "",
            "- 若 **sb_* 管线准确率高** 而 **birdy_rgb** 低 → 主因是预处理/温度，不是权重或解密。",
            "- 若 **所有管线准确率都低** → 权重与 GT 标注体系不匹配，或 bird-seg 裁剪与 SuperBirdID 训练分布差异大。",
            "- 若 **enc 与 pt 不一致** → 检查解密或复制流程。",
            "",
        ]
    )
    readme = "\n".join(lines)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    elapsed = time.monotonic() - t0
    print(f"\n完成，耗时 {elapsed:.1f}s")
    print(f"报告: {out_dir}")
    print("\n准确率排行:")
    for s in summary:
        print(
            f"  {s['pipeline']:28s}  acc={s['top1_accuracy']:.1%}  "
            f"med_conf={s['median_conf']:.3f}  ge05={s['conf_ge_05']}"
        )


if __name__ == "__main__":
    main()
