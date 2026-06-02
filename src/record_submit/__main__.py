# -*- coding: utf-8 -*-
"""python -m record_submit（需在 src 上 PYTHONPATH 或从项目以包方式运行）。"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="从 classification 目录导出 eBird Checklist .csv 与中国观鸟记录中心鸟种导入 Excel。"
    )
    p.add_argument("classification_root", help="Birdy 分类归档根目录")
    p.add_argument(
        "-o",
        "--out-dir",
        default="record_export_out",
        help="输出目录（默认 record_export_out）",
    )
    p.add_argument(
        "--species-csv",
        default=None,
        help="bird_species_list.csv 路径（默认项目 data/species/…）",
    )
    p.add_argument("--no-ebird", action="store_true", help="不写出 eBird Checklist .csv")
    p.add_argument(
        "--no-china-bird-record",
        action="store_true",
        help="不写出观鸟记录中心鸟种导入 .xls",
    )
    p.add_argument(
        "--no-birdreport",
        action="store_true",
        help="同 --no-china-bird-record（兼容旧参数）",
    )
    p.add_argument("--ebird-country", default="CN")
    p.add_argument("--ebird-state", default="FJ", help="省/州 1–3 字符，如 FJ（福建）")
    p.add_argument("--ebird-protocol", default="Traveling")
    p.add_argument("--ebird-duration-min", type=int, default=60)
    p.add_argument(
        "--location-name",
        default="",
        help="观鸟地点中文地址，导出 eBird 时转为拼音 Location Name",
    )
    p.add_argument(
        "--no-count-individuals",
        action="store_true",
        help="不按只数累加，每个 checklist 内每物种计 1",
    )
    p.add_argument(
        "--prefer-spatial-gps",
        action="store_true",
        help="有 GPS 时优先按 0.1 km 空间聚类（如已 GPX 写 EXIF）",
    )
    p.add_argument(
        "--spatial-km",
        type=float,
        default=0.1,
        help="空间聚类阈值（公里，默认 0.1）",
    )
    p.add_argument(
        "--time-minutes",
        type=float,
        default=30.0,
        help="时间聚类阈值（分钟，默认 30）",
    )
    p.add_argument(
        "--submit-birdreport-relay",
        action="store_true",
        help="导出后 POST birdreport_batch.json 到 BIRDREPORT_SUBMIT_URL",
    )
    args = p.parse_args(argv)
    from .export import export_from_classification, submit_exports

    written = export_from_classification(
        args.classification_root,
        args.out_dir,
        write_ebird_csv=not args.no_ebird,
        write_china_bird_record_xls=not (
            args.no_china_bird_record or args.no_birdreport
        ),
        species_csv=args.species_csv,
        ebird_country=args.ebird_country,
        ebird_state=args.ebird_state,
        ebird_protocol=args.ebird_protocol,
        ebird_duration_min=args.ebird_duration_min,
        location_name=args.location_name,
        count_individuals=not args.no_count_individuals,
        prefer_spatial_gps=args.prefer_spatial_gps,
        spatial_threshold_km=args.spatial_km,
        time_threshold_minutes=args.time_minutes,
    )
    for k, v in written.items():
        print(f"{k}: {v}")
    if args.submit_birdreport_relay:
        br = written.get("birdreport_batch_json")
        if not br:
            print("未生成 birdreport_batch_json，跳过中继提交。", file=sys.stderr)
            return 1
        resp = submit_exports(
            birdreport_batch_json=br,
            submit_birdreport_relay=True,
        )
        print("birdreport_relay_response:", resp.get("birdreport_relay_response", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
