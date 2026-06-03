# -*- coding: utf-8 -*-
"""从 classification 根目录扫描并导出 eBird CSV / 中国观鸟记录中心鸟种导入 Excel。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from .china_bird_record_xls import export_china_bird_record_workbooks
from .birdreport_submit import submit_batch_json_via_relay
from .ebird_checklist import export_ebird_checklist_files
from .scan import scan_classification_tree
from .taxonomy_cn import default_species_csv_path, load_cn_to_en_sci
from .xls_deps import ensure_xls_dependencies


def export_from_classification(
    classification_root: str,
    out_dir: str,
    *,
    write_ebird_csv: bool = True,
    ebird_checklist_sample: Optional[str] = None,
    write_china_bird_record_xls: bool = True,
    china_bird_record_template: Optional[str] = None,
    species_csv: Optional[str] = None,
    ebird_country: str = "CN",
    ebird_state: str = "FJ",
    ebird_protocol: str = "Traveling",
    ebird_duration_min: int = 60,
    ebird_num_observers: int = 1,
    location_name: str = "",
    province_cn: str = "",
    city_cn: str = "",
    count_individuals: bool = True,
    prefer_spatial_gps: bool = False,
    spatial_threshold_km: float = 0.1,
    time_threshold_minutes: float = 30.0,
    individual_time_threshold_minutes: Optional[float] = None,
    merge_single_checklist: bool = True,
    gpx_file_path: Optional[str] = None,
    gpx_file_paths: Optional[Sequence[str]] = None,
    gpx_exif_tz: str = "Asia/Shanghai",
    gpx_track_tz: str = "UTC",
) -> Dict[str, str]:
    """
    扫描 ``classification_root``，在 ``out_dir`` 写入：

    - ``ebird/ebird_checklist_{日期}_{时间}_{坐标}_exp{导出时刻}.csv``（Checklist Format）
    - ``china_bird_record/china_bird_species_{…}.xls``（中国观鸟记录中心两列模版）

    默认整次扫描合并为 **一份** eBird CSV 与 **一份** 记录中心 xls（连日同点一次观鸟）。

    数量：同 checklist 内按空间（0.1 km，精确 GPS/GPX）或时间窗（默认 **120 分钟**，
    定点观鸟时 GPS 微移仍按时间合并）视为同一批个体，每批取只数最多的一张再累加；
    ``count_individuals=False`` 时每物种计 1。导出后请人工核对 Count。

    返回已写入文件的绝对路径键值对。
    """
    if write_china_bird_record_xls:
        ensure_xls_dependencies()
    csv_p = Path(species_csv) if species_csv else default_species_csv_path()
    table = load_cn_to_en_sci(csv_p)  # 含 species_text_aliases 异写映射
    _leaves, buckets = scan_classification_tree(
        classification_root,
        count_individuals=count_individuals,
        prefer_spatial_gps=prefer_spatial_gps,
        spatial_threshold_km=spatial_threshold_km,
        time_threshold_minutes=time_threshold_minutes,
        individual_time_threshold_minutes=individual_time_threshold_minutes,
        merge_single_checklist=merge_single_checklist,
    )
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    if write_ebird_csv:
        ebird_dir = out / "ebird"
        ebird_files = export_ebird_checklist_files(
            buckets,
            table,
            str(ebird_dir),
            sample_path=ebird_checklist_sample,
            country=ebird_country,
            state_province=ebird_state,
            protocol=ebird_protocol,
            duration_min=ebird_duration_min,
            num_observers=ebird_num_observers,
            location_name=location_name,
            province_cn=province_cn,
            city_cn=city_cn,
            gpx_file_path=gpx_file_path,
            gpx_file_paths=gpx_file_paths,
            gpx_exif_tz=gpx_exif_tz,
            gpx_track_tz=gpx_track_tz,
        )
        written.update(ebird_files)
    if write_china_bird_record_xls:
        xls_dir = out / "china_bird_record"
        paths = export_china_bird_record_workbooks(
            buckets,
            str(xls_dir),
            template_path=china_bird_record_template,
            region_code=ebird_state,
        )
        if paths:
            written["china_bird_record_xls"] = paths[0]
            if len(paths) > 1:
                written["china_bird_record_xls_all"] = ";".join(paths)
    return written


def submit_exports(
    *,
    birdreport_batch_json: Optional[str] = None,
    submit_birdreport_relay: bool = False,
    birdreport_relay_url: Optional[str] = None,
    birdreport_bearer_token: Optional[str] = None,
) -> Dict[str, str]:
    """
    可选提交：eBird 无公开「直接上报」API，仅能通过网页导入，此处不实现。

    ``submit_birdreport_relay=True`` 时，将 ``birdreport_batch.json`` POST 到
    环境变量或参数指定的中继 URL（见 ``birdreport_submit.submit_batch_json_via_relay``）。
    """
    out: Dict[str, str] = {}
    if submit_birdreport_relay:
        if not birdreport_batch_json:
            raise ValueError("submit_birdreport_relay 需要 birdreport_batch_json 路径")
        out["birdreport_relay_response"] = submit_batch_json_via_relay(
            birdreport_batch_json,
            url=birdreport_relay_url,
            bearer_token=birdreport_bearer_token,
        )
    return out
