#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鸟图智慧仓储 (Birdy) - 图形用户界面
支持：参数配置、进度显示、文件夹选择、处理监控

作者: brigchen@gmail.com
版权说明: 基于开源协议，仅限爱好者、公益、科研等非盈利用途，请勿用于商业用途
"""

import sys
import os
import json
import time
import shutil
import threading
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar, QFileDialog,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
        QGroupBox, QFormLayout, QMessageBox, QTableWidget, QTableWidgetItem,
        QDialog, QDialogButtonBox, QRadioButton, QButtonGroup, QScrollArea, QFrame,
        QCompleter,
        QSizePolicy, QSlider, QShortcut, QProgressDialog, QListWidget,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QUrl, QObject
    from PyQt5.QtGui import QColor, QTextCursor, QIcon, QPalette, QDesktopServices, QKeySequence
except ImportError:
    print("错误: 未安装PyQt5。请运行: pip install PyQt5")
    sys.exit(1)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from burst_grouping import process_folder, get_kept_images, screened_paths_for_kept_images
from geo_encoder import batch_write_gps_exif, geocode_location
from detect_bird_and_eye import (
    BirdAndEyeDetector,
    LOCAL_SPECIES_MODEL_EFFICIENTNET,
    LOCAL_SPECIES_MODEL_RESNET34,
    SPECIES_GEO_MODE_AUTO,
    SPECIES_GEO_MODE_CHINA,
    SPECIES_GEO_MODE_NONE,
    SPECIES_GEO_MODE_PROVINCE,
    archive_identified_crop_file,
    group_crop_records_by_source,
    normalize_local_species_model,
    normalize_species_geo_mode,
    save_instance_crops_to_staging,
    save_union_crop_for_birds,
)
from api_config_defaults import ensure_doubao_api_config_file, ensure_amap_api_config_file
from record_submit import export_from_classification
from record_submit.record_portals import (
    CHINA_BIRD_RECORD_HOME_URL,
    EBIRD_IMPORT_URL,
)
from gpx_track import (
    batch_write_gps_from_gpx,
    generate_track_maps,
    show_track_map_preview,
    merge_gpx_files,
    resolve_gpx_path_list,
)
from gpx_track.track_map import iter_skipped_photo_log_lines
from gpx_track.timezone_util import (
    DEFAULT_EXIF_TZ,
    DEFAULT_GPX_TZ,
    normalize_tz_name,
    read_combo_timezone,
    set_combo_timezone,
    timezone_combo_entries,
)
from watermark_generator import (
    WatermarkOptions,
    choose_default_watermark_source,
    collect_images_recursive,
    generate_watermarks,
    render_watermark_for_image,
)
from image_clean import ImageCleanOptions, clean_bird_images, clean_image_list
from image_io import all_supported_extensions, file_filter_all_images, imread_bgr
from dual_format import extensions_for_dual_mode
from flow_eta import (
    FlowEtaEstimator,
    build_eta_phase_estimates,
)
from burst_webp_dialog import open_burst_webp_dialog
from video_stabilize_dialog import open_video_stabilize_dialog


def _open_local_file(path: str) -> None:
    """在资源管理器/系统默认编辑器中打开本地文件（跨平台）。"""
    path = os.path.abspath(path)
    if sys.platform == "win32":
        os.startfile(path)
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class _WatermarkRenderWorker(QObject):
    """水印渲染后台 worker：在独立 QThread 中执行 render_watermark_for_image。

    避免在预览时阻塞 Qt 主事件循环（启用 AI 降噪/锐化时单张可达数秒）。
    通过 pyqtSignal 把 PIL.Image（或 None）回传主线程。
    """

    finished = pyqtSignal(object)  # PIL.Image | None
    failed = pyqtSignal(str)

    def __init__(self, image_path: str, source_folder: str, opts: "WatermarkOptions"):
        super().__init__()
        self._image_path = image_path
        self._source_folder = source_folder
        self._opts = opts

    @pyqtSlot()
    def run(self) -> None:
        try:
            out = render_watermark_for_image(
                image_path=self._image_path,
                source_folder=self._source_folder,
                options=self._opts,
                prefer_folder_name_as_species=True,
            )
            self.finished.emit(out)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit(str(e))



def _count_images_for_eta(folder: str, dual_format_mode: str = "off") -> int:
    """与连拍/物种步骤同级的图片数量预估（递归）。"""
    if not folder or not os.path.isdir(folder):
        return 0
    return len(_collect_image_paths_under(folder, dual_format_mode))


def _apply_gui_flow_policy(config: Dict) -> None:
    """GUI：不生成连拍/物种 HTML 报告；物种步骤始终裁剪归档。"""
    config["generate_burst_report"] = False
    config["generate_species_report"] = False
    if config.get("enable_species_detection", True):
        config["enable_crop"] = True


def _record_export_dirs_from_config(config: Dict) -> Tuple[str, str]:
    """从配置解析分类归档目录与观鸟记录导出目录。"""
    classification = (
        config.get("record_export_classification_folder", "") or ""
    ).strip()
    if not classification:
        classification = (config.get("crop_output_folder", "") or "").strip()
    out = (config.get("record_export_output_folder", "") or "").strip()
    if not out:
        root = (config.get("output_root_folder", "") or "").strip()
        reports = (config.get("reports_output_folder", "") or "").strip()
        if root:
            out = os.path.join(root, "reports", "record_export")
        elif reports:
            out = os.path.join(reports, "record_export")
        else:
            out = os.path.join(
                os.path.dirname(classification or "."), "record_export"
            )
    return classification, out


def _config_gpx_paths(config: Dict) -> List[str]:
    return resolve_gpx_path_list(
        config.get("gpx_file_path"),
        config.get("gpx_file_paths"),
    )


def _record_export_kwargs(config: Dict) -> Dict:
    """观鸟记录导出：批次去重、GPX 空间聚类与 GPX 里程参数。"""
    gpx_paths = _config_gpx_paths(config)
    prefer = config.get("record_export_prefer_spatial_gps")
    if prefer is None:
        prefer = config.get("gps_write_mode") == "gpx" and bool(gpx_paths)
    else:
        prefer = bool(prefer)
    out: Dict = {
        "count_individuals": bool(
            config.get("record_export_count_individuals", True)
        ),
        "prefer_spatial_gps": prefer,
        "spatial_threshold_km": float(
            config.get("record_export_spatial_km", 0.1) or 0.1
        ),
        "time_threshold_minutes": float(
            config.get("record_export_location_time_minutes", 30) or 30
        ),
        "individual_time_threshold_minutes": float(
            config.get("record_export_time_minutes", 60) or 60
        ),
        "merge_single_checklist": True,
    }
    out["location_name"] = (config.get("location_name") or "").strip()
    out["province_cn"] = (config.get("province") or "").strip()
    out["city_cn"] = (config.get("city") or "").strip()
    if gpx_paths:
        out["gpx_file_path"] = gpx_paths[0]
        out["gpx_file_paths"] = gpx_paths
        out["gpx_exif_tz"] = normalize_tz_name(
            config.get("gpx_match_exif_tz")
            or config.get("track_map_exif_tz")
            or DEFAULT_EXIF_TZ
        )
        out["gpx_track_tz"] = normalize_tz_name(
            config.get("gpx_match_gpx_tz")
            or config.get("track_map_gpx_tz")
            or DEFAULT_GPX_TZ
        )
    return out


def _collect_image_paths_under(root: str, dual_format_mode: str = "off") -> List[str]:
    """递归收集 root 下图片路径（双格式模式下可仅 JPG）。"""
    import os

    if not root or not os.path.isdir(root):
        return []
    exts = extensions_for_dual_mode(dual_format_mode)
    out: List[str] = []
    for walk_root, _dirs, files in os.walk(root):
        for file in files:
            suf = Path(file).suffix.lower()
            if suf in exts:
                p = os.path.join(walk_root, file)
                if p not in out:
                    out.append(p)
    return out


def _session_slug_from_image_folder(image_folder: str) -> str:
    """
    由「原始相片目录」最后一段文件夹名生成会话标签，用于
    screened_<slug> / classification_<slug> 子目录名。
    """
    p = (image_folder or "").strip()
    if not p:
        return "session"
    base = os.path.basename(os.path.normpath(p))
    if not base or base in (".", ".."):
        return "session"
    bad = '<>:"/\\|?*'
    parts = []
    for ch in base:
        if ch in bad or ord(ch) < 32:
            parts.append("_")
        elif ch.isspace():
            parts.append("_")
        else:
            parts.append(ch)
    s = "".join(parts).strip("._") or "session"
    return s


def _reports_dir_from_config(config: Dict) -> str:
    """HTML 报告目录：若配置了 reports_output_folder 则用之，否则 output_folder/reports。"""
    r = (config.get("reports_output_folder") or "").strip()
    if r:
        return r
    return os.path.join((config.get("output_folder") or "./outputs").strip(), "reports")


def _config_paths_snapshot(
    *,
    output_root_folder: str = "",
    image_folder: str = "",
    output_folder: str = "",
    crop_output_folder: str = "",
) -> Dict[str, str]:
    """
    由当前 UI/配置推导会话路径（与 _sync_config_from_ui 一致）。
    启用「输出根目录」时，分类/Screened 路径不在 legacy 输入框中显示，须由此函数解析。
    """
    root = (output_root_folder or "").strip()
    img = (image_folder or "").strip()
    if root:
        slug = _session_slug_from_image_folder(img)
        screened_root = os.path.join(root, f"screened_{slug}")
        classification = os.path.join(root, f"classification_{slug}")
        reports = os.path.join(root, "reports")
        return {
            "output_folder": screened_root,
            "crop_output_folder": classification,
            "reports_output_folder": reports,
            "screened_images": os.path.join(screened_root, "Screened_images"),
        }
    out = (output_folder or "").strip()
    crop = (crop_output_folder or "").strip()
    reports = os.path.join(out, "reports") if out else ""
    return {
        "output_folder": out,
        "crop_output_folder": crop,
        "reports_output_folder": reports,
        "screened_images": os.path.join(out, "Screened_images") if out else "",
    }


def _track_map_location_from_config(config: Dict) -> str:
    """轨迹图标题地点：与水印「人工地点」一致，优先 wm_location_text。"""
    manual = (config.get("wm_location_text") or "").strip()
    if manual:
        return manual
    return (config.get("location_name") or "").strip()


def _flow_enabled_flags(config: Dict) -> Dict[str, bool]:
    """当前主流程各阶段是否勾选「加入主流程」。"""
    return {
        "burst": bool(config.get("enable_burst_detection", True)),
        "gps": bool(config.get("enable_gps_write")),
        "species": bool(config.get("enable_species_detection", True)),
        "watermark": bool(config.get("enable_watermark_generation", False)),
        "record_export": bool(config.get("enable_record_export_auto", False)),
        "track_map": bool(config.get("enable_track_map_auto", False)),
    }


def _emit_burst_skipped_status(emit, config: Dict) -> None:
    """连拍未加入主流程时，按实际后续步骤说明数据来源目录。"""
    flags = _flow_enabled_flags(config)
    if flags["burst"]:
        return
    uses: List[str] = []
    if flags["species"]:
        screened = os.path.join(
            (config.get("output_folder") or "").strip(), "Screened_images"
        )
        uses.append(f"物种识别 → Screened_images（{screened}）")
    class_dir = (config.get("crop_output_folder") or "").strip()
    if flags["track_map"] and flags["record_export"] and class_dir:
        uses.append(f"轨迹图 / 观鸟记录导出 → 分类归档（{class_dir}）")
    elif flags["track_map"] and class_dir:
        uses.append(f"轨迹图 → 分类归档（{class_dir}）")
    elif flags["record_export"] and class_dir:
        uses.append(f"观鸟记录导出 → 分类归档（{class_dir}）")
    if flags["watermark"]:
        uses.append("水印生成（见水印卡片输入/输出目录）")
    if uses:
        emit(
            "已跳过连拍检测；"
            + "；".join(uses)
        )
    else:
        emit("已跳过连拍检测")


def _build_processing_stats(
    results: Dict, config: Dict
) -> List[Tuple[str, str]]:
    """仅汇总本次实际执行步骤的统计项。"""
    stats: List[Tuple[str, str]] = []
    if "total_images" in results:
        stats.extend(
            [
                ("连拍·总图片", results.get("total_images", 0)),
                ("连拍·保留", results.get("kept_images", 0)),
                ("连拍·丢弃", results.get("discarded_images", 0)),
            ]
        )
    if "gps_written" in results:
        stats.append(("GPS 已写入", results.get("gps_written", 0)))
    if "crop_result" in results:
        cr = results["crop_result"]
        stats.append(("裁剪归档文件", cr.get("total_crops", 0)))
        stats.append(
            ("物种识别耗时", f"{cr.get('processing_time', 0):.2f} 秒")
        )
    if "record_export" in results:
        written = results["record_export"]
        stats.append(("观鸟记录导出", f"{len(written)} 个文件"))
        for k, p in written.items():
            stats.append((f"  · {k}", os.path.basename(str(p))))
    if "track_map" in results:
        written = results["track_map"]
        stats.append(("轨迹图输出", f"{len(written)} 个文件"))
        for k, p in written.items():
            stats.append((f"  · {k}", os.path.basename(str(p))))
    if "watermark_result" in results:
        wm = results["watermark_result"]
        stats.append(("水印·总数", wm.get("total", 0)))
        stats.append(("水印·成功", wm.get("ok", 0)))
        stats.append(("水印·失败", wm.get("fail", 0)))
    if not stats:
        stats.append(("本次执行", "无统计项（请检查是否勾选了主流程步骤）"))
    return stats


def _build_output_path_logs(results: Dict, config: Dict) -> List[str]:
    """完成时仅列出本次步骤相关的输出路径。"""
    lines: List[str] = []
    if "total_images" in results:
        lines.append(f"📁 连拍输出: {config.get('output_folder', '')}")
    if "crop_result" in results:
        lines.append(f"📁 分类归档: {config.get('crop_output_folder', '')}")
    if "track_map" in results:
        class_dir = (config.get("crop_output_folder") or "").strip()
        if class_dir:
            lines.append(f"🗺 轨迹图鸟图源: {class_dir}")
        lines.append(f"🗺 轨迹图保存: {_reports_dir_from_config(config)}")
    if "record_export" in results:
        _, out_dir = _record_export_dirs_from_config(config)
        lines.append(f"📤 观鸟记录导出: {out_dir}")
    if "watermark_result" in results:
        out_wm = (config.get("watermark_output_folder") or "").strip()
        if out_wm:
            lines.append(f"🖼 水印输出: {out_wm}")
    return lines


def _main_flow_write_gps(config: Dict, screened_dir: str) -> Tuple[int, str]:
    """
    主流程 GPS 写入（二选一）。
    返回 (写入张数, 模式说明)。
    """
    mode = config.get("gps_write_mode", "fixed")
    if mode == "gpx":
        gpx_paths = _config_gpx_paths(config)
        if not gpx_paths:
            raise FileNotFoundError(
                "主流程已选「GPX 按拍摄时间」，但未配置有效 GPX 文件。"
            )
        stats = batch_write_gps_from_gpx(
            screened_dir,
            gpx_paths=gpx_paths,
            exif_tz=normalize_tz_name(config.get("gpx_match_exif_tz", DEFAULT_EXIF_TZ)),
            gpx_tz=normalize_tz_name(config.get("gpx_match_gpx_tz", DEFAULT_GPX_TZ)),
        )
        n = int(stats.get("written", 0))
        detail = (
            f"GPX 时间匹配：JPEG {stats.get('total', 0)} 张，"
            f"匹配 {stats.get('matched', 0)} 张，写入 {n} 张"
        )
        return n, detail
    gps_count = batch_write_gps_exif(
        image_folder=screened_dir,
        latitude=config["gps_latitude"],
        longitude=config["gps_longitude"],
        altitude=config.get("gps_altitude", 0),
    )
    return int(gps_count), "指定地点统一经纬度"


def _run_species_crop_clean_identify(
    worker: "WorkerThread",
    detector: BirdAndEyeDetector,
    image_files: List[str],
    output_root: str,
    config: Dict,
    start_time: float,
    emit_phase_progress,
    results: Dict,
    manual_province: Optional[str],
    manual_city: Optional[str],
) -> None:
    """大图检鸟并切割 → 清洗每张切割图 → 再对留下的切割图认种归档。"""
    staging_dir = os.path.join(output_root, "_crop_staging")
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)
    os.makedirs(staging_dir, exist_ok=True)

    n_src = len(image_files)
    worker.eta_checkpoint.emit({"kind": "species_begin", "n": n_src})
    records: List[Dict[str, Any]] = []

    for idx, image_file in enumerate(image_files):
        if not worker.is_running:
            break
        worker.status_updated.emit(
            f"切割鸟体: {os.path.basename(image_file)} ({idx + 1}/{n_src})"
        )
        try:
            _vis, detection_results = detector.detect(
                image_file,
                manual_province=manual_province,
                manual_city=manual_city,
                skip_species=True,
            )
            birds = detection_results.get("birds") or []
            orig_img = detection_results.get("original_image")
            if orig_img is None:
                orig_img = detector.load_image(image_file)
            recs = save_instance_crops_to_staging(
                orig_img,
                birds,
                staging_dir,
                image_file,
                province=detection_results.get("province"),
                city=detection_results.get("city"),
            )
            records.extend(recs)
        except Exception as e:
            worker.status_updated.emit(
                f"⚠ 切割失败 {os.path.basename(image_file)}: {e}"
            )
        worker.eta_checkpoint.emit(
            {
                "kind": "species_tick",
                "done": idx + 1,
                "total": max(1, n_src),
            }
        )
        emit_phase_progress("species", idx + 1, max(1, n_src * 3))

    crop_paths = [r["path"] for r in records if os.path.isfile(r.get("path") or "")]
    worker.status_updated.emit(
        f"大图切割完成：{len(crop_paths)} 张切割图，开始清洗…"
    )

    def _clean_prog(d: Dict) -> None:
        if d.get("kind") == "tick":
            emit_phase_progress(
                "species",
                n_src + int(d.get("done", 0)),
                max(1, n_src * 3),
            )

    clean_opts = ImageCleanOptions(
        remove_no_bird=bool(config.get("image_clean_remove_no_bird", True)),
        remove_blurry=bool(config.get("image_clean_remove_blurry", True)),
        dedupe=bool(config.get("image_clean_dedupe", True)),
        min_clarity=float(config.get("image_clean_min_clarity", 35)),
        dup_similarity=float(config.get("image_clean_dup_similarity", 92)),
        use_full_frame_for_clarity=True,
    )
    clean_res = clean_image_list(
        crop_paths,
        clean_opts,
        progress_callback=_clean_prog,
        should_cancel=lambda: not worker.is_running,
    )
    cd = clean_res.as_dict()
    results["image_clean_result"] = cd
    worker.status_updated.emit(
        "切割图清洗完成："
        f"保留 {cd['kept']}/{cd['total']}，"
        f"未检出鸟体 {cd['removed_no_bird']}，"
        f"模糊 {cd['removed_blurry']}，"
        f"重复 {cd['removed_duplicate']}"
    )

    records = [r for r in records if os.path.isfile(r.get("path") or "")]
    if not records:
        worker.status_updated.emit("⚠ 清洗后无切割图，已跳过物种识别。")
        results["crop_result"] = {
            "total_crops": 0,
            "processing_time": time.time() - start_time,
        }
        shutil.rmtree(staging_dir, ignore_errors=True)
        return

    n_id = len(records)
    worker.eta_checkpoint.emit({"kind": "species_begin", "n": n_id})
    archive_counter = {"n": 0}
    total_crops = 0
    kept_for_union: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        if not worker.is_running:
            break
        crop_path = rec["path"]
        worker.status_updated.emit(
            f"识别切割图: {os.path.basename(crop_path)} ({idx + 1}/{n_id})"
        )
        try:
            crop_bgr = imread_bgr(crop_path)
            if crop_bgr is None:
                raise ValueError("无法读取切割图")
            identified = detector.classify_bird_crop(
                crop_bgr,
                rec.get("province") or manual_province,
                rec.get("city") or manual_city,
                log_index=idx + 1,
            )
            rec.update(identified)
            saved = archive_identified_crop_file(
                crop_path,
                rec,
                output_root,
                rec.get("source_path") or "",
                province=rec.get("province"),
                city=rec.get("city"),
                counter=archive_counter,
                inst_i=int(rec.get("inst") or 1),
            )
            if saved:
                total_crops += 1
                kept_for_union.append(rec)
        except Exception as e:
            worker.status_updated.emit(
                f"⚠ 识别失败 {os.path.basename(crop_path)}: {e}"
            )
        worker.eta_checkpoint.emit(
            {
                "kind": "species_tick",
                "done": idx + 1,
                "total": max(1, n_id),
            }
        )
        emit_phase_progress("species", n_src * 2 + idx + 1, max(1, n_src * 3))

    for source, recs in group_crop_records_by_source(kept_for_union).items():
        if len(recs) < 2:
            continue
        orig = None
        if source and os.path.isfile(source):
            try:
                orig = detector.load_image(source)
            except Exception:
                orig = None
        if orig is None:
            continue
        birds = [
            {
                "bbox": r.get("bbox") or [0, 0, 1, 1],
                "species": r.get("species") or [],
                "classification": r.get("classification") or {},
            }
            for r in recs
        ]
        union = save_union_crop_for_birds(
            orig,
            birds,
            output_root,
            source,
            min_species_accept_confidence=detector.min_species_accept_confidence,
            counter=archive_counter,
        )
        if union:
            total_crops += 1

    shutil.rmtree(staging_dir, ignore_errors=True)
    processing_time = time.time() - start_time
    worker.status_updated.emit(
        f"✓ 已输出 {total_crops} 个裁剪归档文件，耗时 {processing_time:.2f} 秒"
    )
    results["crop_result"] = {
        "total_crops": total_crops,
        "species_method": detector.get_species_method(),
        "processing_time": processing_time,
    }


class WorkerThread(QThread):
    """后台工作线程 - 处理图片分析"""
    
    # 信号定义
    progress_updated = pyqtSignal(int)  # 进度百分比
    status_updated = pyqtSignal(str)    # 状态信息
    error_occurred = pyqtSignal(str)    # 错误信息
    finished = pyqtSignal(dict)         # 完成，返回结果统计
    eta_checkpoint = pyqtSignal(dict)   # 剩余时间模型：阶段起止、物种逐张进度
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.is_running = True
    
    def run(self):
        """执行主要处理流程"""
        try:
            import os
            config = self.config
            _apply_gui_flow_policy(config)
            burst_on = config.get("enable_burst_detection", True)
            do_species = config.get("enable_species_detection", True)
            flow_flags = _flow_enabled_flags(config)
            total_steps = 0
            if config.get("enable_gps_write"):
                total_steps += 1
            if burst_on:
                total_steps += 1
            if do_species:
                total_steps += 1
            if config.get("enable_watermark_generation", False):
                total_steps += 1
            if config.get("enable_record_export_auto", False):
                total_steps += 1
            if config.get("enable_track_map_auto", False):
                total_steps += 1
            if total_steps < 1:
                total_steps = 1
            current_step = 0
            results: Dict[str, Any] = {"_flow_flags": flow_flags}
            burst_filter_applied = False
            phase_weights: Dict[str, float] = {
                "gps": 0.08,
                "burst": 0.47,
                "species": 0.35,
                "watermark": 0.10,
                "record_export": 0.06,
                "track_map": 0.08,
            }
            enabled_phases: List[str] = []
            if burst_on:
                enabled_phases.append("burst")
            if config.get("enable_gps_write"):
                enabled_phases.append("gps")
            if do_species:
                enabled_phases.append("species")
            if config.get("enable_watermark_generation", False):
                enabled_phases.append("watermark")
            if config.get("enable_record_export_auto", False):
                enabled_phases.append("record_export")
            if config.get("enable_track_map_auto", False):
                enabled_phases.append("track_map")
            if not enabled_phases:
                for _p, _k in (
                    ("track_map", "enable_track_map_auto"),
                    ("record_export", "enable_record_export_auto"),
                    ("watermark", "enable_watermark_generation"),
                    ("species", "enable_species_detection"),
                    ("burst", "enable_burst_detection"),
                ):
                    if config.get(_k, False if _k != "enable_species_detection" else True):
                        enabled_phases.append(_p)
                        break
                if not enabled_phases:
                    enabled_phases = ["track_map"]

            weight_sum = sum(float(phase_weights.get(p, 0.0)) for p in enabled_phases)
            if weight_sum <= 0:
                weight_sum = float(len(enabled_phases))

            phase_ranges: Dict[str, Tuple[float, float]] = {}
            acc = 0.0
            for p in enabled_phases:
                w = float(phase_weights.get(p, 1.0 / len(enabled_phases)))
                slot = 100.0 * (w / weight_sum)
                phase_ranges[p] = (acc, min(100.0, acc + slot))
                acc += slot

            def _emit_phase_progress(phase: str, done: int, total: int) -> None:
                start, end = phase_ranges.get(phase, (0.0, 100.0))
                span = max(0.0, end - start)
                frac = float(done) / float(max(1, total))
                frac = min(1.0, max(0.0, frac))
                v = int(min(99.0, start + span * frac))
                self.progress_updated.emit(v)

            n_eta = int(config.get("_eta_image_estimate", 0) or 0)
            if n_eta <= 0:
                n_eta = _count_images_for_eta(
                    config.get("image_folder", ""),
                    config.get("dual_format_mode", "off"),
                )
            phase_ests, eta_meta = build_eta_phase_estimates(config, n_eta)
            self.eta_checkpoint.emit(
                {
                    "kind": "start",
                    "n_images": n_eta,
                    "n_species_expected": int(eta_meta.get("n_species_expected") or 0),
                    "burst_sec_per": eta_meta.get("burst_sec_per"),
                    "species_sec_per": eta_meta.get("species_sec_per"),
                    "burst_keep_ratio": eta_meta.get("burst_keep_ratio"),
                    "burst_keep_min": eta_meta.get("burst_keep_min"),
                    "phases": [
                        {"name": n, "est": float(e)} for n, e in phase_ests
                    ],
                }
            )

            # GPS 在连拍筛选并复制到 Screened_images 之后再写入（见 burst 分支内）

            if not self.is_running:
                self.finished.emit({"_aborted": True})
                return

            screened_dir = os.path.join(config["output_folder"], "Screened_images")

            if burst_on:
                # 第二步：连拍识别和筛选
                current_step += 1
                self.status_updated.emit(
                    f"[步骤 {current_step}/{total_steps}] 连拍识别与筛选..."
                )
                _emit_phase_progress("burst", 0, 1)
                self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "burst"})

                try:
                    if not os.path.exists(config["image_folder"]):
                        self.error_occurred.emit(
                            f"错误: 图片文件夹不存在: {config['image_folder']}"
                        )
                        results["total_images"] = 0
                        results["kept_images"] = 0
                        results["discarded_images"] = 0
                    else:
                        burst_result = process_folder(
                            image_folder=config["image_folder"],
                            time_threshold=config["time_threshold"],
                            burst_keep_ratio=float(
                                config.get("burst_keep_ratio", 0.2)
                            ),
                            burst_keep_min=int(
                                config.get(
                                    "burst_keep_min",
                                    config.get("keep_top_n", 2),
                                )
                            ),
                            use_bird_detection=config["use_bird_detection"],
                            use_eye_detection=config.get("use_eye_detection", False),
                            output_report=os.path.join(
                                config["output_folder"], "burst_analysis.json"
                            ),
                            fast_mode=config["use_fast_mode"],
                            screened_output_dir=screened_dir,
                            dual_format_mode=config.get("dual_format_mode", "off"),
                            focus_score_weight=float(
                                config.get("focus_score_weight", 9.0)
                            ),
                            area_score_weight=float(
                                config.get("area_score_weight", 1.0)
                            ),
                            progress_callback=lambda d: (
                                _emit_phase_progress(
                                    "burst",
                                    int(d.get("done", 0)),
                                    int(d.get("total", 1)),
                                ),
                                self.eta_checkpoint.emit({
                                    "kind": "phase_tick",
                                    "phase": "burst",
                                    "done": int(d.get("done", 0)),
                                    "total": max(1, int(d.get("total", 1))),
                                }) if d.get("kind") != "start" else None,
                            ),
                            should_cancel=lambda: not self.is_running,
                        )
                        self.status_updated.emit(
                            f"✓ 处理 {burst_result['total_images']} 张图片，保留 {burst_result['kept_images']} 张"
                        )
                        results.update(burst_result)
                        burst_filter_applied = True
                        self.eta_checkpoint.emit(
                            {
                                "kind": "burst_result",
                                "total": int(burst_result.get("total_images") or 0),
                                "kept": int(burst_result.get("kept_images") or 0),
                            }
                        )
                        n_raw = int(burst_result.get("raw_companions_copied") or 0)
                        if n_raw:
                            raw_dir = burst_result.get("screened_raw_dir") or ""
                            self.status_updated.emit(
                                f"✓ 已复制 {n_raw} 个配对 RAW 至 Screened_raw_images"
                            )
                except Exception as e:
                    self.error_occurred.emit(f"连拍识别失败: {str(e)}")
                    traceback.print_exc()
                _emit_phase_progress("burst", 1, 1)
                self.eta_checkpoint.emit({"kind": "phase_done", "phase": "burst"})

                if not self.is_running:
                    self.finished.emit({"_aborted": True})
                    return

                if config.get("enable_gps_write"):
                    current_step += 1
                    gps_mode = config.get("gps_write_mode", "fixed")
                    mode_cn = (
                        "GPX 时间匹配"
                        if gps_mode == "gpx"
                        else "指定地点"
                    )
                    self.status_updated.emit(
                        f"[步骤 {current_step}/{total_steps}] 向 Screened_images 写入 GPS（{mode_cn}）…"
                    )
                    _emit_phase_progress("gps", 0, 1)
                    self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "gps"})
                    try:
                        gps_count, gps_detail = _main_flow_write_gps(
                            config, screened_dir
                        )
                        self.status_updated.emit(
                            f"✓ GPS 已写入 Screened_images：{gps_detail}"
                        )
                        results["gps_written"] = gps_count
                    except Exception as e:
                        self.error_occurred.emit(f"GPS 写入失败: {str(e)}")
                        results["gps_written"] = 0
                    _emit_phase_progress("gps", 1, 1)
                    self.eta_checkpoint.emit({"kind": "phase_done", "phase": "gps"})

                if not self.is_running:
                    self.finished.emit({"_aborted": True})
                    return
            else:
                _emit_burst_skipped_status(self.status_updated.emit, config)

            if config.get("enable_gps_write") and not burst_on:
                if os.path.isdir(screened_dir):
                    current_step += 1
                    gps_mode = config.get("gps_write_mode", "fixed")
                    mode_cn = (
                        "GPX 时间匹配"
                        if gps_mode == "gpx"
                        else "指定地点"
                    )
                    self.status_updated.emit(
                        f"[步骤 {current_step}/{total_steps}] 向 Screened_images 写入 GPS（{mode_cn}）…"
                    )
                    _emit_phase_progress("gps", 0, 1)
                    self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "gps"})
                    try:
                        gps_count, gps_detail = _main_flow_write_gps(
                            config, screened_dir
                        )
                        self.status_updated.emit(
                            f"✓ GPS 已写入 Screened_images：{gps_detail}"
                        )
                        results["gps_written"] = gps_count
                    except Exception as e:
                        self.error_occurred.emit(f"GPS 写入失败: {str(e)}")
                        results["gps_written"] = 0
                    _emit_phase_progress("gps", 1, 1)
                    self.eta_checkpoint.emit({"kind": "phase_done", "phase": "gps"})
                else:
                    self.status_updated.emit(
                        "GPS：未找到 Screened_images 目录，已跳过（无连拍筛选时请先准备该目录）。"
                    )
                    results["gps_written"] = 0

            # 下一步：物种识别与裁剪归档
            if do_species:
                current_step += 1
                self.status_updated.emit(
                    f"[步骤 {current_step}/{total_steps}] 物种检测、裁剪与归档..."
                )
                _emit_phase_progress("species", 0, 1)
                self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "species"})
                
                try:
                    import time
                    start_time = time.time()

                    # 仅豆包API模式才读取配置文件（本地模型无需读取）
                    doubao_config = None
                    if not config.get('use_local_model', True):
                        import json
                        doubao_path = ensure_doubao_api_config_file(
                            Path(__file__).resolve().parent
                        )
                        with open(doubao_path, "r", encoding="utf-8") as f:
                            doubao_config = json.load(f)

                    # 初始化检测器（支持混合识别模式）
                    min_species_thr = None
                    if config.get('species_conf_threshold_enabled', False):
                        min_species_thr = float(
                            config.get('min_species_accept_confidence', 0.5)
                        )
                    detector = BirdAndEyeDetector(
                        enable_species=do_species,
                        use_local_model=config.get('use_local_model', True),
                        local_species_model=config.get(
                            'local_species_model', LOCAL_SPECIES_MODEL_RESNET34
                        ),
                        doubao_config=doubao_config,
                        geo_mode=normalize_species_geo_mode(
                            config.get('species_geo_mode', SPECIES_GEO_MODE_AUTO)
                        ),
                        min_species_accept_confidence=min_species_thr,
                    )
                    
                    # 连拍成功后仅处理筛选保留的图片；连拍失败则回退为扫描整个输入目录
                    output_root = config['crop_output_folder']
                    _img_exts = all_supported_extensions()

                    if burst_filter_applied:
                        image_files = screened_paths_for_kept_images(
                            results,
                            config["image_folder"],
                            screened_dir,
                        )
                        if not image_files:
                            self.status_updated.emit(
                                "⚠ 未在 Screened_images 中找到与筛选保留对应的文件，"
                                "物种步骤将回退为原库路径。"
                            )
                            image_files = [
                                p
                                for p in get_kept_images(results)
                                if os.path.isfile(p)
                            ]
                        else:
                            self.status_updated.emit(
                                f"物种识别/归档使用 Screened_images 共 {len(image_files)} 张"
                                "（与连拍筛选副本一致，可读取已写入的 GPS 以启用地理约束）"
                            )
                    elif not burst_on:
                        image_files = _collect_image_paths_under(screened_dir)
                        if not image_files:
                            msg = (
                                "输出文件夹下的 Screened_images 中未找到图片。\n"
                                f"路径：{screened_dir}\n"
                                "请先完成连拍筛选并生成该目录，或在连拍处理中勾选「加入主流程」。"
                            )
                            self.status_updated.emit(f"⚠ {msg}")
                            self.error_occurred.emit(msg)
                            return
                        self.status_updated.emit(
                            f"物种识别/归档：使用已筛选目录，共 {len(image_files)} 张"
                        )
                    else:
                        image_folder = config['image_folder']
                        _dual = config.get("dual_format_mode", "off")
                        _img_exts = extensions_for_dual_mode(_dual)
                        image_files = []
                        for root, dirs, files in os.walk(image_folder):
                            for file in files:
                                if Path(file).suffix.lower() in _img_exts:
                                    image_path = os.path.join(root, file)
                                    if image_path not in image_files:
                                        image_files.append(image_path)
                        image_files = list(set(image_files))
                        self.status_updated.emit(
                            "物种识别/归档：连拍步骤未成功完成，扫描全部输入图片"
                        )

                    def _cfg_geo_str(v):
                        if v is None:
                            return None
                        s = str(v).strip()
                        return s or None

                    manual_province = _cfg_geo_str(config.get("province"))
                    manual_city = _cfg_geo_str(config.get("city"))

                    if not image_files:
                        self.status_updated.emit(
                            "⚠ 待处理图片列表为空，已跳过物种识别。"
                        )
                        results["crop_result"] = {
                            "total_crops": 0,
                            "processing_time": time.time() - start_time,
                        }
                    elif config.get("enable_image_clean_before_species", False):
                        self.status_updated.emit(
                            f"先切割大图中的鸟体（{len(image_files)} 张），"
                            "再清洗切割图，然后识别鸟种…"
                        )
                        _run_species_crop_clean_identify(
                            self,
                            detector,
                            image_files,
                            output_root,
                            config,
                            start_time,
                            _emit_phase_progress,
                            results,
                            manual_province,
                            manual_city,
                        )
                    else:
                        total_crops = 0
                        archive_counter = {"n": 0}
                        n_spec = len(image_files)
                        self.eta_checkpoint.emit({"kind": "species_begin", "n": n_spec})

                        for idx, image_file in enumerate(image_files):
                            if not self.is_running:
                                break

                            self.status_updated.emit(
                                f"处理中: {os.path.basename(image_file)} ({idx+1}/{len(image_files)})"
                            )

                            try:
                                result_image, detection_results = detector.detect(
                                    image_file,
                                    manual_province=manual_province,
                                    manual_city=manual_city,
                                )

                                if detection_results.get("birds"):
                                    province = detection_results.get("province")
                                    city = detection_results.get("city")
                                    orig_img = detection_results.get("original_image")
                                    if orig_img is None:
                                        orig_img = detector.load_image(image_file)
                                    saved_paths = detector.crop_species(
                                        image=orig_img,
                                        birds=detection_results["birds"],
                                        output_dir=output_root,
                                        source_path=image_file,
                                        province=province,
                                        city=city,
                                        counter=archive_counter,
                                    )
                                    total_crops += len(saved_paths)
                            except Exception as e:
                                self.status_updated.emit(
                                    f"⚠ {os.path.basename(image_file)}: {str(e)}"
                                )
                            self.eta_checkpoint.emit(
                                {
                                    "kind": "species_tick",
                                    "done": idx + 1,
                                    "total": max(1, n_spec),
                                }
                            )
                            if n_spec > 0:
                                _emit_phase_progress("species", idx + 1, n_spec)

                        processing_time = time.time() - start_time
                        self.status_updated.emit(
                            f"✓ 已输出 {total_crops} 个裁剪归档文件，耗时 {processing_time:.2f} 秒"
                        )
                        results["crop_result"] = {
                            "total_crops": total_crops,
                            "species_method": detector.get_species_method(),
                            "processing_time": processing_time,
                        }

                except Exception as e:
                    self.error_occurred.emit(f"物种检测/归档失败: {str(e)}")
                    traceback.print_exc()
                _emit_phase_progress("species", 1, 1)
                self.eta_checkpoint.emit({"kind": "phase_done", "phase": "species"})

            if not self.is_running:
                self.finished.emit({"_aborted": True})
                return

            if config.get("enable_record_export_auto", False):
                current_step += 1
                self.status_updated.emit(
                    f"[步骤 {current_step}/{total_steps}] 导出观鸟记录（eBird / 观鸟记录中心）..."
                )
                _emit_phase_progress("record_export", 0, 1)
                self.eta_checkpoint.emit(
                    {"kind": "phase_begin", "phase": "record_export"}
                )
                try:
                    class_root, out_dir = _record_export_dirs_from_config(config)
                    if class_root and os.path.isdir(class_root):
                        written = export_from_classification(
                            class_root,
                            out_dir,
                            write_ebird_csv=bool(
                                config.get("record_export_ebird", True)
                            ),
                            write_china_bird_record_xls=bool(
                                config.get("record_export_birdreport", True)
                            ),
                            ebird_country=str(
                                config.get("record_export_ebird_country", "CN")
                                or "CN"
                            ),
                            ebird_state=str(
                                config.get("record_export_ebird_state", "FJ")
                                or "FJ"
                            ),
                            **_record_export_kwargs(config),
                        )
                        for _k, _p in written.items():
                            self.status_updated.emit(
                                f"  ✓ {_k}: {os.path.basename(_p)}"
                            )
                        self.status_updated.emit(
                            "  提示：请在导出文件中核对并自行修改数量后再上传各平台。"
                        )
                        results["record_export"] = written
                    else:
                        self.status_updated.emit(
                            "⚠ 跳过观鸟记录自动导出：分类归档目录不存在或为空"
                        )
                except Exception as e:
                    self.error_occurred.emit(f"观鸟记录导出失败: {str(e)}")
                _emit_phase_progress("record_export", 1, 1)
                self.eta_checkpoint.emit(
                    {"kind": "phase_done", "phase": "record_export"}
                )

            if not self.is_running:
                self.finished.emit({"_aborted": True})
                return

            if config.get("enable_track_map_auto", False):
                current_step += 1
                self.status_updated.emit(
                    f"[步骤 {current_step}/{total_steps}] 生成轨迹图..."
                )
                _emit_phase_progress("track_map", 0, 1)
                self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "track_map"})
                try:
                    reports_dir = _reports_dir_from_config(config)
                    photo_folder = config.get("crop_output_folder", "").strip()
                    if config.get("track_map_photo_source") == "screened":
                        photo_folder = os.path.join(
                            config.get("output_folder", ""), "Screened_images"
                        )
                    gpx_paths = _config_gpx_paths(config)
                    use_gpx = bool(config.get("track_map_use_gpx", True))
                    use_exif = bool(config.get("track_map_use_exif", True))
                    if not use_gpx:
                        use_exif = True
                    written = generate_track_maps(
                        reports_dir=reports_dir,
                        gpx_paths=gpx_paths if use_gpx else None,
                        photo_folder=photo_folder,
                        use_gpx_track=use_gpx,
                        use_exif_gps=use_exif,
                        radius_km=float(config.get("track_map_radius_km", 1.0)),
                        include_elevation=bool(
                            config.get("track_map_include_elevation", True)
                        ),
                        basemap_style=str(
                            config.get("track_map_basemap_style", "normal")
                        ),
                        exif_tz=normalize_tz_name(
                            config.get("gpx_match_exif_tz")
                            or config.get("track_map_exif_tz", DEFAULT_EXIF_TZ)
                        ),
                        gpx_tz=normalize_tz_name(
                            config.get("gpx_match_gpx_tz")
                            or config.get("track_map_gpx_tz", DEFAULT_GPX_TZ)
                        ),
                        location_name=_track_map_location_from_config(config),
                        province=str(config.get("province") or ""),
                        city=str(config.get("city") or ""),
                        logo_path=str(config.get("wm_logo_path", "") or ""),
                        logo_width_ratio=float(
                            config.get("wm_logo_width_ratio", 0.30)
                        ),
                        preview_only=False,
                    )
                    for k, p in written.items():
                        self.status_updated.emit(f"  ✓ {k}: {os.path.basename(p)}")
                    results["track_map"] = written
                except Exception as e:
                    self.error_occurred.emit(f"轨迹图生成失败: {str(e)}")
                _emit_phase_progress("track_map", 1, 1)
                self.eta_checkpoint.emit({"kind": "phase_done", "phase": "track_map"})

            if not self.is_running:
                self.finished.emit({"_aborted": True})
                return

            # 水印生成
            if config.get("enable_watermark_generation", False):
                current_step += 1
                self.status_updated.emit(
                    f"[步骤 {current_step}/{total_steps}] 生成水印图片..."
                )
                _emit_phase_progress("watermark", 0, 1)
                self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "watermark"})
                try:
                    # 主流程模式下：优先使用文件夹设置自动生成的 classification 目录（crop_output_folder），
                    # 确保与物种识别归档目录一致；仅在 classification 目录无效时才 fallback 到 watermark_input_folder
                    crop_folder = config.get("crop_output_folder", "").strip()
                    wm_manual_folder = config.get("watermark_input_folder", "").strip()

                    # 优先使用 classification 目录（如果存在且包含图片）
                    if crop_folder and os.path.isdir(crop_folder):
                        imgs_in_crop = collect_images_recursive(crop_folder)
                        if imgs_in_crop:
                            source_folder = crop_folder
                        else:
                            # classification 目录为空，尝试 manual folder 或默认选择
                            source_folder = wm_manual_folder or choose_default_watermark_source(
                                image_folder=config.get("image_folder", ""),
                                crop_output_folder=crop_folder,
                                output_folder=config.get("output_folder", ""),
                            )
                    else:
                        # classification 目录不存在，使用手动设置或默认选择
                        source_folder = wm_manual_folder or choose_default_watermark_source(
                            image_folder=config.get("image_folder", ""),
                            crop_output_folder=crop_folder,
                            output_folder=config.get("output_folder", ""),
                        )
                    output_folder = (
                        config.get("watermark_output_folder", "").strip()
                        or os.path.join(config.get("output_folder", "./outputs"), "watermarked")
                    )
                    _wm_st = str(config.get("wm_watermark_style", "frame") or "frame")
                    if _wm_st not in ("frame", "inline"):
                        _wm_st = "frame"
                    _wm_ai_dn_model = str(
                        config.get("wm_ai_denoise_model", "realesrgan") or "realesrgan"
                    )
                    if _wm_ai_dn_model not in ("realesrgan", "nafnet"):
                        _wm_ai_dn_model = "realesrgan"
                    wopt = WatermarkOptions(
                        enable_location=bool(config.get("wm_enable_location", True)),
                        location_text=str(config.get("wm_location_text", "") or ""),
                        use_gps_city=bool(config.get("wm_use_gps_city", True)),
                        enable_date=bool(config.get("wm_enable_date", True)),
                        enable_species=bool(config.get("wm_enable_species", True)),
                        enable_camera_params=bool(config.get("wm_enable_camera", True)),
                        logo_path=str(config.get("wm_logo_path", "") or ""),
                        logo_width_ratio=float(config.get("wm_logo_width_ratio", 0.30)),
                        watermark_style=_wm_st,  # type: ignore[arg-type]
                        enable_auto_enhance=False,
                        enable_ai_exposure=bool(config.get("wm_enable_ai_exposure", False)),
                        ai_exposure_strength=float(config.get("wm_ai_exposure_strength", 1.0)),
                        enable_ai_denoise=bool(config.get("wm_enable_ai_denoise", False)),
                        enable_ai_sharpen=bool(config.get("wm_enable_ai_sharpen", False)),
                        ai_denoise_model=_wm_ai_dn_model,
                        ai_denoise_strength=float(config.get("wm_ai_denoise_strength", 0.5)),
                        ai_sharpen_strength=float(config.get("wm_ai_sharpen_strength", 0.5)),
                        ai_tile_size=int(config.get("wm_ai_tile_size", 512)),
                    )
                    wm_result = generate_watermarks(
                        source_folder=source_folder,
                        output_folder=output_folder,
                        options=wopt,
                        prefer_folder_name_as_species=True,
                        progress_callback=lambda d: _emit_phase_progress(
                            "watermark",
                            int(d.get("done", 0)),
                            int(d.get("total", 1)),
                        ),
                        should_cancel=lambda: not self.is_running,
                        random_per_species=(
                            int(config.get("wm_random_per_species_count", 3))
                            if config.get("wm_random_per_species", False)
                            else None
                        ),
                    )
                    self.status_updated.emit(
                        f"✓ 水印生成完成: 共 {wm_result['total']}，成功 {wm_result['ok']}，失败 {wm_result['fail']}"
                    )
                    results["watermark_result"] = wm_result
                except Exception as e:
                    self.error_occurred.emit(f"水印生成失败: {str(e)}")
                _emit_phase_progress("watermark", 1, 1)
                self.eta_checkpoint.emit({"kind": "phase_done", "phase": "watermark"})
            
            self.progress_updated.emit(100)
            self.status_updated.emit("✓ 处理完成！")
            self.finished.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(f"发生异常: {str(e)}\n{traceback.format_exc()}")
    
    def stop(self):
        """停止处理"""
        self.is_running = False


class WatermarkBatchThread(QThread):
    """在后台线程运行批量水印，避免阻塞 GUI。"""

    progress = pyqtSignal(int, int, int, int)  # percent(0-100), elapsed_sec, remaining_sec, done
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        source_folder: str,
        output_folder: str,
        options: WatermarkOptions,
        parent=None,
        random_per_species: Optional[int] = None,
    ):
        super().__init__(parent)
        self._source_folder = source_folder
        self._output_folder = output_folder
        self._options = options
        self._random_per_species = random_per_species
        self._last_log_pct = -1
        self._t_start: float = 0.0
        self._last_pct: int = -1

    def run(self) -> None:
        import time as _t
        self._last_log_pct = -1
        self._last_pct = -1
        self._t_start = _t.time()

        def _emit(pct: int, done: int, total: int) -> None:
            pct = max(0, min(100, pct))
            if pct == self._last_pct and pct < 100:
                return
            self._last_pct = pct
            elapsed = int(_t.time() - self._t_start)
            if pct > 0:
                total_est = elapsed / (pct / 100.0)
                remaining = max(0, int(total_est - elapsed))
            else:
                remaining = 0
            self.progress.emit(pct, elapsed, remaining, done)

        def _cb(d: Dict) -> None:
            k = d.get("kind")
            tot = max(1, int(d.get("total", 1)))
            if k == "start":
                extra = ""
                if self._random_per_species and int(self._random_per_species) > 0:
                    extra = f"（每物种目录随机≤{int(self._random_per_species)} 张）"
                self.log_line.emit(f"水印批量：开始，共 {tot} 张{extra}…")
                _emit(0, 0, tot)
            elif k == "tick":
                done = int(d.get("done", 0))
                pct = min(100, (100 * done) // tot)
                _emit(pct, done, tot)
                if pct >= self._last_log_pct + 5 or done >= tot:
                    self.log_line.emit(f"水印批量：进度 {done}/{tot}（{pct}%）")
                    self._last_log_pct = pct
            elif k == "done":
                _emit(100, tot, tot)

        try:
            r = generate_watermarks(
                source_folder=self._source_folder,
                output_folder=self._output_folder,
                options=self._options,
                prefer_folder_name_as_species=True,
                progress_callback=_cb,
                random_per_species=self._random_per_species,
            )
            self.log_line.emit(
                f"水印批量：结束，成功 {r.get('ok', 0)}，失败 {r.get('fail', 0)}。"
            )
            self.finished_ok.emit(r)
        except Exception as e:
            self.failed.emit(str(e))


class ImageCleanThread(QThread):
    """后台清洗鸟图目录，避免阻塞 GUI。"""

    progress = pyqtSignal(int, int)  # done, total
    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        folder: str,
        options: ImageCleanOptions,
        parent=None,
    ):
        super().__init__(parent)
        self._folder = folder
        self._options = options

    def run(self) -> None:
        try:
            def _cb(d: Dict) -> None:
                k = d.get("kind")
                tot = max(1, int(d.get("total", 1)))
                done = int(d.get("done", 0))
                if k == "start":
                    self.log_line.emit(f"图片清洗：开始扫描，共 {tot} 张…")
                elif k == "tick":
                    self.progress.emit(done, tot)
                    if done == tot or done % 10 == 0:
                        phase = d.get("phase", "")
                        self.log_line.emit(
                            f"图片清洗：进度 {done}/{tot}"
                            + (f"（{phase}）" if phase else "")
                        )
                elif k == "done":
                    self.progress.emit(tot, tot)

            r = clean_bird_images(
                self._folder,
                self._options,
                progress_callback=_cb,
            )
            d = r.as_dict()
            self.log_line.emit(
                "图片清洗完成："
                f"总计 {d['total']}，保留 {d['kept']}，"
                f"无鸟体 {d['removed_no_bird']}，模糊 {d['removed_blurry']}，"
                f"重复 {d['removed_duplicate']}，失败 {d['failed']}"
            )
            self.finished_ok.emit(d)
        except Exception as e:
            self.failed.emit(str(e))


class TrackMapThread(QThread):
    """子进程生成轨迹图，避免 matplotlib 与 Qt 同线程死锁。"""

    log_line = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, kwargs: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._kwargs = dict(kwargs)

    @staticmethod
    def _subprocess_popen_kwargs() -> Dict[str, Any]:
        """Windows 下隐藏控制台窗口，避免「闪一下像退出」。"""
        import subprocess as sp

        kw: Dict[str, Any] = {}
        if sys.platform == "win32":
            kw["creationflags"] = getattr(sp, "CREATE_NO_WINDOW", 0)
        return kw

    def run(self) -> None:
        import json
        import subprocess
        import tempfile
        import time

        self.log_line.emit("轨迹图：已启动生成子进程…")
        use_gpx_track = bool(self._kwargs.get("use_gpx_track", True))
        if use_gpx_track:
            self.log_line.emit("轨迹图：GPX 时间匹配模式")
        else:
            self.log_line.emit("轨迹图：照片 EXIF GPS 模式（不使用 GPX）")
        src_dir = Path(__file__).resolve().parent
        try:
            with tempfile.TemporaryDirectory(prefix="birdy_trackmap_") as td:
                kin = Path(td) / "kwargs.json"
                kout = Path(td) / "result.json"
                kin.write_text(
                    json.dumps(self._kwargs, ensure_ascii=False),
                    encoding="utf-8",
                )
                cmd = [
                    sys.executable,
                    "-m",
                    "gpx_track.generate_worker",
                    str(kin),
                    str(kout),
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get(
                    "PYTHONPATH", ""
                )
                popen_kw = self._subprocess_popen_kwargs()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=str(src_dir),
                    env=env,
                    **popen_kw,
                )
                last_ping = time.monotonic()
                ping_msg = (
                    "轨迹图：仍在生成中（匹配 GPX、下载底图、绘制 PNG）…"
                    if use_gpx_track
                    else "轨迹图：仍在生成中（读取照片 GPS、下载底图、绘制 PNG）…"
                )
                while proc.poll() is None:
                    now = time.monotonic()
                    if now - last_ping >= 3.0:
                        self.log_line.emit(ping_msg)
                        last_ping = now
                    time.sleep(0.25)
                stdout, stderr = proc.communicate(timeout=30)
                if proc.returncode != 0:
                    err_body = (stderr or stdout or "").strip()
                    if kout.is_file():
                        try:
                            payload = json.loads(kout.read_text(encoding="utf-8"))
                            if payload.get("error"):
                                err_body = str(payload["error"]).strip()
                        except Exception:
                            pass
                    self.failed.emit(
                        err_body or f"子进程退出码 {proc.returncode}"
                    )
                    return
                if not kout.is_file():
                    self.failed.emit("轨迹图子进程未生成结果文件")
                    return
                written = json.loads(kout.read_text(encoding="utf-8"))
                if written.get("error"):
                    self.failed.emit(str(written["error"]))
                    return
                self.log_line.emit("轨迹图：子进程绘制完成，正在载入结果…")
                self.finished_ok.emit(written)
        except subprocess.TimeoutExpired:
            self.failed.emit("轨迹图生成超时（超过 10 分钟）")
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class BirdDetectionGUI(QMainWindow):
    """鸟图智慧仓储 (Birdy) GUI 主程序"""

    _APP_NAME_CN = "鸟图智慧仓储"
    _APP_NAME_EN = "Birdy"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Birdy")
        self.setGeometry(100, 100, 1200, 800)
        self.app_version = self._load_skill_version()

        # 窗口与任务栏图标（优先使用 birdy_logo_128.png）
        self._set_window_icon()
        
        # 初始化变量
        self.worker_thread: Optional[WorkerThread] = None
        self._wm_batch_thread: Optional[WatermarkBatchThread] = None
        self._image_clean_thread: Optional[ImageCleanThread] = None
        self._track_map_thread: Optional[TrackMapThread] = None
        self._track_map_progress: Optional[QProgressDialog] = None
        self._track_map_preview_dialog = None
        self.config: Dict = self._get_default_config()
        self._process_start_monotonic: Optional[float] = None
        self._process_time_timer = QTimer(self)
        self._process_time_timer.setInterval(500)
        self._process_time_timer.timeout.connect(self._refresh_process_time_labels)
        self._eta_phases: List[Dict[str, Any]] = []
        self._flow_eta = FlowEtaEstimator()
        self._eta_species_t0: Optional[float] = None
        self._eta_species_done = 0
        self._eta_species_total = 0
        self._ema_sec_per_species: Optional[float] = None
        self._eta_ema_alpha = 0.35
        
        # 设置全局样式
        self._set_global_style()
        
        # 构建UI
        self._init_ui()
        self._load_config()
        
        # 默认最大化窗口
        self.showMaximized()

    @staticmethod
    def _load_skill_version() -> str:
        try:
            info = Path(__file__).resolve().parent.parent / "version-info.json"
            if info.exists():
                with open(info, "r", encoding="utf-8") as f:
                    d = json.load(f)
                v = str(d.get("version", "2.0.7"))
                rd = str(d.get("release_date", "") or "").strip()
                if rd:
                    return f"{v}（{rd}）"
                return v
        except Exception:
            pass
        return "2.0.7"

    @staticmethod
    def _primary_screen_dpr() -> float:
        app = QApplication.instance()
        if app:
            scr = app.primaryScreen()
            if scr is not None:
                return max(1.0, float(scr.devicePixelRatio()))
        return 1.0

    @staticmethod
    def _logo_search_directories() -> List[Path]:
        """
        可能存放 Birdy logo 的目录（去重）。
        兼容：从项目根启动、从 src 启动、工作目录在根或 src、resources 在根或误放在 src 下。
        """
        here = Path(__file__).resolve()
        src_dir = here.parent
        root_dir = src_dir.parent
        cwd = Path.cwd()
        candidates = [
            root_dir / "resources",
            root_dir,
            src_dir / "resources",
            src_dir,
            cwd / "resources",
            cwd,
            cwd / "src" / "resources",
            cwd.parent / "resources",
            cwd.parent,
        ]
        out: List[Path] = []
        seen = set()
        for d in candidates:
            try:
                key = str(d.resolve())
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    @staticmethod
    def _resolve_birdy_logo_asset(names: Tuple[str, ...]) -> Optional[Path]:
        for d in BirdDetectionGUI._logo_search_directories():
            for name in names:
                p = d / name
                if p.is_file():
                    return p
        return None

    @staticmethod
    def _logo_path_for_icon() -> Optional[Path]:
        """软件 Birdy 图标（任务栏/窗口），勿用版权人 logo。"""
        return BirdDetectionGUI._resolve_birdy_logo_asset(
            (
                "birdy_logo.ico",
                "birdy_logo_128.png",
                "birdy_logo_640.png",
            )
        )

    @staticmethod
    def _logo_path_for_banner() -> Optional[Path]:
        """顶栏 Banner：仅 Birdy 品牌图。"""
        return BirdDetectionGUI._resolve_birdy_logo_asset(
            (
                "birdy_logo_640.png",
                "birdy_logo_128.png",
            )
        )

    @staticmethod
    def _copyright_holder_logo_path() -> Optional[Path]:
        """版权说明区：仅版权人提供的 logo（默认 resources/logo.png）。"""
        return BirdDetectionGUI._resolve_birdy_logo_asset(("logo.png", "logo.ico"))

    def _set_window_icon(self):
        """任务栏与标题栏左侧图标（多尺寸，利于 Windows 壳）"""
        try:
            from PyQt5.QtGui import QPixmap, QPainter, QColor, QIcon, QPolygon
            from PyQt5.QtCore import QPoint
            from PyQt5.QtWidgets import QStyle

            path = self._logo_path_for_icon()
            if path is not None:
                pm = QPixmap(str(path))
                if not pm.isNull():
                    # Windows 任务栏对非方形 pixmap 兼容差：先缩放到方形透明画布再入 QIcon
                    icon = QIcon()
                    for s in (16, 24, 32, 48, 64, 128, 256):
                        canvas = QPixmap(s, s)
                        canvas.fill(Qt.transparent)
                        painter = QPainter(canvas)
                        scaled = pm.scaled(
                            s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                        x = (s - scaled.width()) // 2
                        y = (s - scaled.height()) // 2
                        painter.drawPixmap(x, y, scaled)
                        painter.end()
                        icon.addPixmap(canvas)
                    self.setWindowIcon(icon)
                    return

            pixmap = QPixmap(256, 256)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(46, 139, 87))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(60, 100, 140, 100)
            painter.drawEllipse(150, 70, 60, 60)
            painter.setBrush(QColor(39, 118, 73))
            painter.drawEllipse(70, 120, 80, 60)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(175, 85, 18, 18)
            painter.setBrush(QColor(0, 0, 0))
            painter.drawEllipse(180, 90, 10, 10)
            painter.setBrush(QColor(255, 140, 0))
            painter.drawPolygon(
                QPolygon([QPoint(205, 95), QPoint(245, 100), QPoint(205, 105)])
            )
            painter.end()
            self.setWindowIcon(QIcon(pixmap))
        except Exception as e:
            print(f"设置图标时出错: {e}")
            from PyQt5.QtWidgets import QStyle
            self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
    
    def _set_global_style(self):
        """设置全局样式

        字体统一用 pt（点）而非 px：高 DPI 下 px 易偏小，且 QSS 会覆盖 setFont()，
        改 QFont 若未同步样式表会看不到效果。
        """
        style = """
            /* 全局样式 */
            QMainWindow {
                background-color: #F5F5F5;
            }
            
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Arial', sans-serif;
                font-size: 10pt;
            }
            
            /* 标签样式 */
            QLabel {
                color: #333333;
            }
            
            /* 输入框样式 */
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 5px 10px;
                color: #333333;
                font-size: 10pt;
                min-height: 1.1em;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #2E8B57;
                outline: none;
            }
            
            /* 按钮样式 */
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 6px 14px;
                color: #333333;
                font-weight: 500;
                font-size: 10pt;
                min-height: 1.2em;
            }
            
            QPushButton:hover {
                background-color: #F0F0F0;
                border: 1px solid #1E90FF;
            }
            
            QPushButton:pressed {
                background-color: #E0E0E0;
            }
            
            QPushButton:disabled {
                background-color: #F5F5F5;
                color: #999999;
                border: 1px solid #E0E0E0;
            }
            
            /* 复选框样式 */
            QCheckBox {
                spacing: 6px;
                font-size: 10pt;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #E0E0E0;
                border-radius: 5px;
                background-color: #FFFFFF;
            }
            
            QCheckBox::indicator:checked {
                background-color: #2E8B57;
                border: 2px solid #2E8B57;
            }
            
            QCheckBox::indicator:hover {
                border: 2px solid #1E90FF;
            }
            
            /* 单选框样式 */
            QRadioButton {
                spacing: 6px;
                font-size: 10pt;
            }
            
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #E0E0E0;
                border-radius: 10px;
                background-color: #FFFFFF;
            }
            
            QRadioButton::indicator:checked {
                background-color: #2E8B57;
                border: 2px solid #2E8B57;
            }
            
            QRadioButton::indicator:hover {
                border: 2px solid #1E90FF;
            }
            
            /* 进度条样式 */
            QProgressBar {
                background-color: #F0F0F0;
                border: none;
                border-radius: 8px;
                text-align: center;
                height: 24px;
                font-size: 10pt;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2E8B57, stop:1 #1E90FF);
                border-radius: 8px;
            }
            
            /* 文本编辑框样式 */
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 6px;
                font-family: 'Consolas', 'Courier New', 'Microsoft YaHei UI', monospace;
                font-size: 9pt;
            }
            
            /* 表格样式 */
            QTableWidget {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                font-size: 10pt;
            }
            
            QTableWidget::item {
                padding: 4px 8px;
            }
            
            QHeaderView::section {
                background-color: #F0F0F0;
                padding: 5px 8px;
                border: none;
                border-bottom: 1px solid #E0E0E0;
                font-size: 10pt;
            }
        """
        self.setStyleSheet(style)
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'image_folder': '',
            'dual_format_mode': 'off',
            'output_root_folder': '',
            'output_folder': './outputs',
            'crop_output_folder': './crops',
            'enable_gps_write': True,
            'gps_latitude': 31.2304,  # 上海
            'gps_longitude': 121.4737,
            'gps_altitude': 0,
            'location_name': '上海',
            'province': '上海',
            'city': '上海',
            'time_threshold': 1.0,
            'burst_keep_ratio': 0.2,
            'burst_keep_min': 2,
            'keep_top_n': 2,
            'focus_score_weight': 9.0,
            'area_score_weight': 1.0,
            'enable_burst_detection': True,
            'use_bird_detection': True,
            'use_eye_detection': False,
            'use_fast_mode': True,
            'generate_burst_report': False,
            'enable_species_detection': True,
            'enable_crop': True,
            'generate_species_report': False,
            # 切割后识别前 / 分类目录图片清洗
            'enable_image_clean_before_species': False,
            'image_clean_remove_no_bird': True,
            'image_clean_remove_blurry': True,
            'image_clean_dedupe': True,
            'image_clean_min_clarity': 35,
            'image_clean_dup_similarity': 92,
            'image_clean_folder': '',
            # 物种识别模式配置
            'use_local_model': True,  # 默认使用本地模型
            'local_species_model': LOCAL_SPECIES_MODEL_RESNET34,
            # 地理约束：基准测试 ResNet34 在 auto/china/province 下约 87.7% top1（浦口集）
            'species_geo_mode': SPECIES_GEO_MODE_AUTO,
            'enable_doubao_api': False,  # 默认不启用豆包API
            'doubao_api_key': '',
            # 未知种类阈值：仅当 species_conf_threshold_enabled 为 True 时在地理后判顶一生效
            'species_conf_threshold_enabled': False,
            'min_species_accept_confidence': 0.5,
            # 水印生成
            'enable_watermark_generation': False,
            'watermark_input_folder': '',
            'watermark_output_folder': './watermarked',
            'wm_logo_path': '',
            'wm_enable_location': True,
            'wm_location_text': '',
            'wm_use_gps_city': True,
            'wm_enable_date': True,
            'wm_enable_species': True,
            'wm_enable_camera': True,
            'wm_logo_width_ratio': 0.30,
            'wm_watermark_style': 'frame',
            'wm_random_per_species': False,
            'wm_random_per_species_count': 3,
            # AI 增强（水印前曝光/降噪/锐化，流水线顺序：曝光→降噪→锐化）
            'wm_enable_ai_exposure': False,
            'wm_ai_exposure_strength': 1.0,
            'wm_enable_ai_denoise': False,
            'wm_enable_ai_sharpen': False,
            'wm_ai_denoise_model': 'realesrgan',  # "realesrgan" 或 "nafnet"
            'wm_ai_denoise_strength': 0.5,
            'wm_ai_sharpen_strength': 0.5,
            'wm_ai_tile_size': 512,
            # 观鸟记录导出（主流程自动导出默认关闭）
            'enable_record_export_auto': False,
            'record_export_classification_folder': '',
            'record_export_output_folder': '',
            'record_export_ebird': True,
            'record_export_birdreport': True,
            'record_export_ebird_country': 'CN',
            'record_export_ebird_state': 'FJ',
            'record_export_count_individuals': True,
            'record_export_spatial_km': 0.1,
            'record_export_time_minutes': 120.0,
            'record_export_location_time_minutes': 30.0,
            # 可折叠区块默认展开
            'ui_section_expanded_geo': True,
            'ui_section_expanded_burst': True,
            'ui_section_expanded_species': True,
            'ui_section_expanded_watermark': False,
            'ui_section_expanded_export': True,
            # GPX / 轨迹图
            'gpx_file_path': '',
            'gpx_file_paths': [],
            'gpx_apply_to_screened': True,
            'enable_track_map_auto': False,
            'track_map_use_gpx': True,
            'track_map_use_exif': True,
            'track_map_photo_source': 'classification',
            'track_map_photo_folder_override': '',
            'track_map_radius_km': 1.0,
            'track_map_include_elevation': True,
            'track_map_basemap_style': 'normal',
            'gps_write_mode': 'fixed',
            'gpx_match_exif_tz': DEFAULT_EXIF_TZ,
            'gpx_match_gpx_tz': DEFAULT_GPX_TZ,
            'ui_section_expanded_track_map': False,
        }
    
    def _init_ui(self):
        """初始化用户界面（顶栏固定，下方为可滚动的功能区）"""
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._create_top_banner())

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setSpacing(12)
        body_layout.setContentsMargins(12, 8, 12, 12)
        body_layout.addWidget(self._create_settings_panel(), 1)
        body_layout.addWidget(self._create_status_panel(), 1)
        outer.addWidget(body, 1)

        self.setCentralWidget(central)

    def _create_top_banner(self) -> QWidget:
        """顶部固定品牌栏（不参与下方滚动）"""
        from PyQt5.QtGui import QPixmap

        banner = QWidget()
        banner.setObjectName("birdyTopBanner")
        banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        banner.setStyleSheet(
            "#birdyTopBanner { background-color: #FFFFFF; "
            "border-bottom: 1px solid #E0E0E0; }"
        )

        row = QHBoxLayout(banner)
        row.setContentsMargins(14, 8, 18, 8)
        row.setSpacing(10)

        logo_h = 56
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_path = self._logo_path_for_banner()
        if logo_path is not None:
            pm = QPixmap(str(logo_path))
            if not pm.isNull():
                dpr = self._primary_screen_dpr()
                scaled = pm.scaledToHeight(
                    max(1, int(logo_h * dpr)), Qt.SmoothTransformation
                )
                scaled.setDevicePixelRatio(dpr)
                logo_label.setPixmap(scaled)
        logo_label.setFixedHeight(logo_h)
        logo_label.setMinimumWidth(logo_h)
        row.addWidget(logo_label, 0, Qt.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.setContentsMargins(0, 0, 0, 0)

        cn = QLabel(self._APP_NAME_CN)
        cn.setStyleSheet(
            "color: #2E3A3F; font-size: 14pt; font-weight: bold;"
        )
        text_col.addWidget(cn)

        en = QLabel(self._APP_NAME_EN)
        en.setStyleSheet("color: #5A6B73; font-size: 10pt;")
        text_col.addWidget(en)

        ver = QLabel(f"版本 {self.app_version}")
        ver.setStyleSheet("color: #7A8A92; font-size: 9pt;")
        text_col.addWidget(ver)

        row.addLayout(text_col)
        row.addStretch(1)

        return banner
    
    def _create_settings_panel(self) -> QWidget:
        """创建设置面板（可滚动，避免大字/高 DPI 下纵向空间不足时控件被压扁裁切）"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )

        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 8, 0)

        # ═════ 文件夹设置 ═════
        folder_card, folder_group = self._create_card("📁 文件夹设置")
        folder_layout = QFormLayout()
        folder_layout.setSpacing(8)
        folder_layout.setContentsMargins(12, 10, 12, 12)
        
        # 图片文件夹
        folder_row = QHBoxLayout()
        self.image_folder_input = QLineEdit()
        self.image_folder_input.setReadOnly(True)
        folder_btn = QPushButton("浏览...")
        folder_btn.clicked.connect(lambda: self._select_folder('image_folder'))
        folder_row.addWidget(self.image_folder_input, 1)
        folder_row.addWidget(folder_btn)
        folder_layout.addRow("图片文件夹:", folder_row)

        dual_format_row = QHBoxLayout()
        self.dual_format_combo = QComboBox()
        self.dual_format_combo.addItem("关", "off")
        self.dual_format_combo.addItem("仅 JPG", "jpg_only")
        self.dual_format_combo.addItem("JPG + 复制 RAW", "jpg_copy_raw")
        _df = self.config.get("dual_format_mode", "off")
        _dfi = self.dual_format_combo.findData(_df)
        self.dual_format_combo.setCurrentIndex(_dfi if _dfi >= 0 else 0)
        self.dual_format_combo.setToolTip(
            "RAW+JPG 同目录时使用：主流程只处理 JPG；"
            "「复制 RAW」在筛选后将配对 RAW 写入 Screened_raw_images。"
        )
        dual_format_row.addWidget(self.dual_format_combo, 1)
        folder_layout.addRow("RAW+JPG:", dual_format_row)

        # 输出根目录（固定本机路径；与相片文件夹名自动生成 screened_* / classification_* / reports）
        output_root_row = QHBoxLayout()
        self.output_root_input = QLineEdit()
        self.output_root_input.setText(self.config.get("output_root_folder", ""))
        self.output_root_input.setPlaceholderText("例如 D:/birdy/output（留空则使用下方手动路径）")
        output_root_btn = QPushButton("浏览...")
        output_root_btn.clicked.connect(lambda: self._select_folder("output_root_folder"))
        output_root_row.addWidget(self.output_root_input, 1)
        output_root_row.addWidget(output_root_btn)
        folder_layout.addRow("输出根目录:", output_root_row)

        self.derived_paths_label = QLabel("")
        self.derived_paths_label.setWordWrap(True)
        self.derived_paths_label.setStyleSheet("color: #555555; font-size: 9pt;")
        self.derived_paths_label.setVisible(False)
        folder_layout.addRow("自动生成:", self.derived_paths_label)

        # 手动路径（仅当输出根目录为空时使用）
        self._legacy_paths_container = QWidget()
        legacy_form = QFormLayout()
        legacy_form.setSpacing(8)
        legacy_form.setContentsMargins(0, 0, 0, 0)
        output_row = QHBoxLayout()
        self.output_folder_input = QLineEdit()
        self.output_folder_input.setText(self.config['output_folder'])
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(lambda: self._select_folder('output_folder'))
        output_row.addWidget(self.output_folder_input, 1)
        output_row.addWidget(output_btn)
        legacy_form.addRow("输出文件夹:", output_row)
        crop_row = QHBoxLayout()
        self.crop_folder_input = QLineEdit()
        self.crop_folder_input.setText(self.config['crop_output_folder'])
        crop_btn = QPushButton("浏览...")
        crop_btn.clicked.connect(lambda: self._select_folder('crop_output_folder'))
        crop_row.addWidget(self.crop_folder_input, 1)
        crop_row.addWidget(crop_btn)
        legacy_form.addRow("分类归档文件夹:", crop_row)
        self._legacy_paths_container.setLayout(legacy_form)
        folder_layout.addRow(self._legacy_paths_container)

        self._refresh_derived_paths_display()
        self._refresh_track_map_path_ui()
        self.output_root_input.textChanged.connect(
            lambda _t: self._refresh_derived_paths_display()
        )
        self.output_root_input.textChanged.connect(
            lambda _t: self._refresh_track_map_path_ui()
        )
        self.dual_format_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_derived_paths_display()
        )

        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_card)
        
        # ═════ 地理位置（可收起；主流程勾选在标题栏）═════
        self.gps_write_checkbox = QCheckBox("加入主流程")
        self.gps_write_checkbox.setChecked(self.config["enable_gps_write"])
        self.gps_write_checkbox.setToolTip(
            "勾选后，连拍筛选完成时按下方「写入方式」自动写入 Screened_images：\n"
            "指定地点统一经纬度，或按 GPX 与 EXIF 时间匹配插值写入。"
        )
        self.gps_write_checkbox.stateChanged.connect(self._on_gps_write_changed)
        self._style_flow_header_checkbox(self.gps_write_checkbox)

        geo_card, geo_group = self._create_collapsible_card(
            "🌍 地理位置",
            "geo",
            header_widgets=[self.gps_write_checkbox],
        )
        geo_layout = QFormLayout()
        geo_layout.setSpacing(8)
        geo_layout.setContentsMargins(12, 10, 12, 12)

        geo_gps_hint = QLabel(
            "GPS 写入二选一（勾选「加入主流程」后自动执行）："
            "指定地点统一经纬度，或 GPX 按拍摄时间匹配。"
            "下方「批量写入照片 GPS」可随时单独对任意文件夹执行（方式与上方写入方式一致）。"
        )
        geo_gps_hint.setWordWrap(True)
        geo_gps_hint.setStyleSheet("color: #555; font-size: 9pt;")
        geo_layout.addRow(geo_gps_hint)

        self.gps_write_mode_group = QButtonGroup(self)
        mode_row = QHBoxLayout()
        self.gps_mode_fixed_radio = QRadioButton("指定地点统一写入")
        self.gps_mode_gpx_radio = QRadioButton("GPX 按拍摄时间匹配")
        self.gps_write_mode_group.addButton(self.gps_mode_fixed_radio, 0)
        self.gps_write_mode_group.addButton(self.gps_mode_gpx_radio, 1)
        _gps_mode = self.config.get("gps_write_mode", "fixed")
        if _gps_mode == "gpx":
            self.gps_mode_gpx_radio.setChecked(True)
        else:
            self.gps_mode_fixed_radio.setChecked(True)
        self.gps_mode_fixed_radio.setToolTip(
            "主流程：向 Screened_images 写入上方查询/填写的统一经纬度。"
        )
        self.gps_mode_gpx_radio.setToolTip(
            "主流程：按 GPX 轨迹与 EXIF 拍摄时间（见下方时区）插值后写入 GPS。"
        )
        self.gps_write_mode_group.buttonClicked.connect(
            self._on_gps_write_mode_changed
        )
        mode_row.addWidget(self.gps_mode_fixed_radio)
        mode_row.addWidget(self.gps_mode_gpx_radio)
        mode_row.addStretch(1)
        geo_layout.addRow("写入方式:", mode_row)

        amap_row = QHBoxLayout()
        amap_cfg_label = QLabel("高德API:")
        amap_cfg_label.setToolTip("地名转坐标优先走高德；密钥在 amap_api_config.json 中配置")
        amap_row.addWidget(amap_cfg_label)
        amap_cfg_btn = QPushButton("打开配置文件")
        amap_cfg_btn.setToolTip("编辑 amap_api_config.json，填写 api_key（与豆包配置方式相同）")
        amap_cfg_btn.clicked.connect(self._open_amap_config_file)
        amap_row.addWidget(amap_cfg_btn)
        amap_row.addStretch()
        geo_layout.addRow("", amap_row)
        
        # 地址输入和查询
        location_row = QHBoxLayout()
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("输入地址，如：厦门大学翔安校区")
        self.location_input.textChanged.connect(self._on_location_text_changed)
        self.location_input.editingFinished.connect(self._query_location_gps)
        location_row.addWidget(self.location_input, 1)
        
        self.location_query_btn = QPushButton("查询")
        self.location_query_btn.clicked.connect(self._query_location_gps)
        self.location_query_btn.setToolTip("点击查询GPS坐标")
        location_row.addWidget(self.location_query_btn)
        geo_layout.addRow("地址:", location_row)
        
        # 纬度
        self.lat_input = QLineEdit()
        self.lat_input.setPlaceholderText("例: 31.230416")
        self.lat_input.setText(str(self.config['gps_latitude']))
        self.lat_input.setMaxLength(12)
        geo_layout.addRow("纬度:", self.lat_input)

        # 经度
        self.lon_input = QLineEdit()
        self.lon_input.setPlaceholderText("例: 121.473682")
        self.lon_input.setText(str(self.config['gps_longitude']))
        self.lon_input.setMaxLength(13)
        geo_layout.addRow("经度:", self.lon_input)
        
        # 省市显示（只读）
        self.province_city_display = QLineEdit()
        self.province_city_display.setReadOnly(True)
        self.province_city_display.setPlaceholderText("查询后自动显示省市")
        geo_layout.addRow("省市:", self.province_city_display)

        self.gps_gpx_group = QGroupBox("GPX 轨迹与时间（主流程 GPX 模式 / 下方独立按钮）")
        gpx_form = QFormLayout()
        gpx_form.setSpacing(8)

        gpx_row = QHBoxLayout()
        self.gpx_list = QListWidget()
        self.gpx_list.setMaximumHeight(84)
        self.gpx_list.setToolTip(
            "可添加多个 GPX（分段记录），生成地图与 GPS 匹配时按时间合并"
        )
        gpx_add_btn = QPushButton("添加 GPX...")
        gpx_add_btn.clicked.connect(self._add_gpx_files)
        gpx_remove_btn = QPushButton("移除")
        gpx_remove_btn.clicked.connect(self._remove_selected_gpx)
        gpx_merge_btn = QPushButton("合并为文件...")
        gpx_merge_btn.setToolTip("将列表中的 GPX 合并保存为一个文件")
        gpx_merge_btn.clicked.connect(self._merge_gpx_files_dialog)
        gpx_btn_row = QHBoxLayout()
        gpx_btn_row.addWidget(gpx_add_btn)
        gpx_btn_row.addWidget(gpx_remove_btn)
        gpx_btn_row.addWidget(gpx_merge_btn)
        gpx_btn_row.addStretch(1)
        gpx_form.addRow("GPX 轨迹:", self.gpx_list)
        gpx_form.addRow("", gpx_btn_row)

        self.gpx_match_exif_tz_combo = self._create_timezone_combo(
            self._config_gpx_match_exif_tz()
        )
        self.gpx_match_exif_tz_combo.setToolTip(
            "EXIF DateTimeOriginal 所对应的 IANA 时区（可输入搜索，如 Asia/Shanghai）。\n"
            "与 GPX 时区一起换算到 UTC 后再匹配。"
        )
        gpx_form.addRow("EXIF 时区:", self.gpx_match_exif_tz_combo)

        self.gpx_match_gpx_tz_combo = self._create_timezone_combo(
            self._config_gpx_match_gpx_tz()
        )
        self.gpx_match_gpx_tz_combo.setToolTip(
            "GPX <time> 所对应的 IANA 时区（默认 UTC）。"
        )
        gpx_form.addRow("GPX 时区:", self.gpx_match_gpx_tz_combo)

        self.gpx_apply_screened_checkbox = QCheckBox(
            "写入到输出目录 Screened_images（否则写入图片文件夹）"
        )
        self.gpx_apply_screened_checkbox.setChecked(
            self.config.get("gpx_apply_to_screened", True)
        )
        gpx_form.addRow("", self.gpx_apply_screened_checkbox)

        self.gpx_apply_btn = QPushButton("批量写入照片 GPS")
        self.gpx_apply_btn.setToolTip(
            "不依赖「开始处理」：对所选文件夹批量写入 GPS。\n"
            "写入方式与上方「写入方式」一致：\n"
            "· 指定地点统一写入 → 使用上方经纬度\n"
            "· GPX 按拍摄时间匹配 → 使用 GPX 与 EXIF 时间（上方时区）"
        )
        self.gpx_apply_btn.clicked.connect(self._apply_batch_gps_to_photos)
        gpx_form.addRow("", self.gpx_apply_btn)

        self.gps_gpx_group.setLayout(gpx_form)
        geo_layout.addRow(self.gps_gpx_group)

        self._on_gps_write_mode_changed()
        
        geo_group.setLayout(geo_layout)
        layout.addWidget(geo_card)
        
        # ═════ 连拍处理（可收起；主流程勾选在标题栏）═════
        self.enable_burst_detection_checkbox = QCheckBox("加入主流程")
        self.enable_burst_detection_checkbox.setToolTip(
            "勾选：对「图片文件夹」做连拍分组与筛选，并写入 Screened_images。\n"
            "不勾选：跳过连拍，后续从已有 Screened_images 读取。"
        )
        self.enable_burst_detection_checkbox.setChecked(
            self.config.get("enable_burst_detection", True)
        )
        self.enable_burst_detection_checkbox.toggled.connect(
            self._on_burst_detection_toggled
        )
        self._style_flow_header_checkbox(self.enable_burst_detection_checkbox)

        process_card, process_group = self._create_collapsible_card(
            "📷 连拍处理",
            "burst",
            header_widgets=[self.enable_burst_detection_checkbox],
        )
        process_layout = QFormLayout()
        process_layout.setSpacing(8)
        process_layout.setContentsMargins(12, 10, 12, 12)
        
        # 时间阈值
        self.time_threshold_input = QDoubleSpinBox()
        self.time_threshold_input.setRange(0.1, 10.0)
        self.time_threshold_input.setSingleStep(0.1)
        self.time_threshold_input.setValue(self.config['time_threshold'])
        self.time_threshold_input.setSuffix(" 秒")
        process_layout.addRow("连拍时间阈值:", self.time_threshold_input)
        
        self.burst_keep_ratio_input = QDoubleSpinBox()
        self.burst_keep_ratio_input.setRange(0.05, 1.0)
        self.burst_keep_ratio_input.setSingleStep(0.05)
        self.burst_keep_ratio_input.setDecimals(2)
        self.burst_keep_ratio_input.setValue(
            float(self.config.get("burst_keep_ratio", 0.2))
        )
        self.burst_keep_ratio_input.setToolTip(
            "同一连拍组内：保留张数 = 组内总张数×比例，再与「最少保留」取较大值、不超过组大小。"
            "快速模式也按全组总张数计算，不会先抽 1/3 再乘这个比例。"
        )
        process_layout.addRow("连拍保留比例:", self.burst_keep_ratio_input)
        
        self.burst_keep_min_input = QSpinBox()
        self.burst_keep_min_input.setRange(1, 50)
        self.burst_keep_min_input.setValue(
            int(self.config.get("burst_keep_min", self.config.get("keep_top_n", 2)))
        )
        self.burst_keep_min_input.setToolTip(
            "每组至少尝试保留的张数；与比例取 max 后不超过该组总张数"
        )
        process_layout.addRow("连拍最少保留:", self.burst_keep_min_input)

        # 对焦/面积标准化权重（总分 10 分）
        self.focus_score_weight_input = QDoubleSpinBox()
        self.focus_score_weight_input.setRange(0.0, 10.0)
        self.focus_score_weight_input.setSingleStep(0.5)
        self.focus_score_weight_input.setDecimals(1)
        self.focus_score_weight_input.setValue(
            float(self.config.get("focus_score_weight", 9.0))
        )
        self.focus_score_weight_input.setToolTip(
            "对焦评分在组内标准化(0-10)后的权重，默认 9（总分 10 中占 9 分）"
        )
        self.area_score_weight_input = QDoubleSpinBox()
        self.area_score_weight_input.setRange(0.0, 10.0)
        self.area_score_weight_input.setSingleStep(0.5)
        self.area_score_weight_input.setDecimals(1)
        self.area_score_weight_input.setValue(
            float(self.config.get("area_score_weight", 1.0))
        )
        self.area_score_weight_input.setToolTip(
            "鸟体面积在组内标准化(0-10)后的权重，默认 1（总分 10 中占 1 分）"
        )

        from PyQt5.QtWidgets import QHBoxLayout as _QHL
        _ratio_row = _QHL()
        _ratio_row.addWidget(self.focus_score_weight_input)
        _ratio_row.addWidget(QLabel(":"))
        _ratio_row.addWidget(self.area_score_weight_input)
        process_layout.addRow("对焦:面积 权重:", _ratio_row)
        
        # 启用鸟体检测
        self.use_bird_detection_checkbox = QCheckBox("启用鸟体检测")
        self.use_bird_detection_checkbox.setChecked(self.config['use_bird_detection'])
        self.use_bird_detection_checkbox.toggled.connect(
            self._on_bird_detection_toggled
        )
        process_layout.addRow("", self.use_bird_detection_checkbox)

        # 启用鸟眼检测（依赖鸟体检测）
        self.use_eye_detection_checkbox = QCheckBox("启用鸟眼检测（需先启用鸟体检测）")
        self.use_eye_detection_checkbox.setChecked(
            self.config.get("use_eye_detection", False)
        )
        process_layout.addRow("", self.use_eye_detection_checkbox)
        
        # 快速模式
        self.use_fast_mode_checkbox = QCheckBox("使用快速模式")
        self.use_fast_mode_checkbox.setChecked(self.config['use_fast_mode'])
        self.use_fast_mode_checkbox.setToolTip(
            "只对部分照片跑鸟检/对焦以加速。候选张数按「全组保留数」抽取（可 2 倍余量），"
            "保留比例仍相对本组总张数（例如 0.1 即约 10%），不会变成 1/3 再乘 0.1。"
        )
        process_layout.addRow("", self.use_fast_mode_checkbox)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_card)
        self._on_burst_detection_toggled(
            self.enable_burst_detection_checkbox.isChecked()
        )
        self._on_bird_detection_toggled(self.use_bird_detection_checkbox.isChecked())
        
        # ═════ 物种识别（可收起；主流程勾选在标题栏）═════
        self.enable_species_checkbox = QCheckBox("加入主流程")
        self.enable_species_checkbox.setToolTip(
            "勾选：对筛选后照片做物种识别并按鸟体裁剪归档至分类目录。\n"
            "不勾选：跳过物种识别与裁剪归档。"
        )
        self.enable_species_checkbox.setChecked(
            self.config.get("enable_species_detection", True)
        )
        self._style_flow_header_checkbox(self.enable_species_checkbox)

        species_card, species_group = self._create_collapsible_card(
            "🦅 物种识别",
            "species",
            header_widgets=[self.enable_species_checkbox],
        )
        species_layout = QFormLayout()
        species_layout.setSpacing(8)
        species_layout.setContentsMargins(12, 10, 12, 12)
        
        # 模型模式选择 - 使用radiobutton
        model_layout = QHBoxLayout()
        model_layout.setSpacing(12)
        
        self.local_model_radio = QRadioButton("本地模型 (离线)")
        self.local_model_radio.setChecked(self.config['use_local_model'])
        self.local_model_radio.toggled.connect(lambda checked: self._on_model_mode_changed(0 if checked else 1))
        model_layout.addWidget(self.local_model_radio)
        
        self.doubao_model_radio = QRadioButton(
            "豆包 Seed 2.0 视觉（在线，约 1 张/秒）"
        )
        self.doubao_model_radio.setChecked(not self.config['use_local_model'])
        self.doubao_model_radio.toggled.connect(lambda checked: self._on_model_mode_changed(1 if checked else 0))
        model_layout.addWidget(self.doubao_model_radio)
        
        species_layout.addRow("物种识别模式:", model_layout)

        self.local_species_model_combo = QComboBox()
        self.local_species_model_combo.addItem(
            "ResNet34（推荐，浦口基准约 87.7%）",
            LOCAL_SPECIES_MODEL_RESNET34,
        )
        self.local_species_model_combo.addItem(
            "EfficientNet-B0（SuperBirdID birdid2024，约 65%）",
            LOCAL_SPECIES_MODEL_EFFICIENTNET,
        )
        self.local_species_model_combo.setToolTip(
            "仅在选择「本地模型」时生效。\n"
            "推荐 ResNet34（浦口基准约 88%）；EfficientNet 为 SuperBirdID JIT 权重，"
            "须使用 BGR+T=0.6 推理（已内置）。\n"
            "权重：models/bird_iden_efficient_b0.pt（与 birdid2024 解密文件相同）。"
        )
        self.local_species_model_combo.currentIndexChanged.connect(
            self._on_local_species_model_changed
        )
        species_layout.addRow("本地物种模型:", self.local_species_model_combo)

        self.species_geo_mode_combo = QComboBox()
        self.species_geo_mode_combo.addItem(
            "自动（有 GPS/省份用省名单，否则全国）", SPECIES_GEO_MODE_AUTO
        )
        self.species_geo_mode_combo.addItem(
            "全国名单", SPECIES_GEO_MODE_CHINA
        )
        self.species_geo_mode_combo.addItem(
            "仅省份名单（无省信息时回退全国）", SPECIES_GEO_MODE_PROVINCE
        )
        self.species_geo_mode_combo.addItem(
            "不限制地理", SPECIES_GEO_MODE_NONE
        )
        self.species_geo_mode_combo.setToolTip(
            "在模型 top10 候选上按地理名单筛选。\n"
            "浦口测试集：自动/全国/省份约 87.7% top1；不限制约 75.1%。\n"
            "省份来自照片 EXIF GPS 或左侧「地理位置」中的省/市。"
        )
        self.species_geo_mode_combo.currentIndexChanged.connect(
            self._on_species_geo_mode_changed
        )
        species_layout.addRow("地理约束:", self.species_geo_mode_combo)
        
        # 豆包API配置文件链接
        config_layout = QHBoxLayout()
        config_label = QLabel("豆包API配置:")
        config_link = QPushButton("打开配置文件")
        config_link.clicked.connect(self._open_config_file)
        config_layout.addWidget(config_label)
        config_layout.addWidget(config_link)
        species_layout.addRow("", config_layout)

        # 未知种类阈值（可选）：不勾选时不对 top10 做置信度初筛，仅按地理规则筛选
        min_species_row = QHBoxLayout()
        self.min_species_threshold_enable_checkbox = QCheckBox("启用")
        self.min_species_threshold_enable_checkbox.setChecked(
            self.config.get('species_conf_threshold_enabled', False)
        )
        self.min_species_conf_input = QDoubleSpinBox()
        self.min_species_conf_input.setRange(0.0, 1.0)
        self.min_species_conf_input.setSingleStep(0.05)
        self.min_species_conf_input.setDecimals(2)
        self.min_species_conf_input.setValue(
            float(self.config.get('min_species_accept_confidence', 0.5))
        )
        self.min_species_conf_input.setEnabled(
            self.min_species_threshold_enable_checkbox.isChecked()
        )
        self.min_species_conf_input.setToolTip(
            "勾选「启用」后：在地理规则完成之后，若**顶一**置信度仍低于设定值，则归为未知。\n"
            "浦口 ResNet34 基准：启用 ≥0.5 阈值会使 top1 准确率降至约 65%，建议保持关闭。\n"
            "本地仍会应用名单外 0.8 等地理规则；豆包名单外须 >0.75。"
        )
        self.min_species_threshold_enable_checkbox.setToolTip(
            "默认关闭（推荐）：浦口测试集 ResNet34 在关闭时约 87.7% top1；"
            "开启后置信度门槛会显著增加「未知种类」并降低准确率。"
        )
        self.min_species_threshold_enable_checkbox.toggled.connect(
            self.min_species_conf_input.setEnabled
        )
        min_species_row.addWidget(self.min_species_threshold_enable_checkbox)
        min_species_row.addWidget(self.min_species_conf_input)
        min_species_row.addStretch()
        species_layout.addRow("未知种类阈值(可选):", min_species_row)

        # ---- 图片清洗（切割后识别前 / 分类目录）----
        self.image_clean_before_species_checkbox = QCheckBox(
            "主流程：切割后、识别前清洗切割图"
        )
        self.image_clean_before_species_checkbox.setChecked(
            bool(self.config.get("enable_image_clean_before_species", False))
        )
        self.image_clean_before_species_checkbox.setToolTip(
            "勾选后，先从大图检出并切出每只鸟，再对切割图清洗"
            "（去掉失焦、无鸟、重复的个体），最后才识别鸟种。\n"
            "不会删除 Screened_images 里的大图。"
        )
        species_layout.addRow("", self.image_clean_before_species_checkbox)

        self.image_clean_no_bird_checkbox = QCheckBox("去除未检出鸟体的图")
        self.image_clean_no_bird_checkbox.setChecked(
            bool(self.config.get("image_clean_remove_no_bird", True))
        )
        self.image_clean_no_bird_checkbox.setToolTip(
            "用鸟体检测模型扫描；未检出鸟体的图片将被删除。"
        )
        species_layout.addRow("", self.image_clean_no_bird_checkbox)

        self.image_clean_blurry_checkbox = QCheckBox("去除鸟体模糊的图")
        self.image_clean_blurry_checkbox.setChecked(
            bool(self.config.get("image_clean_remove_blurry", True))
        )
        species_layout.addRow("", self.image_clean_blurry_checkbox)

        clean_blur_row = QHBoxLayout()
        self.image_clean_clarity_slider = QSlider(Qt.Horizontal)
        self.image_clean_clarity_slider.setRange(0, 100)
        self.image_clean_clarity_slider.setValue(
            int(self.config.get("image_clean_min_clarity", 35))
        )
        self.image_clean_clarity_slider.setToolTip(
            "最低清晰度（0~100）。数值越大越严格，删除的模糊图越多。\n"
            "基于鸟体区域 Laplacian 清晰度；多鸟时取最靠近画面中央的个体。\n"
            "推荐 25~45。"
        )
        self.image_clean_clarity_label = QLabel(
            f"{self.image_clean_clarity_slider.value()}"
        )
        self.image_clean_clarity_label.setMinimumWidth(28)
        self.image_clean_clarity_slider.valueChanged.connect(
            lambda v: self.image_clean_clarity_label.setText(str(v))
        )
        clean_blur_row.addWidget(self.image_clean_clarity_slider, 1)
        clean_blur_row.addWidget(self.image_clean_clarity_label)
        species_layout.addRow("模糊阈值:", clean_blur_row)
        self.image_clean_blurry_checkbox.toggled.connect(
            self.image_clean_clarity_slider.setEnabled
        )
        self.image_clean_clarity_slider.setEnabled(
            self.image_clean_blurry_checkbox.isChecked()
        )

        self.image_clean_dedupe_checkbox = QCheckBox("去除高度重复的鸟图")
        self.image_clean_dedupe_checkbox.setChecked(
            bool(self.config.get("image_clean_dedupe", True))
        )
        self.image_clean_dedupe_checkbox.setToolTip(
            "同一文件夹内按感知哈希去重，保留更清晰的一张。"
        )
        species_layout.addRow("", self.image_clean_dedupe_checkbox)

        clean_dup_row = QHBoxLayout()
        self.image_clean_dup_slider = QSlider(Qt.Horizontal)
        self.image_clean_dup_slider.setRange(50, 100)
        self.image_clean_dup_slider.setValue(
            int(self.config.get("image_clean_dup_similarity", 92))
        )
        self.image_clean_dup_slider.setToolTip(
            "重复相似度（50~100%）。越高越严格，删除的近似重复图越多。\n"
            "推荐 88~96。"
        )
        self.image_clean_dup_label = QLabel(
            f"{self.image_clean_dup_slider.value()}%"
        )
        self.image_clean_dup_label.setMinimumWidth(36)
        self.image_clean_dup_slider.valueChanged.connect(
            lambda v: self.image_clean_dup_label.setText(f"{v}%")
        )
        clean_dup_row.addWidget(self.image_clean_dup_slider, 1)
        clean_dup_row.addWidget(self.image_clean_dup_label)
        species_layout.addRow("去重阈值:", clean_dup_row)
        self.image_clean_dedupe_checkbox.toggled.connect(
            self.image_clean_dup_slider.setEnabled
        )
        self.image_clean_dup_slider.setEnabled(
            self.image_clean_dedupe_checkbox.isChecked()
        )

        clean_folder_row = QHBoxLayout()
        self.image_clean_folder_input = QLineEdit()
        self.image_clean_folder_input.setText(
            self.config.get("image_clean_folder", "")
        )
        self.image_clean_folder_input.setPlaceholderText(
            "留空则使用分类归档目录（crop / classification）"
        )
        clean_folder_btn = QPushButton("浏览...")
        clean_folder_btn.clicked.connect(
            lambda: self._select_folder("image_clean_folder")
        )
        clean_folder_row.addWidget(self.image_clean_folder_input, 1)
        clean_folder_row.addWidget(clean_folder_btn)
        species_layout.addRow("清洗目录:", clean_folder_row)

        self.image_clean_run_btn = QPushButton("清洗选定目录")
        self.image_clean_run_btn.setToolTip(
            "立即清洗上方目录（默认分类归档目录），按勾选项删除"
            "未检出鸟体 / 模糊 / 高度重复图片。不运行完整主流程。"
        )
        self.image_clean_run_btn.clicked.connect(self._run_image_clean_batch)
        species_layout.addRow("", self.image_clean_run_btn)

        species_group.setLayout(species_layout)
        layout.addWidget(species_card)

        # ═════ 水印生成（可收起；主流程勾选在标题栏）═════
        self.enable_watermark_checkbox = QCheckBox("加入主流程")
        self.enable_watermark_checkbox.setChecked(
            self.config.get("enable_watermark_generation", False)
        )
        self.enable_watermark_checkbox.setToolTip(
            "勾选后，「开始处理」流程末尾自动生成水印。\n"
            "与卡片内「预览」「单独批量水印生成」无关。"
        )
        self._style_flow_header_checkbox(self.enable_watermark_checkbox)

        wm_card, wm_group = self._create_collapsible_card(
            "🖼 水印生成",
            "watermark",
            header_widgets=[self.enable_watermark_checkbox],
        )
        wm_layout = QFormLayout()
        wm_layout.setSpacing(8)
        wm_layout.setContentsMargins(12, 10, 12, 12)

        wm_in_row = QHBoxLayout()
        self.wm_input_folder_input = QLineEdit()
        self.wm_input_folder_input.setText(self.config.get("watermark_input_folder", ""))
        self.wm_input_folder_input.setPlaceholderText("可选：指定输入目录（支持多级子目录）")
        wm_in_btn = QPushButton("浏览...")
        wm_in_btn.clicked.connect(lambda: self._select_folder("watermark_input_folder"))
        wm_in_row.addWidget(self.wm_input_folder_input, 1)
        wm_in_row.addWidget(wm_in_btn)
        wm_layout.addRow("指定相片文件夹:", wm_in_row)

        wm_out_row = QHBoxLayout()
        self.wm_output_folder_input = QLineEdit()
        self.wm_output_folder_input.setText(
            self.config.get("watermark_output_folder", "./watermarked")
        )
        wm_out_btn = QPushButton("浏览...")
        wm_out_btn.clicked.connect(lambda: self._select_folder("watermark_output_folder"))
        wm_out_row.addWidget(self.wm_output_folder_input, 1)
        wm_out_row.addWidget(wm_out_btn)
        wm_layout.addRow("水印输出文件夹:", wm_out_row)

        wm_logo_row = QHBoxLayout()
        self.wm_logo_input = QLineEdit()
        self.wm_logo_input.setText(self.config.get("wm_logo_path", ""))
        self.wm_logo_input.setPlaceholderText("可选：签名 logo 图片路径（png/jpg）")
        wm_logo_btn = QPushButton("选择文件...")
        wm_logo_btn.clicked.connect(self._select_wm_logo_file)
        wm_logo_row.addWidget(self.wm_logo_input, 1)
        wm_logo_row.addWidget(wm_logo_btn)
        wm_layout.addRow("签名 Logo:", wm_logo_row)

        self.wm_logo_width_ratio_input = QDoubleSpinBox()
        self.wm_logo_width_ratio_input.setRange(0.05, 0.80)
        self.wm_logo_width_ratio_input.setSingleStep(0.01)
        self.wm_logo_width_ratio_input.setDecimals(2)
        self.wm_logo_width_ratio_input.setValue(
            float(self.config.get("wm_logo_width_ratio", 0.30))
        )
        self.wm_logo_width_ratio_input.setSuffix(" × 图片宽")
        self.wm_logo_width_ratio_input.setToolTip(
            "控制 Logo 宽度占图片宽度比例，默认 0.30（30%）。"
        )
        wm_layout.addRow("Logo 宽度占比:", self.wm_logo_width_ratio_input)

        self.wm_style_combo = QComboBox()
        self.wm_style_combo.addItem("外框 + 底栏文字 + 图内签名", "frame")
        self.wm_style_combo.addItem("无外框 · 图内签名 + 竖线标签", "inline")
        _ws = str(self.config.get("wm_watermark_style", "frame") or "frame")
        self.wm_style_combo.setCurrentIndex(1 if _ws == "inline" else 0)
        self.wm_style_combo.setToolTip(
            "外框模式：白边灰线 + 底部栏（物种/地点/日期 | 相机）+ 图底中间签名。\n"
            "无外框模式：不扩画布；图内中下方为「签名 | 竖线 | 两行标签」"
            "（上：物种；下：GPS 城市 + 人工地点），颜色与签名剪影一致。"
        )
        wm_layout.addRow("水印布局:", self.wm_style_combo)

        self.wm_location_checkbox = QCheckBox("显示地理位置")
        self.wm_location_checkbox.setChecked(self.config.get("wm_enable_location", True))
        wm_layout.addRow("", self.wm_location_checkbox)

        self.wm_use_gps_city_checkbox = QCheckBox("优先使用图片 GPS 自动定位城市")
        self.wm_use_gps_city_checkbox.setChecked(self.config.get("wm_use_gps_city", True))
        wm_layout.addRow("", self.wm_use_gps_city_checkbox)

        self.wm_location_text_input = QLineEdit()
        self.wm_location_text_input.setText(self.config.get("wm_location_text", ""))
        self.wm_location_text_input.setPlaceholderText("人工输入地点（填入后优先使用）")
        wm_layout.addRow("人工地点:", self.wm_location_text_input)

        self.wm_date_checkbox = QCheckBox("显示拍照日期")
        self.wm_date_checkbox.setChecked(self.config.get("wm_enable_date", True))
        wm_layout.addRow("", self.wm_date_checkbox)

        self.wm_species_checkbox = QCheckBox("显示物种名")
        self.wm_species_checkbox.setChecked(self.config.get("wm_enable_species", True))
        self.wm_species_checkbox.setToolTip(
            "指定相片文件夹时：使用图片所在目录名作为物种名；\n"
            "未指定时：使用归档 ROI 图片目录名。"
        )
        wm_layout.addRow("", self.wm_species_checkbox)

        self.wm_camera_checkbox = QCheckBox("显示相机参数")
        self.wm_camera_checkbox.setChecked(self.config.get("wm_enable_camera", True))
        wm_layout.addRow("", self.wm_camera_checkbox)

        self.wm_random_per_species_checkbox = QCheckBox(
            "每物种目录随机抽若干张（不勾选则全部生成）"
        )
        self.wm_random_per_species_checkbox.setChecked(
            bool(self.config.get("wm_random_per_species", False))
        )
        self.wm_random_per_species_checkbox.setToolTip(
            "按图片所在父目录（通常为物种文件夹）分组，每组随机抽取指定张数生成水印；\n"
            "未勾选时对目录内全部图片生成水印。主流程与「单独批量水印生成」均生效。"
        )
        wm_layout.addRow("", self.wm_random_per_species_checkbox)

        self.wm_random_per_species_count = QSpinBox()
        self.wm_random_per_species_count.setRange(1, 999)
        self.wm_random_per_species_count.setValue(
            max(1, int(self.config.get("wm_random_per_species_count", 3)))
        )
        self.wm_random_per_species_count.setSuffix(" 张/物种目录")
        self.wm_random_per_species_count.setToolTip(
            "每个物种目录最多随机抽取的张数；目录内不足该数时全部保留。"
        )
        wm_layout.addRow("抽样张数:", self.wm_random_per_species_count)
        self.wm_random_per_species_checkbox.toggled.connect(
            self.wm_random_per_species_count.setEnabled
        )
        self.wm_random_per_species_count.setEnabled(
            self.wm_random_per_species_checkbox.isChecked()
        )

        # 自动曝光（水印前；基于鸟体测光，避免剪影）
        self.wm_ai_exposure_checkbox = QCheckBox("水印前自动曝光")
        self.wm_ai_exposure_checkbox.setChecked(
            self.config.get("wm_enable_ai_exposure", False)
        )
        self.wm_ai_exposure_checkbox.setToolTip(
            "基于鸟体检测测光，自动调整曝光避免剪影。\n流水线顺序：自动曝光 → AI 降噪 → AI 锐化。"
        )
        wm_layout.addRow("", self.wm_ai_exposure_checkbox)

        wm_ai_ex_row = QHBoxLayout()
        self.wm_ai_exposure_slider = QSlider(Qt.Horizontal)
        self.wm_ai_exposure_slider.setRange(0, 100)
        self.wm_ai_exposure_slider.setSingleStep(5)
        self.wm_ai_exposure_slider.setValue(
            int(float(self.config.get("wm_ai_exposure_strength", 1.0)) * 100)
        )
        self.wm_ai_exposure_value_label = QLabel(
            f"{self.wm_ai_exposure_slider.value() / 100:.2f}"
        )
        self.wm_ai_exposure_value_label.setMinimumWidth(34)
        self.wm_ai_exposure_slider.valueChanged.connect(
            lambda v: self.wm_ai_exposure_value_label.setText(f"{v / 100:.2f}")
        )
        self.wm_ai_exposure_slider.setToolTip("0=原图，1=完全调整；推荐 0.5~1.0。")
        wm_ai_ex_row.addWidget(self.wm_ai_exposure_slider, 1)
        wm_ai_ex_row.addWidget(self.wm_ai_exposure_value_label)
        wm_layout.addRow("曝光强度:", wm_ai_ex_row)

        # AI 降噪（水印前；Real-ESRGAN 或 NAFNet 可选）
        self.wm_ai_denoise_checkbox = QCheckBox("水印前 AI 降噪")
        self.wm_ai_denoise_checkbox.setChecked(
            self.config.get("wm_enable_ai_denoise", False)
        )
        wm_layout.addRow("", self.wm_ai_denoise_checkbox)

        self.wm_ai_denoise_model_combo = QComboBox()
        self.wm_ai_denoise_model_combo.addItem("Real-ESRGAN", "realesrgan")
        self.wm_ai_denoise_model_combo.addItem("NAFNet", "nafnet")
        _wm_dm = str(self.config.get("wm_ai_denoise_model", "realesrgan") or "realesrgan")
        _wm_di = self.wm_ai_denoise_model_combo.findData(_wm_dm)
        self.wm_ai_denoise_model_combo.setCurrentIndex(_wm_di if _wm_di >= 0 else 0)
        wm_layout.addRow("降噪模型:", self.wm_ai_denoise_model_combo)

        wm_ai_dn_row = QHBoxLayout()
        self.wm_ai_denoise_slider = QSlider(Qt.Horizontal)
        self.wm_ai_denoise_slider.setRange(0, 100)
        self.wm_ai_denoise_slider.setSingleStep(5)
        self.wm_ai_denoise_slider.setValue(
            int(float(self.config.get("wm_ai_denoise_strength", 0.5)) * 100)
        )
        self.wm_ai_denoise_value_label = QLabel(
            f"{self.wm_ai_denoise_slider.value() / 100:.2f}"
        )
        self.wm_ai_denoise_value_label.setMinimumWidth(34)
        self.wm_ai_denoise_slider.valueChanged.connect(
            lambda v: self.wm_ai_denoise_value_label.setText(f"{v / 100:.2f}")
        )
        wm_ai_dn_row.addWidget(self.wm_ai_denoise_slider, 1)
        wm_ai_dn_row.addWidget(self.wm_ai_denoise_value_label)
        wm_layout.addRow("降噪强度:", wm_ai_dn_row)

        # AI 锐化（OmniSR；仅锐化不放大）
        self.wm_ai_sharpen_checkbox = QCheckBox("水印前 AI 锐化")
        self.wm_ai_sharpen_checkbox.setChecked(
            self.config.get("wm_enable_ai_sharpen", False)
        )
        wm_layout.addRow("", self.wm_ai_sharpen_checkbox)

        wm_ai_sh_row = QHBoxLayout()
        self.wm_ai_sharpen_slider = QSlider(Qt.Horizontal)
        self.wm_ai_sharpen_slider.setRange(0, 100)
        self.wm_ai_sharpen_slider.setSingleStep(5)
        self.wm_ai_sharpen_slider.setValue(
            int(float(self.config.get("wm_ai_sharpen_strength", 0.5)) * 100)
        )
        self.wm_ai_sharpen_value_label = QLabel(
            f"{self.wm_ai_sharpen_slider.value() / 100:.2f}"
        )
        self.wm_ai_sharpen_value_label.setMinimumWidth(34)
        self.wm_ai_sharpen_slider.valueChanged.connect(
            lambda v: self.wm_ai_sharpen_value_label.setText(f"{v / 100:.2f}")
        )
        self.wm_ai_sharpen_slider.setToolTip("0=原图，1=完全锐化；推荐 0.3~0.5。")
        wm_ai_sh_row.addWidget(self.wm_ai_sharpen_slider, 1)
        wm_ai_sh_row.addWidget(self.wm_ai_sharpen_value_label)
        wm_layout.addRow("锐化强度:", wm_ai_sh_row)

        # 降噪未启用时禁用降噪模型与强度控件
        self.wm_ai_denoise_checkbox.toggled.connect(
            self._update_wm_ai_denoise_enabled
        )
        self._update_wm_ai_denoise_enabled()
        # 锐化未启用时禁用锐化强度滑块
        self.wm_ai_sharpen_checkbox.toggled.connect(
            self.wm_ai_sharpen_slider.setEnabled
        )
        self.wm_ai_sharpen_slider.setEnabled(
            self.wm_ai_sharpen_checkbox.isChecked()
        )
        # 自动曝光未启用时禁用曝光强度滑块
        self.wm_ai_exposure_checkbox.toggled.connect(
            self.wm_ai_exposure_slider.setEnabled
        )
        self.wm_ai_exposure_slider.setEnabled(
            self.wm_ai_exposure_checkbox.isChecked()
        )

        wm_preview_row = QHBoxLayout()
        wm_preview_btn = QPushButton("预览一张效果")
        wm_preview_btn.clicked.connect(self._preview_watermark_one)
        wm_preview_row.addWidget(wm_preview_btn)
        self.wm_run_btn = QPushButton("单独批量水印生成")
        self.wm_run_btn.setToolTip(
            "仅在本卡片内批量生成水印，不运行「开始处理」主流程；"
            "与上方「主流程自动水印」无关。"
        )
        self.wm_run_btn.clicked.connect(self._run_watermark_batch)
        wm_preview_row.addWidget(self.wm_run_btn)
        wm_burst_btn = QPushButton("动图生成")
        wm_burst_btn.setToolTip(
            "将连拍合成为 WebP 动图或 MP4 视频（白平衡、按裁剪框自动曝光、定点/跟踪裁剪），"
            "后续帧用鸟体检测 + 卡尔曼跟踪标定点；帧率按每秒几张设置（默认 2）。"
            "图片列表与定位自动保存在相片目录的项目文件中；不支持动图 WebP 的 App 可选用 MP4。"
        )
        wm_burst_btn.clicked.connect(self._open_burst_webp_dialog)
        wm_preview_row.addWidget(wm_burst_btn)
        wm_video_stab_btn = QPushButton("视频裁剪")
        wm_video_stab_btn.setToolTip(
            "视频裁剪与稳定：支持 OpenCV VideoStab 和 MTools AI 算法，"
            "可设置时间范围和空间裁剪区域，消除手持拍摄抖动。"
        )
        wm_video_stab_btn.clicked.connect(self._open_video_stabilize_dialog)
        wm_preview_row.addWidget(wm_video_stab_btn)
        wm_preview_row.addStretch(1)
        wm_layout.addRow("", wm_preview_row)

        wm_group.setLayout(wm_layout)
        layout.addWidget(wm_card)

        # ═════ 轨迹图生成（可收起，默认收起；主流程勾选在标题栏）═════
        self.enable_track_map_auto_checkbox = QCheckBox("加入主流程")
        self.enable_track_map_auto_checkbox.setChecked(
            self.config.get("enable_track_map_auto", False)
        )
        self.enable_track_map_auto_checkbox.setToolTip(
            "勾选后，「开始处理」在物种归档后生成轨迹 PNG 至 reports。\n"
            "与下方「预览」「单独生成」无关。"
        )
        self._style_flow_header_checkbox(self.enable_track_map_auto_checkbox)

        track_card, track_group = self._create_collapsible_card(
            "🗺 轨迹图生成",
            "track_map",
            expanded=False,
            header_widgets=[self.enable_track_map_auto_checkbox],
        )
        track_layout = QFormLayout()
        track_layout.setSpacing(8)
        track_layout.setContentsMargins(12, 10, 12, 12)

        track_hint = QLabel(
            "从分类归档生成行迹与物种分布 PNG（2K 竖屏，需高德 Key）。"
            "图内标题地点优先使用「水印生成 → 人工地点」。"
        )
        track_hint.setWordWrap(True)
        track_hint.setStyleSheet("color: #555; font-size: 9pt;")
        track_layout.addRow(track_hint)

        self.track_map_use_gpx_checkbox = QCheckBox("使用 GPX 轨迹（未选则仅用照片 EXIF GPS）")
        self.track_map_use_gpx_checkbox.setChecked(
            self.config.get("track_map_use_gpx", True)
        )
        self.track_map_use_gpx_checkbox.stateChanged.connect(
            self._on_track_map_use_gpx_changed
        )
        track_layout.addRow("", self.track_map_use_gpx_checkbox)

        self.track_map_use_exif_checkbox = QCheckBox("使用照片 EXIF 中的 GPS")
        self.track_map_use_exif_checkbox.setChecked(
            self.config.get("track_map_use_exif", True)
        )
        self.track_map_use_exif_checkbox.setToolTip(
            "不使用 GPX 时：从鸟图 EXIF 读取 GPS 绘制标记。\n"
            "使用 GPX 时：与 GPX 插值位置接近时优先采用 EXIF GPS。"
        )
        track_layout.addRow("", self.track_map_use_exif_checkbox)
        self._on_track_map_use_gpx_changed(
            Qt.Checked if self.track_map_use_gpx_checkbox.isChecked() else Qt.Unchecked
        )

        self.track_map_source_combo = QComboBox()
        self.track_map_source_combo.addItem("分类归档（物种目录）", "classification")
        self.track_map_source_combo.addItem("Screened_images 筛选图", "screened")
        _tsrc = self.config.get("track_map_photo_source", "classification")
        _tsi = self.track_map_source_combo.findData(_tsrc)
        self.track_map_source_combo.setCurrentIndex(_tsi if _tsi >= 0 else 0)
        track_layout.addRow("鸟图来源:", self.track_map_source_combo)

        track_map_folder_row = QHBoxLayout()
        self.track_map_folder_override_input = QLineEdit()
        self.track_map_folder_override_input.setPlaceholderText(
            "留空则使用上方来源的自动路径；可浏览指定其它目录"
        )
        self.track_map_folder_override_input.setText(
            self.config.get("track_map_photo_folder_override", "")
        )
        track_map_folder_btn = QPushButton("浏览…")
        track_map_folder_btn.clicked.connect(
            self._select_track_map_photo_folder_override
        )
        track_map_folder_row.addWidget(self.track_map_folder_override_input, 1)
        track_map_folder_row.addWidget(track_map_folder_btn)
        track_layout.addRow("鸟图目录:", track_map_folder_row)

        self.track_map_photo_path_label = QLabel("")
        self.track_map_photo_path_label.setWordWrap(True)
        self.track_map_photo_path_label.setStyleSheet("color: #555; font-size: 9pt;")
        track_layout.addRow("当前使用:", self.track_map_photo_path_label)

        self.track_map_source_combo.currentIndexChanged.connect(
            lambda _i: self._refresh_track_map_path_ui()
        )
        self.track_map_folder_override_input.textChanged.connect(
            lambda _t: self._refresh_track_map_path_ui()
        )

        self.track_map_radius_input = QDoubleSpinBox()
        self.track_map_radius_input.setRange(0.1, 100.0)
        self.track_map_radius_input.setDecimals(1)
        self.track_map_radius_input.setSuffix(" km")
        self.track_map_radius_input.setValue(
            float(self.config.get("track_map_radius_km", 1.0))
        )
        self.track_map_radius_input.setToolTip(
            "同物种在此半径内只保留一张代表图（先出现的优先）"
        )
        track_layout.addRow("物种去重半径:", self.track_map_radius_input)

        self.track_map_basemap_combo = QComboBox()
        try:
            from gpx_track.amap_basemap import (
                BASEMAP_STYLE_CHOICES,
                normalize_basemap_style,
            )
            for _sid, _slabel in BASEMAP_STYLE_CHOICES:
                self.track_map_basemap_combo.addItem(f"高德·{_slabel}", _sid)
        except Exception:
            self.track_map_basemap_combo.addItem("高德·标准（默认）", "normal")
            self.track_map_basemap_combo.addItem("高德·无路网卫星", "satellite")
            self.track_map_basemap_combo.addItem("高德·有路网卫星", "satellite_roads")

            def normalize_basemap_style(s):  # type: ignore
                return s or "normal"

        _bm = normalize_basemap_style(
            self.config.get("track_map_basemap_style", "normal")
        )
        _bmi = self.track_map_basemap_combo.findData(_bm)
        self.track_map_basemap_combo.setCurrentIndex(_bmi if _bmi >= 0 else 0)
        self.track_map_basemap_combo.setToolTip(
            "轨迹图底图使用高德地图瓦片（与「地理位置」中 amap_api_config.json 的 api_key 相同）。\n"
            "可选官方主题风格（标准/幻影黑/月光银等）及无路网/有路网卫星影像。\n"
            "生成时需联网；未配置 Key 或加载失败时退回经纬度网格。"
        )
        track_layout.addRow("底图风格:", self.track_map_basemap_combo)

        self.track_map_elevation_checkbox = QCheckBox("同时生成海拔-距离剖面图")
        self.track_map_elevation_checkbox.setChecked(
            self.config.get("track_map_include_elevation", True)
        )
        track_layout.addRow("", self.track_map_elevation_checkbox)

        track_btn_row = QHBoxLayout()
        self.track_preview_btn = QPushButton("预览轨迹图")
        self.track_preview_btn.clicked.connect(
            lambda: self._run_track_map_generation(preview=True)
        )
        self.track_save_btn = QPushButton("单独生成并保存 PNG")
        self.track_save_btn.clicked.connect(
            lambda: self._run_track_map_generation(preview=False)
        )
        track_btn_row.addWidget(self.track_preview_btn)
        track_btn_row.addWidget(self.track_save_btn)
        track_btn_row.addStretch(1)
        track_layout.addRow("", track_btn_row)

        track_group.setLayout(track_layout)
        layout.addWidget(track_card)
        self._refresh_track_map_path_ui()

        # ═════ 观鸟记录导出（可收起；主流程勾选在标题栏）═════
        self.enable_record_export_auto_checkbox = QCheckBox("加入主流程")
        self.enable_record_export_auto_checkbox.setChecked(
            self.config.get("enable_record_export_auto", False)
        )
        self.enable_record_export_auto_checkbox.setToolTip(
            "勾选后，主流程在物种归档后自动导出观鸟记录。"
        )
        self._style_flow_header_checkbox(self.enable_record_export_auto_checkbox)

        export_card, export_group = self._create_collapsible_card(
            "📤 观鸟记录导出",
            "export",
            header_widgets=[self.enable_record_export_auto_checkbox],
        )
        export_layout = QFormLayout()
        export_layout.setSpacing(8)
        export_layout.setContentsMargins(12, 10, 12, 12)

        export_hint = QLabel(
            "从分类归档导出 eBird Checklist .csv 与中国观鸟记录中心鸟种导入 .xls。"
        )
        export_hint.setWordWrap(True)
        export_hint.setStyleSheet("color: #555555; font-size: 9pt;")
        export_layout.addRow(export_hint)

        class_row = QHBoxLayout()
        self.record_export_class_input = QLineEdit()
        self.record_export_class_input.setPlaceholderText(
            "默认使用上方「分类归档」路径（完成物种识别后可用）"
        )
        class_btn = QPushButton("浏览...")
        class_btn.clicked.connect(
            lambda: self._select_folder_into("record_export_classification_folder")
        )
        class_row.addWidget(self.record_export_class_input, 1)
        class_row.addWidget(class_btn)
        export_layout.addRow("分类归档目录:", class_row)

        out_row = QHBoxLayout()
        self.record_export_out_input = QLineEdit()
        self.record_export_out_input.setPlaceholderText(
            "留空则导出到 <输出根>/reports/record_export"
        )
        out_btn = QPushButton("浏览...")
        out_btn.clicked.connect(
            lambda: self._select_folder_into("record_export_output_folder")
        )
        out_row.addWidget(self.record_export_out_input, 1)
        out_row.addWidget(out_btn)
        export_layout.addRow("导出目录:", out_row)

        self.record_export_ebird_checkbox = QCheckBox(
            "eBird Checklist Format（.csv）"
        )
        self.record_export_ebird_checkbox.setChecked(
            self.config.get("record_export_ebird", True)
        )
        self.record_export_ebird_checkbox.setToolTip(
            "按 ebird_checklist_format_template.xls 布局导出逗号分隔 .csv；"
            "每文件一个 checklist（列 C）。网站导入时选 Checklist Format。"
        )
        export_layout.addRow("", self.record_export_ebird_checkbox)

        self.record_export_birdreport_checkbox = QCheckBox(
            "中国观鸟记录中心（鸟种导入.xls，官方两列模版）"
        )
        self.record_export_birdreport_checkbox.setChecked(
            self.config.get("record_export_birdreport", True)
        )
        export_layout.addRow("", self.record_export_birdreport_checkbox)

        self.record_export_country_input = QLineEdit()
        self.record_export_country_input.setText(
            self.config.get("record_export_ebird_country", "CN")
        )
        self.record_export_country_input.setMaxLength(8)
        export_layout.addRow("eBird 国家代码:", self.record_export_country_input)

        self.record_export_state_input = QLineEdit()
        self.record_export_state_input.setText(
            self.config.get("record_export_ebird_state", "FJ")
        )
        self.record_export_state_input.setMaxLength(3)
        self.record_export_state_input.setPlaceholderText("1–3 字符，如 FJ")
        self.record_export_state_input.setToolTip(
            "eBird 省/州代码（不含国家前缀），如福建为 FJ；填 CN-FJ 导出时会自动转为 FJ。"
        )
        export_layout.addRow("eBird 省/州:", self.record_export_state_input)

        self._record_export_individual_time_minutes = float(
            self.config.get("record_export_time_minutes", 120.0) or 120.0
        )
        self._record_export_spatial_km = float(
            self.config.get("record_export_spatial_km", 0.1) or 0.1
        )
        self.record_export_count_individuals_checkbox = QCheckBox(
            "累计只数（计数）"
        )
        self.record_export_count_individuals_checkbox.setChecked(
            self.config.get("record_export_count_individuals", True)
        )
        self.record_export_count_individuals_checkbox.setToolTip(
            "勾选：按「设置」中的分窗时间与距离，将同一物种的连续拍摄视为同一批个体后累计只数；"
            "整次导出合并为一份 eBird/记录中心文件。\n"
            "取消：每个 checklist 内该物种只计 1（仅记录出现）。"
        )
        count_row = QHBoxLayout()
        count_row.addWidget(self.record_export_count_individuals_checkbox)
        count_row.addStretch(1)
        count_row.addWidget(
            self._make_action_link(
                "设置",
                self._open_record_export_count_settings_dialog,
            )
        )
        export_layout.addRow("计数:", count_row)

        portal_row = QHBoxLayout()
        portal_row.addWidget(
            self._make_external_link("ebird上传网页", EBIRD_IMPORT_URL)
        )
        portal_row.addSpacing(16)
        portal_row.addWidget(
            self._make_external_link("中国观鸟记录中心", CHINA_BIRD_RECORD_HOME_URL)
        )
        portal_row.addStretch(1)
        export_layout.addRow("", portal_row)

        export_btn_row = QHBoxLayout()
        self.record_export_btn = QPushButton("导出观鸟记录")
        self.record_export_btn.setToolTip("仅导出观鸟记录，不运行主流程。")
        self.record_export_btn.clicked.connect(self._run_record_export)
        export_btn_row.addWidget(self.record_export_btn)
        export_btn_row.addStretch(1)
        export_layout.addRow("", export_btn_row)

        export_group.setLayout(export_layout)
        layout.addWidget(export_card)
        self.crop_folder_input.textChanged.connect(
            lambda _t: self._refresh_record_export_classification_default()
        )
        self.output_root_input.textChanged.connect(
            lambda _t: self._refresh_record_export_classification_default()
        )
        
        # ═════ 操作按钮区 ═════
        btn_card, btn_group = self._create_card("操作")
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.setContentsMargins(12, 10, 12, 12)
        
        self.start_btn = QPushButton("▶ 开始处理")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E8B57;
                color: white;
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 10pt;
                border: none;
            }
            QPushButton:hover:enabled {
                background-color: #277A4B;
            }
            QPushButton:pressed:enabled {
                background-color: #226A3F;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)
        self.start_btn.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.start_btn, 1)
        
        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                font-weight: bold;
                padding: 6px 14px;
                border-radius: 6px;
                font-size: 10pt;
                border: none;
            }
            QPushButton:hover:enabled {
                background-color: #C0392B;
            }
            QPushButton:pressed:enabled {
                background-color: #A93226;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_processing)
        btn_layout.addWidget(self.stop_btn, 1)
        
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_card)
        
        # 底部信息
        info_label = QLabel(
            "💡 提示：设置「输出根目录」后每次只需换「图片文件夹」；"
            "或留空根目录并手动指定输出与分类路径。"
        )
        info_label.setStyleSheet("color: #666666; font-size: 10pt; margin-top: 4px;")
        layout.addWidget(info_label)
        
        # 品牌水印
        watermark_label = QLabel("Birdy · 鸟图智慧仓储")
        watermark_label.setStyleSheet("color: #E0E0E0; font-size: 9pt; text-align: right;")
        watermark_label.setAlignment(Qt.AlignRight)
        layout.addWidget(watermark_label)
        
        panel.setLayout(layout)
        scroll.setWidget(panel)
        return scroll
    
    def _create_card(self, title: str) -> tuple:
        """创建卡片式分组框"""
        card = QWidget()
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        card.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                border-radius: 8px;
            }
        """)
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)
        
        # 卡片标题
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                background-color: #FFFFFF;
                color: #333333;
                font-weight: bold;
                font-size: 11pt;
                padding: 6px 12px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom: 1px solid #F0F0F0;
            }
        """)
        card_layout.addWidget(title_label)
        
        # 卡片内容容器
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
            }
        """)
        card_layout.addWidget(content_widget)
        
        card.setLayout(card_layout)
        return card, content_widget

    @staticmethod
    def _section_arrow_text(expanded: bool) -> str:
        return "▼" if expanded else "▶"

    @staticmethod
    def _style_flow_header_checkbox(cb: QCheckBox) -> None:
        """标题栏主流程勾选：紧凑样式，悬停显示完整说明。"""
        cb.setStyleSheet(
            "QCheckBox { font-size: 9pt; color: #2E6B4A; font-weight: 600; "
            "spacing: 4px; padding: 0 4px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; }"
        )

    def _create_collapsible_card(
        self,
        title: str,
        section_id: str,
        expanded: Optional[bool] = None,
        header_widgets: Optional[List[QWidget]] = None,
    ) -> tuple:
        """
        可收起卡片：标题栏左侧 ▶/▼ + 标题，右侧可放主流程 QCheckBox（无需展开即可查看是否参与处理）。
        """
        if expanded is None:
            expanded = bool(
                self.config.get(f"ui_section_expanded_{section_id}", True)
            )

        outer = QWidget()
        outer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        outer.setStyleSheet(
            "QWidget#collapsibleCard { background-color: #FFFFFF; border-radius: 8px; }"
        )
        outer.setObjectName("collapsibleCard")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        header_bar = QWidget()
        header_bar.setStyleSheet(
            "background-color: #FFFFFF; border-top-left-radius: 8px; "
            "border-top-right-radius: 8px; border-bottom: 1px solid #F0F0F0;"
        )
        header_hl = QHBoxLayout(header_bar)
        header_hl.setContentsMargins(4, 6, 10, 6)
        header_hl.setSpacing(6)

        toggle_btn = QPushButton(self._section_arrow_text(expanded))
        toggle_btn.setCheckable(True)
        toggle_btn.setChecked(expanded)
        toggle_btn.setFixedWidth(32)
        toggle_btn.setFlat(True)
        toggle_btn.setCursor(Qt.PointingHandCursor)
        toggle_btn.setStyleSheet(
            "QPushButton { border: none; font-size: 11pt; font-weight: bold; "
            "background: transparent; color: #333; }"
            "QPushButton:hover { background-color: #F0F0F0; border-radius: 4px; }"
        )

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "font-weight: bold; font-size: 11pt; color: #333333; background: transparent;"
        )
        title_lbl.setCursor(Qt.PointingHandCursor)

        header_hl.addWidget(toggle_btn, 0)
        header_hl.addWidget(title_lbl, 0)
        header_hl.addStretch(1)
        if header_widgets:
            sep = QLabel("|")
            sep.setStyleSheet("color: #CCC; font-size: 9pt;")
            header_hl.addWidget(sep, 0)
            for w in header_widgets:
                header_hl.addWidget(w, 0)

        body_shell = QWidget()
        body_shell.setStyleSheet("background-color: #FFFFFF;")
        body_shell.setVisible(expanded)
        body_layout = QVBoxLayout(body_shell)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        content_widget = QWidget()
        content_widget.setStyleSheet("background-color: #FFFFFF;")
        body_layout.addWidget(content_widget)

        def _on_toggle(checked: bool) -> None:
            body_shell.setVisible(checked)
            toggle_btn.setText(self._section_arrow_text(checked))
            self.config[f"ui_section_expanded_{section_id}"] = checked

        def _title_clicked(_event) -> None:
            toggle_btn.toggle()

        toggle_btn.toggled.connect(_on_toggle)
        title_lbl.mousePressEvent = _title_clicked  # type: ignore[method-assign]

        outer_layout.addWidget(header_bar)
        outer_layout.addWidget(body_shell)
        if not hasattr(self, "_collapsible_sections"):
            self._collapsible_sections = {}
        self._collapsible_sections[section_id] = (toggle_btn, title, body_shell)
        return outer, content_widget

    def _apply_collapsible_sections_from_config(self) -> None:
        sections = getattr(self, "_collapsible_sections", None)
        if not sections:
            return
        for sid, (toggle_btn, _title, body) in sections.items():
            expanded = bool(self.config.get(f"ui_section_expanded_{sid}", True))
            toggle_btn.blockSignals(True)
            toggle_btn.setChecked(expanded)
            toggle_btn.setText(self._section_arrow_text(expanded))
            body.setVisible(expanded)
            toggle_btn.blockSignals(False)

    def _gui_paths_snapshot(self) -> Dict[str, str]:
        return _config_paths_snapshot(
            output_root_folder=self.output_root_input.text().strip(),
            image_folder=self.image_folder_input.text().strip(),
            output_folder=self.output_folder_input.text().strip(),
            crop_output_folder=self.crop_folder_input.text().strip(),
        )

    def _refresh_track_map_path_ui(self) -> None:
        """刷新轨迹图鸟图来源下拉项文案（含实际路径）与「当前使用」提示。"""
        if not hasattr(self, "track_map_source_combo"):
            return
        paths = self._gui_paths_snapshot()
        cls_p = paths.get("crop_output_folder") or ""
        scr_p = paths.get("screened_images") or ""
        cur = self.track_map_source_combo.currentData()
        self.track_map_source_combo.blockSignals(True)
        self.track_map_source_combo.clear()
        cls_label = f"分类归档 — {cls_p}" if cls_p else "分类归档（物种目录）"
        scr_label = f"Screened — {scr_p}" if scr_p else "Screened_images 筛选图"
        self.track_map_source_combo.addItem(cls_label, "classification")
        self.track_map_source_combo.addItem(scr_label, "screened")
        idx = self.track_map_source_combo.findData(cur or "classification")
        self.track_map_source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.track_map_source_combo.blockSignals(False)

        effective = self._track_map_photo_folder()
        exists = os.path.isdir(effective)
        if hasattr(self, "track_map_photo_path_label"):
            if effective:
                status = "（目录存在）" if exists else "（目录不存在或为空路径）"
                self.track_map_photo_path_label.setText(f"{effective}\n{status}")
            else:
                self.track_map_photo_path_label.setText(
                    "未解析到路径：请设置输出根目录与图片文件夹，或手动浏览指定鸟图目录。"
                )

    def _select_track_map_photo_folder_override(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择鸟图目录")
        if not folder:
            return
        self.track_map_folder_override_input.setText(folder)
        self.config["track_map_photo_folder_override"] = folder
        self._refresh_track_map_path_ui()

    def _effective_classification_folder(self) -> str:
        """当前有效的分类归档路径（与主流程一致）。"""
        manual = self.record_export_class_input.text().strip()
        if manual:
            return manual
        return self._gui_paths_snapshot().get("crop_output_folder", "").strip()

    def _default_record_export_output_dir(self) -> str:
        paths = self._gui_paths_snapshot()
        _sync = {
            "record_export_output_folder": self.record_export_out_input.text().strip(),
            "output_root_folder": self.output_root_input.text().strip(),
            "reports_output_folder": paths.get("reports_output_folder", ""),
            "crop_output_folder": paths.get("crop_output_folder", ""),
        }
        return _record_export_dirs_from_config(_sync)[1]

    def _refresh_record_export_classification_default(self) -> None:
        """分类路径未手填时，随归档目录联动显示占位提示（不覆盖用户输入）。"""
        if self.record_export_class_input.text().strip():
            return
        crop = self._gui_paths_snapshot().get("crop_output_folder", "")
        if crop:
            self.record_export_class_input.setPlaceholderText(crop)

    def _select_folder_into(self, field_name: str) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not folder:
            return
        if field_name == "record_export_classification_folder":
            self.record_export_class_input.setText(folder)
            self.config["record_export_classification_folder"] = folder
        elif field_name == "record_export_output_folder":
            self.record_export_out_input.setText(folder)
            self.config["record_export_output_folder"] = folder

    def _run_record_export(self) -> None:
        classification_root = self._effective_classification_folder()
        if not classification_root or not os.path.isdir(classification_root):
            QMessageBox.warning(
                self,
                "无法导出",
                "请指定有效的「分类归档」目录（先完成物种识别归档，或在本卡片中浏览选择）。",
            )
            return
        if not self.record_export_ebird_checkbox.isChecked() and not (
            self.record_export_birdreport_checkbox.isChecked()
        ):
            QMessageBox.warning(
                self, "无法导出", "请至少勾选一种导出格式（eBird 或观鸟记录中心）。"
            )
            return
        out_dir = self._default_record_export_output_dir()
        try:
            self._sync_config_from_ui()
            written = export_from_classification(
                classification_root,
                out_dir,
                write_ebird_csv=self.record_export_ebird_checkbox.isChecked(),
                write_china_bird_record_xls=self.record_export_birdreport_checkbox.isChecked(),
                ebird_country=self.record_export_country_input.text().strip() or "CN",
                ebird_state=self.record_export_state_input.text().strip() or "FJ",
                **_record_export_kwargs(self.config),
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"{e}")
            self.add_log(f"观鸟记录导出失败: {e}")
            return
        lines = [f"导出目录: {os.path.abspath(out_dir)}"]
        for k, v in written.items():
            lines.append(f"  · {k}: {v}")
        lines.append(
            "文件名含日期、时间、坐标（英文）及导出时刻 expHHMMSS，避免覆盖未上传记录。"
            "eBird 用 .csv 上传；中国观鸟记录中心用 china_bird_record/*.xls。"
        )
        self.add_log("观鸟记录已导出:\n" + "\n".join(lines))
        self._show_record_export_done_dialog(out_dir, written)

    def _gpx_paths_from_ui(self) -> List[str]:
        paths: List[str] = []
        for i in range(self.gpx_list.count()):
            p = self.gpx_list.item(i).text().strip()
            if p and p not in paths:
                paths.append(p)
        return resolve_gpx_path_list(gpx_paths=paths)

    def _set_gpx_paths_to_ui(self, paths: List[str]) -> None:
        self.gpx_list.clear()
        for p in resolve_gpx_path_list(gpx_paths=paths):
            self.gpx_list.addItem(p)

    def _sync_gpx_paths_to_config(self) -> None:
        paths = self._gpx_paths_from_ui()
        self.config["gpx_file_paths"] = paths
        self.config["gpx_file_path"] = paths[0] if paths else ""

    def _add_gpx_files(self) -> None:
        start = self.gpx_list.item(0).text() if self.gpx_list.count() else ""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择 GPX 轨迹文件（可多选）",
            start,
            "GPX files (*.gpx);;All files (*)",
        )
        if not paths:
            return
        existing = {
            self.gpx_list.item(i).text()
            for i in range(self.gpx_list.count())
        }
        for p in paths:
            if p not in existing:
                self.gpx_list.addItem(p)
        self._sync_gpx_paths_to_config()

    def _remove_selected_gpx(self) -> None:
        for item in self.gpx_list.selectedItems():
            row = self.gpx_list.row(item)
            self.gpx_list.takeItem(row)
        self._sync_gpx_paths_to_config()

    def _merge_gpx_files_dialog(self) -> None:
        paths = self._gpx_paths_from_ui()
        if len(paths) < 2:
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "选择要合并的 GPX 文件",
                "",
                "GPX files (*.gpx);;All files (*)",
            )
        if len(paths) < 2:
            QMessageBox.information(
                self, "提示", "请至少选择两个 GPX 文件进行合并。"
            )
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "保存合并后的 GPX", "merged_track.gpx", "GPX files (*.gpx)"
        )
        if not out_path:
            return
        try:
            merged = merge_gpx_files(paths, out_path)
            self._set_gpx_paths_to_ui([merged])
            self._sync_gpx_paths_to_config()
            QMessageBox.information(self, "合并完成", f"已保存:\n{merged}")
        except Exception as e:
            QMessageBox.critical(self, "合并失败", str(e))

    def _gpx_target_photo_folder(self) -> str:
        if self.gpx_apply_screened_checkbox.isChecked():
            out = self.output_folder_input.text().strip() or self.config.get(
                "output_folder", ""
            )
            if out:
                return os.path.normpath(os.path.join(out, "Screened_images"))
        return os.path.normpath(self.image_folder_input.text().strip())

    def _apply_batch_gps_to_photos(self) -> None:
        self._sync_config_from_ui()
        folder = self._gpx_target_photo_folder()
        if not folder or not os.path.isdir(folder):
            out_raw = self.output_folder_input.text().strip() or self.config.get(
                "output_folder", ""
            )
            screened_checked = self.gpx_apply_screened_checkbox.isChecked()
            detail = f"实际检查路径: {folder}\n"
            if screened_checked:
                detail += f"输出目录(UI): {out_raw!r}\n"
                detail += f"输出目录(config): {self.config.get('output_folder', '')!r}"
            QMessageBox.warning(
                self,
                "提示",
                f"目标照片目录不存在。\n{detail}",
            )
            return

        use_fixed = self.gps_mode_fixed_radio.isChecked()
        if use_fixed:
            try:
                lat = float(self.lat_input.text().strip())
                lon = float(self.lon_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "提示", "请填写有效的纬度与经度。")
                return
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                QMessageBox.warning(
                    self, "提示", "纬度须在 [-90, 90]，经度须在 [-180, 180]。"
                )
                return
            try:
                alt = float(self.config.get("gps_altitude", 0) or 0)
            except (TypeError, ValueError):
                alt = 0.0
            try:
                count = batch_write_gps_exif(
                    folder,
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                )
                msg = (
                    f"目录: {folder}\n"
                    f"统一坐标: {lat:.6f}, {lon:.6f}\n"
                    f"成功写入 GPS: {count}"
                )
                self.add_log("指定地点→EXIF GPS:\n" + msg)
                QMessageBox.information(self, "写入完成", msg)
            except Exception as e:
                QMessageBox.critical(self, "写入失败", str(e))
            return

        gpx_paths = self._gpx_paths_from_ui()
        if not gpx_paths:
            QMessageBox.warning(self, "提示", "已选 GPX 模式，请先添加至少一个有效的 GPX 文件。")
            return
        try:
            stats = batch_write_gps_from_gpx(
                folder,
                gpx_paths=gpx_paths,
                exif_tz=self._gpx_match_exif_tz(),
                gpx_tz=self._gpx_match_gpx_tz(),
            )
            msg = (
                f"目录: {folder}\n"
                f"JPEG 总数: {stats['total']}\n"
                f"时间匹配: {stats['matched']}\n"
                f"成功写入 GPS: {stats['written']}"
            )
            self.add_log("GPX→EXIF GPS:\n" + msg)
            QMessageBox.information(self, "写入完成", msg)
        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))

    def _track_map_photo_folder(self) -> str:
        override = ""
        if hasattr(self, "track_map_folder_override_input"):
            override = self.track_map_folder_override_input.text().strip()
        if not override:
            override = (self.config.get("track_map_photo_folder_override") or "").strip()
        if override:
            return override
        paths = self._gui_paths_snapshot()
        src = self.track_map_source_combo.currentData()
        if src == "screened":
            return paths.get("screened_images", "")
        return paths.get("crop_output_folder", "")

    def _track_map_busy(self) -> bool:
        th = self._track_map_thread
        if th is None:
            return False
        try:
            return th.isRunning()
        except RuntimeError:
            self._track_map_thread = None
            return False

    def _set_track_map_busy(self, busy: bool, *, preview: bool) -> None:
        self.track_preview_btn.setEnabled(not busy)
        self.track_save_btn.setEnabled(not busy)
        if busy:
            self.track_preview_btn.setText(
                "预览生成中…" if preview else "预览轨迹图"
            )
            self.track_save_btn.setText(
                "保存生成中…" if not preview else "单独生成并保存 PNG"
            )
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("轨迹图生成中…")
        else:
            self.track_preview_btn.setText("预览轨迹图")
            self.track_save_btn.setText("单独生成并保存 PNG")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setFormat("%p%")

    def _finish_track_map_progress_dialog(self) -> None:
        dlg = self._track_map_progress
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()
        self._track_map_progress = None
        QApplication.restoreOverrideCursor()

    def _show_track_map_saved_dialog(self, png_path: str) -> None:
        """轨迹图保存完成：简要提示 + 打开图片链接（详情见右侧日志）。"""
        png_path = os.path.abspath(png_path)
        if not os.path.isfile(png_path):
            QMessageBox.warning(self, "提示", "未找到生成的图片文件。")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("轨迹图已生成")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"已保存：{os.path.basename(png_path)}"))
        path_label = QLabel(png_path)
        path_label.setWordWrap(True)
        path_label.setStyleSheet("color: #666;")
        lay.addWidget(path_label)
        link = QLabel('<a href="#open">打开图片</a>')
        link.setTextFormat(Qt.RichText)
        link.setOpenExternalLinks(False)
        link.linkActivated.connect(lambda _href: _open_local_file(png_path))
        link.setCursor(Qt.PointingHandCursor)
        lay.addWidget(link)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec_()

    def _record_export_openable_files(
        self, written: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """从 export_from_classification 返回值解析可打开的 (链接文案, 绝对路径)。"""
        specs = (
            (
                "ebird_checklist_format_csv",
                "ebird_checklist_format_csv_all",
                "打开 eBird CSV",
            ),
            (
                "china_bird_record_xls",
                "china_bird_record_xls_all",
                "打开观鸟记录中心 Excel",
            ),
        )
        seen: set[str] = set()
        out: List[Tuple[str, str]] = []
        for primary, all_key, base_label in specs:
            if all_key in written:
                paths = [
                    p.strip()
                    for p in written[all_key].split(";")
                    if p.strip()
                ]
            elif primary in written:
                paths = [written[primary].strip()]
            else:
                continue
            for p in paths:
                ap = os.path.abspath(p)
                if ap in seen or not os.path.isfile(ap):
                    continue
                seen.add(ap)
                label = base_label
                if len(paths) > 1:
                    label = f"{base_label} · {os.path.basename(ap)}"
                out.append((label, ap))
        return out

    def _show_record_export_done_dialog(
        self, out_dir: str, written: Dict[str, str]
    ) -> None:
        """观鸟记录导出完成：路径摘要 + 打开各导出文件 / 导出目录。"""
        out_dir = os.path.abspath(out_dir)
        files = self._record_export_openable_files(written)
        dlg = QDialog(self)
        dlg.setWindowTitle("导出完成")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("观鸟记录已导出，请核对数量后再上传各平台。"))
        dir_label = QLabel(f"导出目录：{out_dir}")
        dir_label.setWordWrap(True)
        dir_label.setStyleSheet("color: #666;")
        lay.addWidget(dir_label)
        if files:
            for label, path in files:
                path_lbl = QLabel(os.path.basename(path))
                path_lbl.setStyleSheet("color: #666; font-size: 9pt;")
                path_lbl.setToolTip(path)
                lay.addWidget(path_lbl)
                lay.addWidget(
                    self._make_action_link(
                        label, lambda p=path: _open_local_file(p)
                    )
                )
        else:
            lay.addWidget(QLabel("未找到可打开的导出文件，请查看右侧日志。"))
        lay.addWidget(
            self._make_action_link(
                "打开导出目录",
                lambda: _open_local_file(out_dir),
            )
        )
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec_()

    def _on_track_map_finished(self, written: Dict[str, str], preview: bool) -> None:
        main_png = written.get("track_png", "")
        lines = [f"已保存至 reports 目录:", main_png]
        bm = written.get("map_basemap", "")
        if bm == "no_key":
            lines.append(
                "未配置高德 API Key，请在「地理位置」→「打开配置文件」填写 amap_api_config.json"
            )
        elif bm == "fallback":
            lines.append("高德底图未加载（请检查网络与 API Key 权限）")
        elif bm == "ok":
            style = self.track_map_basemap_combo.currentText()
            lines.append(f"已叠加底图: {style}")
        if written.get("elevation_png"):
            lines.append(written["elevation_png"])
        align_desc = written.get("time_align_desc")
        coord_src = written.get("map_coord_source", "")
        if coord_src == "exif":
            lines.append("坐标来源：照片 EXIF GPS")
        elif align_desc:
            lines.append(f"时间匹配：{align_desc}")
        exif_pos = written.get("map_pos_exif_gps")
        if exif_pos:
            lines.append(f"地图坐标：{exif_pos} 张使用 EXIF GPS（与 GPX 插值一致时）")
        sk = written.get("skipped_time_mismatch")
        skipped_lines = iter_skipped_photo_log_lines(written)
        if skipped_lines:
            lines.extend(skipped_lines)
        elif sk and int(sk) > 0:
            if coord_src == "exif":
                lines.append(f"未绘制 {sk} 张：照片中无 EXIF GPS")
            else:
                lines.append(
                    f"未绘制 {sk} 张：与 GPX 时间差超过 1 小时或无拍摄时间"
                )
        self.add_log("\n".join(lines))
        if preview and main_png and os.path.isfile(main_png):
            map_title = written.get("map_title", "")
            if map_title:
                self.add_log(f"地图标题：{map_title}")
            show_track_map_preview(
                self,
                main_png,
                window_title="观鸟地图预览",
            )
        elif main_png:
            self._show_track_map_saved_dialog(main_png)

    def _run_track_map_generation(self, preview: bool = False) -> None:
        if self._track_map_busy():
            QMessageBox.information(
                self,
                "请稍候",
                "轨迹图正在生成中，请等待当前任务完成。",
            )
            return

        self._sync_config_from_ui()
        photo_folder = self._track_map_photo_folder()
        if not photo_folder or not os.path.isdir(photo_folder):
            QMessageBox.warning(
                self,
                "提示",
                "鸟图目录无效。\n\n"
                f"当前路径：\n{photo_folder or '(空)'}\n\n"
                "若已用「输出根目录」完成物种归档，请确认已选择图片文件夹；"
                "或在「鸟图目录」中浏览指定 classification 目录。",
            )
            return
        gpx_paths = self._gpx_paths_from_ui()
        use_gpx = self.track_map_use_gpx_checkbox.isChecked()
        if use_gpx and not gpx_paths:
            QMessageBox.warning(
                self, "提示", "已勾选使用 GPX，请添加至少一个有效的 GPX 文件。"
            )
            return
        use_exif = (
            True if not use_gpx else self.track_map_use_exif_checkbox.isChecked()
        )

        reports_dir = _reports_dir_from_config(self.config)
        kwargs: Dict[str, Any] = dict(
            reports_dir=reports_dir,
            gpx_paths=gpx_paths if use_gpx else None,
            photo_folder=photo_folder,
            use_gpx_track=use_gpx,
            use_exif_gps=use_exif,
            radius_km=float(self.track_map_radius_input.value()),
            include_elevation=self.track_map_elevation_checkbox.isChecked(),
            basemap_style=str(
                self.track_map_basemap_combo.currentData() or "normal"
            ),
            preview_only=preview,
            preview_max_photos=40,
            location_name=_track_map_location_from_config(self.config),
            province=self.config.get("province", ""),
            city=self.config.get("city", ""),
            exif_tz=self._gpx_match_exif_tz(),
            gpx_tz=self._gpx_match_gpx_tz(),
            logo_path=str(self.config.get("wm_logo_path", "") or ""),
            logo_width_ratio=float(self.config.get("wm_logo_width_ratio", 0.30)),
        )

        label = "预览" if preview else "保存"
        busy_msg = (
            f"轨迹图{label}：正在启动…（底图下载与绘制可能需 30–90 秒，请勿关闭窗口）"
        )
        self.add_log(busy_msg)
        self.update_status(busy_msg)
        print(busy_msg)

        self._set_track_map_busy(True, preview=preview)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        dlg = QProgressDialog(
            busy_msg + "\n\n正在准备…",
            None,
            0,
            0,
            self,
        )
        dlg.setWindowTitle("预览轨迹图" if preview else "生成轨迹图 PNG")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setCancelButton(None)
        dlg.setMinimumWidth(420)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        for _ in range(3):
            QApplication.processEvents()
        self._track_map_progress = dlg

        th = TrackMapThread(kwargs, self)
        self._track_map_thread = th

        def _on_log(msg: str) -> None:
            self.add_log(msg)
            self.update_status(msg)
            print(msg)
            prog = self._track_map_progress
            if prog is not None:
                prog.setLabelText(msg)
                QApplication.processEvents()

        self._track_map_last_preview = preview
        self._track_map_pending_ok: Optional[Dict[str, str]] = None
        self._track_map_pending_err: Optional[str] = None

        def _on_ok(written: Dict[str, str]) -> None:
            self._track_map_pending_ok = written

        def _on_fail(msg: str) -> None:
            self._track_map_pending_err = msg

        def _always_unbusy() -> None:
            self._finish_track_map_progress_dialog()
            self._set_track_map_busy(False, preview=self._track_map_last_preview)
            ok = self._track_map_pending_ok
            err = self._track_map_pending_err
            self._track_map_pending_ok = None
            self._track_map_pending_err = None
            self._track_map_thread = None
            try:
                if ok is not None:
                    self.progress_bar.setValue(100)
                    self.update_status("轨迹图生成完成")
                    self._on_track_map_finished(ok, self._track_map_last_preview)
                elif err is not None:
                    self.progress_bar.setValue(0)
                    self.update_status("轨迹图生成失败")
                    self.add_log(f"轨迹图生成失败: {err}")
                    QMessageBox.critical(
                        self,
                        "生成失败",
                        err[:8000] if len(err) > 8000 else err,
                    )
            except Exception as exc:
                tb = traceback.format_exc()
                self.add_log(f"轨迹图完成回调异常: {exc}\n{tb}")
                QMessageBox.critical(
                    self,
                    "错误",
                    f"轨迹图结果处理失败:\n{exc}",
                )

        th.log_line.connect(_on_log)
        th.finished_ok.connect(_on_ok)
        th.failed.connect(_on_fail)
        th.finished.connect(_always_unbusy)
        th.finished.connect(th.deleteLater)
        th.start()

    def _on_model_mode_changed(self, index: int):
        """模型模式切换"""
        self.config['use_local_model'] = (index == 0)
        self._update_local_species_model_combo_enabled()

    def _on_local_species_model_changed(self, _index: int = 0) -> None:
        kind = self.local_species_model_combo.currentData()
        if kind:
            self.config['local_species_model'] = normalize_local_species_model(kind)

    def _on_species_geo_mode_changed(self, _index: int = 0) -> None:
        mode = self.species_geo_mode_combo.currentData()
        if mode:
            self.config['species_geo_mode'] = normalize_species_geo_mode(mode)

    def _update_local_species_model_combo_enabled(self) -> None:
        enabled = self.local_model_radio.isChecked()
        self.local_species_model_combo.setEnabled(enabled)
    
    def _create_timezone_combo(self, saved_tz: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setMinimumContentsLength(32)
        for label, tzid in timezone_combo_entries():
            combo.addItem(label, tzid)
        names = [combo.itemText(i) for i in range(combo.count())]
        completer = QCompleter(names, combo)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        combo.setCompleter(completer)
        set_combo_timezone(combo, saved_tz)
        return combo

    def _config_gpx_match_exif_tz(self) -> str:
        return normalize_tz_name(
            self.config.get("gpx_match_exif_tz")
            or self.config.get("track_map_exif_tz")
            or DEFAULT_EXIF_TZ
        )

    def _config_gpx_match_gpx_tz(self) -> str:
        return normalize_tz_name(
            self.config.get("gpx_match_gpx_tz")
            or self.config.get("track_map_gpx_tz")
            or DEFAULT_GPX_TZ
        )

    def _gpx_match_exif_tz(self) -> str:
        return read_combo_timezone(self.gpx_match_exif_tz_combo)

    def _gpx_match_gpx_tz(self) -> str:
        return read_combo_timezone(self.gpx_match_gpx_tz_combo)

    def _on_track_map_use_gpx_changed(self, state: int) -> None:
        use_gpx = state == Qt.Checked
        if not use_gpx:
            self.track_map_use_exif_checkbox.setChecked(True)
        self.track_map_use_exif_checkbox.setEnabled(use_gpx)
        if use_gpx:
            self.track_map_use_exif_checkbox.setText("补充使用照片 EXIF 中的 GPS")
        else:
            self.track_map_use_exif_checkbox.setText("使用照片 EXIF 中的 GPS")

    def _on_gps_write_mode_changed(self, *_args) -> None:
        """主流程 GPS 二选一：切换时启用/禁用对应表单项。"""
        use_fixed = self.gps_mode_fixed_radio.isChecked()
        self.config["gps_write_mode"] = "fixed" if use_fixed else "gpx"
        for w in (
            self.location_input,
            self.lat_input,
            self.lon_input,
            self.province_city_display,
            self.location_query_btn,
        ):
            w.setEnabled(use_fixed)
        if hasattr(self, "gpx_apply_btn"):
            if use_fixed:
                self.gpx_apply_btn.setToolTip(
                    "对所选文件夹写入上方统一经纬度（不依赖「开始处理」）。"
                )
            else:
                self.gpx_apply_btn.setToolTip(
                    "对所选文件夹按 GPX 与 EXIF 时间（上方时区）插值写入 GPS。"
                )

    def _on_gps_write_changed(self, state: int):
        """GPS 写入开关状态变化"""
        self.config["enable_gps_write"] = state == Qt.Checked
    
    def _make_external_link(self, text: str, url: str) -> QLabel:
        """下划线链接样式，点击在系统浏览器打开。"""
        link = QLabel(f'<a href="{url}">{text}</a>')
        link.setOpenExternalLinks(True)
        link.setTextFormat(Qt.RichText)
        link.setCursor(Qt.PointingHandCursor)
        link.setToolTip(url)
        link.setStyleSheet(
            "QLabel { color: #1565C0; font-size: 9pt; }"
            "QLabel a { text-decoration: underline; }"
        )
        return link

    def _make_action_link(self, text: str, on_click) -> QLabel:
        """下划线链接样式，点击触发回调（非外链）。"""
        link = QLabel(f'<a href="#action">{text}</a>')
        link.setTextFormat(Qt.RichText)
        link.setOpenExternalLinks(False)
        link.linkActivated.connect(lambda _href: on_click())
        link.setCursor(Qt.PointingHandCursor)
        link.setStyleSheet(
            "QLabel { color: #1565C0; font-size: 9pt; }"
            "QLabel a { text-decoration: underline; }"
        )
        return link

    def _open_record_export_count_settings_dialog(self) -> None:
        """累计只数：按 EXIF 拍摄时间分窗与 GPS 距离分窗。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("累计只数 — 分窗设置")
        lay = QVBoxLayout(dlg)
        hint = QLabel(
            "分窗仅依据照片 EXIF 中的拍摄时间（DateTimeOriginal），"
            "不用文件修改时间或其它时间戳。\n"
            "同一物种在「时间窗内」或「距离内」的多次拍摄视为同一批个体，"
            "每批取归档张数最多的一次再累加。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-size: 9pt;")
        lay.addWidget(hint)

        form = QFormLayout()
        time_spin = QDoubleSpinBox()
        time_spin.setRange(5.0, 10080.0)
        time_spin.setDecimals(0)
        time_spin.setSuffix(" 分钟")
        time_spin.setSingleStep(5.0)
        time_spin.setValue(self._record_export_individual_time_minutes)
        time_spin.setToolTip(
            "同一物种两次拍摄相隔不超过此时间（按 EXIF 拍摄时刻）则合并为 1 个个体批次。"
        )
        form.addRow("分窗时间:", time_spin)

        dist_spin = QDoubleSpinBox()
        dist_spin.setRange(0.01, 10.0)
        dist_spin.setDecimals(2)
        dist_spin.setSuffix(" km")
        dist_spin.setSingleStep(0.05)
        dist_spin.setValue(self._record_export_spatial_km)
        dist_spin.setToolTip(
            "有 GPS 时，两次拍摄相距不超过此距离则合并为 1 个个体批次；"
            "若在时间窗内即使略超距离也会合并（定点观鸟）。"
        )
        form.addRow("分窗距离:", dist_spin)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec_() != QDialog.Accepted:
            return
        self._record_export_individual_time_minutes = float(time_spin.value())
        self._record_export_spatial_km = float(dist_spin.value())
        self.config["record_export_time_minutes"] = (
            self._record_export_individual_time_minutes
        )
        self.config["record_export_spatial_km"] = self._record_export_spatial_km

    def _open_record_portal_url(self, url: str) -> None:
        """在系统浏览器打开观鸟记录上传/说明页面。"""
        u = (url or "").strip()
        if not u:
            return
        if not QDesktopServices.openUrl(QUrl(u)):
            QMessageBox.warning(self, "提示", f"无法打开链接：\n{u}")

    def _open_config_file(self):
        """打开豆包API配置文件"""
        src_dir = Path(__file__).resolve().parent
        cfg_path = src_dir / "doubao_api_config.json"
        existed = cfg_path.is_file()
        path = ensure_doubao_api_config_file(src_dir)
        if not existed:
            QMessageBox.information(
                self,
                "提示",
                "已创建默认配置文件 doubao_api_config.json，请填写 api_key，"
                "并按方舟控制台核对 api_base、model / models。",
            )
        _open_local_file(str(path))

    def _open_amap_config_file(self):
        """打开高德地图 API 配置文件（地名地理编码用）"""
        src_dir = Path(__file__).resolve().parent
        cfg_path = src_dir / "amap_api_config.json"
        existed = cfg_path.is_file()
        path = ensure_amap_api_config_file(src_dir)
        if not existed:
            QMessageBox.information(
                self,
                "提示",
                "已创建默认配置文件 amap_api_config.json，请填写 api_key。",
            )
        _open_local_file(str(path))
    
    def _create_status_panel(self) -> QWidget:
        """创建状态面板（可滚动，与左侧同高时避免内容被纵向挤压）"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )

        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 8, 0)
        
        # 进度卡片
        progress_card_container, progress_card = self._create_card("📊 处理进度")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(10)
        progress_layout.setContentsMargins(12, 10, 12, 12)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #F0F0F0;
                border: none;
                border-radius: 8px;
                text-align: center;
                height: 22px;
                font-size: 10pt;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2E8B57, stop:1 #1E90FF);
                border-radius: 8px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        time_row = QHBoxLayout()
        time_row.setSpacing(16)
        self._elapsed_label = QLabel("已用时间：—")
        self._eta_label = QLabel("预计剩余：—")
        for lb in (self._elapsed_label, self._eta_label):
            lb.setStyleSheet("font-size: 10pt; color: #444444;")
        time_row.addWidget(self._elapsed_label)
        time_row.addWidget(self._eta_label)
        time_row.addStretch(1)
        progress_layout.addLayout(time_row)
        
        # 状态信息
        status_label = QLabel("状态信息:")
        status_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #333333;")
        progress_layout.addWidget(status_label)
        
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMinimumHeight(160)
        self.status_log.setStyleSheet("""
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', 'Microsoft YaHei UI', monospace;
                font-size: 9pt;
            }
            QTextEdit::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            QTextEdit::-webkit-scrollbar-track {
                background: #F0F0F0;
                border-radius: 4px;
            }
            QTextEdit::-webkit-scrollbar-thumb {
                background: #BDC3C7;
                border-radius: 4px;
            }
            QTextEdit::-webkit-scrollbar-thumb:hover {
                background: #95A5A6;
            }
        """)
        progress_layout.addWidget(self.status_log)
        
        # 处理统计
        stats_label = QLabel("处理统计:")
        stats_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #333333;")
        progress_layout.addWidget(stats_label)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["项目", "数值"])
        self.stats_table.setMinimumHeight(120)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        progress_layout.addWidget(self.stats_table)
        
        # 清空日志按钮
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.clear_log)
        progress_layout.addWidget(clear_btn)
        
        progress_card.setLayout(progress_layout)
        layout.addWidget(progress_card_container)
        
        # 品牌信息（底部小 Logo + 版权；资源路径见 _logo_search_directories）
        brand_widget = QWidget()
        brand_layout = QVBoxLayout()
        brand_layout.setSpacing(4)
        brand_layout.setContentsMargins(0, 10, 0, 0)

        from PyQt5.QtGui import QPixmap as _QPixmap

        footer_logo_path = self._copyright_holder_logo_path()
        if footer_logo_path is not None:
            fpm = _QPixmap(str(footer_logo_path))
            if not fpm.isNull():
                fh = 44
                dpr = self._primary_screen_dpr()
                fscaled = fpm.scaledToHeight(
                    max(1, int(fh * dpr)), Qt.SmoothTransformation
                )
                fscaled.setDevicePixelRatio(dpr)
                footer_logo = QLabel()
                footer_logo.setAlignment(Qt.AlignCenter)
                footer_logo.setPixmap(fscaled)
                footer_logo.setFixedHeight(fh)
                brand_layout.addWidget(footer_logo)

        # 版权信息
        copyright_label = QLabel("© 2026 brigchen@gmail.com")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("color: #666666; font-size: 10pt;")
        brand_layout.addWidget(copyright_label)
        
        # 开源声明
        license_label = QLabel("基于开源协议，仅限爱好者、公益、科研等非盈利用途，请勿用于商业用途")
        license_label.setAlignment(Qt.AlignCenter)
        license_label.setStyleSheet("color: #999999; font-size: 9pt;")
        brand_layout.addWidget(license_label)
        
        brand_widget.setLayout(brand_layout)
        layout.addWidget(brand_widget)
        
        panel.setLayout(layout)
        scroll.setWidget(panel)
        return scroll
    
    def _refresh_derived_paths_display(self) -> None:
        """根据输出根目录 + 相片文件夹更新「自动生成」说明与手动路径区域显隐。"""
        if not hasattr(self, "output_root_input"):
            return
        root = self.output_root_input.text().strip()
        img = self.image_folder_input.text().strip()
        if root:
            self._legacy_paths_container.setVisible(False)
            self.derived_paths_label.setVisible(True)
            if not img:
                self.derived_paths_label.setText(
                    "已设置输出根目录。请选择「图片文件夹」后，将按该文件夹名自动生成：\n"
                    "· <根>/screened_<文件夹名>/  （内含 Screened_images/、burst_analysis.json）\n"
                    "· <根>/classification_<文件夹名>/  （物种归档）\n"
                    "· <根>/reports/  （连拍与物种 HTML 报告）"
                )
            else:
                slug = _session_slug_from_image_folder(img)
                o = os.path.join(root, f"screened_{slug}")
                c = os.path.join(root, f"classification_{slug}")
                r = os.path.join(root, "reports")
                lines = [
                    f"会话标签「{slug}」：",
                    f"· {o}",
                    f"· {c}",
                    f"· {r}",
                ]
                if hasattr(self, "dual_format_combo"):
                    if self.dual_format_combo.currentData() == "jpg_copy_raw":
                        lines.insert(
                            3,
                            f"· {os.path.join(o, 'Screened_raw_images')}",
                        )
                self.derived_paths_label.setText("\n".join(lines))
        else:
            self._legacy_paths_container.setVisible(True)
            self.derived_paths_label.setVisible(False)
    
    def _select_folder(self, field_name: str):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self, f"选择{field_name}文件夹"
        )
        if folder:
            if field_name == 'image_folder':
                self.config['image_folder'] = folder
                self.image_folder_input.setText(folder)
                self._refresh_derived_paths_display()
                self._refresh_track_map_path_ui()
            elif field_name == 'output_root_folder':
                self.config['output_root_folder'] = folder
                self.output_root_input.setText(folder)
                self._refresh_derived_paths_display()
                self._refresh_track_map_path_ui()
            elif field_name == 'output_folder':
                self.config['output_folder'] = folder
                self.output_folder_input.setText(folder)
            elif field_name == 'crop_output_folder':
                self.config['crop_output_folder'] = folder
                self.crop_folder_input.setText(folder)
                self._refresh_track_map_path_ui()
            elif field_name == 'watermark_input_folder':
                self.config['watermark_input_folder'] = folder
                self.wm_input_folder_input.setText(folder)
            elif field_name == 'watermark_output_folder':
                self.config['watermark_output_folder'] = folder
                self.wm_output_folder_input.setText(folder)
            elif field_name == 'image_clean_folder':
                self.config['image_clean_folder'] = folder
                self.image_clean_folder_input.setText(folder)
            try:
                self._sync_config_from_ui()
                self._save_config()
            except Exception as e:
                print(f"选择文件夹后保存配置失败: {e}")

    def _select_wm_logo_file(self):
        """选择水印签名logo文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择签名 Logo",
            "",
            file_filter_all_images(),
        )
        if file_path:
            self.wm_logo_input.setText(file_path)
            try:
                self._sync_config_from_ui()
                self._save_config()
            except Exception as e:
                print(f"保存 Logo 路径失败: {e}")

    def _update_wm_ai_denoise_enabled(self) -> None:
        """AI 降噪复选框状态联动：禁用降噪模型与强度控件。"""
        enabled = self.wm_ai_denoise_checkbox.isChecked()
        self.wm_ai_denoise_model_combo.setEnabled(enabled)
        self.wm_ai_denoise_slider.setEnabled(enabled)
        self.wm_ai_denoise_value_label.setEnabled(enabled)

    def _build_watermark_options(self) -> WatermarkOptions:
        _style = (
            "inline"
            if self.wm_style_combo.currentIndex() == 1
            else "frame"
        )
        _ai_dn_model = str(
            self.wm_ai_denoise_model_combo.currentData() or "realesrgan"
        )
        return WatermarkOptions(
            enable_location=self.wm_location_checkbox.isChecked(),
            location_text=self.wm_location_text_input.text().strip(),
            use_gps_city=self.wm_use_gps_city_checkbox.isChecked(),
            enable_date=self.wm_date_checkbox.isChecked(),
            enable_species=self.wm_species_checkbox.isChecked(),
            enable_camera_params=self.wm_camera_checkbox.isChecked(),
            logo_path=self.wm_logo_input.text().strip(),
            logo_width_ratio=float(self.wm_logo_width_ratio_input.value()),
            watermark_style=_style,
            enable_auto_enhance=False,
            enable_ai_exposure=self.wm_ai_exposure_checkbox.isChecked(),
            ai_exposure_strength=self.wm_ai_exposure_slider.value() / 100.0,
            enable_ai_denoise=self.wm_ai_denoise_checkbox.isChecked(),
            enable_ai_sharpen=self.wm_ai_sharpen_checkbox.isChecked(),
            ai_denoise_model=_ai_dn_model,
            ai_denoise_strength=self.wm_ai_denoise_slider.value() / 100.0,
            ai_sharpen_strength=self.wm_ai_sharpen_slider.value() / 100.0,
            ai_tile_size=int(self.config.get("wm_ai_tile_size", 512)),
        )

    def _resolve_watermark_source_folder(self) -> str:
        source_folder = self.wm_input_folder_input.text().strip()
        if not source_folder:
            source_folder = choose_default_watermark_source(
                image_folder=self.image_folder_input.text().strip(),
                crop_output_folder=self.crop_folder_input.text().strip(),
                output_folder=self.output_folder_input.text().strip(),
            )
        return source_folder

    def _open_burst_webp_dialog(self) -> None:
        """连拍 → WebP / MP4（弹窗内选图、调参与预览）。"""
        # 「添加图片」起始目录用原始相片文件夹，不用水印默认目录（可能指向 ROI/Screened）。
        img_dir = self.image_folder_input.text().strip()
        print(
            f"[Birdy 动图GUI] 主界面：打开动图对话框，相片文件夹={img_dir or '(空)'}",
            flush=True,
        )
        try:
            open_burst_webp_dialog(self, default_dir=img_dir or "")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "动图",
                f"打开连拍动图窗口失败：\n{e}",
            )

    def _open_video_stabilize_dialog(self) -> None:
        """视频裁剪与稳定（弹窗内选文件、调参、预览与处理）。"""
        # 起始目录使用原始相片文件夹或输出根目录
        img_dir = self.image_folder_input.text().strip()
        output_root = self.output_root_input.text().strip() if hasattr(self, 'output_root_input') else ""
        default_dir = img_dir or output_root or ""

        # 输出目录使用水印输出文件夹（watermarked）
        wm_output_dir = self.wm_output_folder_input.text().strip() if hasattr(self, 'wm_output_folder_input') else ""
        if not wm_output_dir:
            wm_output_dir = self.config.get("watermark_output_folder", "./watermarked").strip()

        print(
            f"[Birdy 视频稳定GUI] 主界面：打开视频稳定对话框，默认目录={default_dir or '(空)'}, 输出目录={wm_output_dir}",
            flush=True,
        )
        try:
            open_video_stabilize_dialog(self, default_dir=default_dir, default_output_dir=wm_output_dir)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "视频稳定",
                f"打开视频稳定窗口失败：\n{e}",
            )

    def get_burst_webp_bird_detector(self):
        """
        懒加载仅鸟类检测（无物种/鸟眼），供动图弹窗推断首张裁剪中心。
        失败时缓存 False，避免重复加载。
        使用线程锁：动图首张鸟检在后台线程调用本方法时，避免与预览线程双开竞态。
        """
        lock = getattr(self, "_burst_webp_detector_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._burst_webp_detector_lock = lock
        with lock:
            d = getattr(self, "_burst_webp_bird_detector", None)
            if d is False:
                print(
                    "[Birdy 动图GUI] 鸟检测器不可用（此前加载失败，动图首张将用几何中心）。",
                    flush=True,
                )
                return None
            if d is not None:
                return d
            print(
                "[Birdy 动图GUI] 鸟检测器首次加载（仅鸟体 YOLO，无物种/鸟眼，可能卡顿数秒）…",
                flush=True,
            )
            try:
                self._burst_webp_bird_detector = BirdAndEyeDetector(
                    enable_species=False,
                    enable_eye=False,
                )
            except Exception as e:
                print(f"[Birdy 动图GUI] 鸟检测器加载失败：{e}", flush=True)
                self._burst_webp_bird_detector = False
                return None
            print("[Birdy 动图GUI] 鸟检测器加载成功。", flush=True)
            return self._burst_webp_bird_detector

    def _build_image_clean_options(self) -> ImageCleanOptions:
        return ImageCleanOptions(
            remove_no_bird=self.image_clean_no_bird_checkbox.isChecked(),
            remove_blurry=self.image_clean_blurry_checkbox.isChecked(),
            dedupe=self.image_clean_dedupe_checkbox.isChecked(),
            min_clarity=float(self.image_clean_clarity_slider.value()),
            dup_similarity=float(self.image_clean_dup_slider.value()),
        )

    def _resolve_image_clean_folder(self) -> str:
        folder = self.image_clean_folder_input.text().strip()
        if folder:
            return folder
        folder = self.crop_folder_input.text().strip()
        if folder:
            return folder
        return (self.config.get("crop_output_folder") or "").strip()

    def _run_image_clean_batch(self) -> None:
        """单独清洗选定目录（默认分类归档目录）。"""
        if self._image_clean_thread is not None and self._image_clean_thread.isRunning():
            QMessageBox.information(self, "提示", "图片清洗正在运行中，请稍候。")
            return
        folder = self._resolve_image_clean_folder()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self,
                "提示",
                "未找到可清洗的目录。\n"
                "请填写「清洗目录」，或先设置分类归档目录（classification）。",
            )
            return
        opts = self._build_image_clean_options()
        if not (opts.remove_no_bird or opts.remove_blurry or opts.dedupe):
            QMessageBox.warning(self, "提示", "请至少勾选一项清洗规则。")
            return
        reply = QMessageBox.question(
            self,
            "确认清洗",
            f"将对以下目录执行清洗，不合格图片会被直接删除：\n\n{folder}\n\n"
            f"· 未检出鸟体：{'是' if opts.remove_no_bird else '否'}\n"
            f"· 模糊（清晰度 < {int(opts.min_clarity)}）："
            f"{'是' if opts.remove_blurry else '否'}\n"
            f"· 去重（相似度 ≥ {int(opts.dup_similarity)}%）："
            f"{'是' if opts.dedupe else '否'}\n\n"
            "确定继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._sync_config_from_ui()
        self._save_config()
        self.image_clean_run_btn.setEnabled(False)
        self.add_log(f"已启动图片清洗：{folder}")
        th = ImageCleanThread(folder, opts, self)
        self._image_clean_thread = th

        def _on_prog(done: int, total: int) -> None:
            tot = max(1, total)
            self.progress_bar.setValue(min(100, int(100 * done / tot)))

        def _on_log(msg: str) -> None:
            self.add_log(msg)

        def _on_ok(r: Dict) -> None:
            self.image_clean_run_btn.setEnabled(True)
            self.progress_bar.setValue(100)
            QMessageBox.information(
                self,
                "清洗完成",
                f"总计 {r.get('total', 0)}，保留 {r.get('kept', 0)}\n"
                f"删除：未检出鸟体 {r.get('removed_no_bird', 0)}，"
                f"模糊 {r.get('removed_blurry', 0)}，"
                f"重复 {r.get('removed_duplicate', 0)}\n"
                f"失败 {r.get('failed', 0)}\n\n目录：{folder}",
            )

        def _on_fail(msg: str) -> None:
            self.image_clean_run_btn.setEnabled(True)
            QMessageBox.critical(self, "清洗失败", msg)

        th.progress.connect(_on_prog)
        th.log_line.connect(_on_log)
        th.finished_ok.connect(_on_ok)
        th.failed.connect(_on_fail)
        th.start()

    def _run_watermark_batch(self):
        """仅执行批量水印生成（不触发完整主流程）。"""
        if self._wm_batch_thread is not None and self._wm_batch_thread.isRunning():
            QMessageBox.information(self, "提示", "批量水印正在运行中，请稍候。")
            return
        source_folder = self._resolve_watermark_source_folder()
        if not source_folder or not os.path.isdir(source_folder):
            QMessageBox.warning(
                self, "提示", "未找到可处理的图片目录，请先设置相片文件夹。"
            )
            return
        output_folder = self.wm_output_folder_input.text().strip()
        if not output_folder:
            output_folder = os.path.join(
                self.output_folder_input.text().strip() or "./outputs",
                "watermarked",
            )
            self.wm_output_folder_input.setText(output_folder)
        try:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "水印生成失败", f"无法创建输出目录：{e}")
            return

        options = self._build_watermark_options()
        self._sync_config_from_ui()
        self._save_config()

        self.wm_run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        if hasattr(self, "_elapsed_label"):
            self._elapsed_label.setText("已用时间：0秒")
        if hasattr(self, "_eta_label"):
            self._eta_label.setText("预计剩余：…")
        self.add_log("已启动后台批量水印线程…")
        print("已启动后台批量水印线程…")

        th = WatermarkBatchThread(
            source_folder,
            output_folder,
            options,
            self,
            random_per_species=(
                int(self.wm_random_per_species_count.value())
                if self.wm_random_per_species_checkbox.isChecked()
                else None
            ),
        )
        self._wm_batch_thread = th

        def _on_prog(pct: int, elapsed: int, remaining: int, done: int) -> None:
            self.progress_bar.setValue(pct)
            if hasattr(self, "_elapsed_label"):
                self._elapsed_label.setText(f"已用时间：{self._format_duration_hms(elapsed)}")
            if hasattr(self, "_eta_label"):
                if pct >= 100:
                    self._eta_label.setText("预计剩余：完成")
                else:
                    self._eta_label.setText(f"预计剩余：{self._format_duration_hms(remaining)}")

        def _on_log(msg: str) -> None:
            self.add_log(msg)
            print(msg)

        def _on_ok(r: Dict) -> None:
            self.wm_run_btn.setEnabled(True)
            self.progress_bar.setValue(100)
            if hasattr(self, "_eta_label"):
                self._eta_label.setText("预计剩余：完成")
            QMessageBox.information(
                self,
                "水印生成完成",
                f"总计 {r.get('total', 0)}，成功 {r.get('ok', 0)}，失败 {r.get('fail', 0)}\n"
                f"输出目录：{output_folder}",
            )
            try:
                self._sync_config_from_ui()
                self._save_config()
            except Exception as e:
                print(f"水印完成后保存配置失败: {e}")

        def _on_fail(msg: str) -> None:
            self.wm_run_btn.setEnabled(True)
            if hasattr(self, "_eta_label"):
                self._eta_label.setText("预计剩余：失败")
            QMessageBox.critical(self, "水印生成失败", msg)

        th.progress.connect(_on_prog)
        th.log_line.connect(_on_log)
        th.finished_ok.connect(_on_ok)
        th.failed.connect(_on_fail)
        th.start()

    def _preview_watermark_one(self):
        """预览水印效果。"""
        source_folder = self._resolve_watermark_source_folder()
        if not source_folder or not os.path.isdir(source_folder):
            QMessageBox.warning(
                self, "提示", "未找到可预览的图片目录，请先设置相片文件夹或先跑归档流程。"
            )
            return

        imgs = collect_images_recursive(source_folder)
        paths_all: List[str] = list(imgs)
        if not paths_all:
            fp, _ = QFileDialog.getOpenFileName(
                self,
                "选择一张用于预览的图片",
                source_folder,
                file_filter_all_images(),
            )
            if not fp:
                return
            paths_all = [fp]

        nav_state: Dict[str, Any] = {"paths": paths_all, "idx": 0}

        def _current_path() -> str:
            ps = nav_state["paths"]
            i = int(nav_state["idx"])
            return str(ps[i])

        from PyQt5.QtGui import QImage, QPixmap

        dlg = QDialog(self)
        dlg.setWindowTitle("水印预览")
        dlg.resize(1400, 920)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(8, 8, 8, 8)

        nav_row = QHBoxLayout()
        prev_im_btn = QPushButton("上一张")
        next_im_btn = QPushButton("下一张")
        prev_im_btn.setToolTip("上一张 (←)")
        next_im_btn.setToolTip("下一张 (→)")
        nav_pos_label = QLabel()
        nav_pos_label.setMinimumWidth(120)
        nav_pos_label.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(prev_im_btn)
        nav_row.addStretch(1)
        nav_row.addWidget(nav_pos_label)
        nav_row.addStretch(1)
        nav_row.addWidget(next_im_btn)
        v.addLayout(nav_row)

        tools = QHBoxLayout()
        zoom_out_btn = QPushButton("缩小")
        zoom_in_btn = QPushButton("放大")
        fit_btn = QPushButton("适应窗口")
        one_btn = QPushButton("100%")
        tools.addWidget(zoom_out_btn)
        tools.addWidget(zoom_in_btn)
        tools.addWidget(fit_btn)
        tools.addWidget(one_btn)
        tools.addStretch(1)
        v.addLayout(tools)

        sc = QScrollArea(dlg)
        sc.setWidgetResizable(True)
        sc.setMinimumHeight(700)
        holder = QWidget()
        hv = QVBoxLayout(holder)
        img_lb = QLabel()
        img_lb.setAlignment(Qt.AlignCenter)
        zoom_state = {"scale": 1.0}
        pix_state: Dict[str, Any] = {"pix": None}

        def _apply_scaled():
            pix = pix_state.get("pix")
            if pix is None or pix.isNull():
                return
            scale = max(0.1, float(zoom_state["scale"]))
            nw = max(1, int(pix.width() * scale))
            nh = max(1, int(pix.height() * scale))
            img_lb.setPixmap(
                pix.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        def _fit_to_view():
            pix = pix_state.get("pix")
            if pix is None or pix.isNull():
                return
            vw = max(1, sc.viewport().width() - 16)
            vh = max(1, sc.viewport().height() - 16)
            sx = vw / max(1, pix.width())
            sy = vh / max(1, pix.height())
            zoom_state["scale"] = min(sx, sy, 1.0)
            _apply_scaled()

        # ---- 后台渲染（QThread + Worker，避免 AI 推理阻塞 UI） ----
        render_state: Dict[str, Any] = {"busy": False, "pending": None, "thread": None, "worker": None}

        def _update_nav_ui() -> None:
            n = len(nav_state["paths"])
            i = int(nav_state["idx"])
            busy = bool(render_state["busy"])
            prev_im_btn.setEnabled(not busy and n > 1 and i > 0)
            next_im_btn.setEnabled(not busy and n > 1 and i < n - 1)
            cur = _current_path()
            if busy:
                nav_pos_label.setText(f"渲染中… {i + 1} / {n}  {os.path.basename(cur)}")
            else:
                nav_pos_label.setText(f"{i + 1} / {n}  {os.path.basename(cur)}")

        def _cleanup_render_thread() -> None:
            th = render_state.get("thread")
            wk = render_state.get("worker")
            if wk is not None:
                try:
                    wk.deleteLater()
                except Exception:
                    pass
            if th is not None:
                try:
                    th.quit()
                    th.wait(3000)
                except Exception:
                    pass
                try:
                    th.deleteLater()
                except Exception:
                    pass
            render_state["thread"] = None
            render_state["worker"] = None

        def _on_render_done(pil_img) -> None:
            _cleanup_render_thread()
            render_state["busy"] = False
            if pil_img is None:
                img_lb.setText("渲染失败或无水印内容")
                _update_nav_ui()
                _consume_pending()
                return
            arr = np.array(pil_img.convert("RGB"))
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            pix_state["pix"] = QPixmap.fromImage(qimg)
            _fit_to_view()
            _update_nav_ui()
            _consume_pending()

        def _on_render_failed(msg: str) -> None:
            _cleanup_render_thread()
            render_state["busy"] = False
            img_lb.setText(f"渲染失败: {msg}")
            _update_nav_ui()
            _consume_pending()

        def _consume_pending() -> None:
            """当前渲染完成后，若用户在期间点了下一张，则继续处理。"""
            pending = render_state.get("pending")
            render_state["pending"] = None
            if pending is not None:
                zoom_state["scale"] = 1.0
                _start_render(idx=int(pending))

        def _start_render(idx: Optional[int] = None) -> None:
            if idx is not None:
                nav_state["idx"] = idx
            if render_state["busy"]:
                # 当前正在渲染，记录 pending（取最新请求）
                render_state["pending"] = int(nav_state["idx"])
                return
            render_state["busy"] = True
            _update_nav_ui()
            opts = self._build_watermark_options()
            wk = _WatermarkRenderWorker(_current_path(), source_folder, opts)
            th = QThread()
            wk.moveToThread(th)
            th.started.connect(wk.run)
            wk.finished.connect(_on_render_done)
            wk.failed.connect(_on_render_failed)
            wk.finished.connect(th.quit)
            wk.failed.connect(th.quit)
            th.finished.connect(wk.deleteLater)
            th.finished.connect(th.deleteLater)
            render_state["thread"] = th
            render_state["worker"] = wk
            th.start()

        def _do_refresh() -> None:
            _start_render()

        def _step_image(delta: int) -> None:
            n = len(nav_state["paths"])
            if n <= 1:
                return
            ni = max(0, min(n - 1, int(nav_state["idx"]) + int(delta)))
            if ni == int(nav_state["idx"]):
                return
            zoom_state["scale"] = 1.0
            nav_state["idx"] = ni
            _update_nav_ui()
            _start_render()

        prev_im_btn.clicked.connect(lambda: _step_image(-1))
        next_im_btn.clicked.connect(lambda: _step_image(1))
        _sc_prev = QShortcut(QKeySequence(Qt.Key_Left), dlg)
        _sc_prev.activated.connect(lambda: _step_image(-1))
        _sc_next = QShortcut(QKeySequence(Qt.Key_Right), dlg)
        _sc_next.activated.connect(lambda: _step_image(1))

        zoom_in_btn.clicked.connect(
            lambda: (
                zoom_state.__setitem__("scale", zoom_state["scale"] * 1.15),
                _apply_scaled(),
            )
        )
        zoom_out_btn.clicked.connect(
            lambda: (
                zoom_state.__setitem__("scale", zoom_state["scale"] / 1.15),
                _apply_scaled(),
            )
        )
        fit_btn.clicked.connect(_fit_to_view)
        one_btn.clicked.connect(
            lambda: (zoom_state.__setitem__("scale", 1.0), _apply_scaled())
        )
        hv.addWidget(img_lb)
        sc.setWidget(holder)
        v.addWidget(sc, 1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)

        # 对话框关闭时清理后台线程，避免泄露
        dlg.finished.connect(lambda *_: _cleanup_render_thread())

        _update_nav_ui()
        _start_render()

        dlg.exec_()
    
    def _on_location_text_changed(self, text):
        """地址文本改变时的处理"""
        # 可选：实时查询或延迟查询
        pass
    
    def _get_gps_coords(self):
        """从经纬度输入框安全读取坐标，超出范围则弹出提示并返回 None"""
        try:
            lat = float(self.lat_input.text().strip())
            lon = float(self.lon_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "经纬度必须是有效数字")
            return None, None
        if not (-90 <= lat <= 90):
            QMessageBox.warning(self, "输入错误", "纬度必须在 -90 ~ 90 之间")
            return None, None
        if not (-180 <= lon <= 180):
            QMessageBox.warning(self, "输入错误", "经度必须在 -180 ~ 180 之间")
            return None, None
        return lat, lon

    def _query_location_gps(self):
        """查询地址的GPS坐标并更新界面"""
        location_name = self.location_input.text().strip()
        if not location_name:
            return
        
        try:
            result = geocode_location(location_name)
            if result:
                lat, lon = result
                # 更新经纬度输入框
                self.lat_input.setText(f"{lat:.6f}")
                self.lon_input.setText(f"{lon:.6f}")
                
                # 保存到配置
                self.config['gps_latitude'] = lat
                self.config['gps_longitude'] = lon
                self.config['location_name'] = location_name
                
                # 查询省市信息
                self._update_province_city(lat, lon)
                
                self.add_log(f"✓ 查询成功: {location_name} -> 纬度: {lat:.6f}, 经度: {lon:.6f}")
            else:
                QMessageBox.warning(self, "查询失败", f"无法找到 '{location_name}' 的GPS坐标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"查询失败: {str(e)}")
    
    def _update_province_city(self, lat, lon):
        """根据经纬度更新省市信息"""
        try:
            from detect_bird_and_eye import locate_province, locate_city
            
            province = locate_province(lon, lat)
            if province:
                city = locate_city(lon, lat, province)
                self.config['province'] = province
                self.config['city'] = city
                
                # 更新省市显示
                province_city_text = f"{province} {city}" if city else province
                self.province_city_display.setText(province_city_text)
                self.add_log(f"✓ 定位到: {province_city_text}")
            else:
                self.province_city_display.setText("未知")
                self.config['province'] = None
                self.config['city'] = None
        except Exception as e:
            self.province_city_display.setText("定位失败")
            self.add_log(f"⚠ 省市定位失败: {e}")

    @staticmethod
    def _format_duration_hms(total_sec: float) -> str:
        """将秒数格式化为中文可读时长（用于界面显示）。"""
        s = int(max(0, round(total_sec)))
        if s < 60:
            return f"{s}秒"
        m, s = s // 60, s % 60
        if m < 60:
            return f"{m}分{s}秒" if s else f"{m}分钟"
        h, m = m // 60, m % 60
        out = f"{h}小时"
        if m:
            out += f"{m}分"
        if s:
            out += f"{s}秒"
        return out

    def _reset_eta_model(self) -> None:
        self._eta_phases = []
        self._flow_eta = FlowEtaEstimator()
        self._eta_phase_rates: Dict[str, Optional[float]] = {}
        self._eta_phase_t0: Dict[str, Optional[float]] = {}
        self._eta_phase_done: Dict[str, int] = {}
        self._eta_phase_total: Dict[str, int] = {}
        self._eta_species_t0 = None
        self._eta_species_done = 0
        self._eta_species_total = 0
        self._ema_sec_per_species = None

    def _persist_eta_learned(self) -> None:
        est = getattr(self, "_flow_eta", None)
        if est is None or not getattr(est, "learned", None):
            return
        try:
            self.config["_eta_learned"] = est.learned_for_config(self.config)
        except Exception:
            pass

    def _compute_eta_remaining_sec(self) -> Optional[float]:
        est = getattr(self, "_flow_eta", None)
        if est is None or not est.phases:
            return None
        return float(est.remaining_sec())

    def _on_eta_checkpoint(self, d: Dict[str, Any]) -> None:
        kind = d.get("kind")
        est = getattr(self, "_flow_eta", None)
        if est is None:
            est = FlowEtaEstimator()
            self._flow_eta = est
        if kind == "start":
            self._reset_eta_model()
            self._flow_eta = FlowEtaEstimator.from_start(
                list(d.get("phases") or []),
                n_images=int(d.get("n_images") or 0),
                n_species_expected=int(d.get("n_species_expected") or 0),
                burst_sec_per=d.get("burst_sec_per"),
                species_sec_per=d.get("species_sec_per"),
                keep_ratio=float(d.get("burst_keep_ratio") or 0.2),
                keep_min=int(d.get("burst_keep_min") or 2),
            )
            self._eta_phases = self._flow_eta.phases
            return
        est = self._flow_eta
        if kind == "phase_begin":
            name = str(d.get("phase") or "")
            if name:
                est.phase_begin(name)
            return
        if kind == "phase_done":
            name = str(d.get("phase") or "")
            if name:
                est.phase_done(name)
                self._persist_eta_learned()
            return
        if kind == "phase_tick":
            name = str(d.get("phase") or "")
            if name:
                est.phase_tick(
                    name,
                    int(d.get("done", 0)),
                    max(1, int(d.get("total", 1))),
                )
            return
        if kind == "burst_result":
            est.burst_result(
                int(d.get("total") or 0),
                int(d.get("kept") or 0),
            )
            return
        if kind == "species_begin":
            n = int(d.get("n", 0))
            est.species_begin(n)
            self._eta_species_total = n
            self._eta_species_done = 0
            return
        if kind == "species_tick":
            done = int(d.get("done", 0))
            total = max(1, int(d.get("total", 1)))
            est.phase_tick("species", done, total)
            self._eta_species_done = done
            self._eta_species_total = total

    def _refresh_process_time_labels(self):
        """已用时间 + 预计剩余（分阶段先验，连拍/识别按各自张数与速度校正）。"""
        if self._process_start_monotonic is None:
            return
        elapsed = time.monotonic() - self._process_start_monotonic
        self._elapsed_label.setText(f"已用时间：{self._format_duration_hms(elapsed)}")
        rem = self._compute_eta_remaining_sec()
        if rem is not None:
            if rem > 86400 * 7:
                self._eta_label.setText("预计剩余：>7天")
            else:
                self._eta_label.setText(f"预计剩余：{self._format_duration_hms(rem)}")
            return
        p = self.progress_bar.value()
        if p <= 0:
            self._eta_label.setText("预计剩余：—")
        elif p >= 100:
            self._eta_label.setText("预计剩余：0秒")
        else:
            remaining = elapsed * (100.0 - float(p)) / float(p)
            if remaining > 86400 * 7:
                self._eta_label.setText("预计剩余：>7天")
            else:
                self._eta_label.setText(f"预计剩余：{self._format_duration_hms(remaining)}")

    def _idle_process_time_labels(self):
        self._elapsed_label.setText("已用时间：—")
        self._eta_label.setText("预计剩余：—")

    @staticmethod
    def _count_images_in_screened(output_folder: str) -> int:
        screened = os.path.join(output_folder.strip(), "Screened_images")
        return len(_collect_image_paths_under(screened))

    def _on_burst_detection_toggled(self, checked: bool):
        """关闭连拍检测时禁用连拍相关参数。"""
        en = checked
        self.time_threshold_input.setEnabled(en)
        self.burst_keep_ratio_input.setEnabled(en)
        self.burst_keep_min_input.setEnabled(en)
        self.use_bird_detection_checkbox.setEnabled(en)
        self.use_eye_detection_checkbox.setEnabled(
            en and self.use_bird_detection_checkbox.isChecked()
        )
        self.use_fast_mode_checkbox.setEnabled(en)

    def _on_bird_detection_toggled(self, checked: bool):
        """鸟眼检测依赖鸟体检测。"""
        can_eye = (
            checked and self.enable_burst_detection_checkbox.isChecked()
        )
        self.use_eye_detection_checkbox.setEnabled(can_eye)
        if not can_eye:
            self.use_eye_detection_checkbox.setChecked(False)

    def start_processing(self):
        """开始处理"""
        self._sync_config_from_ui()
        output_folder = self.config["output_folder"].strip()
        if not output_folder or not self.config.get("crop_output_folder", "").strip():
            QMessageBox.warning(
                self,
                "提示",
                "请填写「输出根目录」（推荐），或在留空根目录时填写「输出文件夹」与「分类归档文件夹」。",
            )
            self._save_config()
            return
        burst_on = self.config["enable_burst_detection"]
        need_species = self.config["enable_species_detection"]
        need_track = self.config.get("enable_track_map_auto", False)
        need_export = self.config.get("enable_record_export_auto", False)
        class_dir = (self.config.get("crop_output_folder") or "").strip()

        if (need_track or need_export) and class_dir and not os.path.isdir(class_dir):
            QMessageBox.warning(
                self,
                "提示",
                "轨迹图 / 观鸟记录导出需要有效的「分类归档」目录。\n\n"
                f"当前路径不存在：\n{class_dir}\n\n"
                "请先完成物种识别归档，或在本工具卡片中指定已有归档目录。",
            )
            self._save_config()
            return

        if not burst_on and need_species:
            if self._count_images_in_screened(output_folder) < 1:
                QMessageBox.warning(
                    self,
                    "提示",
                    "连拍处理未加入主流程时，物种识别将只处理输出文件夹下的 "
                    "Screened_images 中的已筛选照片。\n\n"
                    "当前该目录下没有图片，请先运行一次连拍流程生成筛选结果，"
                    "或在连拍处理中勾选「加入主流程」。",
                )
                self._save_config()
                return

        need_image_folder = burst_on or self.config["enable_gps_write"]
        if self.config["enable_watermark_generation"] and not self.config.get(
            "watermark_input_folder", ""
        ):
            need_image_folder = True
        if need_image_folder and not self.config.get("image_folder"):
            QMessageBox.warning(self, "提示", "请选择图片文件夹")
            self._save_config()
            return

        n_eta = _count_images_for_eta(
            self.config.get("image_folder", ""),
            self.config.get("dual_format_mode", "off"),
        )
        if not burst_on and need_species:
            n_eta = max(
                n_eta,
                self._count_images_in_screened(output_folder),
            )
        self.config["_eta_image_estimate"] = n_eta

        if self.config.get("enable_gps_write"):
            if self.config.get("gps_write_mode", "fixed") == "gpx":
                if not _config_gpx_paths(self.config):
                    QMessageBox.warning(
                        self,
                        "提示",
                        "主流程 GPS 已选「GPX 按拍摄时间」，请先添加有效 GPX 文件。",
                    )
                    self._save_config()
                    return
            else:
                lat, lon = self._get_gps_coords()
                if lat is None:
                    self._save_config()
                    return
                self.config["gps_latitude"] = lat
                self.config["gps_longitude"] = lon
        else:
            try:
                self.config["gps_latitude"] = float(self.lat_input.text().strip())
                self.config["gps_longitude"] = float(self.lon_input.text().strip())
            except ValueError:
                pass
        
        # 创建输出文件夹
        Path(self.config['output_folder']).mkdir(parents=True, exist_ok=True)
        Path(self.config['crop_output_folder']).mkdir(parents=True, exist_ok=True)
        Path(_reports_dir_from_config(self.config)).mkdir(parents=True, exist_ok=True)
        if self.config['watermark_output_folder']:
            Path(self.config['watermark_output_folder']).mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        self._save_config()
        
        # 启动工作线程
        self.worker_thread = WorkerThread(self.config)
        self.worker_thread.progress_updated.connect(self.update_progress)
        self.worker_thread.status_updated.connect(self.update_status)
        self.worker_thread.error_occurred.connect(self.handle_error)
        self.worker_thread.finished.connect(self.processing_finished)
        self.worker_thread.eta_checkpoint.connect(self._on_eta_checkpoint)
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_log.clear()
        self.stats_table.setRowCount(0)
        self._idle_process_time_labels()
        self._reset_eta_model()
        self._process_start_monotonic = time.monotonic()
        self._process_time_timer.start()
        
        self.add_log("开始处理，请稍候...")
        self.worker_thread.start()
        self._refresh_process_time_labels()
    
    def stop_processing(self):
        """停止处理（异步：设置标志位后立即返回，UI 不阻塞；
        worker 线程在下一个检查点退出后通过 finished 信号触发 processing_finished 统一清理）"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.add_log("✗ 正在中止处理，等待当前步骤完成...")
            self.stop_btn.setEnabled(False)  # 防止重复点击
            # 开始按钮保持禁用，直到 processing_finished 被调用
            # UI 状态（计时器、elapsed、eta、按钮）由 processing_finished 统一清理
    
    def update_progress(self, value: int):
        """更新进度条"""
        self.progress_bar.setValue(value)
        self._refresh_process_time_labels()
    
    def update_status(self, message: str):
        """更新状态信息"""
        self.add_log(message)
    
    def handle_error(self, error_msg: str):
        """处理错误"""
        self.add_log(f"❌ 错误: {error_msg}")
        self._process_time_timer.stop()
        if self._process_start_monotonic is not None:
            elapsed = time.monotonic() - self._process_start_monotonic
            self._elapsed_label.setText(f"已用时间：{self._format_duration_hms(elapsed)}")
        self._process_start_monotonic = None
        self._eta_label.setText("预计剩余：—")
        self._reset_eta_model()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        try:
            self._sync_config_from_ui()
            self._save_config()
        except Exception as e:
            print(f"出错后保存配置失败: {e}")
        QMessageBox.critical(self, "处理出错", error_msg)
    
    def processing_finished(self, results: Dict):
        """处理完成（或已中止）"""
        aborted = bool(results.get("_aborted", False))
        self.add_log("\n" + "=" * 60)
        self.add_log("✓ 处理已中止" if aborted else "✓ 处理完成！")
        self.add_log("=" * 60)

        if not aborted:
            self.stats_table.setRowCount(0)
            for i, (key, value) in enumerate(
                _build_processing_stats(results, self.config)
            ):
                self.stats_table.insertRow(i)
                self.stats_table.setItem(i, 0, QTableWidgetItem(str(key)))
                self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))

            path_lines = _build_output_path_logs(results, self.config)
            if path_lines:
                self.add_log("")
                for line in path_lines:
                    self.add_log(line)

        # 恢复UI状态
        self._process_time_timer.stop()
        if self._process_start_monotonic is not None:
            elapsed = time.monotonic() - self._process_start_monotonic
            suffix = "（已中止）" if aborted else ""
            self._elapsed_label.setText(
                f"已用时间：{self._format_duration_hms(elapsed)}{suffix}"
            )
        self._process_start_monotonic = None
        self._eta_label.setText("预计剩余：—" if aborted else "预计剩余：0秒")
        self._persist_eta_learned()
        self._reset_eta_model()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        try:
            self._sync_config_from_ui()
            self._save_config()
            if not aborted:
                self._refresh_track_map_path_ui()
        except Exception as e:
            print(f"完成后保存配置失败: {e}")
        if aborted:
            return
        done_parts: List[str] = []
        flags = results.get("_flow_flags") or _flow_enabled_flags(self.config)
        if "total_images" in results:
            done_parts.append("连拍筛选")
        if "crop_result" in results:
            done_parts.append("物种归档")
        if "track_map" in results:
            done_parts.append("轨迹图")
        if "record_export" in results:
            done_parts.append("观鸟记录导出")
        if "watermark_result" in results:
            done_parts.append("水印")
        summary = "、".join(done_parts) if done_parts else "主流程"
        QMessageBox.information(
            self,
            "处理完成",
            f"{summary} 已完成。\n请查看右侧日志与输出路径。",
        )
    
    def add_log(self, message: str):
        """添加日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.status_log.append(log_message)
        
        # 自动滚动到底部
        cursor = self.status_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.status_log.setTextCursor(cursor)
    
    def clear_log(self):
        """清空日志"""
        self.status_log.clear()

    def _sync_config_from_ui(self) -> None:
        """把当前界面上的选项全部写回 self.config（与「开始处理」写入项保持一致）。"""
        self.config["image_folder"] = self.image_folder_input.text().strip()
        _df = self.dual_format_combo.currentData()
        if _df:
            self.config["dual_format_mode"] = str(_df)
        root = self.output_root_input.text().strip()
        self.config["output_root_folder"] = root
        img = self.config["image_folder"]
        if root:
            slug = _session_slug_from_image_folder(img)
            self.config["output_folder"] = os.path.join(root, f"screened_{slug}")
            self.config["crop_output_folder"] = os.path.join(
                root, f"classification_{slug}"
            )
            self.config["reports_output_folder"] = os.path.join(root, "reports")
        else:
            self.config["output_folder"] = self.output_folder_input.text().strip()
            self.config["crop_output_folder"] = self.crop_folder_input.text().strip()
            self.config.pop("reports_output_folder", None)
        self.config["enable_gps_write"] = self.gps_write_checkbox.isChecked()
        self.config["location_name"] = self.location_input.text().strip()
        try:
            lat = float(self.lat_input.text().strip())
            if -90.0 <= lat <= 90.0:
                self.config["gps_latitude"] = lat
        except ValueError:
            pass
        try:
            lon = float(self.lon_input.text().strip())
            if -180.0 <= lon <= 180.0:
                self.config["gps_longitude"] = lon
        except ValueError:
            pass
        self.config["time_threshold"] = float(self.time_threshold_input.value())
        self.config["burst_keep_ratio"] = float(self.burst_keep_ratio_input.value())
        self.config["burst_keep_min"] = int(self.burst_keep_min_input.value())
        self.config["keep_top_n"] = int(self.config["burst_keep_min"])
        self.config["focus_score_weight"] = float(self.focus_score_weight_input.value())
        self.config["area_score_weight"] = float(self.area_score_weight_input.value())
        self.config["enable_burst_detection"] = (
            self.enable_burst_detection_checkbox.isChecked()
        )
        self.config["use_bird_detection"] = self.use_bird_detection_checkbox.isChecked()
        self.config["use_eye_detection"] = (
            self.use_eye_detection_checkbox.isChecked()
            and self.config["use_bird_detection"]
        )
        self.config["use_fast_mode"] = self.use_fast_mode_checkbox.isChecked()
        self.config["enable_species_detection"] = (
            self.enable_species_checkbox.isChecked()
        )
        self.config["enable_image_clean_before_species"] = (
            self.image_clean_before_species_checkbox.isChecked()
        )
        self.config["image_clean_remove_no_bird"] = (
            self.image_clean_no_bird_checkbox.isChecked()
        )
        self.config["image_clean_remove_blurry"] = (
            self.image_clean_blurry_checkbox.isChecked()
        )
        self.config["image_clean_dedupe"] = (
            self.image_clean_dedupe_checkbox.isChecked()
        )
        self.config["image_clean_min_clarity"] = int(
            self.image_clean_clarity_slider.value()
        )
        self.config["image_clean_dup_similarity"] = int(
            self.image_clean_dup_slider.value()
        )
        self.config["image_clean_folder"] = (
            self.image_clean_folder_input.text().strip()
        )
        _apply_gui_flow_policy(self.config)
        self.config["enable_watermark_generation"] = (
            self.enable_watermark_checkbox.isChecked()
        )
        self.config["watermark_input_folder"] = (
            self.wm_input_folder_input.text().strip()
        )
        self.config["watermark_output_folder"] = (
            self.wm_output_folder_input.text().strip()
        )
        self.config["wm_logo_path"] = self.wm_logo_input.text().strip()
        self.config["wm_enable_location"] = self.wm_location_checkbox.isChecked()
        self.config["wm_location_text"] = self.wm_location_text_input.text().strip()
        self.config["wm_use_gps_city"] = self.wm_use_gps_city_checkbox.isChecked()
        self.config["wm_enable_date"] = self.wm_date_checkbox.isChecked()
        self.config["wm_enable_species"] = self.wm_species_checkbox.isChecked()
        self.config["wm_enable_camera"] = self.wm_camera_checkbox.isChecked()
        self.config["wm_random_per_species"] = (
            self.wm_random_per_species_checkbox.isChecked()
        )
        self.config["wm_random_per_species_count"] = int(
            self.wm_random_per_species_count.value()
        )
        self.config["wm_logo_width_ratio"] = float(
            self.wm_logo_width_ratio_input.value()
        )
        self.config["wm_watermark_style"] = (
            "inline"
            if self.wm_style_combo.currentIndex() == 1
            else "frame"
        )
        self.config["wm_enable_ai_exposure"] = (
            self.wm_ai_exposure_checkbox.isChecked()
        )
        self.config["wm_ai_exposure_strength"] = (
            self.wm_ai_exposure_slider.value() / 100.0
        )
        self.config["wm_enable_ai_denoise"] = (
            self.wm_ai_denoise_checkbox.isChecked()
        )
        self.config["wm_enable_ai_sharpen"] = (
            self.wm_ai_sharpen_checkbox.isChecked()
        )
        _wm_ai_dn_model = str(
            self.wm_ai_denoise_model_combo.currentData() or "realesrgan"
        )
        self.config["wm_ai_denoise_model"] = _wm_ai_dn_model
        self.config["wm_ai_denoise_strength"] = (
            self.wm_ai_denoise_slider.value() / 100.0
        )
        self.config["wm_ai_sharpen_strength"] = (
            self.wm_ai_sharpen_slider.value() / 100.0
        )
        self.config["use_local_model"] = self.local_model_radio.isChecked()
        _lsm = self.local_species_model_combo.currentData()
        if _lsm:
            self.config["local_species_model"] = _lsm
        self.config["species_conf_threshold_enabled"] = (
            self.min_species_threshold_enable_checkbox.isChecked()
        )
        self.config["min_species_accept_confidence"] = float(
            self.min_species_conf_input.value()
        )
        _geo = self.species_geo_mode_combo.currentData()
        if _geo:
            self.config["species_geo_mode"] = normalize_species_geo_mode(_geo)
        self.config["enable_record_export_auto"] = (
            self.enable_record_export_auto_checkbox.isChecked()
        )
        self.config["record_export_classification_folder"] = (
            self.record_export_class_input.text().strip()
        )
        self.config["record_export_output_folder"] = (
            self.record_export_out_input.text().strip()
        )
        self.config["record_export_ebird"] = (
            self.record_export_ebird_checkbox.isChecked()
        )
        self.config["record_export_birdreport"] = (
            self.record_export_birdreport_checkbox.isChecked()
        )
        self.config["record_export_count_individuals"] = (
            self.record_export_count_individuals_checkbox.isChecked()
        )
        self.config["record_export_time_minutes"] = float(
            getattr(self, "_record_export_individual_time_minutes", 120.0)
        )
        self.config["record_export_spatial_km"] = float(
            getattr(self, "_record_export_spatial_km", 0.1)
        )
        self.config["record_export_ebird_country"] = (
            self.record_export_country_input.text().strip() or "CN"
        )
        self.config["record_export_ebird_state"] = (
            self.record_export_state_input.text().strip() or "FJ"
        )
        self._sync_gpx_paths_to_config()
        self.config["gpx_apply_to_screened"] = (
            self.gpx_apply_screened_checkbox.isChecked()
        )
        self.config["enable_track_map_auto"] = (
            self.enable_track_map_auto_checkbox.isChecked()
        )
        self.config["track_map_use_gpx"] = self.track_map_use_gpx_checkbox.isChecked()
        self.config["track_map_use_exif"] = (
            self.track_map_use_exif_checkbox.isChecked()
        )
        _ts = self.track_map_source_combo.currentData()
        if _ts:
            self.config["track_map_photo_source"] = _ts
        if hasattr(self, "track_map_folder_override_input"):
            self.config["track_map_photo_folder_override"] = (
                self.track_map_folder_override_input.text().strip()
            )
        self.config["track_map_radius_km"] = float(
            self.track_map_radius_input.value()
        )
        self.config["track_map_include_elevation"] = (
            self.track_map_elevation_checkbox.isChecked()
        )
        _bm = self.track_map_basemap_combo.currentData()
        if _bm:
            self.config["track_map_basemap_style"] = _bm
        self.config["gpx_match_exif_tz"] = self._gpx_match_exif_tz()
        self.config["gpx_match_gpx_tz"] = self._gpx_match_gpx_tz()
        self.config["gps_write_mode"] = (
            "gpx" if self.gps_mode_gpx_radio.isChecked() else "fixed"
        )

    def _save_config(self):
        """保存配置到文件"""
        config_file = Path(__file__).parent / 'gui_config.json'
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def _load_config(self):
        """从文件加载配置"""
        config_file = Path(__file__).parent / 'gui_config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                    if 'burst_keep_min' not in saved_config and 'keep_top_n' in saved_config:
                        self.config['burst_keep_min'] = saved_config['keep_top_n']
                    if 'burst_keep_ratio' not in saved_config:
                        self.config['burst_keep_ratio'] = 0.2
                    if 'use_local_model' not in saved_config:
                        self.config['use_local_model'] = True
                    if 'local_species_model' not in saved_config:
                        self.config['local_species_model'] = LOCAL_SPECIES_MODEL_RESNET34
                    else:
                        self.config['local_species_model'] = normalize_local_species_model(
                            self.config['local_species_model']
                        )
                    if 'enable_species_detection' not in saved_config:
                        self.config['enable_species_detection'] = self.config.get(
                            'enable_crop', True
                        )
                    if 'species_conf_threshold_enabled' not in saved_config:
                        self.config['species_conf_threshold_enabled'] = False
                    if 'species_geo_mode' not in saved_config:
                        if saved_config.get('use_geo_constraint') is False:
                            self.config['species_geo_mode'] = SPECIES_GEO_MODE_NONE
                        else:
                            self.config['species_geo_mode'] = SPECIES_GEO_MODE_AUTO
                    else:
                        self.config['species_geo_mode'] = normalize_species_geo_mode(
                            self.config['species_geo_mode']
                        )
                    if "output_root_folder" not in saved_config:
                        self.config["output_root_folder"] = ""
                    _apply_gui_flow_policy(self.config)
                    self._update_ui_from_config()
            except Exception as e:
                print(f"加载配置失败: {e}")
    
    def _update_ui_from_config(self):
        """从配置更新UI"""
        _apply_gui_flow_policy(self.config)
        self.image_folder_input.setText(self.config.get('image_folder', ''))
        if hasattr(self, "dual_format_combo"):
            _df = self.config.get("dual_format_mode", "off")
            _dfi = self.dual_format_combo.findData(_df)
            self.dual_format_combo.setCurrentIndex(_dfi if _dfi >= 0 else 0)
        self.output_root_input.setText(self.config.get("output_root_folder", ""))
        self.output_folder_input.setText(self.config.get('output_folder', ''))
        self.crop_folder_input.setText(self.config.get('crop_output_folder', ''))
        self._refresh_derived_paths_display()
        self.gps_write_checkbox.setChecked(self.config.get('enable_gps_write', True))
        self.lat_input.setText(str(self.config.get('gps_latitude', 31.2304)))
        self.lon_input.setText(str(self.config.get('gps_longitude', 121.4737)))
        self.time_threshold_input.setValue(self.config.get('time_threshold', 1.0))
        self.burst_keep_ratio_input.setValue(
            float(self.config.get('burst_keep_ratio', 0.2))
        )
        self.burst_keep_min_input.setValue(
            int(self.config.get('burst_keep_min', self.config.get('keep_top_n', 2)))
        )
        self.focus_score_weight_input.setValue(
            float(self.config.get('focus_score_weight', 9.0))
        )
        self.area_score_weight_input.setValue(
            float(self.config.get('area_score_weight', 1.0))
        )
        self.enable_burst_detection_checkbox.setChecked(
            self.config.get("enable_burst_detection", True)
        )
        self._on_burst_detection_toggled(
            self.enable_burst_detection_checkbox.isChecked()
        )
        self.use_bird_detection_checkbox.setChecked(self.config.get('use_bird_detection', True))
        self.use_eye_detection_checkbox.setChecked(self.config.get('use_eye_detection', False))
        self.use_fast_mode_checkbox.setChecked(self.config.get('use_fast_mode', True))
        self._on_bird_detection_toggled(self.use_bird_detection_checkbox.isChecked())
        self.enable_species_checkbox.setChecked(
            self.config.get('enable_species_detection', True)
        )
        self.image_clean_before_species_checkbox.setChecked(
            bool(self.config.get("enable_image_clean_before_species", False))
        )
        self.image_clean_no_bird_checkbox.setChecked(
            bool(self.config.get("image_clean_remove_no_bird", True))
        )
        self.image_clean_blurry_checkbox.setChecked(
            bool(self.config.get("image_clean_remove_blurry", True))
        )
        self.image_clean_dedupe_checkbox.setChecked(
            bool(self.config.get("image_clean_dedupe", True))
        )
        self.image_clean_clarity_slider.setValue(
            int(self.config.get("image_clean_min_clarity", 35))
        )
        self.image_clean_clarity_label.setText(
            str(self.image_clean_clarity_slider.value())
        )
        self.image_clean_clarity_slider.setEnabled(
            self.image_clean_blurry_checkbox.isChecked()
        )
        self.image_clean_dup_slider.setValue(
            int(self.config.get("image_clean_dup_similarity", 92))
        )
        self.image_clean_dup_label.setText(
            f"{self.image_clean_dup_slider.value()}%"
        )
        self.image_clean_dup_slider.setEnabled(
            self.image_clean_dedupe_checkbox.isChecked()
        )
        self.image_clean_folder_input.setText(
            self.config.get("image_clean_folder", "")
        )
        self.enable_watermark_checkbox.setChecked(
            self.config.get('enable_watermark_generation', False)
        )
        self.wm_input_folder_input.setText(self.config.get('watermark_input_folder', ''))
        self.wm_output_folder_input.setText(
            self.config.get('watermark_output_folder', './watermarked')
        )
        self.wm_logo_input.setText(self.config.get('wm_logo_path', ''))
        self.wm_location_checkbox.setChecked(self.config.get('wm_enable_location', True))
        self.wm_location_text_input.setText(self.config.get('wm_location_text', ''))
        self.wm_use_gps_city_checkbox.setChecked(
            self.config.get('wm_use_gps_city', True)
        )
        self.wm_date_checkbox.setChecked(self.config.get('wm_enable_date', True))
        self.wm_species_checkbox.setChecked(self.config.get('wm_enable_species', True))
        self.wm_camera_checkbox.setChecked(self.config.get('wm_enable_camera', True))
        self.wm_random_per_species_checkbox.setChecked(
            bool(self.config.get("wm_random_per_species", False))
        )
        self.wm_random_per_species_count.setValue(
            max(1, int(self.config.get("wm_random_per_species_count", 3)))
        )
        self.wm_random_per_species_count.setEnabled(
            self.wm_random_per_species_checkbox.isChecked()
        )
        self.wm_logo_width_ratio_input.setValue(
            float(self.config.get('wm_logo_width_ratio', 0.30))
        )
        _wst = str(self.config.get("wm_watermark_style", "frame") or "frame")
        _wi = self.wm_style_combo.findData(_wst)
        self.wm_style_combo.setCurrentIndex(_wi if _wi >= 0 else 0)
        # AI 曝光/降噪/锐化（水印前）
        self.wm_ai_exposure_checkbox.setChecked(
            bool(self.config.get("wm_enable_ai_exposure", False))
        )
        self.wm_ai_denoise_checkbox.setChecked(
            bool(self.config.get("wm_enable_ai_denoise", False))
        )
        self.wm_ai_sharpen_checkbox.setChecked(
            bool(self.config.get("wm_enable_ai_sharpen", False))
        )
        _wm_dm = str(
            self.config.get("wm_ai_denoise_model", "realesrgan") or "realesrgan"
        )
        _wm_di = self.wm_ai_denoise_model_combo.findData(_wm_dm)
        self.wm_ai_denoise_model_combo.setCurrentIndex(
            _wm_di if _wm_di >= 0 else 0
        )
        self.wm_ai_denoise_slider.setValue(
            int(float(self.config.get("wm_ai_denoise_strength", 0.5)) * 100)
        )
        self.wm_ai_sharpen_slider.setValue(
            int(float(self.config.get("wm_ai_sharpen_strength", 0.5)) * 100)
        )
        self.wm_ai_exposure_slider.setValue(
            int(float(self.config.get("wm_ai_exposure_strength", 1.0)) * 100)
        )
        self._update_wm_ai_denoise_enabled()
        self.wm_ai_sharpen_slider.setEnabled(
            self.wm_ai_sharpen_checkbox.isChecked()
        )
        self.wm_ai_exposure_slider.setEnabled(
            self.wm_ai_exposure_checkbox.isChecked()
        )

        # 物种识别配置
        use_local = self.config.get('use_local_model', True)
        self.local_model_radio.setChecked(use_local)
        self.doubao_model_radio.setChecked(not use_local)
        _lsm = normalize_local_species_model(
            self.config.get('local_species_model', LOCAL_SPECIES_MODEL_RESNET34)
        )
        self.config['local_species_model'] = _lsm
        _li = self.local_species_model_combo.findData(_lsm)
        self.local_species_model_combo.setCurrentIndex(_li if _li >= 0 else 0)
        self._update_local_species_model_combo_enabled()
        self.min_species_threshold_enable_checkbox.setChecked(
            self.config.get('species_conf_threshold_enabled', False)
        )
        self.min_species_conf_input.setEnabled(
            self.min_species_threshold_enable_checkbox.isChecked()
        )
        self.min_species_conf_input.setValue(
            float(self.config.get('min_species_accept_confidence', 0.5))
        )
        _geo = normalize_species_geo_mode(
            self.config.get('species_geo_mode', SPECIES_GEO_MODE_AUTO)
        )
        self.config['species_geo_mode'] = _geo
        _gi = self.species_geo_mode_combo.findData(_geo)
        self.species_geo_mode_combo.setCurrentIndex(_gi if _gi >= 0 else 0)

        self.enable_record_export_auto_checkbox.setChecked(
            self.config.get("enable_record_export_auto", False)
        )
        self.record_export_class_input.setText(
            self.config.get("record_export_classification_folder", "")
        )
        self.record_export_out_input.setText(
            self.config.get("record_export_output_folder", "")
        )
        self.record_export_ebird_checkbox.setChecked(
            self.config.get("record_export_ebird", True)
        )
        self.record_export_birdreport_checkbox.setChecked(
            self.config.get("record_export_birdreport", True)
        )
        self.record_export_country_input.setText(
            self.config.get("record_export_ebird_country", "CN")
        )
        self.record_export_state_input.setText(
            self.config.get("record_export_ebird_state", "FJ")
        )
        self.record_export_count_individuals_checkbox.setChecked(
            self.config.get("record_export_count_individuals", True)
        )
        self._record_export_individual_time_minutes = float(
            self.config.get("record_export_time_minutes", 120.0) or 120.0
        )
        self._record_export_spatial_km = float(
            self.config.get("record_export_spatial_km", 0.1) or 0.1
        )
        self._refresh_record_export_classification_default()
        self._apply_collapsible_sections_from_config()

        self._set_gpx_paths_to_ui(_config_gpx_paths(self.config))
        self.gpx_apply_screened_checkbox.setChecked(
            self.config.get("gpx_apply_to_screened", True)
        )
        self.enable_track_map_auto_checkbox.setChecked(
            self.config.get("enable_track_map_auto", False)
        )
        self.track_map_use_gpx_checkbox.setChecked(
            self.config.get("track_map_use_gpx", True)
        )
        self.track_map_use_exif_checkbox.setChecked(
            self.config.get("track_map_use_exif", True)
        )
        self._on_track_map_use_gpx_changed(
            Qt.Checked if self.track_map_use_gpx_checkbox.isChecked() else Qt.Unchecked
        )
        _tsrc = self.config.get("track_map_photo_source", "classification")
        _tsi = self.track_map_source_combo.findData(_tsrc)
        self.track_map_source_combo.setCurrentIndex(_tsi if _tsi >= 0 else 0)
        if hasattr(self, "track_map_folder_override_input"):
            self.track_map_folder_override_input.setText(
                self.config.get("track_map_photo_folder_override", "")
            )
        self._refresh_track_map_path_ui()
        self.track_map_radius_input.setValue(
            float(self.config.get("track_map_radius_km", 1.0))
        )
        self.track_map_elevation_checkbox.setChecked(
            self.config.get("track_map_include_elevation", True)
        )
        try:
            from gpx_track.amap_basemap import normalize_basemap_style as _norm_bm
            _bm = _norm_bm(self.config.get("track_map_basemap_style", "normal"))
        except Exception:
            _bm = self.config.get("track_map_basemap_style", "normal")
            if _bm == "digital":
                _bm = "normal"
        _bmi = self.track_map_basemap_combo.findData(_bm)
        self.track_map_basemap_combo.setCurrentIndex(_bmi if _bmi >= 0 else 0)
        set_combo_timezone(
            self.gpx_match_exif_tz_combo, self._config_gpx_match_exif_tz()
        )
        set_combo_timezone(
            self.gpx_match_gpx_tz_combo, self._config_gpx_match_gpx_tz()
        )
        _gps_mode = self.config.get("gps_write_mode", "fixed")
        if _gps_mode == "gpx":
            self.gps_mode_gpx_radio.setChecked(True)
        else:
            self.gps_mode_fixed_radio.setChecked(True)
        self._on_gps_write_mode_changed()

        # 更新地理位置相关UI
        self.location_input.setText(self.config.get('location_name', ''))
        province = self.config.get('province', '')
        city = self.config.get('city', '')
        if province or city:
            self.province_city_display.setText(f"{province} {city}".strip())
    
    def closeEvent(self, event):
        """关闭窗口事件"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认关闭", 
                "处理正在进行中，确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            else:
                self.worker_thread.stop()
                self.worker_thread.wait()
                self._process_time_timer.stop()
                self._process_start_monotonic = None
        if self._wm_batch_thread is not None and self._wm_batch_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认关闭",
                "批量水印仍在后台运行，关闭窗口将中止该任务。确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._wm_batch_thread.wait()
        if self._image_clean_thread is not None and self._image_clean_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "图片清洗仍在进行，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._image_clean_thread.wait()
        try:
            self._sync_config_from_ui()
            self._save_config()
        except Exception as e:
            print(f"关闭时保存配置失败: {e}")
        event.accept()


if __name__ == "__main__":
    # Windows：与宿主 python.exe（如 Anaconda 带 Jupyter 图标）区分任务栏身份，否则壳层可能显示错误图标
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "WorkBuddy.Birdy.NiaotuSmartStorage.GUI.1.0"
            )
        except Exception:
            pass

    # 必须在创建 QApplication 之前启用，否则 Windows 高缩放比例下整窗会偏小
    from PyQt5.QtCore import Qt as _QtCoreQt

    QApplication.setAttribute(_QtCoreQt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(_QtCoreQt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = BirdDetectionGUI()
    # 部分环境下需同时设置到 QApplication，任务栏才采用窗口图标
    app.setWindowIcon(window.windowIcon())
    sys.exit(app.exec_())
