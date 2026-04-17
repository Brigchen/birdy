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
        QDialog, QDialogButtonBox, QRadioButton, QScrollArea, QFrame,
        QSizePolicy,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
    from PyQt5.QtGui import QColor, QTextCursor, QIcon, QPalette, QDesktopServices
except ImportError:
    print("错误: 未安装PyQt5。请运行: pip install PyQt5")
    sys.exit(1)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from burst_grouping import process_folder, get_kept_images
from html_report_generator import generate_html_report
from geo_encoder import batch_write_gps_exif, geocode_location
from detect_bird_and_eye import BirdAndEyeDetector
from api_config_defaults import ensure_doubao_api_config_file, ensure_amap_api_config_file
from watermark_generator import (
    WatermarkOptions,
    choose_default_watermark_source,
    collect_images_recursive,
    generate_watermarks,
    render_watermark_for_image,
)


def _open_local_file(path: str) -> None:
    """在资源管理器/系统默认编辑器中打开本地文件（跨平台）。"""
    path = os.path.abspath(path)
    if sys.platform == "win32":
        os.startfile(path)
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _count_images_for_eta(folder: str) -> int:
    """与连拍/物种步骤同级的图片数量预估（递归 jpg/jpeg/png）。"""
    if not folder or not os.path.isdir(folder):
        return 0
    return len(_collect_image_paths_under(folder))


def _build_eta_phase_estimates(config: Dict, n_images: int) -> List[Tuple[str, float]]:
    """
    各阶段耗时粗估（秒），用于初始剩余时间与虚拟进度。
    n_images：输入目录或用于物种步骤的图片数预估。
    """
    n = max(0, int(n_images))
    burst_on = config.get("enable_burst_detection", True)
    do_species = config.get("enable_species_detection", True)
    do_crop = config.get("enable_crop", False)
    use_local = config.get("use_local_model", True)
    phases: List[Tuple[str, float]] = []
    if config.get("enable_gps_write"):
        phases.append(("gps", max(3.0, n * 0.35)))
    if burst_on:
        phases.append(("burst", max(18.0, n * 2.8)))
        if config.get("generate_burst_report", True):
            phases.append(("burst_report", max(10.0, min(120.0, 15.0 + n * 0.12))))
    if do_species or do_crop:
        per = 5.0 if use_local else 14.0
        phases.append(("species", max(25.0, n * per)))
    if config.get("enable_watermark_generation", False):
        phases.append(("watermark", max(12.0, n * 0.8)))
    return phases


def _collect_image_paths_under(root: str) -> List[str]:
    """递归收集 root 下 jpg/jpeg/png 路径（去重）。"""
    import fnmatch
    import os
    if not root or not os.path.isdir(root):
        return []
    patterns = ("*.jpg", "*.jpeg", "*.png")
    out: List[str] = []
    for pattern in patterns:
        for walk_root, _dirs, files in os.walk(root):
            for file in files:
                if fnmatch.fnmatch(file.lower(), pattern.lower()):
                    p = os.path.join(walk_root, file)
                    if p not in out:
                        out.append(p)
    return out


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
            burst_on = config.get("enable_burst_detection", True)
            total_steps = 0
            if config.get("enable_gps_write"):
                total_steps += 1
            if burst_on:
                total_steps += 2  # 连拍筛选 + 连拍报告
            if config.get("enable_species_detection", True) or config.get(
                "enable_crop", False
            ):
                total_steps += 1
            if config.get("enable_watermark_generation", False):
                total_steps += 1
            if total_steps < 1:
                total_steps = 1
            current_step = 0
            results = {}
            burst_filter_applied = False
            phase_weights: Dict[str, float] = {
                "gps": 0.08,
                "burst": 0.40,
                "burst_report": 0.07,
                "species": 0.35,
                "watermark": 0.10,
            }
            enabled_phases: List[str] = []
            if config.get("enable_gps_write"):
                enabled_phases.append("gps")
            if burst_on:
                enabled_phases.append("burst")
                if config.get("generate_burst_report", True):
                    enabled_phases.append("burst_report")
            if config.get("enable_species_detection", True) or config.get(
                "enable_crop", False
            ):
                enabled_phases.append("species")
            if config.get("enable_watermark_generation", False):
                enabled_phases.append("watermark")
            if not enabled_phases:
                enabled_phases = ["species"]

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
                n_eta = _count_images_for_eta(config.get("image_folder", ""))
            phase_ests = _build_eta_phase_estimates(config, n_eta)
            self.eta_checkpoint.emit(
                {
                    "kind": "start",
                    "n_images": n_eta,
                    "phases": [
                        {"name": n, "est": float(e)} for n, e in phase_ests
                    ],
                }
            )

            # 第一步：GPS写入
            if config.get('enable_gps_write'):
                current_step += 1
                self.status_updated.emit(f"[步骤 {current_step}/{total_steps}] 写入GPS EXIF...")
                _emit_phase_progress("gps", 0, 1)
                self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "gps"})
                
                try:
                    gps_count = batch_write_gps_exif(
                        image_folder=config['image_folder'],
                        latitude=config['gps_latitude'],
                        longitude=config['gps_longitude'],
                        altitude=config.get('gps_altitude', 0)
                    )
                    self.status_updated.emit(f"✓ 成功写入 {gps_count} 张图片的 GPS")
                    results['gps_written'] = gps_count
                except Exception as e:
                    self.error_occurred.emit(f"GPS写入失败: {str(e)}")
                    results['gps_written'] = 0
                _emit_phase_progress("gps", 1, 1)
                self.eta_checkpoint.emit({"kind": "phase_done", "phase": "gps"})
            
            if not self.is_running:
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
                            progress_callback=lambda d: _emit_phase_progress(
                                "burst",
                                int(d.get("done", 0)),
                                int(d.get("total", 1)),
                            ),
                        )
                        self.status_updated.emit(
                            f"✓ 处理 {burst_result['total_images']} 张图片，保留 {burst_result['kept_images']} 张"
                        )
                        results.update(burst_result)
                        burst_filter_applied = True
                except Exception as e:
                    self.error_occurred.emit(f"连拍识别失败: {str(e)}")
                    traceback.print_exc()
                _emit_phase_progress("burst", 1, 1)
                self.eta_checkpoint.emit({"kind": "phase_done", "phase": "burst"})

                if not self.is_running:
                    return

                # 第三步：生成连拍报告
                current_step += 1
                if config.get("generate_burst_report", True):
                    self.status_updated.emit(
                        f"[步骤 {current_step}/{total_steps}] 生成可视化报告..."
                    )
                    _emit_phase_progress("burst_report", 0, 1)
                    self.eta_checkpoint.emit(
                        {"kind": "phase_begin", "phase": "burst_report"}
                    )

                    try:
                        json_report_file = os.path.join(
                            config["output_folder"], "burst_analysis.json"
                        )
                        if os.path.exists(json_report_file):
                            reports_dir = os.path.join(
                                config["output_folder"], "reports"
                            )
                            os.makedirs(reports_dir, exist_ok=True)

                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            html_report_path = os.path.join(
                                reports_dir, f"连拍分析报告_{timestamp}.html"
                            )

                            generate_html_report(
                                json_report_path=json_report_file,
                                output_html_path=html_report_path,
                                image_folder=config["image_folder"],
                            )
                            self.status_updated.emit(
                                f"✓ HTML报告已生成: {os.path.basename(html_report_path)}"
                            )
                        else:
                            self.status_updated.emit(
                                "⚠ 跳过报告生成：JSON报告文件不存在"
                            )
                    except Exception as e:
                        self.error_occurred.emit(f"报告生成失败: {str(e)}")
                    _emit_phase_progress("burst_report", 1, 1)
                    self.eta_checkpoint.emit(
                        {"kind": "phase_done", "phase": "burst_report"}
                    )
                else:
                    self.status_updated.emit(
                        f"[步骤 {current_step}/{total_steps}] 跳过报告生成"
                    )
                    _emit_phase_progress("burst_report", 1, 1)
                    self.eta_checkpoint.emit(
                        {"kind": "phase_done", "phase": "burst_report"}
                    )

                if not self.is_running:
                    return
            else:
                self.status_updated.emit(
                    "已跳过连拍检测，物种等后续步骤将使用输出文件夹下的 Screened_images"
                )

            # 下一步：物种识别 / 裁剪或原图归档
            do_species = config.get('enable_species_detection', True)
            do_crop = config.get('enable_crop', False)
            if do_species or do_crop:
                current_step += 1
                step_title = (
                    "物种检测、裁剪与归档"
                    if do_species and do_crop
                    else (
                        "物种识别（原图按物种归档）"
                        if do_species and not do_crop
                        else "鸟体检测与裁剪（无物种识别）"
                    )
                )
                self.status_updated.emit(
                    f"[步骤 {current_step}/{total_steps}] {step_title}..."
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
                        doubao_config=doubao_config,
                        min_species_accept_confidence=min_species_thr,
                    )
                    
                    # 连拍成功后仅处理筛选保留的图片；连拍失败则回退为扫描整个输入目录
                    import fnmatch
                    
                    output_root = config['crop_output_folder']
                    image_patterns = ('*.jpg', '*.jpeg', '*.png')
                    
                    if burst_filter_applied:
                        image_files = [
                            p for p in get_kept_images(results)
                            if os.path.isfile(p)
                        ]
                        self.status_updated.emit(
                            f"物种识别/归档输入为连拍筛选保留的 {len(image_files)} 张（非全部原始图）"
                        )
                    elif not burst_on:
                        image_files = _collect_image_paths_under(screened_dir)
                        if not image_files:
                            msg = (
                                "输出文件夹下的 Screened_images 中未找到图片。\n"
                                f"路径：{screened_dir}\n"
                                "请先完成连拍筛选并生成该目录，或勾选「连拍检测」。"
                            )
                            self.status_updated.emit(f"⚠ {msg}")
                            self.error_occurred.emit(msg)
                            return
                        self.status_updated.emit(
                            f"物种识别/归档：使用已筛选目录，共 {len(image_files)} 张"
                        )
                    else:
                        image_folder = config['image_folder']
                        image_files = []
                        for pattern in image_patterns:
                            for root, dirs, files in os.walk(image_folder):
                                for file in files:
                                    if fnmatch.fnmatch(file.lower(), pattern.lower()):
                                        image_path = os.path.join(root, file)
                                        if image_path not in image_files:
                                            image_files.append(image_path)
                        image_files = list(set(image_files))
                        self.status_updated.emit(
                            "物种识别/归档：连拍步骤未成功完成，扫描全部输入图片"
                        )
                    
                    total_crops = 0
                    archive_counter = {"n": 0}
                    # 收集物种识别结果
                    species_results = []
                    n_spec = len(image_files)
                    self.eta_checkpoint.emit({"kind": "species_begin", "n": n_spec})
                    def _cfg_geo_str(v):
                        if v is None:
                            return None
                        s = str(v).strip()
                        return s or None

                    manual_province = _cfg_geo_str(config.get("province"))
                    manual_city = _cfg_geo_str(config.get("city"))

                    for idx, image_file in enumerate(image_files):
                        if not self.is_running:
                            break
                        
                        self.status_updated.emit(f"处理中: {os.path.basename(image_file)} ({idx+1}/{len(image_files)})")
                        
                        try:
                            result_image, detection_results = detector.detect(
                                image_file,
                                manual_province=manual_province,
                                manual_city=manual_city,
                            )
                            
                            crop_paths = []
                            if detection_results.get('birds'):
                                province = detection_results.get("province")
                                city = detection_results.get("city")
                                if do_crop:
                                    saved_paths = detector.crop_species(
                                        image=detector.load_image(image_file),
                                        birds=detection_results['birds'],
                                        output_dir=output_root,
                                        source_path=image_file,
                                        province=province,
                                        city=city,
                                        counter=archive_counter,
                                    )
                                    crop_paths = saved_paths
                                    total_crops += len(saved_paths)
                                elif do_species:
                                    saved_paths = (
                                        detector.copy_original_by_top_species(
                                            source_path=image_file,
                                            birds=detection_results['birds'],
                                            output_dir=output_root,
                                            province=province,
                                            city=city,
                                            counter=archive_counter,
                                        )
                                    )
                                    crop_paths = saved_paths
                                    total_crops += len(saved_paths)
                            
                            # 收集物种识别结果（含未知种类）
                            if detection_results.get('birds'):
                                for bird in detection_results['birds']:
                                    sp_list = bird.get('species') or []
                                    if not sp_list:
                                        sp_list = [
                                            {
                                                "chinese_name": "未知种类",
                                                "english_name": "",
                                                "scientific_name": "",
                                                "confidence": 0.0,
                                            }
                                        ]
                                    species_results.append({
                                        'image': os.path.basename(image_file),
                                        'species': sp_list,
                                        'method': bird.get('species_method', '未知'),
                                        'crop_paths': crop_paths,
                                    })
                            else:
                                # 即使没有检测到鸟，也添加到结果中，确保所有图片都出现在报告中
                                species_results.append({
                                    'image': os.path.basename(image_file),
                                    'species': [],
                                    'method': '未检测到鸟',
                                    'crop_paths': []
                                })
                        except Exception as e:
                            self.status_updated.emit(f"⚠ {os.path.basename(image_file)}: {str(e)}")
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
                        f"✓ 已输出 {total_crops} 个文件（裁剪或原图归档），耗时 {processing_time:.2f} 秒"
                    )
                    results_dict = {
                        'total_crops': total_crops,
                        'species_method': detector.get_species_method(),
                        'processing_time': processing_time
                    }
                    results['crop_result'] = results_dict
                    
                    # 生成物种识别报告
                    if config.get('generate_species_report', True):
                        try:
                            # 确保outputs\reports目录存在
                            reports_dir = os.path.join(config['output_folder'], 'reports')
                            os.makedirs(reports_dir, exist_ok=True)
                            
                            # 生成带时间戳的中文报告名称
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            species_report_path = os.path.join(reports_dir, f'物种识别报告_{timestamp}.html')
                            
                            # 生成HTML报告
                            if species_results:
                                html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>物种识别报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .image {{ font-weight: bold; }}
        .species {{ margin-left: 20px; }}
        .method {{ font-style: italic; color: #666; }}
        .crop-images {{ margin-left: 20px; margin-top: 10px; }}
        .crop-image {{ margin: 5px; display: inline-block; }}
        .crop-image img {{ max-width: 200px; max-height: 200px; }}
    </style>
</head>
<body>
    <h1>物种识别报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>处理图片数量: {len(image_files)}</p>
    <p>检测到鸟体数量: {total_crops}</p>
    
    <h2>识别结果</h2>
'''
                            
                            for item in species_results:
                                html_content += f'''
    <div class="result">
        <div class="image">图片: {item['image']}</div>
        <div class="method">识别方法: {item['method']}</div>
'''
                                for species in item['species']:
                                    # 统一物种名称到本地数据库
                                    from detect_bird_and_eye import lookup_classification
                                    chinese_name = species.get('chinese_name', '未知')
                                    scientific_name = species.get('scientific_name', '')
                                    classification = lookup_classification(chinese_name, scientific_name)
                                    unified_chinese_name = classification.get('species_cn', chinese_name)
                                    
                                    html_content += f'''
        <div class="species">
            中文名: {unified_chinese_name}<br>
            英文名: {species.get('english_name', '未知')}<br>
            学名: {species.get('scientific_name', '未知')}<br>
            置信度: {species.get('confidence', 0):.2f}
        </div>
'''
                                # 添加裁剪图
                                if item.get('crop_paths'):
                                    html_content += '''
        <div class="crop-images">
            <strong>裁剪图:</strong><br>
'''
                                    for crop_path in item['crop_paths']:
                                        # 计算裁剪图相对于报告目录的路径
                                        relative_path = os.path.relpath(crop_path, reports_dir)
                                        html_content += f'''
            <div class="crop-image">
                <img src="{relative_path}" alt="裁剪图">
            </div>
'''
                                    html_content += '''
        </div>
'''
                                html_content += '''
    </div>
'''
                            
                            html_content += '''
</body>
</html>
'''
                            
                            with open(species_report_path, 'w', encoding='utf-8') as f:
                                f.write(html_content)
                            
                            self.status_updated.emit(f"✓ 物种识别报告已生成: {os.path.basename(species_report_path)}")
                        except Exception as e:
                            self.error_occurred.emit(f"物种识别报告生成失败: {str(e)}")
                    
                except Exception as e:
                    self.error_occurred.emit(f"物种检测/归档失败: {str(e)}")
                    traceback.print_exc()
                _emit_phase_progress("species", 1, 1)
                self.eta_checkpoint.emit({"kind": "phase_done", "phase": "species"})

            # 水印生成
            if config.get("enable_watermark_generation", False):
                current_step += 1
                self.status_updated.emit(
                    f"[步骤 {current_step}/{total_steps}] 生成水印图片..."
                )
                _emit_phase_progress("watermark", 0, 1)
                self.eta_checkpoint.emit({"kind": "phase_begin", "phase": "watermark"})
                try:
                    source_folder = config.get("watermark_input_folder", "").strip()
                    if not source_folder:
                        source_folder = choose_default_watermark_source(
                            image_folder=config.get("image_folder", ""),
                            crop_output_folder=config.get("crop_output_folder", ""),
                            output_folder=config.get("output_folder", ""),
                        )
                    output_folder = (
                        config.get("watermark_output_folder", "").strip()
                        or os.path.join(config.get("output_folder", "./outputs"), "watermarked")
                    )
                    wopt = WatermarkOptions(
                        enable_location=bool(config.get("wm_enable_location", True)),
                        location_text=str(config.get("wm_location_text", "") or ""),
                        use_gps_city=bool(config.get("wm_use_gps_city", True)),
                        enable_date=bool(config.get("wm_enable_date", True)),
                        enable_species=bool(config.get("wm_enable_species", True)),
                        enable_camera_params=bool(config.get("wm_enable_camera", True)),
                        logo_path=str(config.get("wm_logo_path", "") or ""),
                        logo_width_ratio=float(config.get("wm_logo_width_ratio", 0.30)),
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
        self.config: Dict = self._get_default_config()
        self._process_start_monotonic: Optional[float] = None
        self._process_time_timer = QTimer(self)
        self._process_time_timer.setInterval(500)
        self._process_time_timer.timeout.connect(self._refresh_process_time_labels)
        self._eta_phases: List[Dict[str, Any]] = []
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
                v = str(d.get("version", "2.0.0"))
                rd = str(d.get("release_date", "") or "").strip()
                if rd:
                    return f"{v}（{rd}）"
                return v
        except Exception:
            pass
        return "2.0.0"

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
            'enable_burst_detection': True,
            'use_bird_detection': True,
            'use_eye_detection': False,
            'use_fast_mode': True,
            'generate_burst_report': True,
            'enable_species_detection': True,
            'enable_crop': True,
            'use_geo_constraint': True,
            'generate_species_report': True,
            # 物种识别模式配置
            'use_local_model': True,  # 默认使用本地模型
            'enable_doubao_api': False,  # 默认不启用豆包API
            'doubao_api_key': '',
            # 未知种类阈值：仅当 species_conf_threshold_enabled 为 True 时生效
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
        
        # 输出文件夹
        output_row = QHBoxLayout()
        self.output_folder_input = QLineEdit()
        self.output_folder_input.setText(self.config['output_folder'])
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(lambda: self._select_folder('output_folder'))
        output_row.addWidget(self.output_folder_input, 1)
        output_row.addWidget(output_btn)
        folder_layout.addRow("输出文件夹:", output_row)
        
        # 分类归档文件夹（物种目录输出根路径）
        crop_row = QHBoxLayout()
        self.crop_folder_input = QLineEdit()
        self.crop_folder_input.setText(self.config['crop_output_folder'])
        crop_btn = QPushButton("浏览...")
        crop_btn.clicked.connect(lambda: self._select_folder('crop_output_folder'))
        crop_row.addWidget(self.crop_folder_input, 1)
        crop_row.addWidget(crop_btn)
        folder_layout.addRow("分类归档文件夹:", crop_row)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_card)
        
        # ═════ 地理位置设置 ═════
        geo_card, geo_group = self._create_card("🌍 地理位置")
        geo_layout = QFormLayout()
        geo_layout.setSpacing(8)
        geo_layout.setContentsMargins(12, 10, 12, 12)
        
        # GPS写入Exif开关
        gps_write_row = QHBoxLayout()
        self.gps_write_checkbox = QCheckBox("写入GPS到Exif")
        self.gps_write_checkbox.setChecked(self.config['enable_gps_write'])
        self.gps_write_checkbox.setToolTip("是否将GPS坐标写入图片的Exif信息中")
        self.gps_write_checkbox.stateChanged.connect(self._on_gps_write_changed)
        gps_write_row.addWidget(self.gps_write_checkbox)
        gps_write_row.addStretch()
        amap_cfg_label = QLabel("高德API:")
        amap_cfg_label.setToolTip("地名转坐标优先走高德；密钥在 amap_api_config.json 中配置")
        gps_write_row.addWidget(amap_cfg_label)
        amap_cfg_btn = QPushButton("打开配置文件")
        amap_cfg_btn.setToolTip("编辑 amap_api_config.json，填写 api_key（与豆包配置方式相同）")
        amap_cfg_btn.clicked.connect(self._open_amap_config_file)
        gps_write_row.addWidget(amap_cfg_btn)
        geo_layout.addRow("GPS写入:", gps_write_row)
        
        # 地址输入和查询
        location_row = QHBoxLayout()
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("输入地址，如：厦门大学翔安校区")
        self.location_input.textChanged.connect(self._on_location_text_changed)
        self.location_input.editingFinished.connect(self._query_location_gps)
        location_row.addWidget(self.location_input, 1)
        
        query_btn = QPushButton("查询")
        query_btn.clicked.connect(self._query_location_gps)
        query_btn.setToolTip("点击查询GPS坐标")
        location_row.addWidget(query_btn)
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
        
        geo_group.setLayout(geo_layout)
        layout.addWidget(geo_card)
        
        # ═════ 连拍处理设置 ═════
        process_card, process_group = self._create_card("📷 连拍处理")
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
            "同一连拍组内张数较多时，按 组内张数×比例 与「最少保留」取较大值后封顶组大小"
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
        
        # 连拍检测（关闭则跳过连拍，直接用输出目录下 Screened_images）
        self.enable_burst_detection_checkbox = QCheckBox("连拍检测")
        self.enable_burst_detection_checkbox.setToolTip(
            "勾选：对「图片文件夹」做连拍分组与筛选，并写入输出目录下的 Screened_images。\n"
            "不勾选：跳过连拍，物种识别等步骤仅从「输出文件夹」下的 Screened_images 读取已筛选照片。"
        )
        self.enable_burst_detection_checkbox.setChecked(
            self.config.get("enable_burst_detection", True)
        )
        self.enable_burst_detection_checkbox.toggled.connect(
            self._on_burst_detection_toggled
        )
        process_layout.addRow("", self.enable_burst_detection_checkbox)
        
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
        process_layout.addRow("", self.use_fast_mode_checkbox)
        
        # 生成连拍报告
        self.generate_burst_report_checkbox = QCheckBox("生成连拍报告")
        self.generate_burst_report_checkbox.setChecked(self.config.get('generate_burst_report', True))
        process_layout.addRow("", self.generate_burst_report_checkbox)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_card)
        self._on_burst_detection_toggled(
            self.enable_burst_detection_checkbox.isChecked()
        )
        self._on_bird_detection_toggled(self.use_bird_detection_checkbox.isChecked())
        
        # ═════ 物种识别设置 ═════
        species_card, species_group = self._create_card("🦅 物种识别")
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
            "勾选「启用」后：低于该值的候选不会进入地理筛选；顶一低于该值也会视为未知。\n"
            "不勾选：不对 top10 做该阈值筛选，仅按地理名单与名单外>0.75 规则处理。"
        )
        self.min_species_threshold_enable_checkbox.setToolTip(
            "默认关闭：不限制 top10 置信度，直接按地理规则选种。"
        )
        self.min_species_threshold_enable_checkbox.toggled.connect(
            self.min_species_conf_input.setEnabled
        )
        min_species_row.addWidget(self.min_species_threshold_enable_checkbox)
        min_species_row.addWidget(self.min_species_conf_input)
        min_species_row.addStretch()
        species_layout.addRow("未知种类阈值(可选):", min_species_row)

        # 物种识别与裁剪拆分为两项
        self.enable_species_checkbox = QCheckBox(
            "启用物种识别（地理限制 ）"
        )
        self.enable_species_checkbox.setChecked(
            self.config.get('enable_species_detection', True)
        )
        species_layout.addRow("", self.enable_species_checkbox)

        self.enable_crop_checkbox = QCheckBox("启用按鸟体裁剪输出")
        self.enable_crop_checkbox.setToolTip(
            "关闭时：不裁剪；将整张原图复制到「分类归档文件夹」下，"
            "按置信度最高的物种归入 目/科/属/种 目录（<70% 归入未知）。"
        )
        self.enable_crop_checkbox.setChecked(self.config.get('enable_crop', True))
        species_layout.addRow("", self.enable_crop_checkbox)
        
        # 生成物种识别报告
        self.generate_species_report_checkbox = QCheckBox("生成物种识别报告")
        self.generate_species_report_checkbox.setChecked(self.config.get('generate_species_report', True))
        species_layout.addRow("", self.generate_species_report_checkbox)
        
        species_group.setLayout(species_layout)
        layout.addWidget(species_card)

        # ═════ 水印生成设置 ═════
        wm_card, wm_group = self._create_card("🖼 水印生成")
        wm_layout = QFormLayout()
        wm_layout.setSpacing(8)
        wm_layout.setContentsMargins(12, 10, 12, 12)

        self.enable_watermark_checkbox = QCheckBox("启用水印生成")
        self.enable_watermark_checkbox.setChecked(
            self.config.get("enable_watermark_generation", False)
        )
        wm_layout.addRow("", self.enable_watermark_checkbox)

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

        wm_preview_row = QHBoxLayout()
        wm_preview_btn = QPushButton("预览一张效果")
        wm_preview_btn.clicked.connect(self._preview_watermark_one)
        wm_preview_row.addWidget(wm_preview_btn)
        wm_run_btn = QPushButton("批量水印生成")
        wm_run_btn.clicked.connect(self._run_watermark_batch)
        wm_preview_row.addWidget(wm_run_btn)
        wm_preview_row.addStretch(1)
        wm_layout.addRow("", wm_preview_row)

        wm_group.setLayout(wm_layout)
        layout.addWidget(wm_card)
        
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
        info_label = QLabel("💡 提示：选择图片文件夹后点击'开始处理'")
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

    
    def _on_model_mode_changed(self, index: int):
        """模型模式切换"""
        self.config['use_local_model'] = (index == 0)
    
    def _on_gps_write_changed(self, state: int):
        """GPS写入开关状态变化"""
        self.config['enable_gps_write'] = (state == Qt.Checked)
    
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
    
    def _select_folder(self, field_name: str):
        """选择文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self, f"选择{field_name}文件夹"
        )
        if folder:
            if field_name == 'image_folder':
                self.config['image_folder'] = folder
                self.image_folder_input.setText(folder)
            elif field_name == 'output_folder':
                self.config['output_folder'] = folder
                self.output_folder_input.setText(folder)
            elif field_name == 'crop_output_folder':
                self.config['crop_output_folder'] = folder
                self.crop_folder_input.setText(folder)
            elif field_name == 'watermark_input_folder':
                self.config['watermark_input_folder'] = folder
                self.wm_input_folder_input.setText(folder)
            elif field_name == 'watermark_output_folder':
                self.config['watermark_output_folder'] = folder
                self.wm_output_folder_input.setText(folder)
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
            "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if file_path:
            self.wm_logo_input.setText(file_path)
            try:
                self._sync_config_from_ui()
                self._save_config()
            except Exception as e:
                print(f"保存 Logo 路径失败: {e}")

    def _build_watermark_options(self) -> WatermarkOptions:
        return WatermarkOptions(
            enable_location=self.wm_location_checkbox.isChecked(),
            location_text=self.wm_location_text_input.text().strip(),
            use_gps_city=self.wm_use_gps_city_checkbox.isChecked(),
            enable_date=self.wm_date_checkbox.isChecked(),
            enable_species=self.wm_species_checkbox.isChecked(),
            enable_camera_params=self.wm_camera_checkbox.isChecked(),
            logo_path=self.wm_logo_input.text().strip(),
            logo_width_ratio=float(self.wm_logo_width_ratio_input.value()),
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

    def _run_watermark_batch(self):
        """仅执行批量水印生成（不触发完整主流程）。"""
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
            options = self._build_watermark_options()
            wm_result = generate_watermarks(
                source_folder=source_folder,
                output_folder=output_folder,
                options=options,
                prefer_folder_name_as_species=True,
            )
            QMessageBox.information(
                self,
                "水印生成完成",
                f"总计 {wm_result['total']}，成功 {wm_result['ok']}，失败 {wm_result['fail']}\n输出目录：{output_folder}",
            )
            try:
                self._sync_config_from_ui()
                self._save_config()
            except Exception as e:
                print(f"水印完成后保存配置失败: {e}")
        except Exception as e:
            QMessageBox.critical(self, "水印生成失败", str(e))

    def _preview_watermark_one(self):
        """按当前水印配置预览一张效果图。"""
        source_folder = self._resolve_watermark_source_folder()
        if not source_folder or not os.path.isdir(source_folder):
            QMessageBox.warning(
                self, "提示", "未找到可预览的图片目录，请先设置相片文件夹或先跑归档流程。"
            )
            return

        imgs = collect_images_recursive(source_folder)
        preview_path = imgs[0] if imgs else ""
        if not preview_path:
            fp, _ = QFileDialog.getOpenFileName(
                self,
                "选择一张用于预览的图片",
                source_folder,
                "Image Files (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)",
            )
            preview_path = fp
        if not preview_path:
            return

        options = self._build_watermark_options()
        out_img = render_watermark_for_image(
            image_path=preview_path,
            source_folder=source_folder,
            options=options,
            prefer_folder_name_as_species=True,
        )
        if out_img is None:
            QMessageBox.warning(self, "提示", "预览失败：无法读取图片。")
            return

        from PyQt5.QtGui import QImage, QPixmap

        arr = np.array(out_img.convert("RGB"))
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)

        dlg = QDialog(self)
        dlg.setWindowTitle("水印预览")
        dlg.resize(980, 720)
        v = QVBoxLayout(dlg)
        info = QLabel(f"预览文件：{os.path.basename(preview_path)}")
        info.setWordWrap(True)
        v.addWidget(info)
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
        holder = QWidget()
        hv = QVBoxLayout(holder)
        img_lb = QLabel()
        img_lb.setAlignment(Qt.AlignCenter)
        zoom_state = {"scale": 1.0}

        def _apply_scaled():
            scale = max(0.1, float(zoom_state["scale"]))
            nw = max(1, int(pix.width() * scale))
            nh = max(1, int(pix.height() * scale))
            img_lb.setPixmap(
                pix.scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        def _fit_to_view():
            vw = max(1, sc.viewport().width() - 16)
            vh = max(1, sc.viewport().height() - 16)
            sx = vw / max(1, pix.width())
            sy = vh / max(1, pix.height())
            zoom_state["scale"] = min(sx, sy, 1.0)
            _apply_scaled()

        zoom_in_btn.clicked.connect(
            lambda: (zoom_state.__setitem__("scale", zoom_state["scale"] * 1.15), _apply_scaled())
        )
        zoom_out_btn.clicked.connect(
            lambda: (zoom_state.__setitem__("scale", zoom_state["scale"] / 1.15), _apply_scaled())
        )
        fit_btn.clicked.connect(_fit_to_view)
        one_btn.clicked.connect(
            lambda: (zoom_state.__setitem__("scale", 1.0), _apply_scaled())
        )
        hv.addWidget(img_lb)
        sc.setWidget(holder)
        v.addWidget(sc, 1)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        btns.button(QDialogButtonBox.Close).clicked.connect(dlg.reject)
        v.addWidget(btns)
        QTimer.singleShot(0, _fit_to_view)
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
        self._eta_species_t0 = None
        self._eta_species_done = 0
        self._eta_species_total = 0
        self._ema_sec_per_species = None

    def _compute_eta_remaining_sec(self) -> Optional[float]:
        if not self._eta_phases:
            return None
        mono = time.monotonic()
        rem = 0.0
        seen_current = False
        for p in self._eta_phases:
            if p.get("done"):
                continue
            if not seen_current:
                seen_current = True
                name = p["name"]
                est = float(p.get("est", 0.0))
                if name == "species" and self._eta_species_total > 0:
                    left = max(0, self._eta_species_total - self._eta_species_done)
                    if left <= 0:
                        continue
                    rate = self._ema_sec_per_species
                    if rate is not None and self._eta_species_done > 0:
                        rem += left * rate
                    else:
                        rem += est * (left / max(1, self._eta_species_total))
                else:
                    t0 = p.get("t0")
                    if t0 is None:
                        rem += est
                    else:
                        rem += max(0.0, est - (mono - t0))
            else:
                rem += float(p.get("est", 0.0))
        return max(0.0, rem)

    def _on_eta_checkpoint(self, d: Dict[str, Any]) -> None:
        kind = d.get("kind")
        if kind == "start":
            self._reset_eta_model()
            for p in d.get("phases") or []:
                self._eta_phases.append(
                    {
                        "name": p["name"],
                        "est": float(p.get("est", 1.0)),
                        "done": False,
                        "t0": None,
                    }
                )
            return
        if kind == "phase_begin":
            name = d.get("phase")
            for p in self._eta_phases:
                if p["name"] == name:
                    p["t0"] = time.monotonic()
                    if name == "species":
                        self._eta_species_t0 = p["t0"]
                    break
            return
        if kind == "phase_done":
            name = d.get("phase")
            for p in self._eta_phases:
                if p["name"] == name:
                    p["done"] = True
                    break
            return
        if kind == "species_begin":
            n = int(d.get("n", 0))
            self._eta_species_total = n
            self._eta_species_done = 0
            use_local = self.config.get("use_local_model", True)
            per = 5.0 if use_local else 14.0
            for p in self._eta_phases:
                if p["name"] == "species":
                    p["est"] = max(8.0, n * per) if n > 0 else 3.0
                    break
            return
        if kind == "species_tick":
            done = int(d.get("done", 0))
            total = max(1, int(d.get("total", 1)))
            self._eta_species_done = done
            if self._eta_species_t0 is not None and done > 0:
                inst = (time.monotonic() - self._eta_species_t0) / float(done)
                a = self._eta_ema_alpha
                if self._ema_sec_per_species is None:
                    self._ema_sec_per_species = inst
                else:
                    self._ema_sec_per_species = (
                        (1.0 - a) * self._ema_sec_per_species + a * inst
                    )

    def _refresh_process_time_labels(self):
        """已用时间 + 预计剩余（阶段预估 + 物种逐张 EMA 修正）。"""
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
        self.generate_burst_report_checkbox.setEnabled(en)

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
        burst_on = self.config["enable_burst_detection"]
        need_species_or_crop = (
            self.config["enable_species_detection"] or self.config["enable_crop"]
        )

        if not burst_on and need_species_or_crop:
            if self._count_images_in_screened(output_folder) < 1:
                QMessageBox.warning(
                    self,
                    "提示",
                    "未勾选「连拍检测」时，物种识别与裁剪将只处理输出文件夹下的 "
                    "Screened_images 中的已筛选照片。\n\n"
                    "当前该目录下没有图片，请先运行一次连拍流程生成筛选结果，"
                    "或勾选「连拍检测」。",
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

        n_eta = _count_images_for_eta(self.config.get("image_folder", ""))
        if not burst_on and need_species_or_crop:
            n_eta = max(
                n_eta,
                self._count_images_in_screened(output_folder),
            )
        self.config["_eta_image_estimate"] = n_eta

        lat, lon = self._get_gps_coords()
        if lat is None:
            self._save_config()
            return  # _get_gps_coords 已弹出警告
        self.config["gps_latitude"] = lat
        self.config["gps_longitude"] = lon
        
        # 创建输出文件夹
        Path(self.config['output_folder']).mkdir(parents=True, exist_ok=True)
        Path(self.config['crop_output_folder']).mkdir(parents=True, exist_ok=True)
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
        """停止处理"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
            self.add_log("✗ 处理已中止")
            self._process_time_timer.stop()
            if self._process_start_monotonic is not None:
                elapsed = time.monotonic() - self._process_start_monotonic
                self._elapsed_label.setText(
                    f"已用时间：{self._format_duration_hms(elapsed)}（已中止）"
                )
            self._process_start_monotonic = None
            self._eta_label.setText("预计剩余：—")
            self._reset_eta_model()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            try:
                self._sync_config_from_ui()
                self._save_config()
            except Exception as e:
                print(f"中止后保存配置失败: {e}")
    
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
        """处理完成"""
        self.add_log("\n" + "="*60)
        self.add_log("✓ 处理完成！")
        self.add_log("="*60)
        
        # 显示统计信息
        self.stats_table.setRowCount(0)
        stats = [
            ("总处理图片", results.get('total_images', 'N/A')),
            ("保留图片", results.get('kept_images', 'N/A')),
            ("丢弃图片", results.get('discarded_images', 'N/A')),
            ("GPS已写入", results.get('gps_written', 0)),
        ]
        
        if 'crop_result' in results:
            crop_result = results['crop_result']
            stats.append(("检测到的鸟体", crop_result.get('total_crops', 'N/A')))
            stats.append(("处理耗时", f"{crop_result.get('processing_time', 0):.2f}秒"))
        if 'watermark_result' in results:
            wm = results['watermark_result']
            stats.append(("水印图片总数", wm.get('total', 'N/A')))
            stats.append(("水印成功", wm.get('ok', 'N/A')))
            stats.append(("水印失败", wm.get('fail', 'N/A')))
        
        for i, (key, value) in enumerate(stats):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.stats_table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        # 输出文件夹信息
        self.add_log(f"\n📁 输出文件夹: {self.config['output_folder']}")
        self.add_log(f"📁 分类归档: {self.config['crop_output_folder']}")
        
        # 恢复UI状态
        self._process_time_timer.stop()
        if self._process_start_monotonic is not None:
            elapsed = time.monotonic() - self._process_start_monotonic
            self._elapsed_label.setText(f"已用时间：{self._format_duration_hms(elapsed)}")
        self._process_start_monotonic = None
        self._eta_label.setText("预计剩余：0秒")
        self._reset_eta_model()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        try:
            self._sync_config_from_ui()
            self._save_config()
        except Exception as e:
            print(f"完成后保存配置失败: {e}")
        QMessageBox.information(self, "处理完成", "图片处理已完成！\n请检查输出文件夹中的结果。")
    
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
        self.config["output_folder"] = self.output_folder_input.text().strip()
        self.config["crop_output_folder"] = self.crop_folder_input.text().strip()
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
        self.config["enable_burst_detection"] = (
            self.enable_burst_detection_checkbox.isChecked()
        )
        self.config["use_bird_detection"] = self.use_bird_detection_checkbox.isChecked()
        self.config["use_eye_detection"] = (
            self.use_eye_detection_checkbox.isChecked()
            and self.config["use_bird_detection"]
        )
        self.config["use_fast_mode"] = self.use_fast_mode_checkbox.isChecked()
        self.config["generate_burst_report"] = (
            self.generate_burst_report_checkbox.isChecked()
        )
        self.config["enable_species_detection"] = (
            self.enable_species_checkbox.isChecked()
        )
        self.config["enable_crop"] = self.enable_crop_checkbox.isChecked()
        self.config["generate_species_report"] = (
            self.generate_species_report_checkbox.isChecked()
        )
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
        self.config["wm_logo_width_ratio"] = float(
            self.wm_logo_width_ratio_input.value()
        )
        self.config["use_local_model"] = self.local_model_radio.isChecked()
        self.config["species_conf_threshold_enabled"] = (
            self.min_species_threshold_enable_checkbox.isChecked()
        )
        self.config["min_species_accept_confidence"] = float(
            self.min_species_conf_input.value()
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
                    if 'enable_species_detection' not in saved_config:
                        self.config['enable_species_detection'] = self.config.get(
                            'enable_crop', True
                        )
                    if 'species_conf_threshold_enabled' not in saved_config:
                        self.config['species_conf_threshold_enabled'] = False
                    self._update_ui_from_config()
            except Exception as e:
                print(f"加载配置失败: {e}")
    
    def _update_ui_from_config(self):
        """从配置更新UI"""
        self.image_folder_input.setText(self.config.get('image_folder', ''))
        self.output_folder_input.setText(self.config.get('output_folder', ''))
        self.crop_folder_input.setText(self.config.get('crop_output_folder', ''))
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
        self.enable_burst_detection_checkbox.setChecked(
            self.config.get("enable_burst_detection", True)
        )
        self._on_burst_detection_toggled(
            self.enable_burst_detection_checkbox.isChecked()
        )
        self.use_bird_detection_checkbox.setChecked(self.config.get('use_bird_detection', True))
        self.use_eye_detection_checkbox.setChecked(self.config.get('use_eye_detection', False))
        self.use_fast_mode_checkbox.setChecked(self.config.get('use_fast_mode', True))
        self.generate_burst_report_checkbox.setChecked(
            self.config.get("generate_burst_report", True)
        )
        self._on_bird_detection_toggled(self.use_bird_detection_checkbox.isChecked())
        self.enable_species_checkbox.setChecked(
            self.config.get('enable_species_detection', True)
        )
        self.enable_crop_checkbox.setChecked(self.config.get('enable_crop', True))
        self.generate_species_report_checkbox.setChecked(
            self.config.get("generate_species_report", True)
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
        self.wm_logo_width_ratio_input.setValue(
            float(self.config.get('wm_logo_width_ratio', 0.30))
        )
        
        # 物种识别配置
        use_local = self.config.get('use_local_model', True)
        self.local_model_radio.setChecked(use_local)
        self.doubao_model_radio.setChecked(not use_local)
        self.min_species_threshold_enable_checkbox.setChecked(
            self.config.get('species_conf_threshold_enabled', False)
        )
        self.min_species_conf_input.setEnabled(
            self.min_species_threshold_enable_checkbox.isChecked()
        )
        self.min_species_conf_input.setValue(
            float(self.config.get('min_species_accept_confidence', 0.5))
        )
        
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
