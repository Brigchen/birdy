#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鸟类检测Skill - 命令行界面
支持：参数配置、进度显示、文件夹处理、全流程自动化

作者: brigchen@gmail.com
版权说明: 基于开源协议，请勿商用
"""

import sys
import os
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_skill_info_meta() -> Tuple[str, str]:
    """从项目根目录 skill-info.json 读取版本号与发布日期。"""
    root = Path(__file__).resolve().parent.parent
    info_path = root / "skill-info.json"
    version, release_date = "2.0.0", ""
    try:
        if info_path.is_file():
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = str(data.get("version", version))
            release_date = str(data.get("release_date", "") or "")
    except Exception:
        pass
    return version, release_date

from burst_grouping import process_folder, get_kept_images
from html_report_generator import generate_html_report
from geo_encoder import batch_write_gps_exif, geocode_location
from detect_bird_and_eye import BirdAndEyeDetector
from api_config_defaults import ensure_doubao_api_config_file


class BirdDetectionCLI:
    """鸟类检测命令行工具"""
    
    def __init__(self):
        self.config = {}
    
    def print_header(self):
        """打印程序头部信息"""
        print("\n" + "="*70)
        print("     Bird Detection Skill - Command Line Interface")
        print("="*70)
        ver, rdate = _load_skill_info_meta()
        print(f"     Version: {ver}")
        if rdate:
            print(f"     Release date: {rdate}")
        print(f"     Python: {sys.version.split()[0]}")
        print("="*70 + "\n")
    
    def load_config(self, config_file: Optional[str] = None):
        """加载配置文件"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config.update(json.load(f))
                print(f"[+] 配置已从 {config_file} 加载\n")
            except Exception as e:
                print(f"[!] 配置文件加载失败: {e}")
        else:
            print("[!] 使用默认配置\n")
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'image_folder': '',
            'output_folder': './outputs',
            'crop_output_folder': './crops',
            'enable_gps_write': False,
            'gps_latitude': 31.2304,
            'gps_longitude': 121.4737,
            'gps_altitude': 0,
            'location_name': '',
            'province': '',
            'city': '',
            'time_threshold': 1.0,
            'burst_keep_ratio': 0.2,
            'burst_keep_min': 2,
            'keep_top_n': 2,
            'use_bird_detection': True,
            'use_eye_detection': False,
            'use_fast_mode': True,
            'generate_burst_report': True,
            'enable_species_detection': True,
            'enable_crop': True,
            'generate_species_report': True,
            # 物种识别模式配置
            'use_local_model': True,
            # 本地模型最低接受置信度（低于此值视为未知种类）
            'min_species_accept_confidence': 0.5,
        }
    
    def validate_config(self) -> bool:
        """验证配置"""
        if not self.config.get('image_folder'):
            print("[!] 错误: 未指定图片文件夹")
            return False
        
        if not os.path.exists(self.config['image_folder']):
            print(f"[!] 错误: 图片文件夹不存在: {self.config['image_folder']}")
            return False
        
        return True
    
    def run(self):
        """执行主流程"""
        self.print_header()
        
        # 初始化默认配置
        self.config = self.get_default_config()
        
        # 解析命令行参数
        self._parse_args()
        
        # 加载配置文件（如果指定）
        if self.config.get('config_file'):
            self.load_config(self.config['config_file'])
        
        # 验证配置
        if not self.validate_config():
            print("\n[!] 配置验证失败，请检查参数")
            print("\n使用 --help 查看帮助信息\n")
            return False
        
        # 显示配置摘要
        self._print_config_summary()
        
        # 创建输出文件夹
        Path(self.config['output_folder']).mkdir(parents=True, exist_ok=True)
        Path(self.config['crop_output_folder']).mkdir(parents=True, exist_ok=True)
        
        # 执行处理流程
        try:
            results = self._process_images()
            
            # 显示结果摘要
            self._print_results_summary(results)
            
            return True
        except KeyboardInterrupt:
            print("\n\n[!] 用户中断处理")
            return False
        except Exception as e:
            print(f"\n[!] 处理失败: {e}")
            traceback.print_exc()
            return False
    
    def _parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description='Bird Detection Skill - 命令行界面',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  # 基本用法
  python birdy_cli.py -i ./images
  
  # 指定输出文件夹
  python birdy_cli.py -i ./images -o ./output
  
  # 启用GPS写入
  python birdy_cli.py -i ./images --gps --lat 31.23 --lon 121.47
  
  # 使用Baidu API模式
  python birdy_cli.py -i ./images --api-mode baidu
  
  # 使用配置文件
  python birdy_cli.py -i ./images --config config.json
            """
        )
        
        # 必需参数
        parser.add_argument('-i', '--input', type=str, required=False,
                          help='输入图片文件夹路径')
        
        # 可选参数
        parser.add_argument('-o', '--output', type=str,
                          help='输出文件夹路径 (默认: ./outputs)')
        parser.add_argument('--crop-output', type=str,
                          help='裁剪图片输出文件夹路径 (默认: ./crops)')
        parser.add_argument('--config', type=str, dest='config_file',
                          help='配置文件路径 (JSON格式)')
        
        # GPS相关
        parser.add_argument('--gps', action='store_true',
                          help='启用GPS EXIF写入')
        parser.add_argument('--lat', type=float, dest='gps_latitude',
                          help='GPS纬度')
        parser.add_argument('--lon', type=float, dest='gps_longitude',
                          help='GPS经度')
        parser.add_argument('--alt', type=float, dest='gps_altitude',
                          help='GPS高度 (默认: 0)')
        parser.add_argument('--location', type=str, dest='location_name',
                          help='位置名称 (如: 厦门大学翔安校区)')
        
        # 连拍处理
        parser.add_argument('--time-threshold', type=float,
                          help='连拍时间阈值，单位秒 (默认: 1.0)')
        parser.add_argument(
            '--burst-keep-ratio',
            type=float,
            dest='burst_keep_ratio',
            help='连拍组保留比例，如 0.2 约等于 1/5 (默认: 0.2)',
        )
        parser.add_argument(
            '--burst-keep-min',
            type=int,
            dest='burst_keep_min',
            help='每组最少保留张数，与比例取较大值后不超过组大小 (默认: 2)',
        )
        parser.add_argument('--keep-top-n', type=int,
                          help='已弃用：等同 --burst-keep-min')
        parser.add_argument('--no-bird-detection', action='store_true',
                          help='禁用鸟体检测')
        parser.add_argument('--eye-detection', action='store_true',
                          help='启用鸟眼检测（仅在启用鸟体检测时有效）')
        parser.add_argument('--no-fast-mode', action='store_true',
                          help='禁用快速模式')
        parser.add_argument('--no-burst-report', action='store_true',
                          help='不生成连拍报告')
        
        # 物种识别
        parser.add_argument('--no-species', action='store_true',
                          help='禁用物种识别（仅鸟体检测与裁剪时归入未知分级）')
        parser.add_argument('--no-crop', action='store_true',
                          help='不裁剪；若仍启用物种识别，则按顶一物种将原图复制到裁剪输出目录')
        parser.add_argument('--no-species-report', action='store_true',
                          help='不生成物种识别报告')
        parser.add_argument('--api-mode', type=str, choices=['local', 'baidu', 'doubao'],
                          help='物种识别API模式: local(本地), baidu(Baidu), doubao(豆包)')
        parser.add_argument(
            '--species-conf',
            type=float,
            default=None,
            dest='species_conf',
            help='未知种类阈值（0~1），与 GUI「未知种类阈值」及 detect_bird_and_eye.py --species-conf 一致',
        )
        parser.add_argument(
            '--min-species-conf',
            type=float,
            default=None,
            dest='min_species_accept_confidence',
            help='同 --species-conf（保留旧参数名）',
        )
        
        args = parser.parse_args()
        
        # 更新配置
        if args.input:
            self.config['image_folder'] = args.input
        if args.output:
            self.config['output_folder'] = args.output
        if args.crop_output:
            self.config['crop_output_folder'] = args.crop_output
        
        # GPS配置
        if args.gps:
            self.config['enable_gps_write'] = True
        if args.gps_latitude is not None:
            self.config['gps_latitude'] = args.gps_latitude
        if args.gps_longitude is not None:
            self.config['gps_longitude'] = args.gps_longitude
        if args.gps_altitude is not None:
            self.config['gps_altitude'] = args.gps_altitude
        if args.location_name is not None:
            self.config['location_name'] = args.location_name
            
            # 如果提供了位置名称，尝试查询GPS坐标
            if args.gps_latitude is None or args.gps_longitude is None:
                try:
                    result = geocode_location(args.location_name)
                    if result:
                        lat, lon = result
                        self.config['gps_latitude'] = lat
                        self.config['gps_longitude'] = lon
                        print(f"[+] 查询到GPS: {args.location_name} -> 纬度: {lat:.6f}, 经度: {lon:.6f}\n")
                except Exception as e:
                    print(f"[!] GPS查询失败: {e}\n")
        
        # 连拍处理配置
        if args.time_threshold is not None:
            self.config['time_threshold'] = args.time_threshold
        if args.burst_keep_ratio is not None:
            self.config['burst_keep_ratio'] = args.burst_keep_ratio
        if args.burst_keep_min is not None:
            self.config['burst_keep_min'] = args.burst_keep_min
            self.config['keep_top_n'] = args.burst_keep_min
        if args.keep_top_n is not None:
            self.config['burst_keep_min'] = args.keep_top_n
            self.config['keep_top_n'] = args.keep_top_n
        if args.no_bird_detection:
            self.config['use_bird_detection'] = False
        if args.eye_detection:
            self.config['use_eye_detection'] = True
        if args.no_fast_mode:
            self.config['use_fast_mode'] = False
        if args.no_burst_report:
            self.config['generate_burst_report'] = False
        
        # 物种识别配置
        if args.no_species:
            self.config['enable_species_detection'] = False
        if args.no_crop:
            self.config['enable_crop'] = False
        if args.no_species_report:
            self.config['generate_species_report'] = False
        if args.api_mode:
            if args.api_mode == 'local':
                self.config['use_local_model'] = True
            elif args.api_mode == 'baidu':
                self.config['use_local_model'] = False
            elif args.api_mode == 'doubao':
                self.config['use_local_model'] = False
                self.config['enable_doubao_api'] = True
        if args.min_species_accept_confidence is not None:
            self.config['min_species_accept_confidence'] = (
                args.min_species_accept_confidence
            )
        elif args.species_conf is not None:
            self.config['min_species_accept_confidence'] = args.species_conf
    
    def _print_config_summary(self):
        """打印配置摘要"""
        print("配置摘要:")
        print("-" * 70)
        print(f"  输入文件夹:     {self.config['image_folder']}")
        print(f"  输出文件夹:     {self.config['output_folder']}")
        print(f"  裁剪输出:       {self.config['crop_output_folder']}")
        print(f"  GPS写入:        {'是' if self.config['enable_gps_write'] else '否'}")
        print(f"  连拍时间阈值:   {self.config['time_threshold']}秒")
        print(
            f"  连拍保留:       比例 {self.config.get('burst_keep_ratio', 0.2)}, "
            f"最少 {self.config.get('burst_keep_min', self.config.get('keep_top_n', 2))} 张/组"
        )
        print(f"  鸟体检测:       {'是' if self.config['use_bird_detection'] else '否'}")
        print(
            f"  鸟眼检测:       {'是' if (self.config.get('use_eye_detection') and self.config['use_bird_detection']) else '否'}"
        )
        print(f"  快速模式:       {'是' if self.config['use_fast_mode'] else '否'}")
        print(
            f"  物种识别:       {'是' if self.config.get('enable_species_detection', True) else '否'}"
        )
        print(
            f"  鸟体裁剪:       {'是' if self.config.get('enable_crop', True) else '否'}"
        )
        print(f"  识别模式:       {'本地' if self.config['use_local_model'] else 'API'}")
        if self.config['use_local_model']:
            print(
                f"  未知种类阈值:   {self.config.get('min_species_accept_confidence', 0.5):.2f} "
                f"（低于此值视为未知，本参数对豆包API模式无效）"
            )
        print("-" * 70 + "\n")
    
    def _process_images(self) -> Dict:
        """执行图像处理流程"""
        results = {}
        total_steps = 0
        
        # 计算总步骤数
        if self.config['enable_gps_write']:
            total_steps += 1
        total_steps += 1  # 连拍识别
        if self.config['generate_burst_report']:
            total_steps += 1  # 生成报告
        if self.config.get('enable_species_detection', True) or self.config.get(
            'enable_crop', False
        ):
            total_steps += 1  # 物种/裁剪/归档
        
        current_step = 0
        start_time = datetime.now()
        
        print(f"[步骤 {current_step + 1}/{total_steps}] 开始处理...")
        print("-" * 70)
        
        # 第一步：GPS写入
        if self.config['enable_gps_write']:
            current_step += 1
            print(f"\n[步骤 {current_step}/{total_steps}] 写入GPS EXIF...")
            
            try:
                gps_count = batch_write_gps_exif(
                    image_folder=self.config['image_folder'],
                    latitude=self.config['gps_latitude'],
                    longitude=self.config['gps_longitude'],
                    altitude=self.config.get('gps_altitude', 0)
                )
                print(f"    [+] 成功写入 {gps_count} 张图片的 GPS")
                results['gps_written'] = gps_count
            except Exception as e:
                print(f"    [!] GPS写入失败: {e}")
                results['gps_written'] = 0
        
        # 第二步：连拍识别和筛选
        current_step += 1
        print(f"\n[步骤 {current_step}/{total_steps}] 连拍识别与筛选...")
        burst_filter_applied = False
        try:
            screened_dir = os.path.join(self.config['output_folder'], 'Screened_images')
            burst_result = process_folder(
                image_folder=self.config['image_folder'],
                time_threshold=self.config['time_threshold'],
                burst_keep_ratio=float(self.config.get('burst_keep_ratio', 0.2)),
                burst_keep_min=int(
                    self.config.get(
                        'burst_keep_min', self.config.get('keep_top_n', 2)
                    )
                ),
                use_bird_detection=self.config['use_bird_detection'],
                use_eye_detection=(
                    self.config.get('use_eye_detection', False)
                    and self.config['use_bird_detection']
                ),
                output_report=os.path.join(
                    self.config['output_folder'], 'burst_analysis.json'
                ),
                fast_mode=self.config['use_fast_mode'],
                screened_output_dir=screened_dir,
            )
            print(f"    [+] 处理 {burst_result['total_images']} 张图片")
            print(f"    [+] 保留 {burst_result['kept_images']} 张")
            print(f"    [+] 丢弃 {burst_result['discarded_images']} 张")
            results.update(burst_result)
            burst_filter_applied = True
        except Exception as e:
            print(f"    [!] 连拍识别失败: {e}")
            traceback.print_exc()
            results['total_images'] = 0
            results['kept_images'] = 0
            results['discarded_images'] = 0
        
        # 第三步：生成报告
        if self.config['generate_burst_report']:
            current_step += 1
            print(f"\n[步骤 {current_step}/{total_steps}] 生成可视化报告...")
            
            try:
                json_report_file = os.path.join(self.config['output_folder'], 'burst_analysis.json')
                if os.path.exists(json_report_file):
                    reports_dir = os.path.join(self.config['output_folder'], 'reports')
                    os.makedirs(reports_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    html_report_path = os.path.join(reports_dir, f'连拍分析报告_{timestamp}.html')
                    
                    generate_html_report(
                        json_report_path=json_report_file,
                        output_html_path=html_report_path,
                        image_folder=self.config['image_folder']
                    )
                    print(f"    [+] HTML报告已生成: {os.path.basename(html_report_path)}")
                else:
                    print(f"    [!] 跳过报告生成：JSON报告文件不存在")
            except Exception as e:
                print(f"    [!] 报告生成失败: {e}")
        else:
            print(f"\n[步骤 {current_step + 1}/{total_steps}] 跳过报告生成")
        
        # 第四步：物种识别 / 裁剪或原图归档
        do_species = self.config.get('enable_species_detection', True)
        do_crop = self.config.get('enable_crop', False)
        if do_species or do_crop:
            current_step += 1
            title = (
                "物种检测、裁剪与归档"
                if do_species and do_crop
                else (
                    "物种识别（原图按物种归档）"
                    if do_species and not do_crop
                    else "鸟体检测与裁剪（无物种识别）"
                )
            )
            print(f"\n[步骤 {current_step}/{total_steps}] {title}...")
            
            try:
                import time
                step_start_time = time.time()
                
                # 仅豆包API模式才读取配置文件（本地模型无需读取）
                doubao_config = None
                if not self.config.get('use_local_model', True):
                    doubao_path = ensure_doubao_api_config_file(
                        Path(__file__).resolve().parent
                    )
                    with open(doubao_path, "r", encoding="utf-8") as f:
                        doubao_config = json.load(f)

                # 初始化检测器
                detector = BirdAndEyeDetector(
                    enable_species=do_species,
                    use_local_model=self.config.get('use_local_model', True),
                    doubao_config=doubao_config,
                    min_species_accept_confidence=self.config.get(
                        'min_species_accept_confidence', 0.5
                    ),
                )
                
                output_root = self.config['crop_output_folder']
                import fnmatch
                
                if burst_filter_applied:
                    image_files = [
                        p for p in get_kept_images(results)
                        if os.path.isfile(p)
                    ]
                    print(
                        f"    [+] 物种识别/归档仅处理连拍筛选保留的 {len(image_files)} 张"
                    )
                else:
                    image_folder = self.config['image_folder']
                    image_patterns = ['*.jpg', '*.jpeg', '*.png']
                    image_files = []
                    for pattern in image_patterns:
                        for root, dirs, files in os.walk(image_folder):
                            for file in files:
                                if fnmatch.fnmatch(file.lower(), pattern.lower()):
                                    image_path = os.path.join(root, file)
                                    if image_path not in image_files:
                                        image_files.append(image_path)
                    image_files = list(set(image_files))
                    print(
                        f"    [+] 连拍步骤未完成，扫描输入目录共 {len(image_files)} 张"
                    )
                
                total_crops = 0
                archive_counter = {"n": 0}
                species_results = []
                
                for idx, image_file in enumerate(image_files):
                    progress = (idx + 1) / len(image_files) * 100
                    print(f"    进度: [{idx+1}/{len(image_files)}] {progress:.0f}% - {os.path.basename(image_file)}", end='\r')
                    
                    try:
                        result_image, detection_results = detector.detect(image_file)
                        
                        if detection_results.get('birds'):
                            from detect_bird_and_eye import gps_to_location
                            province, city = gps_to_location(image_file)
                            saved_paths = []
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
                            elif do_species:
                                saved_paths = detector.copy_original_by_top_species(
                                    source_path=image_file,
                                    birds=detection_results['birds'],
                                    output_dir=output_root,
                                    province=province,
                                    city=city,
                                    counter=archive_counter,
                                )
                            total_crops += len(saved_paths)
                        
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
                                })
                    except Exception as e:
                        pass  # 静默处理单张图片错误
                
                print()  # 换行
                
                processing_time = time.time() - step_start_time
                print(f"    [+] 已输出 {total_crops} 个文件（裁剪或原图归档）")
                print(f"    [+] 处理耗时 {processing_time:.2f} 秒")
                
                results_dict = {
                    'total_crops': total_crops,
                    'species_method': detector.get_species_method(),
                    'processing_time': processing_time
                }
                results['crop_result'] = results_dict
                
                # 生成物种识别报告
                if self.config['generate_species_report']:
                    print(f"\n    [+] 生成物种识别报告...")
                    self._generate_species_report(image_files, total_crops, detector)
                
            except Exception as e:
                print(f"\n    [!] 物种检测/归档失败: {e}")
                traceback.print_exc()
        
        # 总体时间统计
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n[完成] 总耗时: {total_time:.2f} 秒")
        
        return results
    
    def _generate_species_report(self, image_files: list, total_crops: int, detector):
        """生成物种识别报告"""
        try:
            from detect_bird_and_eye import gps_to_location, lookup_classification
            
            reports_dir = os.path.join(self.config['output_folder'], 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            species_report_path = os.path.join(reports_dir, f'物种识别报告_{timestamp}.html')
            
            # 收集物种识别结果
            species_results = []
            for image_file in image_files:
                try:
                    _, detection_results = detector.detect(image_file)
                    
                    if detection_results.get('birds'):
                        for bird in detection_results['birds']:
                            if bird.get('species'):
                                for species in bird['species']:
                                    chinese_name = species.get('chinese_name', '未知')
                                    scientific_name = species.get('scientific_name', '')
                                    classification = lookup_classification(chinese_name, scientific_name)
                                    unified_chinese_name = classification.get('species_cn', chinese_name)
                                    
                                    species_results.append({
                                        'image': os.path.basename(image_file),
                                        'species': unified_chinese_name,
                                        'english_name': species.get('english_name', '未知'),
                                        'scientific_name': species.get('scientific_name', '未知'),
                                        'confidence': species.get('confidence', 0),
                                        'method': bird.get('species_method', '未知'),
                                    })
                except Exception:
                    pass
            
            # 生成HTML报告
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
    </style>
</head>
<body>
    <h1>物种识别报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>处理图片数量: {len(image_files)}</p>
    <p>检测到鸟体数量: {total_crops}</p>
    <p>识别到物种数量: {len(species_results)}</p>
    
    <h2>识别结果</h2>
'''
            
            for item in species_results:
                html_content += f'''
    <div class="result">
        <div class="image">图片: {item['image']}</div>
        <div class="method">识别方法: {item['method']}</div>
        <div class="species">
            中文名: {item['species']}<br>
            英文名: {item['english_name']}<br>
            学名: {item['scientific_name']}<br>
            置信度: {item['confidence']:.2f}
        </div>
    </div>
'''
            
            html_content += '''
</body>
</html>
'''
            
            with open(species_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"    [+] 物种识别报告已生成: {os.path.basename(species_report_path)}")
        except Exception as e:
            print(f"    [!] 物种识别报告生成失败: {e}")
    
    def _print_results_summary(self, results: Dict):
        """打印结果摘要"""
        print("\n" + "="*70)
        print("处理完成！")
        print("="*70)
        
        print(f"总处理图片:     {results.get('total_images', 'N/A')}")
        print(f"保留图片:       {results.get('kept_images', 'N/A')}")
        print(f"丢弃图片:       {results.get('discarded_images', 'N/A')}")
        
        if results.get('gps_written'):
            print(f"GPS已写入:       {results['gps_written']}")
        
        if 'crop_result' in results:
            crop_result = results['crop_result']
            print(f"检测到的鸟体:   {crop_result.get('total_crops', 'N/A')}")
            print(f"识别方法:       {crop_result.get('species_method', 'N/A')}")
            print(f"处理耗时:       {crop_result.get('processing_time', 0):.2f}秒")
        
        print("\n输出文件夹:")
        print(f"  主输出:   {self.config['output_folder']}")
        print(f"  裁剪图片: {self.config['crop_output_folder']}")
        
        print("\n" + "="*70 + "\n")


def main():
    """主入口"""
    cli = BirdDetectionCLI()
    success = cli.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
