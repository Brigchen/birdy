# -*- coding: utf-8 -*-
"""
HTML可视化报告生成器：
生成交互式HTML报告，包含检测框对比图、统计信息等

作者: brigchen@gmail.com
版权说明: 基于开源协议，请勿商用
"""
import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def draw_bird_boxes(image_path: str, birds: List[Dict], label: str = "") -> np.ndarray:
    """
    在图片上绘制鸟体检测框
    
    Args:
        image_path: 图片路径
        birds: 鸟检测信息列表 [{"bbox": [x1, y1, x2, y2], "conf": 0.9, "area": 1000}, ...]
        label: 图片标签（用于标注）
    
    Returns:
        绘制了检测框的图片数组
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 绘制检测框
        for bird in birds:
            if "bbox" not in bird:
                continue
            
            x1, y1, x2, y2 = bird["bbox"]
            conf = bird.get("conf", 0)
            
            # 绘制矩形框
            color = (0, 255, 0) if conf > 0.7 else (255, 165, 0)  # 绿色高置信度，橙色低置信度
            thickness = 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制置信度标签
            text = f"Bird {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # 背景矩形
            cv2.rectangle(img, (x1, y1 - text_size[1] - 4), 
                         (x1 + text_size[0], y1), color, -1)
            # 文字
            cv2.putText(img, text, (x1, y1 - 2), font, font_scale, (255, 255, 255), font_thickness)
        
        return img
    except Exception as e:
        print(f"绘制检测框失败: {e}")
        return None


def create_comparison_image(
    kept_image_path: str,
    discarded_image_path: str,
    kept_birds: List[Dict] = None,
    discarded_birds: List[Dict] = None,
    kept_score: float = 0,
    discarded_score: float = 0
) -> Optional[str]:
    """
    创建保留vs丢弃的对比图，并转换为base64
    
    Args:
        kept_image_path: 保留图片路径
        discarded_image_path: 丢弃图片路径
        kept_birds: 保留图片的鸟检测信息
        discarded_birds: 丢弃图片的鸟检测信息
        kept_score: 保留图片的对焦评分
        discarded_score: 丢弃图片的对焦评分
    
    Returns:
        base64编码的对比图，或None如果失败
    """
    try:
        if kept_birds is None:
            kept_birds = []
        if discarded_birds is None:
            discarded_birds = []
        
        # 绘制两张图片的检测框
        kept_img = draw_bird_boxes(kept_image_path, kept_birds, "保留")
        discarded_img = draw_bird_boxes(discarded_image_path, discarded_birds, "丢弃")
        
        if kept_img is None or discarded_img is None:
            return None
        
        # 调整大小使其相同
        h = max(kept_img.shape[0], discarded_img.shape[0])
        w = max(kept_img.shape[1], discarded_img.shape[1])
        
        # 缩放到相同大小
        kept_img_resized = cv2.resize(kept_img, (w, h))
        discarded_img_resized = cv2.resize(discarded_img, (w, h))
        
        # 创建对比图（上下排列）
        comparison = np.vstack([
            kept_img_resized,
            discarded_img_resized
        ])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (0, 255, 0)  # 绿色
        
        cv2.putText(comparison, f"KEPT (Score: {kept_score:.2f})", 
                   (20, 40), font, font_scale, color, font_thickness)
        cv2.putText(comparison, f"DISCARDED (Score: {discarded_score:.2f})", 
                   (20, h + 40), font, font_scale, (0, 0, 255), font_thickness)
        
        # 转换为base64
        _, buffer = cv2.imencode('.jpg', comparison)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str
    except Exception as e:
        print(f"创建对比图失败: {e}")
        return None


def image_to_base64(image_path: str, max_width: int = 400) -> Optional[str]:
    """
    将图片转换为base64字符串（可选缩放）
    
    Args:
        image_path: 图片路径
        max_width: 最大宽度（用于缩放）
    
    Returns:
        base64编码的图片字符串
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # 如果图片过大，缩放一下
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (max_width, new_h))
        
        _, buffer = cv2.imencode('.jpg', img)
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str
    except Exception as e:
        print(f"转换图片失败: {e}")
        return None


def generate_html_report(
    json_report_path: str,
    output_html_path: str,
    image_folder: str = None
) -> bool:
    """
    从JSON报告生成HTML可视化报告
    
    Args:
        json_report_path: JSON报告路径
        output_html_path: 输出HTML报告路径
        image_folder: 图片文件夹路径（用于生成对比图）
    
    Returns:
        是否生成成功
    """
    try:
        # 读取JSON报告
        with open(json_report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # 开始生成HTML
        html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>鸟图分组筛选报告</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-card .label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .section h2 {
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        
        .group-container {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }
        
        .group-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            cursor: pointer;
            user-select: none;
        }
        
        .group-header:hover {
            background: rgba(102, 126, 234, 0.1);
            border-radius: 4px;
            padding: 5px;
        }
        
        .group-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        
        .group-stats {
            display: flex;
            gap: 30px;
            font-size: 0.9em;
        }
        
        .group-stats span {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .stat-badge {
            background: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        .stat-badge.kept {
            color: #28a745;
            border: 1px solid #28a745;
        }
        
        .stat-badge.discarded {
            color: #dc3545;
            border: 1px solid #dc3545;
        }
        
        .group-content {
            display: none;
        }
        
        .group-content.active {
            display: block;
        }
        
        .toggle-icon {
            font-size: 1.2em;
            transition: transform 0.3s;
        }
        
        .group-content.active .toggle-icon {
            transform: rotate(90deg);
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .image-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            background: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .image-card.kept {
            border-left: 4px solid #28a745;
        }
        
        .image-card.discarded {
            border-left: 4px solid #dc3545;
            opacity: 0.7;
        }
        
        .image-preview {
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #f0f0f0;
        }
        
        .image-info {
            padding: 15px;
        }
        
        .image-name {
            font-weight: bold;
            margin-bottom: 8px;
            word-break: break-all;
            font-size: 0.9em;
        }
        
        .image-stats {
            font-size: 0.85em;
            color: #666;
            line-height: 1.6;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
        }
        
        .stat-label {
            font-weight: 500;
        }
        
        .stat-value {
            color: #667eea;
            font-weight: bold;
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-top: 8px;
        }
        
        .status-badge.kept {
            background: #d4edda;
            color: #155724;
        }
        
        .status-badge.discarded {
            background: #f8d7da;
            color: #721c24;
        }
        
        .comparison-container {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            border: 1px solid #ddd;
        }
        
        .comparison-image {
            width: 100%;
            border-radius: 4px;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }
        
        .expand-all {
            margin-bottom: 20px;
        }
        
        .expand-all button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
        }
        
        .expand-all button:hover {
            background: #764ba2;
        }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .stats {
                grid-template-columns: 1fr;
            }
            
            .image-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐦 鸟图分组筛选报告</h1>
            <p>连拍图片预分组及质量评估结果</p>
        </div>
        
        <div class="content">
"""
        
        # 添加统计卡片
        html += """            <div class="stats">
"""
        
        total = report.get("total_images", 0)
        kept = report.get("kept_images", 0)
        discarded = report.get("discarded_images", 0)
        processing_time = report.get("processing_time", 0)
        
        stats = [
            ("总图片数", total, "#667eea"),
            ("保留图片", kept, "#28a745"),
            ("丢弃图片", discarded, "#dc3545"),
            ("处理时间", f"{processing_time}s", "#ff9800")
        ]
        
        for label, value, color in stats:
            html += f"""                <div class="stat-card" style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%);">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
"""
        
        html += """            </div>
"""
        
        # 添加连拍组详情
        groups = report.get("groups", [])
        
        if groups:
            html += """            <div class="section">
                <h2>📸 连拍组详情</h2>
                <div class="expand-all">
                    <button onclick="toggleAllGroups()">展开/折叠全部</button>
                </div>
"""
            
            for group in groups:
                group_id = group.get("group_id", 0)
                total_in_group = group.get("total", 0)
                kept_in_group = group.get("kept", 0)
                images = group.get("images", [])
                
                html += f"""                <div class="group-container">
                    <div class="group-header" onclick="toggleGroup(this)">
                        <div class="group-title">
                            <span class="toggle-icon">▶</span>
                            连拍组 #{group_id} ({kept_in_group}/{total_in_group})
                        </div>
                        <div class="group-stats">
                            <span>拍摄时间: {images[0].get('time', 'N/A') if images else 'N/A'}</span>
                            <span><span class="stat-badge kept">保留: {kept_in_group}</span></span>
                            <span><span class="stat-badge discarded">丢弃: {total_in_group - kept_in_group}</span></span>
                        </div>
                    </div>
                    
                    <div class="group-content">
                        <div class="image-grid">
"""
                
                for img in images:
                    img_path = img.get("path", "")
                    img_name = img.get("name", "")
                    focus_score = img.get("focus_score", 0)
                    birds_detected = img.get("birds_detected", 0)
                    bird_area = img.get("bird_area", 0)
                    is_kept = img.get("kept", False)
                    time_diff = img.get("time_diff", 0)
                    
                    # 转换图片为base64
                    img_base64 = image_to_base64(img_path)
                    
                    card_class = "kept" if is_kept else "discarded"
                    status_class = "kept" if is_kept else "discarded"
                    status_text = "✓ 保留" if is_kept else "✗ 丢弃"
                    
                    img_preview = f'<img src="data:image/jpeg;base64,{img_base64}" class="image-preview">' if img_base64 else '<div class="image-preview" style="display:flex;align-items:center;justify-content:center;color:#ccc;">图片加载失败</div>'
                    
                    html += f"""                            <div class="image-card {card_class}">
                                {img_preview}
                                <div class="image-info">
                                    <div class="image-name">{img_name}</div>
                                    <div class="image-stats">
                                        <div class="stat-row">
                                            <span class="stat-label">对焦评分:</span>
                                            <span class="stat-value">{focus_score:.2f}</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">鸟体检测:</span>
                                            <span class="stat-value">{birds_detected} 只</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">鸟体面积:</span>
                                            <span class="stat-value">{bird_area:.0f} px²</span>
                                        </div>
                                        <div class="stat-row">
                                            <span class="stat-label">时间差:</span>
                                            <span class="stat-value">{time_diff:.1f}s</span>
                                        </div>
                                    </div>
                                    <div class="status-badge {status_class}">{status_text}</div>
                                </div>
                            </div>
"""
                
                html += """                        </div>
                    </div>
                </div>
"""
            
            html += """            </div>
"""
        
        # 添加非连拍图片
        non_burst = report.get("non_burst", [])
        if non_burst:
            html += """            <div class="section">
                <h2>📷 单张照片（非连拍）</h2>
                <div class="image-grid">
"""
            
            for img in non_burst:
                img_path = img.get("path", "")
                img_name = img.get("name", "")
                focus_score = img.get("focus_score", 0)
                kept = img.get("kept", True)
                bird_n = img.get("birds_detected", 0)
                card_cls = "kept" if kept else "discarded"
                badge_cls = "kept" if kept else "discarded"
                badge_txt = "✓ 保留（单张）" if kept else "✗ 丢弃（无有效鸟体）"
                
                img_base64 = image_to_base64(img_path)
                
                img_preview = f'<img src="data:image/jpeg;base64,{img_base64}" class="image-preview">' if img_base64 else '<div class="image-preview" style="display:flex;align-items:center;justify-content:center;color:#ccc;">图片加载失败</div>'
                
                html += f"""                    <div class="image-card {card_cls}">
                        {img_preview}
                        <div class="image-info">
                            <div class="image-name">{img_name}</div>
                            <div class="image-stats">
                                <div class="stat-row">
                                    <span class="stat-label">对焦评分:</span>
                                    <span class="stat-value">{focus_score:.2f}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">检出鸟框:</span>
                                    <span class="stat-value">{bird_n}</span>
                                </div>
                            </div>
                            <div class="status-badge {badge_cls}">{badge_txt}</div>
                        </div>
                    </div>
"""
            
            html += """                </div>
            </div>
"""
        
        # 页脚
        html += f"""        </div>
        
        <div class="footer">
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>鸟类检测系统 v1.0 | 基于YOLOv8和Laplacian对焦评分</p>
        </div>
    </div>
    
    <script>
        function toggleGroup(header) {{
            const content = header.nextElementSibling;
            content.classList.toggle('active');
        }}
        
        function toggleAllGroups() {{
            const headers = document.querySelectorAll('.group-header');
            const allActive = Array.from(headers).every(h => 
                h.nextElementSibling.classList.contains('active')
            );
            
            headers.forEach(header => {{
                const content = header.nextElementSibling;
                if (allActive) {{
                    content.classList.remove('active');
                }} else {{
                    content.classList.add('active');
                }}
            }});
        }}
    </script>
</body>
</html>
"""
        
        # 保存HTML文件
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ HTML报告已生成: {output_html_path}")
        return True
    
    except Exception as e:
        print(f"✗ 生成HTML报告失败: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python html_report_generator.py <JSON报告> <输出HTML> [图片文件夹]")
        print("示例: python html_report_generator.py report.json report.html ./photos")
        sys.exit(1)
    
    json_report = sys.argv[1]
    output_html = sys.argv[2]
    image_folder = sys.argv[3] if len(sys.argv) > 3 else None
    
    generate_html_report(json_report, output_html, image_folder)
