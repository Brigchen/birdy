#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GPS EXIF写入速度的脚本

使用方法：
python test_gps_speed.py <测试图片文件夹> <测试次数>

示例：
python test_gps_speed.py ./test 3
"""
import os
import sys
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from geo_encoder import batch_write_gps_exif

def test_gps_write_speed(image_folder, test_count=3):
    """测试GPS写入速度"""
    folder = Path(image_folder)
    if not folder.exists():
        print(f"错误: 文件夹不存在: {image_folder}")
        return
    
    # 测试坐标（杭州西湖）
    latitude = 30.2741
    longitude = 120.1551
    
    print("=" * 70)
    print(f"开始测试 GPS EXIF 写入速度")
    print(f"测试文件夹: {image_folder}")
    print(f"测试坐标: ({latitude:.6f}, {longitude:.6f})")
    print(f"测试次数: {test_count}")
    print("=" * 70)
    
    # 计算图片数量
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions and f.is_file()]
    image_count = len(image_files)
    
    if image_count == 0:
        print("错误: 测试文件夹中没有图片文件")
        return
    
    print(f"找到 {image_count} 张图片")
    print("=" * 70)
    
    # 测试不同线程数的性能
    thread_counts = [1, 4, 8, 16, None]  # None表示自动
    
    for thread_count in thread_counts:
        print(f"\n测试线程数: {thread_count or '自动'}")
        print("-" * 50)
        
        total_times = []
        
        for i in range(test_count):
            start_time = time.time()
            success_count = batch_write_gps_exif(
                image_folder, 
                latitude, 
                longitude, 
                max_workers=thread_count
            )
            end_time = time.time()
            elapsed = end_time - start_time
            total_times.append(elapsed)
            
            print(f"测试 {i+1}/{test_count}: {elapsed:.2f} 秒, 成功: {success_count}/{image_count}")
        
        # 计算平均时间
        avg_time = sum(total_times) / len(total_times)
        speed_per_image = avg_time / image_count
        
        print("-" * 50)
        print(f"平均时间: {avg_time:.2f} 秒")
        print(f"每张图片平均时间: {speed_per_image:.4f} 秒")
        print(f"速度提升: {1.0 / speed_per_image:.2f} 张/秒")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_gps_speed.py <测试图片文件夹> [测试次数]")
        print("示例: python test_gps_speed.py ./test 3")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    test_count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    test_gps_write_speed(image_folder, test_count)
