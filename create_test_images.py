#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试图片的脚本

使用方法：
python create_test_images.py <输出目录> <图片数量>

示例：
python create_test_images.py ./test_images 10
"""
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

def create_test_images(output_dir, count=10):
    """创建测试用的图片文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始创建 {count} 张测试图片到 {output_dir}")
    
    for i in range(count):
        # 创建一个简单的彩色图片
        img = Image.new('RGB', (1024, 768), color=(73, 109, 137))
        
        # 保存为JPEG文件
        img_path = output_path / f"test_{i+1:03d}.jpg"
        img.save(img_path, "JPEG", quality=85)
        
        print(f"创建: {img_path.name}")
    
    print(f"完成！在 {output_dir} 中创建了 {count} 张测试图片")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python create_test_images.py <输出目录> [图片数量]")
        print("示例: python create_test_images.py ./test_images 10")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    create_test_images(output_dir, count)
