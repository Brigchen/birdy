#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPX 轨迹报告生成器 - 命令行工具

用法:
    python gpx_report_cli.py <gpx文件路径> [输出文件名] [标题]

示例:
    python gpx_report_cli.py data/20260516户外步行_合并.gpx
    python gpx_report_cli.py data/20260516户外步行_合并.gpx 我的徒步报告 "周末徒步"

作者: brigchen@gmail.com
版权说明: 基于开源协议，仅限爱好者、公益、科研等非盈利用途，请勿用于商业用途
"""

import sys
from pathlib import Path
from gpx_visualizer import generate_html_report


def main():
    if len(sys.argv) < 2:
        print("用法: python gpx_report_cli.py <gpx文件路径> [输出文件名] [标题]")
        print("示例: python gpx_report_cli.py data/20260516户外步行_合并.gpx")
        sys.exit(1)
    
    # 获取输入文件
    gpx_path = Path(sys.argv[1])
    if not gpx_path.exists():
        print(f"错误: 文件不存在 - {gpx_path}")
        sys.exit(1)
    
    # 获取输出文件名
    if len(sys.argv) >= 3:
        output_name = sys.argv[2]
        if not output_name.endswith('.html'):
            output_name += '.html'
    else:
        output_name = gpx_path.stem + '_轨迹报告.html'
    
    # 获取标题
    title = sys.argv[3] if len(sys.argv) >= 4 else gpx_path.stem
    
    # 确定输出路径
    project_root = Path(__file__).parent.parent
    output_path = project_root / 'data' / output_name
    
    # 生成报告
    print(f"正在生成轨迹报告...")
    print(f"输入文件: {gpx_path}")
    print(f"输出文件: {output_path}")
    print(f"报告标题: {title}")
    print()
    
    success = generate_html_report(gpx_path, output_path, title)
    
    if success:
        print()
        print("=" * 50)
        print("报告生成成功!")
        print(f"请在浏览器中打开查看: {output_path}")
        print("=" * 50)
    else:
        print("报告生成失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()
