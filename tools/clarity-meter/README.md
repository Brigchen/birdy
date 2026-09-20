# 清晰度打分（clarity-meter）

只读工具：调用 Birdy **认种前模糊评分**（鸟体分割掩膜 + 长边 640 + 抑噪 Laplacian + 峰度/噪点修正），**不删除、不改写**源图片。打完后生成 HTML 表格报告并用浏览器打开。

须在 Birdy 仓库内运行（需要 `src/` 与 `models/bird-seg.pt`）。

## 快速开始

```bat
Clarity.bat
```

或命令行指定目录：

```bat
Clarity.bat --folder "C:\Users\brigc\Pictures\test" --threshold 35
```

Linux / macOS：`chmod +x start.sh && ./start.sh`

## 报告列

- 预览（红线为计分掩膜）、相对路径、尺寸、检出鸟数
- **整框分**：鸟框内整块区域（含地面）
- **鸟体掩膜分**：与主程序清洗相同的分数
- **判定**：通过 / 模糊 / 未检出鸟体 / 读图失败（对照阈值，默认 35）

勾选「已是切割图」时按主流程切割后再清洗的方式计分。

报告写在**原图片目录**下的 `clarity_report_日期时间.html`，表格显示叠了掩膜红线的预览与文件名（点击打开原图）。
