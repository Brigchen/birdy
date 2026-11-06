# Video Stabilizer（观鸟录像稳定）

独立于 Birdy 主程序的 **MP4 稳定工具**，面向手持/长焦观鸟录像。两种效果与 Birdy GUI「视频裁剪」里的模式一致，但本工具只做稳定、无水印/识别耦合。

| 模式 | 用途 | 默认实现 |
|------|------|----------|
| **三脚架固定视角** | 背景/地平线钉死，只消手抖 | 光流 + 首帧仿射锚点；可选 FFmpeg `libvidstab` `opttripod` |
| **追踪稳定器** | 目标鸟保持在画面中心 | YOLO `bird-seg.pt` + ByteTrack；无模型时用框跟踪（CSRT，OpenCV 5 为 MIL） |

## 快速开始

在 Birdy 仓库内（需已 `pip install -r requirements.txt`，含 `vidstab` / `ultralytics`）：

```bat
cd tools\video-stabilizer
pip install -r requirements.txt
Stabilize.bat
```

CLI 示例：

```bat
python -m video_stab --mode tripod input.mp4 -o output_tripod.mp4
python -m video_stab --mode track --smooth 40 input.mp4
python -m video_stab --mode track --bbox 0.40,0.35,0.18,0.22 input.mp4
python -m video_stab --mode tripod D:\clips --recursive -o D:\clips_out
python -m video_stab --gui
```

GUI 里可选择文件或目录（可选含子目录），追踪模式下在预览画面拖拽框选目标。按钮 **创建桌面快捷方式** 会在当前用户桌面生成「观鸟录像稳定」，用 `pythonw` 打开本 GUI。

- Windows 可自动使用仓库 `bin/ffmpeg.exe`（Git LFS）合并原音轨。
- 追踪模式默认加载 `models/bird-seg.pt`（与 Birdy 鸟检相同）。

输出固定在 `C:\Users\brigc\Pictures\watermarked`。

---

## 方案选型（现成算法 / 插件）

### 模式 A：三脚架固定视角

目标：整幅画面相对**第一帧**（或全局参考）锁定，适合「站定拍固定背景上的鸟」。

| 方案 | 成熟度 | 优点 | 缺点 | 本工具 |
|------|--------|------|------|--------|
| **FFmpeg libvidstab**（`vidstabdetect` + `vidstabtransform`，`smoothing=0`，`opttripod=1`） | 高 | 全 GPU/原生 C，两 pass；仓库 ffmpeg 已带 `--enable-libvidstab` | 大平移/变焦会裁切或黑边；参数需试 | `--tripod-engine ffmpeg` |
| **Python vidstab**（`pip install vidstab`） | 高 | 与 Birdy 水印模块相同引擎，平滑窗口可调 | 偏「手持防抖」而非硬锁；大抖动仍漂 | Birdy 内「标准防抖」 |
| **OpenCV 光流锁定首帧**（相位相关初值 + LK + 相似变换） | 中 | 大位移不设上限；锁定后再磨掉几像素高频残差，裁到全程公共可视区，允许裁掉一半以上面积，无黑边 | CPU 逐帧，4K 较慢 | **默认 `optical`** |
| OpenCV `videostab`（contrib） | 中 | 官方 VideoStabilizer 管线 | Windows 轮子常未编译 contrib | 未采用 |
| Gyroflow / DaVinci / Premiere | 高（GUI） | 专业工作流 | 难批处理、难与 Birdy 脚本集成 | 人工备选 |

**推荐**：批处理默认 **光流首帧锚点**；若已熟悉 FFmpeg 且片源抖动频率稳定，可试 **`ffmpeg` 引擎**。

### 模式 B：追踪稳定器（鸟在中心）

目标：检测/跟踪鸟体，平滑中心轨迹，动态裁切（虚拟云台）。

| 方案 | 成熟度 | 优点 | 缺点 | 本工具 |
|------|--------|------|------|--------|
| **Ultralytics YOLO + `model.track()`**（ByteTrack/BoT-SORT） | 高 | 与 Birdy 鸟检模型一致；换种/遮挡可重检 | 需 GPU/CPU 推理；极小目标可能丢 | **默认** |
| OpenCV **CSRT/KCF** 相关跟踪 | 中 | 无深度学习依赖 | 初始框不准易丢；鸟形变化大 | `--track-engine csrt` 回退 |
| 检测 + **Kalman** 平滑中心 + crop | 高 | 可控、可解释 | 需自写管线 | 已实现为移动平均 + 可选光流微抖补偿 |
| **SiamRPN / NanoTrack** 等单目标跟踪 | 中 | 专盯一个 ID | 额外模型；观鸟多鸟场景弱 | 后续可选 |
| 纯光流质心 | 低 | 快 | 背景干扰大 | 不采用 |

**推荐**：有 `bird-seg.pt` 时用 **YOLO 跟踪**；临时无模型时用 **CSRT + 画面中心 40% 初始框**（需鸟大致在中心）。

### 其他可扩展

- **标准手持防抖**（非三脚架）：`vidstab` / FFmpeg，`smoothing_window=15~30` — Birdy「水印与分享 → 视频裁剪」已提供；本工具专注上述两种观鸟场景。
- **深度学习端到端 stabilizer**（FuSta、PWStableNet 等）：画质好但依赖重、实时性差，适合研究而非当前批处理工具。

---

## 架构

```
input.mp4
  → [可选 ffmpeg 按时间截取]
  → tripod: optical | ffmpeg vidstab
    track:  YOLO track 或 CSRT → 轨迹平滑 → 裁切输出
  → [ffmpeg 合并原音轨]
  → C:\Users\brigc\Pictures\watermarked\*.mp4
```

与 Birdy 关系：**代码独立**于 `tools/video-stabilizer/`；算法与 `src/video_stabilize_dialog.py` 同源思路，但不 import Birdy `src/`，不修改主程序版本号。

## 依赖

见 `requirements.txt`。追踪模式需要 `torch` + `ultralytics` 及 `models/bird-seg.pt`（Birdy LFS）。

## 打包分享

压缩整个 `video-stabilizer/` 目录；接收方在 Birdy 仓库根附近保留 `bin/ffmpeg.exe` 与 `models/bird-seg.pt`，或自行安装系统 ffmpeg / 指定 `--bird-model`。
