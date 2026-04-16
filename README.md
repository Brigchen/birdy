# 鸟图智慧仓储（Birdy）

**Birdy** 是一套面向观鸟与生态摄影的桌面工具：鸟体检测与分割、鸟眼检测、可选物种识别（本地模型或云端 API）、连拍智能筛选、GPS/地名写入与 HTML 报告。提供 **PyQt5 图形界面** 与 **命令行** 两种使用方式。

> **版本**：以项目根目录 **`skill-info.json`** 中的 `version` 与 `release_date` 为准（当前文档编写时对应 **2.0.0 稳定版**）。  
> **许可**：整体以仓库 **LICENSE** 为准；**Ultralytics YOLOv8** 为 **AGPL-3.0**，商用请自行评估。请勿将含真实 API Key 的配置文件公开分发。  
> **GUI**：界面依赖 **PyQt5**，请遵守 [PyQt5 / Riverbank 的许可条款](https://www.riverbankcomputing.com/software/pyqt/)（通常为 GPL v3，或商业授权）。

克隆仓库后若缺少 `src/doubao_api_config.json` 或 `src/amap_api_config.json`，**无需手动新建**：首次在 GUI 中点击打开对应配置、或运行 CLI/GPS 相关流程时，程序会在 `src/` 下**自动生成**带完整字段、`api_key` 为空的 JSON 模板（逻辑见 `src/api_config_defaults.py`）。公开推送前请使用根目录 **`.gitignore`**，避免将密钥与权重提交到 Git。

---

## 功能概览

| 模块 | 说明 |
|------|------|
| **鸟体检测** | YOLOv8 分割，支持置信度与面积过滤、可选鸟眼辅助 |
| **物种识别** | 本地 ResNet 或 **百度** / **火山方舟豆包** 视觉 API；豆包支持 **`doubao_api_config.json` 多模型轮换** 与按模型日 token 统计 |
| **连拍筛选** | 时间聚类 + 清晰度（可结合鸟 ROI）；**`burst_keep_ratio` + `burst_keep_min`**；非连拍单张可在开启鸟检时按策略丢弃 |
| **地理信息** | EXIF GPS、**高德** / 其它地理编码（`src/amap_api_config.json` + `geocoding_config.py`） |
| **报告** | 连拍报告、物种识别报告；GUI 含 **ETA** 与各阶段耗时估算 |

---

## 环境要求

- **Python 3.8+**（建议 3.10～3.12；3.13 请以本机实测为准）  
- **PyTorch** + **CUDA**（可选，用于 GPU 加速）  
- 依赖见 **`requirements.txt`**；详细步骤见 **`安装说明.md`**（中文）

---

## 快速开始

### 1. 安装依赖

在项目**根目录**（含 `src/`、`requirements.txt`）：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

GPU 用户建议先到 [pytorch.org](https://pytorch.org) 安装匹配 CUDA 的 `torch` / `torchvision`，再安装其余依赖（见 `安装说明.md`）。

### 2. 模型文件

将以下文件放在 **`models/`**（与 `src/` 同级；体积大，通常需单独拷贝或网盘分发）：

| 文件 | 用途 |
|------|------|
| `yolov8x-seg.pt` | 鸟体检测 |
| `model20240824.pth` + `bird_info.json` | 本地物种分类 |
| `birdeye.pt` | 鸟眼检测（可选） |

### 3. 启动 GUI

**从项目根目录**运行（或使用根目录下的启动脚本，脚本会进入 `src` 再启动）：

```bash
# Windows
python src\birdy_gui.py
# 或双击 start_gui.bat

# macOS / Linux
python src/birdy_gui.py
# 或 ./start_gui.sh
```

### 4. 命令行示例

```bash
python src/birdy_cli.py -i ./images -o ./outputs
python src/birdy_cli.py -i ./images --api-mode doubao --burst-keep-ratio 0.2 --burst-keep-min 2
python src/birdy_cli.py --help
```

---

## 配置说明

| 路径 | 说明 |
|------|------|
| **`src/doubao_api_config.json`** | 豆包：`api_key`、`api_base`、`models` 列表、`daily_token_limit_per_model`、`token_switch_ratio`、非鸟归档标签等 |
| **`src/amap_api_config.json`** | 高德 Web Key（GUI「打开配置文件」） |
| **`src/geocoding_config.py`** | 地理编码回退与开关 |
| **`src/gui_config.json`** | GUI 保存的参数（运行后生成） |

豆包用量统计默认写入运行目录下的 **`doubao_api_usage.json`**（可按配置调整路径）。

---

## 目录结构（要点）

```
birdy-skill/
├── README.md                 # 本文件
├── 安装说明.md               # 中文安装与分发
├── SKILL.md                  # Cursor Agent Skill 说明
├── requirements.txt
├── skill-info.json           # 版本与变更摘要
├── start_gui.bat / start_gui.sh
├── models/                    # 权重与 bird_info.json（需自备）
├── data/                      # 地理与物种数据
├── resources/                 # logo 等静态资源
├── test/                      # 测试脚本与样例
└── src/
    ├── birdy_gui.py          # 图形界面入口
    ├── birdy_cli.py          # 命令行入口
    ├── doubao_bird_api.py    # 豆包视觉 API
    ├── doubao_api_config.json
    ├── amap_api_config.json
    └── ...
```

---

## 连拍与输出目录

- 勾选连拍且启用筛选时，保留图片写入 **`{输出目录}/Screened_images/`**，后续物种识别默认使用该目录。  
- **`burst_keep_ratio`**：每组保留比例（如 `0.2` 约等于五选一）。  
- **`burst_keep_min`**：每组至少保留张数（与比例取较大值，且不超过组大小）。  
- CLI 中 **`--keep-top-n`** 已弃用，语义等同 **`--burst-keep-min`**。

---

## 引用

若本工作对您的研究有帮助，可引用：

```bibtex
@misc{birdy2024,
  title={Birdy: Bird Detection and Species Identification Tool},
  author={Chen, Brig},
  year={2024},
  url={https://github.com/brigchen-yi/Birdy}
}
```

---

## 相关文档

- **`安装说明.md`** — 安装、GPU、配置模板、分发清单、常见问题  
- **`SKILL.md`** — 供 Cursor Agent 调用的技能说明  
- **`SKILL_README.md`** — 技能包结构与字段说明  
- **`CHANGELOG.md`** — 版本变更记录  

---

*README 随功能迭代更新；具体版本号请以 **`skill-info.json`** 为准。*
