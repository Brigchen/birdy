# 鸟图智慧仓储（Birdy）

**Birdy** 的定位是：面向拍鸟场景的**海量相片自动化整理与归档工具**。它不是单一检测脚本，而是一条从批量照片输入到可用结果输出的完整流水线，目标是减少手工筛图、命名、分类与出图成本。

围绕“海量相片自动化整理与归档”这一目标，核心功能包括：

1. **连拍自动筛选**：按拍摄时间自动聚类连拍序列，结合清晰度与鸟体信息进行智能保留，先把可用照片快速筛出来。
2. **鸟体检测与分割**：批量定位鸟体目标，支持置信度/面积过滤，为后续裁剪、分类与归档提供结构化结果。
3. **物种自动识别与分级归档**：支持本地模型与豆包视觉 API 两套识别路径，按目/科/属/种自动整理输出目录。
4. **地理信息写入与反查**：支持 GPS EXIF 批量写入与地名编码，为归档检索和水印文案提供位置数据。
5. **水印图批量生成与报告输出**：可批量生成带时间/地点/物种等信息的水印成片，并输出处理报告，便于分享与追溯。

提供 **PyQt5 图形界面** 与 **命令行** 两种使用方式，既可交互式操作，也可用于脚本化批处理。

> **当前发布版本**：**2.0.2**（稳定版）  
> **版本发布日期**：**2026-04-17**（与根目录 `**version-info.json`** 中的 `version`、`release_date` 保持一致；后续迭代以此文件为准。）  
> **许可**：整体以仓库 **LICENSE** 为准；项目基于开源协议发布，**仅限爱好者、公益、科研等非盈利用途**，请勿用于商业用途。涉及第三方组件（如 Ultralytics YOLOv8 / PyQt5）时，也请同时遵守其各自许可证要求。请勿将含真实 API Key 的配置文件公开分发。  
> **GUI**：界面依赖 **PyQt5**，请遵守 [PyQt5 / Riverbank 的许可条款](https://www.riverbankcomputing.com/software/pyqt/)（通常为 GPL v3，或商业授权）。

克隆仓库后若缺少 `src/doubao_api_config.json` 或 `src/amap_api_config.json`，**无需手动新建**：首次在 GUI 中点击打开对应配置、或运行 CLI/GPS 相关流程时，程序会在 `src/` 下**自动生成**带完整字段、`api_key` 为空的 JSON 模板（逻辑见 `src/api_config_defaults.py`）。

---

## 使用须知（下载包、模型与 API）

### RAW 格式

当前版本**暂不支持 RAW**（如 `.cr2`、`.nef`、`.arw` 等）。请先将照片导出为 **JPEG / PNG** 等常见格式再处理。仓库内与 `rawpy` 相关的说明属于历史或预留能力，**不作为正式承诺**；后续若开放 RAW 支持会在本 README 与 `CHANGELOG.md` 中单独说明。

### GitHub 源码包与本地模型

从 GitHub 克隆或下载的压缩包**不包含**下列**大体积**本地推理权重（受分发方式限制，需自备或向作者索取）。为便于自备或自训，下面给出与当前代码兼容的最低规格：


| 文件                | 用途                | 兼容规格（最低要求）                                                                                                                                                                                               |
| ----------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bird-seg.pt`     | 鸟体检测与分割           | **Ultralytics YOLO v8 `.pt`** 权重（与项目依赖 `ultralytics>=8.0.0` 一致），可被 `YOLO(path)` 直接加载；推理结果需提供 `boxes.xyxy / boxes.conf / boxes.cls`。建议训练为单类 bird（或以 bird 为主类）的检测/分割模型。                                    |
| `birdeye.pt`      | 鸟眼检测（可选）          | **Ultralytics YOLO v8 `.pt`** 权重（与项目依赖 `ultralytics>=8.0.0` 一致），可被 `YOLO(path)` 直接加载；推理结果需提供边界框（同上 `boxes.*` 字段）。建议训练为鸟眼目标检测模型（单类或少类均可）。                                                                 |
| `bird-iden.pth` | 本地物种识别（ResNet 权重） | **PyTorch `state_dict`**，网络结构需与代码一致：`torchvision.models.resnet34(weights=None)`，并使用 `fc.weight` 维度定义类别数（代码会据此自动构建 `Linear(512, num_classes)`）。输入预处理固定为 `Resize(256)+CenterCrop(224)+ImageNet Normalize`。 |


`**models/bird_info.json**` 为与上述权重配套的物种索引与名称映射表。若你本地缺失该文件，可与 `**bird-iden.pth**` 成套向作者索取。

`bird_info.json` 与 `bird-iden.pth` 的类别维度需配套：建议 `len(bird_info)` 与分类器输出类别数一致（至少不小于输出类别数），并保持索引顺序一一对应（第 `i` 类对应该 JSON 的第 `i` 条）。

将三个权重文件放入项目根目录下的 `**models/**`（与 `src/` 同级）即可与程序默认路径一致。若手中暂无权重，**可邮件联系作者**：**[brigchen@gmail.com](mailto:brigchen@gmail.com)**，说明用途与平台，便于单独获取或约定分发方式。

### 建议申请的 API Key（均为常见「按量 / 试用」档，个人学习可视为低成本或免费额度）

在仅使用本地模型、且不写入 GPS、不做地名反查时，可以不配置任何云端 Key。若你希望**批量写入 GPS、地名反查、水印中的地理位置**，或**在本地物种识别不满意时改用云端视觉识别**，建议提前申请：

1. **高德开放平台 Web 服务 Key**（[https://lbs.amap.com/](https://lbs.amap.com/)）
  用于：批量地理编码（地名 → 坐标）、配合 EXIF 的 GPS 写入，以及水印生成时的城市/地点等地理文案。将 Key 填入 `src/amap_api_config.json`（GUI 内也可打开该文件）。
2. **火山引擎方舟「豆包」视觉模型接入**（[https://www.volcengine.com/ark/](https://www.volcengine.com/ark/)）
  用于：在 `**doubao_api_config.json`** 中配置后，通过豆包视觉 API 做鸟类及相关主体的识别；适合本地 ResNet 结果不理想、或需要更广物种覆盖时的补充方案。具体字段说明见下文「配置说明」表。

两项均请在各平台控制台完成实名/应用创建后获取密钥；请妥善保管真实 Key，避免泄露。

---

## 功能概览


| 模块            | 说明                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| **鸟体检测**      | YOLOv8 分割，支持置信度与面积过滤、可选鸟眼辅助                                                                                  |
| **物种识别**      | 本地 ResNet 或 **百度** / **火山方舟豆包** 视觉 API；豆包支持 `**doubao_api_config.json` 多模型轮换** 与按模型日 token 统计                |
| **物种名称规范与检索** | 基于 `bird_classification.json` 整理出的 `data/species/bird_species_list.csv`（中文名 / 英文名 / 学名对照），用于名称不一致时的统一规范与快速检索 |
| **连拍筛选**      | 时间聚类 + 清晰度（可结合鸟 ROI）；`**burst_keep_ratio` + `burst_keep_min`**；非连拍单张可在开启鸟检时按策略丢弃                             |
| **地理信息**      | EXIF GPS、**高德** / 其它地理编码（`src/amap_api_config.json` + `geocoding_config.py`）                                 |
| **报告**        | 连拍报告、物种识别报告；GUI 含 **ETA** 与各阶段耗时估算                                                                           |


---

## 环境要求

- **Python 3.8+**（建议 3.10～3.12；3.13 请以本机实测为准）  
- **PyTorch** + **CUDA**（可选，用于 GPU 加速）  
- 依赖见 `**requirements.txt`**；详细步骤见 `**安装说明.md**`（中文）

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

将以下文件放在 `**models/**`（与 `src/` 同级；其中 `.pt` / `bird-iden.pth` 体积大，通常需单独拷贝或网盘分发）：


| 文件                                   | 用途               |
| ------------------------------------ | ---------------- |
| `bird-seg.pt`                        | 鸟体检测             |
| `bird-iden.pth` + `bird_info.json` | 本地物种分类（权重 + 索引表） |
| `birdeye.pt`                         | 鸟眼检测（可选）         |


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


| 路径                               | 说明                                                                                             |
| -------------------------------- | ---------------------------------------------------------------------------------------------- |
| `**src/doubao_api_config.json**` | 豆包：`api_key`、`api_base`、`models` 列表、`daily_token_limit_per_model`、`token_switch_ratio`、非鸟归档标签等 |
| `**src/amap_api_config.json**`   | 高德 Web Key（GUI「打开配置文件」）                                                                        |
| `**src/geocoding_config.py**`    | 地理编码回退与开关                                                                                      |
| `**src/gui_config.json**`        | GUI 保存的参数（运行后生成）                                                                               |


豆包用量统计默认写入运行目录下的 `**doubao_api_usage.json**`（可按配置调整路径）。

---

## 目录结构（要点）

```
birdy-skill/
├── README.md                 # 本文件
├── 安装说明.md               # 中文安装与分发
├── requirements.txt
├── version-info.json             # 版本与变更摘要
├── start_gui.bat / start_gui.sh
├── models/                    # 模型文件目录（见上文模型规格表）
├── data/                      # 地理与物种数据（含 bird_species_list.csv 名称对照表）
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

- 勾选连拍且启用筛选时，保留图片写入 `**{输出目录}/Screened_images/**`，后续物种识别默认使用该目录。  
- `**burst_keep_ratio**`：每组保留比例（如 `0.2` 约等于五选一）。  
- `**burst_keep_min**`：每组至少保留张数（与比例取较大值，且不超过组大小）。  
- CLI 中 `**--keep-top-n**` 已弃用，语义等同 `**--burst-keep-min**`。

---

## 引用

若本工作对您的研究有帮助，可引用。英文 **title** 与中文产品名「**鸟图智慧仓储**」含义相近：面向鸟类相片的**自动化处理流水线**（检测、识别等），并写明基于拍摄时间戳的**连拍分组与智能筛选**。

```bibtex
@misc{birdy2026,
  title={Birdy: Intelligent Automation for Bird Photographs--Detection, Species Identification, and Temporal Burst-Sequence Smart Filtering},
  author={Chen, Brig},
  year={2026},
  url={https://github.com/Brigchen/birdy}
}
```

---

## 相关文档

- `**安装说明.md**` — 安装、GPU、配置模板、分发清单、常见问题  
- `**CHANGELOG.md**` — 版本变更记录

---

*README 随功能迭代更新。当前文档对应 **2.0.2**，发布日期 **2026-04-17**；之后请以根目录 `**version-info.json*`* 中的 `version` 与 `release_date` 为准。*