---
name: bird-detection-skill
description: 鸟图智慧仓储（Birdy）— 鸟体检测、物种识别（本地/百度/豆包）、连拍筛选、GPS 与 HTML 报告；含 PyQt5 GUI 与 CLI。使用本技能处理鸟类照片文件夹、配置豆包/高德 API、解释输出目录与连拍参数。
---

# Birdy / 鸟图智慧仓储 — Agent 技能

面向 **Cursor Agent**：在仓库 **`birdy-skill-*`**（或含 `src/birdy_gui.py` 的 Birdy 项目）中协助用户完成安装、配置、批处理与排错。

## 何时使用本技能

用户提到：鸟图、Birdy、鸟体检测、物种识别、连拍、豆包/火山方舟、`doubao_api_config`、`amap_api_config`、`birdy_gui`、`birdy_cli`、`Screened_images`、YOLO 鸟分割等。

## 项目要点（给 Agent）

- **入口**：`src/birdy_gui.py`（GUI）、`src/birdy_cli.py`（CLI）。工作目录可为**仓库根**；启动脚本 `start_gui.bat` / `start_gui.sh` 会 `cd` 到 `src` 再执行 `python birdy_gui.py`。  
- **版本**：以根目录 **`skill-info.json`** 的 `version` 为准。  
- **模型**：`models/`（与 `src/` 同级）— `bird-seg.pt`、`birdiden_v1.pth`、`birdeye.pt`（大文件，常不随 Git）；**`bird_info.json`** 为索引元数据，**应随仓库跟踪**，勿加入 `.gitignore`。  
- **豆包**：`src/doubao_api_config.json` — `api_key`、`api_base`、**`models`** 多模型轮换、`daily_token_limit_per_model`、`token_switch_ratio`；用量可写入 `doubao_api_usage.json`。非鸟/非动物图可配置归档到「人像」「其它动物」「其它」等标签。若文件缺失，由 **`api_config_defaults.ensure_doubao_api_config_file`** 自动生成空模板。  
- **高德**：`src/amap_api_config.json`；GUI 有「打开配置文件」。`geocoding_config.py` 可作回退。缺失时 **`ensure_amap_api_config_file`**（或首次调用 `_effective_amap_key`）会生成仅含空 `api_key` 的 JSON。  
- **连拍**：`burst_keep_ratio` + `burst_keep_min`；结果在 **`{输出}/Screened_images/`**。CLI：`--burst-keep-ratio`、`--burst-keep-min`；`--keep-top-n` 已弃用（等同 `--burst-keep-min`）。  
- **依赖**：根目录 **`requirements.txt`**；安装细节见 **`安装说明.md`**。  
- **YOLO**：Ultralytics，**AGPL-3.0**。

## 用户操作速查

1. `python -m venv .venv` → 激活 → `pip install -r requirements.txt`  
2. 放好 `models/` 下权重  
3. 按需填写 `doubao_api_config.json`、`amap_api_config.json`（勿把密钥提交到公开仓库）  
4. `python src/birdy_gui.py` 或 `python src/birdy_cli.py -i <文件夹> -o <输出>`

## 自动化 / 编程调用

**推荐**：全流程与 GUI 行为一致时，在仓库根目录用子进程调用 CLI（或让用户直接运行）：

```bash
python src/birdy_cli.py -i ./images -o ./outputs --crop-output ./crops --api-mode doubao \
  --burst-keep-ratio 0.2 --burst-keep-min 2
```

JSON 配置可通过 `birdy_cli.py --config your.json` 传入；字段含义与 `BirdDetectionCLI.get_default_config()`（见 `src/birdy_cli.py`）一致。

**仅连拍筛选**：在已将 `src` 加入 `sys.path` 的前提下，可直接调用 `burst_grouping.process_folder`（与 CLI 内部第二步相同），例如将保留图写入 `{输出}/Screened_images`：

```python
import sys
from pathlib import Path

src = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src))

from burst_grouping import process_folder

out = Path("./outputs")
out.mkdir(parents=True, exist_ok=True)
screened = out / "Screened_images"

process_folder(
    image_folder=str(Path("./images").resolve()),
    time_threshold=1.0,
    burst_keep_ratio=0.2,
    burst_keep_min=2,
    use_bird_detection=True,
    output_report=str(out / "burst_analysis.json"),
    fast_mode=True,
    screened_output_dir=str(screened),
)
```

物种识别与裁剪需继续使用 `detect_bird_and_eye.BirdAndEyeDetector`（逻辑较复杂，优先走 CLI）。

## 文档索引

| 文件 | 用途 |
|------|------|
| `README.md` | 功能与快速开始 |
| `安装说明.md` | 中文安装、GPU、分发清单、排错 |
| `requirements.txt` | pip 依赖 |
| `skill-info.json` | 版本与 changelog 摘要 |
| `SKILL_README.md` | 技能包元数据与打包说明 |
| `CHANGELOG.md` | 详细变更 |

## Agent 行为建议

- 路径与命令中的 **`src/`** 前缀要与用户实际 cwd 一致。  
- 修改 API 配置时提醒用户不要泄露密钥。  
- 连拍/物种步骤「没有图」时，检查是否生成了 `Screened_images` 或未勾选连拍。  
- 豆包 404：核对方舟 **`api_base`** 与控制台 **推理接入点 ID**（配置里 `model` / `models` 项）。
