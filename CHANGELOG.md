# 🐦 鸟类检测Skill - 更新日志

## v2.0.2 (2026-04-17)

### 产品定位、许可与模型命名

- README 介绍重写：定位为拍鸟场景下的海量相片自动化整理与归档工具，并按连拍筛选、检测、识别归档、地理写入、水印与报告等流程组织核心能力。
- 许可说明统一为：仅限爱好者、公益、科研等非盈利用途，请勿用于商业用途。
- 鸟体检测权重文件名由 **`yolov8x-seg.pt`** 更名为 **`bird-seg.pt`**，代码默认路径、CLI 默认参数与文档已同步更新。

---

## v2.0.1 (2026-04-17)

### 模型与仓库元数据

- 本地物种分类权重文件由 **`model20240824.pth`** 重命名为 **`birdiden_v1.pth`**，代码、CLI 默认值与文档已同步更新。
- 明确 **`models/bird_info.json`** 为应纳入版本控制的索引文件；**`.gitignore`** 不再保留「可忽略 bird_info」类表述，避免误配。

---

## v2.0.0 (2026-04-17) 🎉

### 稳定版发布

- 阶段性功能迭代收敛，发布 **2.0.0** 作为当前稳定基线版本；**GitHub 公开发布日期：2026-04-17**（与 `skill-info.json` 的 `release_date` 一致）。

### 物种识别与地理约束

- 本地物种识别地理规则升级为 **top10 地理筛选**：
  - 在地理名单内（省级或中国索引）按规则优先；
  - 不在地理名单的候选仅当 **`confidence > 0.75`** 才保留。
- 新增可解释日志，未知判定会输出 Top1 与过滤原因，便于参数调优与排查。

### 阈值策略（GUI）

- GUI 的「未知种类阈值」改为**可选开关**：
  - 默认不启用：不做 top10 置信度阈值初筛，仅按地理规则筛选；
  - 启用后：按阈值进行候选过滤并执行低置信度未知判定。

### 豆包 API 优化

- 请求侧禁用 thinking/reasoning 输出，减少不必要 token 开销。
- 识别结果简化为「鸟类 / 其它」，非鸟统一归档为「其它」。
- 404/401/403 场景下，失效模型会话内自动加入黑名单，后续调用会跳过，避免反复命中同一坏模型。

### 水印输出

- 水印批处理输出不再镜像原目录层级，统一直接保存到目标输出目录根下；同名文件自动追加序号避让。

---

## v1.2.0 (2026-04-15) 🚀

### 鸟眼检测召回增强

- 鸟眼检测由整图推理改为**逐鸟体 ROI 推理**，并支持 ROI 外扩参数，提升小目标与远距离场景召回。
- 有效眼判定升级为“眼中心落入鸟体框 + 容差比例”，降低框偏移导致的漏判。
- 新增每鸟高置信 TopN 保留策略，减少噪声框干扰。

### 连拍评分与筛选优化

- 连拍综合分引入 **`BIRD_AREA_WEIGHT`**，将鸟体面积项改为可配置权重并降低默认值，减少“大鸟体”对排序的过度主导。
- 鸟眼检出继续以加权项参与筛选，不破坏原有清晰度主逻辑。

### 进度条与耗时感知优化

- GUI 进度条从“按步骤跳变”升级为“按阶段内真实处理量推进”：
  - 连拍阶段按已处理图片/总图片实时推进；
  - 物种阶段按逐图处理实时推进；
  - 水印阶段按逐图生成实时推进。
- 预计剩余时间与用户体感一致性提升，长阶段不再出现“长时间卡住不动”。

### 新增水印功能卡片

- 增加「水印生成」卡片，支持输入/输出目录、签名 Logo、地理位置（手填或 GPS 城市）、日期、物种名、相机参数等选项。
- 新增「预览一张效果」按钮，支持先预览再批量生成。

### 目录结构规范化

- `data/`、`models/`、`test/` 调整到项目根目录与 `src/` 同级。
- logo 等静态资源统一归档到 `resources/`，并同步修正代码路径与文档说明。

---

## v1.1.10 (2026-04-12) 🐛

### 经纬度输入改为数字文本框

- **GUI 经纬度输入**：`QDoubleSpinBox` → `QLineEdit`（避免鼠标滚轮误触改变数值）
- 新增 `_get_gps_coords()` 辅助方法，支持范围校验（纬度 -90~90，经度 -180~180），非法输入弹出警告并阻断处理流程。
- 输入框占位提示：`"例: 31.230416"` / `"例: 121.473682"`

### 本地模型与豆包 API 完全解耦

- **本地模型模式**：不再读取 `doubao_api_config.json`，彻底消除不必要的文件 I/O 和配置依赖。
- **豆包 API 模式**：同样不再强制加载本地权重配置。

### 本地模型「未知种类」阈值可配置

- 新增 `min_species_accept_confidence` 参数（默认 **0.5**，低于此值视为未知）。
- **GUI**：物种识别卡片新增「未知种类阈值」旋钮（0.0~1.0，步进 0.05）。
- **CLI**：新增 `--min-species-conf` 参数（例：`--min-species-conf 0.3`）。
- 此参数**仅对本地模型生效**，豆包 API 模式不受影响。
- 旧版硬编码的 0.7 阈值已废除。

---

## v1.1.9 (2026-04-11) ✨

### 连拍输出

- 连拍筛选**保留**的图片复制到 **`{output_folder}/Screened_images/`**，并尽量保持相对输入目录的子路径，避免重名覆盖。

### 物种置信度与「未知」

- 顶一物种候选**置信度低于 70%**（或地理过滤后无候选）视为**未知种类**，分类为 **未知目 / 未知科 / 未知属 / 未知**，裁剪或原图归档均归入该分级目录。

### 物种识别与裁剪拆分（GUI / CLI）

- 配置项 **`enable_species_detection`** 与 **`enable_crop`** 独立：可只做识别不归档裁剪、或只裁剪不跑物种模型等组合。
- **仅识别、不勾选裁剪**：将**整张原图**复制到「裁剪输出文件夹」下，按**置信度最高的物种**对应 目/科/属/种 目录（多鸟取全局最高置信度候选）；低于 70% 走未知分级。
- CLI：`--no-species` 禁用物种识别；`--no-crop` 关闭裁剪（保留物种时走原图复制）。

---

## v1.1.8 (2026-04-11) 🔧

### 模型路径与报错说明

- **`BirdAndEyeDetector` 默认权重路径**：统一到项目根 **`models/`**（与 `src/` 同级）。从项目根执行 `python src/birdy_gui.py` 或 `start_gui.bat` 时会正确加载 `models/bird-seg.pt` 等，避免误读错误/空文件导致 `PytorchStreamReader … zip … central directory` 一类错误。
- **加载失败提示**：对 YOLO 与 `torch.load` 物种权重捕获上述 zip/损坏类异常，抛出带中文说明的 `RuntimeError`（含绝对路径与排查建议）。

---

## v1.1.7 (2026-04-11) 🔧

### Windows 任务栏图标

- **GUI 入口**：在创建 `QApplication` 前调用 `SetCurrentProcessExplicitAppUserModelID`，使 Birdy 与宿主 `python.exe`（如 Anaconda/Jupyter 共用解释器时）在任务栏上分离，从而采用 `setWindowIcon` 的 Birdy Logo。
- 创建主窗口后对 `QApplication.setWindowIcon` 同步主窗图标，提高壳层识别一致性。
- **`start_gui.bat`**：补充说明若仍异常可检查 `resources\birdy_logo_128.png` 或使用快捷方式指定 `.ico`。

---

## v1.1.6 (2026-04-11) ✨

- **GUI**：处理进度卡片在进度条下方增加 **已用时间**（`time.monotonic()` + 500ms 刷新）与 **预计剩余**（按当前进度百分比线性估算：已用 × (100−p) / p；p=0 时显示「—」，完成时「0秒」）。

---

## v1.1.5 (2026-04-11) 🔧

- **GUI**：整体收紧间距与内边距（主区、卡片、表单、全局 QSS、顶栏、进度区），控件更紧凑。
- **默认模型**：`gui_config.json` 中 `use_local_model` 改为 `true`；若旧配置缺少该字段，加载时默认本地模型。

---

## v1.1.4 (2026-04-11) ✨

### 品牌与界面（GUI）

- **产品名称**：中文「鸟图智慧仓储」，英文「Birdy」；窗口标题栏显示 **Birdy**。
- **固定顶栏**：顶部白底品牌栏展示 Logo、中英文名称与版本号（版本读取 `skill-info.json`），与下方功能区分离，**不参与左右侧滚动**。
- **图标**：窗口与 Windows 任务栏图标优先使用 `resources/birdy_logo_128.png`，其次 `birdy_logo_640.png` / `logo.png`；无文件时沿用内置矢量回退绘制。
- **Banner Logo**：优先 `resources/birdy_logo_640.png`，其次 `birdy_logo_128.png` / `logo.png`。
- 右侧底部重复大图 Logo 已移除（改由顶栏统一展示）。

---

## v1.1.3 (2026-04-11) 🔧

### GUI 布局

- **可滚动面板**：左右两栏内容分别放入 `QScrollArea`（`setWidgetResizable(True)`），内容按自然高度排版；窗口高度不足时纵向滚动，避免 `QVBoxLayout` 把输入框、按钮压扁导致裁切。
- **尺寸策略**：设置面板与卡片使用 `QSizePolicy` 纵向 `Minimum`，优先占满内容所需高度而非被强行压缩。

---

## v1.1.2 (2026-04-11) 🔧

### GUI 显示

- **高 DPI / 缩放**：在创建 `QApplication` 前启用 `AA_EnableHighDpiScaling` 与 `AA_UseHighDpiPixmaps`，改善 Windows 在 125%/150%/200% 缩放下界面与字体过小的问题。
- **字体与样式表**：全局样式由 `px` 改为 `pt`，并略增大正文字号；卡片标题、主副标题、按钮、日志与表格统一在 QSS 中指定字号，避免仅改 `QFont` 被样式表覆盖而不生效。
- **控件与资源**：复选框/单选框指示器加大；日志等宽字体补充 `Microsoft YaHei UI` 回退；底部 logo 按主屏 `devicePixelRatio` 缩放并设置 pixmap DPR，高分屏上更清晰。

---

## v1.1.1 (2026-03-19) 🔧

### 修复与改进

#### 1. 依赖库问题修复 ⭐

- **PyQt5 替换 PyQt6**
  - 将 `requirements.txt` 中的 PyQt6 改为 PyQt5>=5.15.0
  - 与实际代码使用的 PyQt5 保持一致
  - 移除了不必要的 timm 依赖

#### 2. 文件结构标准化 ⭐

- **源文件组织**
  - 所有 Python 源代码移至 `src/` 目录
  - 模型文件移至 `models/` 目录
  - 数据资源移至 `data/` 目录
- **启动脚本**
  - `start_gui.bat` - Windows GUI 启动脚本
  - `start_gui.sh` - Linux/macOS GUI 启动脚本

#### 3. 文件重命名 ⭐

- **GUI 主程序**
  - 从 `gui_app.py` 重命名为 `birdy_gui.py`
  - 保持 PyQt5 框架不变
- **CLI 主程序**
  - 新增 `birdy_cli.py` - 命令行界面
  - 支持完整的参数配置和批处理

#### 4. 文档完善 ⭐

- **README.md**
  - 完整的项目介绍和功能说明
  - 详细的目录结构说明
  - 快速开始指南和参数说明
  - 工作流程和API集成说明
- **skill.md**
  - 更新所有路径引用到新结构
  - 添加 CLI 版本使用说明
  - 更新文件结构和API示例
- **skill-info.json**
  - 版本号升级到 1.1.1
  - 更新发布日期到 2026-03-19
  - 添加 v1.1.1 的变更日志

### 技术细节

#### 路径引用更新

所有代码中的路径引用已更新：

```python
# 旧路径
from gui_app import ...
from detect_bird_and_eye import ...

# 新路径（保持相对导入）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from burst_grouping import process_folder
from html_report_generator import generate_html_report
```

#### 模型路径

```python
# 模型文件在 models/
models/bird-seg.pt
models/birdeye.pt
models/birdiden_v1.pth
models/bird_info.json
```

#### 数据路径

```python
# 数据文件在 data/
data/species/bird_classification.json
data/species/geo_species_index.json
data/species/species_text_aliases.json
data/geo/world-area.json
```

### 文件变更

#### 新增文件 ⭐

```
birdy-skill-1.1.1/
├── README.md                  ✨ 新增完整README
├── src/                       ✨ 新增源文件目录
│   ├── birdy_gui.py          ⭐ 重命名（原gui_app.py）
│   ├── birdy_cli.py          ⭐ 新增CLI主程序
├── models/                   ✨ 新增模型目录
├── data/                     ✨ 新增数据目录
├── resources/                ✨ 新增资源目录
├── test/                     ✨ 新增测试目录
├── start_gui.bat             ✨ 新增Windows启动脚本
└── start_gui.sh              ✨ 新增Linux/macOS启动脚本
```

#### 修改文件 🔧

```
├── requirements.txt           🔧 PyQt6→PyQt5，移除timm
├── skill.md                  🔧 更新路径和版本
├── skill-info.json           🔧 版本号→1.1.1
└── CHANGELOG.md              🔧 添加v1.1.1日志
```

### 向后兼容性

✅ **完全兼容**

- 所有原有API保持不变
- 命令行脚本继续工作
- Python代码集成无影响
- 配置文件格式相同
- 只需更新启动脚本路径

### 测试覆盖

- ✅ GUI启动和加载
- ✅ CLI命令行处理
- ✅ 参数配置保存
- ✅ GPS查询功能
- ✅ 工作线程处理
- ✅ 异常处理机制
- ✅ 跨平台兼容性
- ✅ EXIF信息保留

