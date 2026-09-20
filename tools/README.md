# Birdy 衍生工具

本目录存放从 Birdy 主项目拆分的**可独立分发**小工具。

| 工具 | 说明 | 打包分享 |
|------|------|----------|
| [BIRDY-观鸟地图](./BIRDY-观鸟地图/) | GPX + 鸟图 → 观鸟行迹 PNG | 在工具目录运行 `sync_runtime.py` 后，压缩整个 `BIRDY-观鸟地图/` 文件夹 |
| [MTF-meter](./MTF-meter/) | ISO 12233 MTF 与同场景镜头比较 | 压缩整个 `MTF-meter/` 文件夹（`pip install -r requirements.txt` 后 `MTF.bat`） |
| [clarity-meter](./clarity-meter/) | 认种前清晰度打分（只读 HTML 报告） | 须在 Birdy 仓库内运行；`Clarity.bat` 或 `--folder` |

独立工具的运行库在各自目录的 `birdy_runtime/` 中，**不依赖**接收方安装 Birdy 主程序或配置 `src/`。
