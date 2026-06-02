# birdy_runtime

本目录为 **BIRDY-观鸟地图** 独立运行库，**纳入 Git 版本管理**。

克隆仓库后可直接运行 `start.bat`，无需再执行 `sync_runtime.py`。

## 内容

由 `../sync_runtime.py` 从 Birdy 主项目 `src/` 同步：

- `gpx_track/` — 轨迹图生成（工具固定使用经纬度网格底图，无高德）
- `record_submit/exif_read.py` — 照片 EXIF 读取

## 何时重新 sync

修改主项目 `src/gpx_track/` 或 `record_submit/exif_read.py` 后，在工具目录执行：

```bat
python sync_runtime.py
```

然后将本目录变更一并提交，以便分发包与主程序保持一致。
