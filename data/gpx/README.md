# GPX 轨迹工具

本目录提供 GPX 合并与地图展示的示例脚本；Birdy GUI 与主程序使用 `src/gpx_track/` 模块。

## 合并多个 GPX

```bash
# 在仓库根目录
python data/gpx/merge_gpx.py -o data/gpx/merged.gpx track1.gpx track2.gpx
```

## 在地图上查看轨迹（HTML）

需安装可选依赖：`pip install folium`

```bash
python data/gpx/view_track_map.py data/gpx/merged.gpx -o data/gpx/track_map.html
```

浏览器打开生成的 HTML 即可查看轨迹。

## 与 Birdy 集成

- **地理位置** 卡片：选择 GPX →「按 GPX 时间写入照片 GPS」
- **轨迹图生成** 卡片：导入 GPX 或读取照片 EXIF，生成行迹/物种分布 PNG 至 `reports/`
