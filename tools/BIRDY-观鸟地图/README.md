# BIRDY-观鸟地图

独立 GUI：根据 **GPX 轨迹**与**鸟图目录**生成观鸟行迹 PNG（2K 竖屏，内嵌海拔剖面、标题与 Logo）。

**本目录即为完整分发包**——压缩整个文件夹即可分享，无需 Birdy 主程序、无需高德 Key。

---

## 目录结构

```
BIRDY-观鸟地图/
├── start.bat / start.sh
├── sync_runtime.py           # 维护者：从主项目同步 birdy_runtime/
├── requirements.txt
├── config.json               # 界面参数（本地，勿提交）
├── output/                   # 默认 PNG 输出
├── birdy_runtime/            # 绘图运行库（已纳入 Git）
└── birdy_track_map/          # 本工具界面
```

---

## 快速开始

```bat
python -m pip install -r requirements.txt
start.bat
```

1. 选择鸟图目录、GPX（可多段）
2. **地图标题（可选）**：填写地点名称 → PNG 图内为「日期 / 地点 / 观鸟地图 / 签名 Logo」；留空地点时仍显示「观鸟地图」
3. 预览或生成并保存

---

## 地图样式

- 使用 **经纬度网格底图**（轨迹 + 鸟种标注），不依赖高德在线底图
- 图内标题、透明底白/深色签名 Logo、物种名左右侧标与主程序 **2.0.7** 观鸟地图版式一致（独立工具无底图来源行）
- 无需配置 API Key，离线可生成（仅 EXIF/GPX 处理需本地数据）

---

## 维护者

修改主项目 `src/gpx_track/` 后：

```bat
python sync_runtime.py
```

提交 `birdy_runtime/` 变更后再打包分享。

---

## 许可

与 Birdy 主项目相同：仅限爱好者、公益、科研等非盈利用途。
