# models/ 本地推理权重

与 `src/` 同级的模型目录。克隆仓库后请在**项目根目录**执行：

```bash
git lfs pull
```

## 已纳入仓库（Git LFS）

| 文件 | 用途 | 约大小 |
| ---- | ---- | ------ |
| `bird-seg.pt` | YOLOv8 鸟体检测与分割 | 88 MB |
| `bird_iden_res34.pth` | ResNet34 本地物种分类 | 103 MB |
| `bird_info.json` | 物种索引与名称映射（普通 Git 文件） | < 1 MB |

若文件只有几 KB，说明 LFS 未拉取完整，请重试 `git lfs pull`。

## 可选（需自备）

| 文件 | 用途 |
| ---- | ---- |
| `birdeye.pt` | 鸟眼检测（GUI 勾选「鸟眼检测」时） |
| `bird_iden_efficient_b0.pt` | EfficientNet-B0 物种分类（GUI 切换模型时） |

默认 GUI 配置使用 **ResNet34**，无需上述可选文件即可运行主流程。
