# -*- coding: utf-8 -*-
"""
连拍图片预分组配置文件
直接修改此文件即可更改参数设置

作者: brigchen@gmail.com
版权说明: 基于开源协议，请勿商用
"""

# ==================== 连拍分组设置 ====================

# 连拍时间阈值（秒）
# 时间间隔小于此值视为连拍
# 默认: 1秒内拍摄的照片为一组连拍
BURST_TIME_THRESHOLD = 1.0

# 保留数量（与 GUI/代码中 burst_keep_min 对应；旧名 KEEP_TOP_N 仍作最小保留默认值）
KEEP_TOP_N = 2

# 连拍组较大时按组内张数×比例保留；按比例不足最小张数时至少保留 burst_keep_min 张
BURST_KEEP_RATIO = 0.2
BURST_KEEP_MIN = 2

# 最小鸟体面积阈值
# 只有大于此面积的鸟体才被视为有效检测
# 用于判断是否为有效的鸟照
# 默认: 1000像素
MIN_BIRD_AREA = 1000

# ==================== 对焦评估设置 ====================

# 是否启用对焦排序
# 如果启用，会按对焦质量排序并保留最佳图片
# 如果禁用，所有图片都会保留
ENABLE_FOCUS_SORT = True

# 对焦评分权重
# 综合排序分 = FOCUS_WEIGHT * focus_score + BIRD_AREA_WEIGHT * (鸟体面积/10000)
# 增加此值可更重视对焦质量
FOCUS_WEIGHT = 1.0

# 鸟体面积项权重（降低此值可减少“大鸟体”对组内排序的主导）
BIRD_AREA_WEIGHT = 0.45

# 对焦指标模式（连拍筛选用）
# - laplacian：仅在 ROI 内计算 Laplacian 方差（与早期版本一致）
# - hybrid：Laplacian 方差 + Tenengrad（Sobel 能量）+ 中心加权清晰度 − 背景边缘环惩罚（减轻树枝/背景清晰抢分）
# - mask_hybrid：在 YOLO 分割掩膜内计分（需 seg 模型有 masks）；无掩膜时自动回退 hybrid
FOCUS_METRIC_MODE = "mask_hybrid"

# 鸟体框外扩比例（用于对焦 ROI；越小越贴鸟体，减少树枝背景）
FOCUS_ROI_MARGIN_FRAC = 0.04

# hybrid / mask_hybrid 内部权重（经验默认，可按题材微调）
FOCUS_HYBRID_W_LAP = 1.0
FOCUS_HYBRID_W_TEN = 0.55
FOCUS_HYBRID_W_CENTER = 0.35  # 中心加权 Laplacian 能量（抑制边缘树枝抢焦）
FOCUS_HYBRID_RING_PX = 10  # 近似环带宽度（像素，随 ROI 尺度裁剪）
FOCUS_HYBRID_BG_PENALTY = 1.15  # 环带/鸟体 Sobel 能量比超阈值后的惩罚强度

# 鸟体检测置信度阈值
# 只保留置信度大于此值的鸟体检测结果
BIRD_CONF_THRESHOLD = 0.3

# 是否启用鸟眼检测（仅在启用鸟体检测时生效）
ENABLE_EYE_DETECTION = False

# 鸟眼检测置信度阈值
EYE_CONF_THRESHOLD = 0.25

# 鸟眼检测时对鸟体框外扩比例（每只鸟单独 ROI 检测，提升小目标召回）
EYE_ROI_MARGIN_FRAC = 0.12

# 鸟眼中心点落入鸟体判定容差比例（相对鸟框宽高）
EYE_INSIDE_TOL_FRAC = 0.06

# 每只鸟最多保留的鸟眼数（按置信度排序）
EYE_MAX_PER_BIRD = 2

# 鸟眼加权（在原有排序分基础上增加）
# 综合排序分 = FOCUS_WEIGHT * focus_score + BIRD_AREA_WEIGHT * (鸟体面积/10000) + (有鸟眼? EYE_BONUS_WEIGHT:0)
EYE_BONUS_WEIGHT = 0.8

# ==================== 输出设置 ====================

# 是否输出详细日志
VERBOSE = True

# 是否生成JSON报告
GENERATE_REPORT = True
