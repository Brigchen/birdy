# -*- coding: utf-8 -*-
"""
地理编码配置文件
直接修改此文件即可更改配置，无需修改代码

作者: brigchen@gmail.com
版权说明: 基于开源协议，仅限爱好者、公益、科研等非盈利用途，请勿用于商业用途
"""

# ==================== 高德地图 API 配置 ====================
# 优先使用与程序同目录的 amap_api_config.json 中的 api_key（GUI「高德API」可打开该文件）。
# 若 JSON 中未填写，则使用下方 AMAP_KEY。
# 免费注册：https://console.amap.com ；Web 服务 Key；每天免费额度以控制台为准。
AMAP_KEY = ""

# ==================== 腾讯地图 API 配置 ====================
# 腾讯地图API密钥（可选，不设置则不使用）
# 免费注册：https://lbs.qq.com/
TENCENT_MAP_KEY = ""

# ==================== 百度地图 API 配置 ====================
# 百度地图API密钥（可选，不设置则不使用）
# 免费注册：http://lbsyun.baidu.com/
BAIDU_MAP_KEY = ""

# ==================== 查询优先级配置 ====================
# 设置各API的启用状态（True=启用，False=禁用）
ENABLE_AMAP = True           # 高德地图（推荐）
ENABLE_TENCENT = False       # 腾讯地图
ENABLE_BAIDU = False         # 百度地图
ENABLE_OPEN_METEO = True     # Open-Meteo（免费，无需key）
ENABLE_PHOTON = True         # Photon（基于OSM）
ENABLE_NOMINATIM = True      # Nominatim（OSM官方）

# API超时设置（秒）
API_TIMEOUT = 5

# ==================== 本地数据库配置 ====================
# 是否自动保存新查询到的地址到本地数据库
AUTO_SAVE_TO_LOCAL_DB = True

# 本地数据库文件路径（可选）
# 如果不设置，默认保存在代码同目录
# LOCAL_DB_FILE = "locations_db.json"
LOCAL_DB_FILE = None
