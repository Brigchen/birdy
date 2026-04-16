# -*- coding: utf-8 -*-
"""
地理编码模块：地名 → GPS 坐标 → EXIF 写入
支持离线 Nominatim 地理编码，无需 API 密钥

配置说明：
- 高德 Key 推荐写在 amap_api_config.json（api_key），与 GUI「高德API → 打开配置文件」一致
- 其它开关与回退 Key 在 geocoding_config.py 中设置

作者: brigchen@gmail.com
版权说明: 基于开源协议，请勿商用
"""
import json
import math
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import piexif
import concurrent.futures
import os

# 导入配置文件
try:
    from geocoding_config import (
        AMAP_KEY, TENCENT_MAP_KEY, BAIDU_MAP_KEY,
        ENABLE_AMAP, ENABLE_TENCENT, ENABLE_BAIDU,
        ENABLE_OPEN_METEO, ENABLE_PHOTON, ENABLE_NOMINATIM,
        API_TIMEOUT, AUTO_SAVE_TO_LOCAL_DB
    )
except ImportError:
    # 如果配置文件不存在，使用默认配置
    AMAP_KEY = ""
    TENCENT_MAP_KEY = ""
    BAIDU_MAP_KEY = ""
    ENABLE_AMAP = True
    ENABLE_TENCENT = False
    ENABLE_BAIDU = False
    ENABLE_OPEN_METEO = True
    ENABLE_PHOTON = True
    ENABLE_NOMINATIM = True
    API_TIMEOUT = 5
    AUTO_SAVE_TO_LOCAL_DB = True

def _effective_amap_key() -> str:
    """
    高德 Web 服务 Key：优先 amap_api_config.json（与 GUI「打开配置文件」一致），
    否则回退 geocoding_config.AMAP_KEY。
    """
    cfg_path = Path(__file__).resolve().parent / "amap_api_config.json"
    if not cfg_path.is_file():
        try:
            from api_config_defaults import ensure_amap_api_config_file

            ensure_amap_api_config_file(cfg_path.parent)
        except Exception:
            pass
    try:
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            k = (data.get("api_key") or "").strip()
            if k:
                return k
    except Exception:
        pass
    return (AMAP_KEY or "").strip()


# 常见中文地点坐标库（本地）- 避免网络超时
CHINESE_LOCATIONS = {
    "杭州西湖": (30.27, 120.16),
    "西湖": (30.27, 120.16),
    "北京故宫": (39.92, 116.40),
    "故宫": (39.92, 116.40),
    "厦门美峰体育公园": (24.474, 117.941),
    "美峰体育公园": (24.474, 117.941),
    "上海外滩": (31.24, 121.49),
    "广州塔": (23.11, 113.33),
    "深圳中心": (22.53, 114.06),
    "成都宽窄巷子": (30.57, 104.07),
    "福建平潭竹屿湖": (25.539720, 119.772810),
    "平潭竹屿湖": (25.539720, 119.772810),
    "竹屿湖": (25.539720, 119.772810),
}

# 中文地名 -> 英文翻译映射（用于 Open-Meteo API）
CHINESE_TO_ENGLISH = {
    # 具体景点 -> 城市或地点
    "杭州西湖": "Hangzhou",
    "西湖": "Hangzhou",
    "北京故宫": "Beijing",
    "故宫": "Beijing",
    "厦门美峰体育公园": "Xiamen",
    "美峰体育公园": "Xiamen",
    "上海外滩": "Shanghai",
    "广州塔": "Guangzhou",
    "深圳中心": "Shenzhen",
    "成都宽窄巷子": "Chengdu",
    "天安门": "Beijing",
    "南京路步行街": "Shanghai",
    "西安城墙": "Xi'an",
    "苏州园林": "Suzhou",
    "福建平潭竹屿湖": "Pingtan Island",
    "平潭竹屿湖": "Pingtan Island",
    "竹屿湖": "Pingtan Island",
    "平潭": "Pingtan Island",
    # 主要城市
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "成都": "Chengdu",
    "西安": "Xi'an",
    "苏州": "Suzhou",
    "武汉": "Wuhan",
    "重庆": "Chongqing",
    "南京": "Nanjing",
    "天津": "Tianjin",
    "厦门": "Xiamen",
}


def _save_to_local_db(location_name: str, latitude: float, longitude: float) -> bool:
    """
    将新查询到的地址保存到本地数据库（CHINESE_LOCATIONS字典）
    
    Args:
        location_name: 地名
        latitude: 纬度
        longitude: 经度
    
    Returns:
        True 如果保存成功或已存在，False 如果保存失败
    """
    global CHINESE_LOCATIONS
    
    # 检查是否已存在
    if location_name in CHINESE_LOCATIONS:
        return True
    
    # 添加新地址到本地库
    CHINESE_LOCATIONS[location_name] = (latitude, longitude)
    print("    [LOCAL DB] 新地址已保存: {} -> ({:.6f}, {:.6f})".format(
        location_name, latitude, longitude))
    
    # 同时添加到翻译映射（如果不存在）
    if location_name not in CHINESE_TO_ENGLISH:
        # 尝试提取城市名作为翻译
        english_name = _extract_english_name(location_name)
        if english_name:
            CHINESE_TO_ENGLISH[location_name] = english_name
    
    return True


def _extract_english_name(location_name: str) -> Optional[str]:
    """
    从中文地址中提取可能的英文名称
    这是一个简单的启发式方法
    """
    # 已知的城市英文名映射
    city_keywords = {
        "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou",
        "深圳": "Shenzhen", "杭州": "Hangzhou", "成都": "Chengdu",
        "重庆": "Chongqing", "武汉": "Wuhan", "西安": "Xi'an",
        "南京": "Nanjing", "天津": "Tianjin", "厦门": "Xiamen",
        "苏州": "Suzhou", "青岛": "Qingdao", "大连": "Dalian",
        "沈阳": "Shenyang", "哈尔滨": "Harbin", "长春": "Changchun",
        "福州": "Fuzhou", "昆明": "Kunming", "贵阳": "Guiyang",
        "长沙": "Changsha", "南昌": "Nanchang", "合肥": "Hefei",
        "郑州": "Zhengzhou", "济南": "Jinan", "石家庄": "Shijiazhuang",
        "太原": "Taiyuan", "兰州": "Lanzhou", "西宁": "Xining",
        "拉萨": "Lhasa", "乌鲁木齐": "Urumqi", "呼和浩特": "Hohhot",
        "银川": "Yinchuan", "乌鲁木齐": "Urumqi", "平潭": "Pingtan Island",
    }
    
    # 查找城市名
    for cn_name, en_name in city_keywords.items():
        if cn_name in location_name:
            return en_name
    
    return None


def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """
    将地名转换为 GPS 坐标 (纬度, 经度)
    
    支持以下输入格式：
    1. 中文地名（优先使用本地库，无网络）
    2. 地名字符串（首选高德API，免费额度50000次/天）
    3. 坐标字符串，如 "30.27,120.16"
    
    查询顺序：本地库 > 直接坐标解析 > 高德API > 其他免费API（备选）
    
    自动保存：新查询到的地址会自动保存到本地数据库（CHINESE_LOCATIONS）
    
    配置文件：所有配置在 geocoding_config.py 中设置
    
    Args:
        location_name: 地名字符串或坐标字符串
    
    Returns:
        (latitude, longitude) 或 None（如果地理编码失败）
    """
    # 移除可能的多余引号
    location_name = location_name.strip().strip('"').strip("'")
    
    # [优先] 1. 检查本地中文地点库
    if location_name in CHINESE_LOCATIONS:
        lat, lon = CHINESE_LOCATIONS[location_name]
        print("[LOCAL] Found in database: {} -> ({:.6f}, {:.6f})".format(location_name, lat, lon))
        return (lat, lon)
    
    # [次选] 2. 尝试解析为直接坐标
    if ',' in location_name:
        try:
            parts = location_name.split(',')
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                print("[DIRECT] Parse coordinates: {} -> ({:.6f}, {:.6f})".format(location_name, lat, lon))
                return (lat, lon)
        except (ValueError, IndexError):
            pass
    
    # [最后] 3. 尝试通过多个在线地理编码 API
    print("[ONLINE] Trying geocoding for: '{}'...".format(location_name))
    
    # 尝试顺序：高德 > 腾讯 > 百度 > 自定义免费 API > Nominatim
    result = _geocode_with_amap(location_name)
    if result:
        return result
    
    result = _geocode_with_tencent(location_name)
    if result:
        return result
    
    result = _geocode_with_baidu(location_name)
    if result:
        return result
    
    result = _geocode_with_free_api(location_name)
    if result:
        return result
    
    result = _geocode_with_nominatim(location_name)
    if result:
        return result
    
    print("[FAIL] Could not geocode: '{}'".format(location_name))
    print("[HINT] Try using coordinate format: --location '30.27,120.16'")
    return None


def _geocode_with_amap(location_name: str) -> Optional[Tuple[float, float]]:
    """
    使用高德地图 API（高德开放平台）
    API密钥和配置在 geocoding_config.py 中设置
    免费额度：50000次/天
    """
    # 检查是否启用
    if not ENABLE_AMAP:
        return None
        
    try:
        import requests
    except ImportError:
        return None
    
    try:
        amap_key = _effective_amap_key()
        if not amap_key:
            print(
                "  [高德] API key 未设置，请编辑 src/amap_api_config.json（推荐）"
                " 或在 geocoding_config.py 中配置 AMAP_KEY"
            )
            return None
        
        print("  [高德] Querying Amap...")
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {
            "address": location_name,
            "output": "json",
            "key": amap_key
        }
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        data = response.json()
        
        if data.get("status") == "1" and data.get("geocodes"):
            location = data["geocodes"][0]
            lon, lat = map(float, location["location"].split(","))
            print("    [高德] Success: ({:.6f}, {:.6f})".format(lat, lon))
            
            # 自动保存到本地数据库
            if AUTO_SAVE_TO_LOCAL_DB:
                _save_to_local_db(location_name, lat, lon)
            
            return (lat, lon)
        else:
            print("    [高德] 失败: {}".format(data.get("info", "未知错误")))
    except Exception as e:
        pass
    
    return None


def _geocode_with_tencent(location_name: str) -> Optional[Tuple[float, float]]:
    """
    使用腾讯地图 API
    API密钥和配置在 geocoding_config.py 中设置
    """
    # 检查是否启用
    if not ENABLE_TENCENT:
        return None
        
    try:
        import requests
    except ImportError:
        return None
    
    try:
        if not TENCENT_MAP_KEY:
            return None
        
        print("  [腾讯] Querying Tencent Map...")
        url = "https://apis.map.qq.com/ws/geocoder/v1"
        params = {
            "address": location_name,
            "key": TENCENT_MAP_KEY,
            "output": "json"
        }
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        data = response.json()
        
        if data.get("status") == 0 and data.get("result"):
            loc = data["result"]["location"]
            lat, lon = loc["lat"], loc["lng"]
            print("    [腾讯] Success: ({:.6f}, {:.6f})".format(lat, lon))
            
            # 自动保存到本地数据库
            if AUTO_SAVE_TO_LOCAL_DB:
                _save_to_local_db(location_name, lat, lon)
            
            return (lat, lon)
    except Exception as e:
        pass
    
    return None


def _geocode_with_baidu(location_name: str) -> Optional[Tuple[float, float]]:
    """
    使用百度地图 API
    API密钥和配置在 geocoding_config.py 中设置
    """
    # 检查是否启用
    if not ENABLE_BAIDU:
        return None
        
    try:
        import requests
    except ImportError:
        return None
    
    try:
        if not BAIDU_MAP_KEY:
            return None
        
        print("  [百度] Querying Baidu Map...")
        url = "https://api.map.baidu.com/geocoding/v3"
        params = {
            "address": location_name,
            "output": "json",
            "ak": BAIDU_MAP_KEY
        }
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        data = response.json()
        
        if data.get("status") == 0 and data.get("result"):
            loc = data["result"]["location"]
            lat, lon = loc["lat"], loc["lng"]
            print("    [百度] Success: ({:.6f}, {:.6f})".format(lat, lon))
            
            # 自动保存到本地数据库
            if AUTO_SAVE_TO_LOCAL_DB:
                _save_to_local_db(location_name, lat, lon)
            
            return (lat, lon)
    except Exception as e:
        pass
    
    return None


def _geocode_with_free_api(location_name: str) -> Optional[Tuple[float, float]]:
    """
    使用完全免费的第三方地理编码 API
    配置在 geocoding_config.py 中设置
    
    尝试以下源：
    1. Open-Meteo - 完全免费，无需 key
    2. Photon - 基于 OSM 的快速服务
    """
    # 如果所有免费API都禁用，直接返回
    if not ENABLE_OPEN_METEO and not ENABLE_PHOTON:
        return None
        
    try:
        import requests
    except ImportError:
        return None
    
    # 方案 1: Open-Meteo
    if ENABLE_OPEN_METEO:
        try:
            print("  [Open-Meteo] Querying free geocoding service...")
            
            # 获取英文翻译（如果有）
            search_name = CHINESE_TO_ENGLISH.get(location_name, location_name)
            
            url = "https://geocoding-api.open-meteo.com/v1/search"
            params = {
                "name": search_name,
                "count": 1,
                "format": "json",
                "language": "en"
            }
            
            response = requests.get(url, params=params, timeout=API_TIMEOUT)
            data = response.json()
            
            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                lat = result.get("latitude")
                lon = result.get("longitude")
                if lat and lon:
                    print("    [Open-Meteo] Success: ({:.6f}, {:.6f})".format(lat, lon))
                    # 自动保存到本地数据库
                    if AUTO_SAVE_TO_LOCAL_DB:
                        _save_to_local_db(location_name, lat, lon)
                    return (lat, lon)
            
            # 如果指定翻译没有工作，再试一次原始字符串
            if search_name != location_name:
                print("    [Open-Meteo] Retrying with original name...")
                params["name"] = location_name
                response = requests.get(url, params=params, timeout=API_TIMEOUT)
                data = response.json()
                
                if data.get("results") and len(data["results"]) > 0:
                    result = data["results"][0]
                    lat = result.get("latitude")
                    lon = result.get("longitude")
                    if lat and lon:
                        print("    [Open-Meteo] Success: ({:.6f}, {:.6f})".format(lat, lon))
                        # 自动保存到本地数据库
                        if AUTO_SAVE_TO_LOCAL_DB:
                            _save_to_local_db(location_name, lat, lon)
                        return (lat, lon)
                    
        except Exception as e:
            pass
    
    # 方案 2: Photon - 基于 OSM 的快速地理编码
    if ENABLE_PHOTON:
        try:
            print("  [Photon] Querying free OSM-based geocoder...")
            url = "https://photon.komoot.io/api"
            params = {
                "q": location_name,
                "limit": 1,
                "lang": "en"
            }
            response = requests.get(url, params=params, timeout=API_TIMEOUT)
            data = response.json()
            
            if data.get("features") and len(data["features"]) > 0:
                coords = data["features"][0]["geometry"]["coordinates"]
                lon, lat = coords[0], coords[1]
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    print("    [Photon] Success: ({:.6f}, {:.6f})".format(lat, lon))
                    # 自动保存到本地数据库
                    if AUTO_SAVE_TO_LOCAL_DB:
                        _save_to_local_db(location_name, lat, lon)
                    return (lat, lon)
        except Exception as e:
            pass
    
    return None


def _geocode_with_nominatim(location_name: str) -> Optional[Tuple[float, float]]:
    """
    降级方案：使用 OpenStreetMap Nominatim
    配置在 geocoding_config.py 中设置
    """
    # 检查是否启用
    if not ENABLE_NOMINATIM:
        return None
        
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        return None
    
    try:
        print("  [Nominatim] Querying (slowest, fallback)...")
        geolocator = Nominatim(user_agent="birdy_detector_v2", timeout=10)
        location = geolocator.geocode(location_name, country_codes="cn", timeout=10)
        
        if location:
            lat, lon = location.latitude, location.longitude
            print("    [Nominatim] Success: ({:.6f}, {:.6f})".format(lat, lon))
            # 自动保存到本地数据库
            if AUTO_SAVE_TO_LOCAL_DB:
                _save_to_local_db(location_name, lat, lon)
            return (lat, lon)
    except Exception as e:
        pass
    
    return None


def decimal_to_dms(decimal: float) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    将十进制坐标转换为 度/分/秒 格式（EXIF GPS 所需）
    
    Args:
        decimal: 十进制度数
    
    Returns:
        ((度, 1), (分, 1), (秒*100, 100)) - EXIF GPS 格式
    """
    is_negative = decimal < 0
    decimal = abs(decimal)
    
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = ((decimal - degrees) * 60 - minutes) * 60
    
    # EXIF GPS 格式：(分子, 分母)
    return (
        (degrees, 1),
        (minutes, 1),
        (int(seconds * 100), 100)
    )


def write_gps_exif(image_path: str, latitude: float, longitude: float, altitude: Optional[float] = None, verbose: bool = False) -> bool:
    """
    将 GPS 坐标写入图片的 EXIF 元数据
    
    Args:
        image_path:  图片路径
        latitude:    纬度 (-90 ~ 90)
        longitude:   经度 (-180 ~ 180)
        altitude:    海拔（可选，单位米）
        verbose:     是否打印详细信息
    
    Returns:
        True 成功，False 失败
    """
    try:
        from PIL import Image
    except ImportError:
        if verbose:
            print("错误: 未安装 Pillow，请运行: pip install Pillow")
        return False

    try:
        image_path = Path(image_path)
        
        # 确定 GPS 方向
        gps_latitude_ref = "N" if latitude >= 0 else "S"
        gps_longitude_ref = "E" if longitude >= 0 else "W"
        
        # 转换为 DMS 格式
        lat_dms = decimal_to_dms(latitude)
        lon_dms = decimal_to_dms(longitude)
        
        # 构建 GPS IFD (Image File Directory)
        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: gps_latitude_ref.encode(),
            piexif.GPSIFD.GPSLatitude: lat_dms,
            piexif.GPSIFD.GPSLongitudeRef: gps_longitude_ref.encode(),
            piexif.GPSIFD.GPSLongitude: lon_dms,
        }
        
        # 如果有海拔信息，添加海拔数据
        if altitude is not None:
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0  # 0 = 海平面上方
            gps_ifd[piexif.GPSIFD.GPSAltitude] = (int(altitude), 1)
        
        # 读取现有 EXIF（如果有）
        try:
            exif_dict = piexif.load(str(image_path))
        except:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
        
        # 更新 GPS 数据
        exif_dict["GPS"] = gps_ifd
        
        # 转换为字节并写回
        exif_bytes = piexif.dump(exif_dict)
        
        # 直接保存到原文件（移除备份操作以提高速度）
        img = Image.open(image_path)
        img.save(str(image_path), "jpeg", exif=exif_bytes, quality=95, optimize=True)
        
        if verbose:
            print("[OK] EXIF write success: {}".format(image_path.name))
            print("     GPS: ({:.6f}, {:.6f})".format(latitude, longitude))
        return True
        
    except Exception as e:
        if verbose:
            print("[ERROR] EXIF write failed: {} - {}".format(image_path, str(e)))
        return False


def batch_write_gps_exif(image_folder: str, latitude: float, longitude: float, altitude: Optional[float] = None, max_workers: int = None) -> int:
    """
    批量将 GPS 坐标写入文件夹中所有图片的 EXIF
    
    Args:
        image_folder: 图片文件夹路径
        latitude:     纬度
        longitude:    经度
        altitude:     海拔（可选）
        max_workers:  最大线程数，None 表示自动根据系统核心数调整
    
    Returns:
        成功写入的图片数量
    """
    folder = Path(image_folder)
    if not folder.exists():
        print(f"错误: 文件夹不存在: {image_folder}")
        return 0
    
    # 支持的图片格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    
    image_files = [f for f in folder.iterdir() 
                   if f.suffix.lower() in image_extensions and f.is_file()]
    
    if not image_files:
        print(f"警告: 文件夹中没有发现图片文件: {image_folder}")
        return 0
    
    print(f"开始批量写入 GPS EXIF...")
    print(f"文件夹: {image_folder}")
    print(f"坐标: ({latitude:.6f}, {longitude:.6f})")
    print(f"图片数: {len(image_files)}")
    print(f"使用并行处理，最大线程数: {max_workers or '自动'}")
    print("=" * 60)
    
    success_count = 0
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(write_gps_exif, str(img_file), latitude, longitude, altitude, verbose=False): img_file
            for img_file in sorted(image_files)
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_file):
            img_file = future_to_file[future]
            try:
                if future.result():
                    success_count += 1
                    print(f"[OK] {img_file.name}")
            except Exception as e:
                print(f"[ERROR] {img_file.name}: {e}")
    
    print("=" * 60)
    print(f"完成: {success_count}/{len(image_files)} 张图片 GPS EXIF 写入成功")
    
    return success_count


def _gps_ref_to_str(ref) -> str:
    if ref is None:
        return "N"
    if isinstance(ref, bytes):
        return ref.decode(errors="ignore")[:1].upper() or "N"
    s = str(ref).strip()
    return (s[:1].upper() if s else "N")


def wgs84_to_gcj02(lat: float, lon: float) -> Tuple[float, float]:
    """
    WGS84 → GCJ-02（国测局火星坐标）。
    相机 EXIF 多为 WGS84；若离线省界数据为火星坐标，需转换后再做点内判断。
    境外或明显非中国范围则原样返回。
    """
    if lon < 72.004 or lon > 137.8347 or lat < 0.8293 or lat > 55.8271:
        return lat, lon

    a = 6378245.0
    ee = 0.00669342162296594323

    def _transform_lat(x: float, y: float) -> float:
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (
            160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)
        ) * 2.0 / 3.0
        return ret

    def _transform_lon(x: float, y: float) -> float:
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (
            150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)
        ) * 2.0 / 3.0
        return ret

    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    return lat + dlat, lon + dlon


def read_gps_exif(
    image_path: str, *, quiet: bool = False
) -> Optional[Tuple[float, float, Optional[float]]]:
    """
    从图片 EXIF 中读取 GPS 坐标（piexif 直读文件，与多数相机 JPEG 兼容）。

    Args:
        image_path: 图片路径
        quiet: 为 True 时不打印解析异常（供上层多策略回退时调用）

    Returns:
        (latitude, longitude, altitude) 或 None
    """
    try:
        exif_dict = piexif.load(str(image_path))
        gps_ifd = exif_dict.get("GPS", {})

        if not gps_ifd:
            return None

        lat_data = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
        lon_data = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
        if not lat_data or not lon_data:
            return None

        lat_ref = _gps_ref_to_str(
            gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b"N")
        )
        lon_ref = _gps_ref_to_str(
            gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b"E")
        )

        def dms_to_decimal(dms_tuple, direction: str) -> float:
            def comp(i: int) -> float:
                if i >= len(dms_tuple):
                    return 0.0
                t = dms_tuple[i]
                if isinstance(t, (tuple, list)) and len(t) >= 2:
                    return float(t[0]) / float(t[1]) if t[1] else 0.0
                if hasattr(t, "numerator") and hasattr(t, "denominator"):
                    return (
                        float(t.numerator) / float(t.denominator)
                        if t.denominator
                        else 0.0
                    )
                return float(t)

            d, m, s = comp(0), comp(1), comp(2)
            decimal = d + m / 60.0 + s / 3600.0
            if direction in ("S", "W"):
                decimal = -decimal
            return decimal

        latitude = dms_to_decimal(lat_data, lat_ref)
        longitude = dms_to_decimal(lon_data, lon_ref)

        altitude = None
        alt_data = gps_ifd.get(piexif.GPSIFD.GPSAltitude)
        if alt_data:
            if isinstance(alt_data, (tuple, list)) and len(alt_data) > 0:
                t0 = alt_data[0]
                if isinstance(t0, (tuple, list)) and len(t0) >= 2:
                    altitude = float(t0[0]) / float(t0[1]) if t0[1] else None
                elif hasattr(t0, "numerator"):
                    altitude = (
                        float(t0.numerator) / float(t0.denominator)
                        if t0.denominator
                        else None
                    )

        return (latitude, longitude, altitude)

    except Exception as e:
        if not quiet:
            print(f"错误: 读取 GPS EXIF 失败: {e}")
        return None


if __name__ == "__main__":
    # 简单的测试脚本
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python geo_encoder.py <地名> [图片文件夹]")
        print("示例: python geo_encoder.py '杭州西湖' ./test")
        sys.exit(1)
    
    location_name = sys.argv[1]
    image_folder = sys.argv[2] if len(sys.argv) > 2 else "."
    
    # 地理编码
    coords = geocode_location(location_name)
    if not coords:
        sys.exit(1)
    
    # 批量写入 EXIF
    batch_write_gps_exif(image_folder, coords[0], coords[1])
