#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPX 轨迹可视化生成器
修复：支持 WGS84 转 GCJ-02 坐标系，支持卫星底图切换，正确处理多段轨迹
"""

import json
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional


# WGS84 转 GCJ-02 坐标系转换
def wgs84_to_gcj02(lat: float, lon: float) -> Tuple[float, float]:
    """
    将 WGS84 坐标转换为 GCJ-02（高德地图使用）坐标
    """
    PI = 3.14159265358979323846
    a = 6378137.0
    ee = 0.00669342162296594323
    
    def transform(x, y):
        dLat = transform_lat(x - 105.0, y - 35.0)
        dLon = transform_lon(x - 105.0, y - 35.0)
        radLat = y / 180.0 * PI
        magic = math.sin(radLat)
        magic = 1 - ee * magic * magic
        sqrtMagic = math.sqrt(magic)
        dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * PI)
        dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * PI)
        return dLat, dLon
    
    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * PI) + 40.0 * math.sin(y / 3.0 * PI)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * PI) + 320 * math.sin(y * PI / 30.0)) * 2.0 / 3.0
        return ret
    
    def transform_lon(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * PI) + 20.0 * math.sin(2.0 * x * PI)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * PI) + 40.0 * math.sin(x / 3.0 * PI)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * PI) + 300.0 * math.sin(x / 30.0 * PI)) * 2.0 / 3.0
        return ret
    
    dLat, dLon = transform(lon, lat)
    return lat + dLat, lon + dLon


def parse_gpx(gpx_file: Path) -> Tuple[List[List[Dict]], List[Dict]]:
    """解析 GPX 文件，提取轨迹段和所有轨迹点"""
    segments = []
    all_points = []
    
    import re
    with open(gpx_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配每个 trkseg
    trkseg_pattern = r'<trkseg>(.*?)</trkseg>'
    trkpt_pattern = r'<trkpt\s+lat="([^"]+)"\s+lon="([^"]+)">(.*?)</trkpt>'
    ele_pattern = r'<ele>([^<]+)</ele>'
    time_pattern = r'<time>([^<]+)</time>'
    
    for seg_match in re.finditer(trkseg_pattern, content, re.DOTALL):
        seg_content = seg_match.group(1)
        seg_points = []
        
        for match in re.finditer(trkpt_pattern, seg_content, re.DOTALL):
            lat = float(match.group(1))
            lon = float(match.group(2))
            point_content = match.group(3)
            
            gcj_lat, gcj_lon = wgs84_to_gcj02(lat, lon)
            
            point = {
                'lat': lat,
                'lon': lon,
                'gcj_lat': gcj_lat,
                'gcj_lon': gcj_lon,
                'ele': 0.0,
                'time': None
            }
            
            ele_match = re.search(ele_pattern, point_content)
            if ele_match:
                point['ele'] = float(ele_match.group(1))
            
            time_match = re.search(time_pattern, point_content)
            if time_match:
                point['time'] = time_match.group(1)
            
            seg_points.append(point)
            all_points.append(point)
        
        if seg_points:
            segments.append(seg_points)
    
    return segments, all_points


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 公式计算两点距离（米）"""
    R = 6371000
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def calculate_statistics(segments: List[List[Dict]]) -> Dict:
    """计算轨迹统计信息（正确处理多段轨迹）"""
    total_distance = 0
    total_climb = 0
    total_descent = 0
    all_points = []
    
    for seg_points in segments:
        all_points.extend(seg_points)
        
        for i in range(1, len(seg_points)):
            dist = calculate_distance(seg_points[i-1]['lat'], seg_points[i-1]['lon'],
                                     seg_points[i]['lat'], seg_points[i]['lon'])
            total_distance += dist
            
            ele_diff = seg_points[i]['ele'] - seg_points[i-1]['ele']
            if ele_diff > 0:
                total_climb += ele_diff
            else:
                total_descent += abs(ele_diff)
    
    elevations = [p['ele'] for p in all_points]
    
    return {
        'total_points': len(all_points),
        'total_distance_m': round(total_distance, 2),
        'total_distance_km': round(total_distance / 1000, 2),
        'total_climb_m': round(total_climb, 2),
        'total_descent_m': round(total_descent, 2),
        'max_elevation_m': round(max(elevations), 2),
        'min_elevation_m': round(min(elevations), 2),
        'avg_elevation_m': round(sum(elevations) / len(elevations), 2),
        'start_lat': all_points[0]['gcj_lat'] if all_points else 0,
        'start_lon': all_points[0]['gcj_lon'] if all_points else 0,
        'end_lat': all_points[-1]['gcj_lat'] if all_points else 0,
        'end_lon': all_points[-1]['gcj_lon'] if all_points else 0
    }


def generate_elevation_data(segments: List[List[Dict]]) -> Tuple[List[float], List[float]]:
    """生成海拔-距离数据（正确处理多段轨迹）"""
    distances = []
    elevations = []
    total_distance = 0.0
    
    for seg_points in segments:
        if not seg_points:
            continue
        
        # 添加第一段的第一个点
        if not distances:
            distances.append(total_distance)
            elevations.append(seg_points[0]['ele'])
        
        # 计算本段内的距离
        for i in range(1, len(seg_points)):
            dist = calculate_distance(seg_points[i-1]['lat'], seg_points[i-1]['lon'],
                                     seg_points[i]['lat'], seg_points[i]['lon'])
            total_distance += dist
            distances.append(total_distance)
            elevations.append(seg_points[i]['ele'])
        
        # 段间不连接，下一段重新开始
    
    distances_km = [d / 1000 for d in distances]
    return distances_km, elevations


def load_amap_key() -> str:
    """加载高德 API Key"""
    config_path = Path(__file__).parent / 'amap_api_config.json'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('api_key', '')
    except Exception as e:
        print(f"加载高德 API Key 失败: {e}")
        return ''


def generate_html_report(gpx_file: Path, output_file: Path, title: str = "户外步行轨迹") -> bool:
    """生成 HTML 轨迹报告"""
    try:
        segments, all_points = parse_gpx(gpx_file)
        if not segments or not all_points:
            print("未找到轨迹点")
            return False
        
        stats = calculate_statistics(segments)
        distances, elevations = generate_elevation_data(segments)
        amap_key = load_amap_key()
        
        # 为每个轨迹段创建坐标数组
        all_path_coords = []
        for seg_points in segments:
            coords = [[p['gcj_lon'], p['gcj_lat']] for p in seg_points]
            sample_interval = max(1, len(coords) // 200)
            sampled_coords = coords[::sample_interval]
            all_path_coords.append(sampled_coords)
        
        js_code = generate_javascript_code(all_path_coords, distances, elevations, stats)
        
        html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://webapi.amap.com/maps?v=2.0&key={amap_key}"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; color: white; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
        .header .subtitle {{ font-size: 1.1em; opacity: 0.9; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-icon {{ font-size: 2em; margin-bottom: 10px; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #333; margin-bottom: 5px; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
        .map-container {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .map-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .map-title {{ font-size: 1.3em; font-weight: bold; color: #333; display: flex; align-items: center; gap: 10px; }}
        .base-layer-switch {{
            display: flex;
            gap: 8px;
        }}
        .base-layer-btn {{
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }}
        .base-layer-btn:hover {{ background: #f0f0f0; }}
        .base-layer-btn.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        #map {{ width: 100%; height: 0; padding-bottom: 177.78%; border-radius: 10px; background: #f5f5f5; position: relative; }}
        #map > div {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-radius: 10px; }}
        .chart-container {{
            background: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .chart-title {{ font-size: 1.1em; font-weight: bold; margin-bottom: 10px; color: #333; display: flex; align-items: center; gap: 10px; }}
        #elevation-chart {{ width: 100%; height: 250px; }}
        .footer {{ text-align: center; color: white; margin-top: 30px; opacity: 0.8; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚶 {title}</h1>
            <div class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-icon">📏</div><div class="stat-value">{stats['total_distance_km']:.2f}</div><div class="stat-label">总距离 (公里)</div></div>
            <div class="stat-card"><div class="stat-icon">📍</div><div class="stat-value">{stats['total_points']}</div><div class="stat-label">轨迹点数</div></div>
            <div class="stat-card"><div class="stat-icon">⛰️</div><div class="stat-value">{stats['total_climb_m']:.1f}</div><div class="stat-label">累计爬升 (米)</div></div>
            <div class="stat-card"><div class="stat-icon">⬇️</div><div class="stat-value">{stats['total_descent_m']:.1f}</div><div class="stat-label">累计下降 (米)</div></div>
            <div class="stat-card"><div class="stat-icon">🏔️</div><div class="stat-value">{stats['max_elevation_m']:.1f}</div><div class="stat-label">最高海拔 (米)</div></div>
            <div class="stat-card"><div class="stat-icon">🏔️</div><div class="stat-value">{stats['min_elevation_m']:.1f}</div><div class="stat-label">最低海拔 (米)</div></div>
        </div>
        
        <div class="map-container">
            <div class="map-header">
                <div class="map-title">🗺️ 轨迹地图</div>
                <div class="base-layer-switch">
                    <button class="base-layer-btn" onclick="switchBaseLayer('road')">🗺️ 地图</button>
                    <button class="base-layer-btn active" onclick="switchBaseLayer('satellite')">🛰️ 卫星</button>
                </div>
            </div>
            <div id="map"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📈 海拔高度变化</div>
            <div id="elevation-chart"></div>
        </div>
        
        <div class="footer">
            <p>Generated by Birdy GPX Visualizer | 使用高德地图 API</p>
        </div>
    </div>
    
    {js_code}
</body>
</html>'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML 报告生成成功!")
        print(f"📄 输出文件: {output_file}")
        print(f"📊 轨迹点数: {stats['total_points']}")
        print(f"📏 总距离: {stats['total_distance_km']:.2f} km")
        print(f"⛰️ 累计爬升: {stats['total_climb_m']:.1f} m")
        
        return True
        
    except Exception as e:
        print(f"生成 HTML 报告失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_javascript_code(all_path_coords, distances, elevations, stats):
    """生成 JavaScript 代码"""
    
    js = '''
    <script>
        const allPathCoords = ''' + json.dumps(all_path_coords) + ''';
        const distances = ''' + json.dumps(distances) + ''';
        const elevations = ''' + json.dumps(elevations) + ''';
        const startLat = ''' + str(stats['start_lat']) + ''';
        const startLon = ''' + str(stats['start_lon']) + ''';
        const endLat = ''' + str(stats['end_lat']) + ''';
        const endLon = ''' + str(stats['end_lon']) + ''';
        
        var map = null;
        var polylines = [];
        var roadLayer = null;
        var satelliteLayer = null;
        var roadNetLayer = null;
        var currentLayer = 'satellite';
        
        function initMap() {
            if (typeof AMap === 'undefined') {
                document.getElementById('map').innerHTML = '<div style="text-align:center; padding-top:100px; color:#666;">高德地图 API 加载失败</div>';
                return;
            }
            
            try {
                map = new AMap.Map('map', {
                    zoom: 14,
                    center: [startLon, startLat],
                    viewMode: '2D',
                    resizeEnable: true
                });
                
                roadLayer = new AMap.TileLayer({zIndex: 1});
                satelliteLayer = new AMap.TileLayer.Satellite({zIndex: 1});
                roadNetLayer = new AMap.TileLayer.RoadNet({zIndex: 2});
                
                // 默认显示卫星地图
                map.add(satelliteLayer);
                map.add(roadNetLayer);
                
                // 为每个轨迹段创建独立的轨迹线
                var colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'];
                allPathCoords.forEach(function(coords, index) {
                    var polyline = new AMap.Polyline({
                        path: coords,
                        strokeColor: colors[index % colors.length],
                        strokeWeight: 4,
                        strokeOpacity: 0.8,
                        strokeStyle: 'solid',
                        showDir: true,
                        zIndex: 100 + index
                    });
                    polylines.push(polyline);
                    map.add(polyline);
                });
                
                var startMarker = new AMap.Marker({
                    position: [startLon, startLat],
                    title: '起点',
                    label: { content: '起点', offset: new AMap.Pixel(0, -25) },
                    zIndex: 200
                });
                
                var endMarker = new AMap.Marker({
                    position: [endLon, endLat],
                    title: '终点',
                    label: { content: '终点', offset: new AMap.Pixel(0, -25) },
                    zIndex: 200
                });
                
                map.add([startMarker, endMarker]);
                
                // 调整视野以包含所有轨迹
                var allMarkers = [startMarker, endMarker];
                map.setFitView(polylines.concat(allMarkers));
                
            } catch (e) {
                console.error('地图初始化失败:', e);
                document.getElementById('map').innerHTML = '<div style="text-align:center; padding-top:100px; color:#666;">地图初始化失败</div>';
            }
        }
        
        function switchBaseLayer(type) {
            if (currentLayer === type || !map) return;
            
            if (currentLayer === 'road') {
                map.remove(roadLayer);
            } else {
                map.remove(satelliteLayer);
                map.remove(roadNetLayer);
            }
            
            if (type === 'road') {
                map.add(roadLayer);
            } else {
                map.add(satelliteLayer);
                map.add(roadNetLayer);
            }
            
            currentLayer = type;
            
            document.querySelectorAll('.base-layer-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        function initChart() {
            var chartDom = document.getElementById('elevation-chart');
            var myChart = echarts.init(chartDom);
            
            var option = {
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        return '距离: ' + params[0].value[0].toFixed(2) + ' km<br>海拔: ' + params[0].value[1].toFixed(1) + ' m';
                    }
                },
                grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
                xAxis: {
                    type: 'value',
                    name: '距离 (km)',
                    nameLocation: 'middle',
                    nameGap: 30,
                    axisLine: { lineStyle: { color: '#666' } },
                    axisLabel: { formatter: '{value}' }
                },
                yAxis: {
                    type: 'value',
                    name: '海拔 (m)',
                    nameLocation: 'middle',
                    nameGap: 40,
                    axisLine: { lineStyle: { color: '#666' } },
                    axisLabel: { formatter: '{value}' },
                    splitLine: { lineStyle: { type: 'dashed', color: '#eee' } }
                },
                series: [{
                    name: '海拔',
                    type: 'line',
                    smooth: true,
                    symbol: 'none',
                    sampling: 'lttb',
                    itemStyle: { color: '#667eea' },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: 'rgba(102, 126, 234, 0.5)' },
                            { offset: 1, color: 'rgba(102, 126, 234, 0.1)' }
                        ])
                    },
                    data: distances.map(function(d, i) { return [d, elevations[i]]; })
                }]
            };
            
            myChart.setOption(option);
            window.addEventListener('resize', function() { myChart.resize(); });
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            initChart();
            initMap();
        });
    </script>
    '''
    
    return js


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="从 GPX 生成 HTML 轨迹报告")
    ap.add_argument("gpx", type=Path, help="GPX 文件路径")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 HTML 路径（默认与 GPX 同目录同名 .html）",
    )
    ap.add_argument("--title", default="", help="报告标题")
    args = ap.parse_args()
    gpx_file = args.gpx.expanduser().resolve()
    if not gpx_file.is_file():
        raise SystemExit(f"GPX 文件不存在: {gpx_file}")
    output_file = args.output
    if output_file is None:
        output_file = gpx_file.with_suffix(".html")
    else:
        output_file = output_file.expanduser().resolve()
    generate_html_report(gpx_file, output_file, args.title or gpx_file.stem)
    print(f"已生成: {output_file}")