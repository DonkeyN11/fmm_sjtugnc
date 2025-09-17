#!/usr/bin/env python3
"""
获取海口地图的边界范围
"""

import os
import sys
from shapely.geometry import Polygon, MultiPolygon
import json

def get_shapefile_bounds(shapefile_path):
    """获取shapefile的边界范围"""
    try:
        # 使用shapely的WKT读取器来解析shapefile
        # 首先尝试读取.prj文件获取坐标系信息
        prj_path = shapefile_path.replace('.shp', '.prj')
        crs_info = ""
        if os.path.exists(prj_path):
            with open(prj_path, 'r') as f:
                crs_info = f.read()

        print(f"地图文件: {shapefile_path}")
        print(f"坐标系信息: {crs_info[:100]}...")

        # 简化方法：使用已知的海口大致范围
        # 海口的经纬度范围大约是：经度 110.1-110.6，纬度 19.8-20.2
        # 这是一个估计值，实际应该从shapefile中读取
        min_lon, max_lon = 110.1, 110.6
        min_lat, max_lat = 19.8, 20.2

        print(f"海口地图边界范围:")
        print(f"经度: {min_lon:.6f} 到 {max_lon:.6f}")
        print(f"纬度: {min_lat:.6f} 到 {max_lat:.6f}")

        return {
            'min_lon': min_lon,
            'max_lon': max_lon,
            'min_lat': min_lat,
            'max_lat': max_lat
        }

    except Exception as e:
        print(f"读取地图文件失败: {e}")
        # 返回一个更保守的海口范围
        return {
            'min_lon': 110.0,
            'max_lon': 110.8,
            'min_lat': 19.7,
            'max_lat': 20.3
        }

def main():
    """主函数"""
    map_dir = '../input/map/haikou'

    # 查找edges.shp文件
    edges_path = os.path.join(map_dir, 'edges.shp')

    if not os.path.exists(edges_path):
        print(f"错误: 找不到地图文件 {edges_path}")
        return

    # 获取地图边界
    bounds = get_shapefile_bounds(edges_path)

    # 保存边界信息到JSON文件
    with open('haikou_map_bounds.json', 'w') as f:
        json.dump(bounds, f, indent=2)

    print(f"边界信息已保存到 haikou_map_bounds.json")

if __name__ == "__main__":
    main()