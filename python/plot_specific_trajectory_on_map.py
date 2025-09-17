#!/usr/bin/env python3
"""
在Haikou地图上绘制特定轨迹的脚本
使用shapefile地图作为背景，绘制指定的LINESTRING轨迹
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shapely.wkt import loads
from shapely.geometry import LineString
import geopandas as gpd
import os
import json


def parse_linestring(wkt_str):
    """解析WKT格式的LineString为坐标点数组"""
    try:
        line = loads(wkt_str)
        if isinstance(line, LineString):
            return list(line.coords)
        return None
    except Exception as e:
        print(f"解析错误: {e}")
        return None


def load_haikou_map(shapefile_path):
    """加载Haikou地图shapefile"""
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"成功加载地图数据: {len(gdf)} 条道路")
        return gdf
    except Exception as e:
        print(f"加载地图失败: {e}")
        return None


def create_trajectory_on_map(shapefile_path, trajectory_wkt, output_html_path=None):
    """创建特定轨迹在地图上的可视化"""
    print(f"开始创建轨迹可视化...")

    # 加载地图
    map_gdf = load_haikou_map(shapefile_path)
    if map_gdf is None:
        return False

    # 解析轨迹
    trajectory_coords = parse_linestring(trajectory_wkt)
    if not trajectory_coords:
        print("轨迹解析失败")
        return False

    print(f"成功解析轨迹，包含 {len(trajectory_coords)} 个坐标点")

    # 分离经纬度
    lons = [coord[0] for coord in trajectory_coords]
    lats = [coord[1] for coord in trajectory_coords]

    # 计算轨迹范围
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    print(f"轨迹范围:")
    print(f"  经度: {min_lon:.6f} 到 {max_lon:.6f}")
    print(f"  纬度: {min_lat:.6f} 到 {max_lat:.6f}")

    # 创建图形
    fig = go.Figure()

    # 添加地图道路
    print("正在添加地图道路...")
    road_count = 0
    for idx, road in map_gdf.iterrows():
        if road.geometry.geom_type == 'LineString':
            road_coords = list(road.geometry.coords)
            road_lons = [coord[0] for coord in road_coords]
            road_lats = [coord[1] for coord in road_coords]

            fig.add_trace(go.Scatter(
                x=road_lons,
                y=road_lats,
                mode='lines',
                line=dict(
                    color='gray',
                    width=1
                ),
                name='Road Network',
                hoverinfo='none',
                showlegend=False,
                opacity=0.6
            ))
            road_count += 1

            if road_count % 1000 == 0:
                print(f"已处理 {road_count} 条道路...")

    print(f"总共处理了 {road_count} 条道路")

    # 添加轨迹
    print("正在添加轨迹...")
    fig.add_trace(go.Scatter(
        x=lons,
        y=lats,
        mode='lines',
        line=dict(
            color='red',
            width=4
        ),
        name='Specific Trajectory',
        hoverinfo='text',
        showlegend=True,
        opacity=0.9
    ))

    # 添加轨迹起点和终点
    fig.add_trace(go.Scatter(
        x=[lons[0]],
        y=[lats[0]],
        mode='markers',
        marker=dict(
            color='green',
            size=8,
            symbol='circle'
        ),
        name='Start Point',
        hovertext=f'起点: ({lons[0]:.6f}, {lats[0]:.6f})',
        hoverinfo='text',
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        x=[lons[-1]],
        y=[lats[-1]],
        mode='markers',
        marker=dict(
            color='blue',
            size=8,
            symbol='circle'
        ),
        name='End Point',
        hovertext=f'终点: ({lons[-1]:.6f}, {lats[-1]:.6f})',
        hoverinfo='text',
        showlegend=True
    ))

    # 计算合适的显示范围（添加边距）
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    lon_margin = max(lon_range * 0.1, 0.01)
    lat_margin = max(lat_range * 0.1, 0.01)

    # 设置图形布局
    fig.update_layout(
        title={
            'text': 'Specific Trajectory on Haikou Road Network',
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20, family='Arial')
        },
        xaxis=dict(
            title='Longitude (°)',
            range=[min_lon - lon_margin, max_lon + lon_margin],
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='Latitude (°)',
            range=[min_lat - lat_margin, max_lat + lat_margin],
            gridcolor='lightgray',
            showgrid=True
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        width=1200,
        height=800,
        showlegend=True
    )

    # 创建输出目录
    os.makedirs('output/specific', exist_ok=True)

    # 保存为HTML文件
    if output_html_path:
        fig.write_html(output_html_path)
        print(f"交互式可视化已保存到: {output_html_path}")
    else:
        # 默认保存路径
        default_path = 'output/specific/specific_trajectory_on_map.html'
        fig.write_html(default_path)
        print(f"交互式可视化已保存到: {default_path}")

    # 在浏览器中显示
    fig.show()

    return True


def main():
    """主函数"""
    # 设置地图文件路径
    shapefile_path = 'input/map/haikou/edges.shp'

    # 检查地图文件是否存在
    if not os.path.exists(shapefile_path):
        print(f"错误: 地图文件 {shapefile_path} 不存在")
        print("请确保haikou/edges.shp文件存在")
        return

    # 设置轨迹WKT
    trajectory_wkt = "LINESTRING (110.332213 19.980554, 110.332213 19.980554, 110.33223 19.980546, 110.33223 19.980546, 110.332239 19.980536, 110.332239 19.980536, 110.332236 19.980535, 110.332236 19.980535, 110.332236 19.980535, 110.332236 19.980535, 110.332235 19.980114, 110.332235 19.980114, 110.332235 19.980114, 110.332257 19.978453, 110.332257 19.978453, 110.332242 19.978027, 110.332242 19.978027, 110.332257 19.977992, 110.332257 19.977992, 110.332258 19.977806, 110.332248 19.97692, 110.332238 19.975616, 110.334929 19.974915, 110.33754 19.975601, 110.3392 19.975349, 110.3392 19.975349, 110.3392 19.975349, 110.339986 19.971713, 110.340077 19.968573, 110.340077 19.968573, 110.340077 19.968573, 110.340079 19.964029, 110.340079 19.964029, 110.340306 19.957381, 110.341362 19.953158, 110.342121 19.950345, 110.342121 19.950345, 110.342768 19.945971, 110.343302 19.94181, 110.343608 19.939438, 110.344046 19.936129, 110.344335 19.933809, 110.344508 19.93111, 110.344508 19.93111, 110.342494 19.925791, 110.341032 19.923026, 110.339658 19.920459, 110.339658 19.920459, 110.339658 19.920459, 110.338433 19.918356, 110.338433 19.918356, 110.337607 19.91657, 110.337607 19.91657, 110.336778 19.911616, 110.336778 19.911616, 110.335311 19.908852, 110.335311 19.908852, 110.335311 19.908852, 110.331242 19.90521, 110.331242 19.90521, 110.331227 19.903875, 110.331227 19.903875, 110.332877 19.900928, 110.334371 19.899608, 110.334981 19.898732, 110.335018 19.897284, 110.333842 19.896648, 110.333842 19.896648, 110.331951 19.895024, 110.331951 19.895024, 110.331951 19.895024, 110.330899 19.893087, 110.328113 19.891862, 110.328113 19.891862, 110.326602 19.889418, 110.326602 19.889418, 110.326602 19.889418, 110.322893 19.887582, 110.322004 19.886931, 110.319692 19.887234, 110.317292 19.886971, 110.313845 19.885123, 110.311381 19.88458, 110.311381 19.88458, 110.311048 19.883005, 110.310754 19.883124, 110.310754 19.883124, 110.310708 19.883112, 110.310708 19.883112, 110.311046 19.882979, 110.311046 19.882979, 110.311499 19.883226, 110.311489 19.883234, 110.311511 19.88324, 110.311511 19.88324, 110.311479 19.883275, 110.311479 19.883275, 110.311479 19.883275, 110.311479 19.883275, 110.311499 19.883252, 110.311499 19.883252, 110.31144 19.883301, 110.31144 19.883301, 110.31144 19.883301, 110.31144 19.883301, 110.31141 19.883328, 110.311175 19.883055, 110.310921 19.883002, 110.310065 19.883348, 110.30846 19.884384, 110.30689 19.884495, 110.305465 19.885369, 110.303884 19.885724, 110.302101 19.885643, 110.299666 19.885555, 110.297743 19.885642, 110.296498 19.885651, 110.291774 19.885208, 110.290013 19.885732, 110.288205 19.886107, 110.287172 19.886523, 110.284502 19.889047, 110.28282 19.889612, 110.280669 19.890083, 110.280669 19.890083, 110.280669 19.890083, 110.276653 19.89078, 110.276653 19.89078, 110.276653 19.89078, 110.271921 19.891273, 110.269743 19.891557, 110.267897 19.891819, 110.266496 19.892227, 110.265495 19.892664, 110.264913 19.892704, 110.264707 19.892765, 110.264716 19.892768, 110.264716 19.892768, 110.264717 19.892768, 110.264717 19.892768, 110.264717 19.892769, 110.264717 19.892769, 110.264717 19.892769, 110.264717 19.892769, 110.264717 19.892769, 110.264717 19.89277, 110.264717 19.89277, 110.264718 19.89277, 110.264718 19.89277, 110.264718 19.89277, 110.264718 19.89277, 110.264718 19.892771, 110.264718 19.892771, 110.264718 19.892771, 110.264719 19.892771, 110.264719 19.892771, 110.264718 19.892771, 110.264718 19.892771, 110.264718 19.892771, 110.264719 19.892771, 110.264719 19.892771, 110.264719 19.892771, 110.264719 19.892772, 110.264727 19.892772, 110.264727 19.892772, 110.264863 19.892733, 110.264863 19.892733, 110.265575 19.892657)"

    # 可选：设置输出HTML路径
    output_html_path = None  # 设为None使用默认路径

    # 执行可视化
    success = create_trajectory_on_map(shapefile_path, trajectory_wkt, output_html_path)

    if success:
        print("\n特定轨迹在地图上的可视化完成！")
        print("使用说明：")
        print("- 红色粗线：特定轨迹")
        print("- 灰色细线：Haikou道路网络")
        print("- 绿色圆点：轨迹起点")
        print("- 蓝色圆点：轨迹终点")
        print("- 使用鼠标滚轮缩放")
        print("- 拖动鼠标平移视图")
        print("- 双击重置视图")
    else:
        print("特定轨迹可视化失败！")


if __name__ == "__main__":
    main()