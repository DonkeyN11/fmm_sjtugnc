#!/usr/bin/env python3
"""
交互式轨迹可视化脚本
使用Plotly创建可交互的轨迹地图
鼠标悬停显示轨迹ID和其他详细信息
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shapely.wkt import loads
from shapely.geometry import LineString
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


def create_interactive_trajectory_visualization(csv_file_path, output_html_path=None, max_trajectories=2000):
    """创建交互式轨迹可视化"""
    print(f"开始处理轨迹数据: {csv_file_path}")

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path, sep=';')
        print(f"成功读取 {len(df)} 条轨迹数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return False

    # 检查数据格式
    if 'id' not in df.columns or 'geom' not in df.columns:
        print("CSV文件格式错误，需要包含 'id' 和 'geom' 列")
        return False

    # 限制轨迹数量以提高性能
    if len(df) > max_trajectories:
        print(f"限制轨迹数量: 从 {len(df)} 减少到 {max_trajectories}")
        df = df.sample(n=max_trajectories, random_state=42)

    print(f"将显示 {len(df)} 条轨迹")

    # 创建图形
    fig = go.Figure()

    # 为每条轨迹创建一个轨迹线
    valid_trajectories = 0
    all_coords = []

    print("开始创建交互式轨迹...")
    for idx, row in df.iterrows():
        coords = parse_linestring(row['geom'])

        if coords and len(coords) >= 2:
            # 分离经纬度
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]

            # 收集坐标用于计算范围
            all_coords.extend(coords)

            # 创建悬停文本
            trajectory_id = row['id']
            original_id = row.get('original_id', trajectory_id)
            point_count = len(coords)

            # 计算轨迹统计信息
            lon_range = max(lons) - min(lons)
            lat_range = max(lats) - min(lats)

            hover_text = f"""
            <b>轨迹ID: {trajectory_id}</b><br>
            原始ID: {original_id}<br>
            坐标点数: {point_count}<br>
            经度范围: {min(lons):.6f} ~ {max(lons):.6f}<br>
            纬度范围: {min(lats):.6f} ~ {max(lats):.6f}<br>
            经度跨度: {lon_range:.6f}°<br>
            纬度跨度: {lat_range:.6f}°
            """

            # 生成随机颜色
            random_color = np.random.rand(3,)
            color_str = f'rgb({int(random_color[0]*255)}, {int(random_color[1]*255)}, {int(random_color[2]*255)})'

            # 添加轨迹线
            fig.add_trace(go.Scatter(
                x=lons,
                y=lats,
                mode='lines',
                line=dict(
                    color=color_str,  # 使用RGB字符串格式
                    width=2
                ),
                name=f'Trajectory {trajectory_id}',
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False,  # 不显示图例
                opacity=0.7
            ))

            valid_trajectories += 1

            if valid_trajectories % 500 == 0:
                print(f"已处理 {valid_trajectories} 条轨迹...")

    print(f"成功处理 {valid_trajectories} 条轨迹")

    # 计算坐标范围
    if all_coords:
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]

        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # 计算合适的边距
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        lon_margin = max(lon_range * 0.05, 0.001)
        lat_margin = max(lat_range * 0.05, 0.001)

        print(f"经度范围: {min_lon:.6f} 到 {max_lon:.6f}")
        print(f"纬度范围: {min_lat:.6f} 到 {max_lat:.6f}")
    else:
        # 使用默认的海口范围
        min_lon, max_lon = 110.1, 110.6
        min_lat, max_lat = 19.8, 20.2
        lon_margin, lat_margin = 0.1, 0.1

    # 设置图形布局
    fig.update_layout(
        title={
            'text': 'Interactive Trajectory Visualization',
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
        hovermode='closest',  # 悬停模式：显示最近的数据点
        width=1200,
        height=800,
        showlegend=False
    )

    # 添加地图背景色块
    fig.add_shape(
        type="rect",
        x0=min_lon, y0=min_lat,
        x1=max_lon, y1=max_lat,
        fillcolor="lightblue",
        opacity=0.1,
        layer="below",
        line_width=0
    )

    # 保存为HTML文件
    if output_html_path:
        fig.write_html(output_html_path)
        print(f"交互式可视化已保存到: {output_html_path}")
    else:
        # 默认保存路径
        default_path = 'output/interactive_trajectory_visualization.html'
        fig.write_html(default_path)
        print(f"交互式可视化已保存到: {default_path}")

    # 在浏览器中显示
    fig.show()

    return True


def main():
    """主函数"""
    # 设置输入文件路径（使用过滤后的数据）
    csv_file_path = 'input/trajectory/all_2hour_data/all_2hour_data_Jan_parallel_filtered.csv'

    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        print("请先运行 parallel_trajectory_filter.py 来过滤轨迹数据")
        return

    # 可选：设置输出HTML路径
    output_html_path = None  # 设为None使用默认路径

    # 可选：调整显示的轨迹数量
    max_trajectories = 1500  # 减少轨迹数量以提高交互性能

    # 执行可视化
    success = create_interactive_trajectory_visualization(csv_file_path, output_html_path, max_trajectories)

    if success:
        print("交互式轨迹可视化完成！")
        print("使用说明：")
        print("- 鼠标悬停在轨迹上可查看详细信息")
        print("- 使用鼠标滚轮缩放")
        print("- 拖动鼠标平移视图")
        print("- 双击重置视图")
    else:
        print("交互式轨迹可视化失败！")


if __name__ == "__main__":
    main()