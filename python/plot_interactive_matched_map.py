#!/usr/bin/env python3
"""
并行计算的可交互轨迹可视化脚本
将mr_pgeom_cumu.txt中的轨迹数据绘制在haikou地图上
根据cumu_prob值进行颜色映射，使用透明度叠加效果
使用Plotly创建交互式可视化
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shapely.wkt import loads
from shapely.geometry import LineString
import multiprocessing as mp
from functools import partial
import os
import warnings
warnings.filterwarnings('ignore')

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

def extract_last_cumu_prob(cumu_prob_str):
    """提取cumu_prob字符串的最后一个值"""
    try:
        if pd.isna(cumu_prob_str):
            return None
        str_values = str(cumu_prob_str).split(',')
        if str_values and str_values[-1].strip():
            return float(str_values[-1].strip())
        return None
    except Exception as e:
        print(f"解析cumu_prob错误: {e}")
        return None

def parse_trajectory_line(line):
    """解析单行轨迹数据"""
    try:
        # 跳过空行和标题行
        if not line.strip() or line.strip().startswith('id;'):
            return None

        parts = line.strip().split(';')
        if len(parts) != 5:  # 实际有5个字段
            return None

        trajectory_id = parts[0]
        geom_wkt = parts[1]

        # 解析cumu_prob值（最后一个字段）
        cumu_probs = parts[4].split(',')

        # 使用最后一个cumu_prob值作为该轨迹的代表值
        final_prob = float(cumu_probs[-1])

        # 解析几何图形
        coords = parse_linestring(geom_wkt)

        if not coords:
            return None

        return {
            'id': trajectory_id,
            'coords': coords,
            'cumu_prob': final_prob,
            'all_probs': [float(x) for x in cumu_probs]
        }
    except Exception as e:
        print(f"解析行时出错: {e}")
        return None

def read_trajectories_parallel(file_path, num_processes=None):
    """并行读取轨迹数据"""
    if num_processes is None:
        # 使用较少的进程数以避免过多进程创建开销
        num_processes = min(16, mp.cpu_count())

    print(f"使用 {num_processes} 个进程并行读取数据...")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过标题行

    print(f"总共 {len(lines)} 条轨迹数据")

    # 使用进程池并行处理
    with mp.Pool(processes=num_processes) as pool:
        results = list(pool.map(parse_trajectory_line, lines))

    # 过滤掉None值
    valid_results = [r for r in results if r is not None]

    print(f"成功解析 {len(valid_results)} 条有效轨迹")

    return pd.DataFrame(valid_results)

def create_interactive_plotly_visualization(trajectories_df, output_html_path=None):
    """使用Plotly创建交互式轨迹可视化"""
    print("开始创建交互式Plotly可视化...")

    # 创建图形
    fig = go.Figure()

    # 收集所有坐标和概率值
    all_coords = []
    cumu_prob_values = []
    valid_trajectories = 0

    # 限制轨迹数量以提高性能
    max_trajectories = min(len(trajectories_df), 5000)  # 限制为5000条轨迹
    if len(trajectories_df) > max_trajectories:
        print(f"限制轨迹数量: 从 {len(trajectories_df)} 减少到 {max_trajectories}")
        trajectories_df = trajectories_df.sample(n=max_trajectories, random_state=42)

    print(f"将显示 {len(trajectories_df)} 条轨迹")

    # 为每条轨迹创建一个轨迹线
    for idx, row in trajectories_df.iterrows():
        coords = row['coords']
        cumu_prob = row['cumu_prob']

        if coords and len(coords) >= 2:
            # 分离经纬度
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]

            # 收集坐标用于计算范围
            all_coords.extend(coords)
            cumu_prob_values.append(cumu_prob)

            # 创建悬停文本
            trajectory_id = row['id']
            point_count = len(coords)

            # 计算轨迹统计信息
            lon_range = max(lons) - min(lons)
            lat_range = max(lats) - min(lats)

            hover_text = f"""
            <b>轨迹ID: {trajectory_id}</b><br>
            <b>累积概率: {cumu_prob:.6e}</b><br>
            坐标点数: {point_count}<br>
            经度范围: {min(lons):.6f} ~ {max(lons):.6f}<br>
            纬度范围: {min(lats):.6f} ~ {max(lats):.6f}<br>
            经度跨度: {lon_range:.6f}°<br>
            纬度跨度: {lat_range:.6f}°
            """

            # 根据cumu_prob值设置颜色（值越小越红，越大越蓝）
            # cumu_prob都是负值，所以需要特殊处理
            min_prob = min(cumu_prob_values) if cumu_prob_values else cumu_prob
            max_prob = max(cumu_prob_values) if cumu_prob_values else cumu_prob

            # 归一化到0-1范围
            if max_prob != min_prob:
                normalized_prob = (cumu_prob - min_prob) / (max_prob - min_prob)
            else:
                normalized_prob = 0.5

            # 颜色映射：红色(低值) -> 蓝色(高值)
            red_component = int(255 * (1 - normalized_prob))
            blue_component = int(255 * normalized_prob)
            color_str = f'rgb({red_component}, 0, {blue_component})'

            # 设置透明度（概率值越高越不透明）
            opacity = 0.3 + 0.7 * normalized_prob

            # 添加轨迹线
            fig.add_trace(go.Scatter(
                x=lons,
                y=lats,
                mode='lines',
                line=dict(
                    color=color_str,
                    width=2
                ),
                name=f'Trajectory {trajectory_id}',
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=False,
                opacity=opacity
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

    # 统计cumu_prob信息
    if cumu_prob_values:
        print(f"\n=== cumu_prob统计 ===")
        print(f"最小值: {min(cumu_prob_values):.6e}")
        print(f"最大值: {max(cumu_prob_values):.6e}")
        print(f"平均值: {np.mean(cumu_prob_values):.6e}")
        print(f"中位数: {np.median(cumu_prob_values):.6e}")

    # 设置图形布局
    fig.update_layout(
        title={
            'text': 'Trajectory Visualization with Cumulative Probability',
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
        width=1400,
        height=900,
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
        default_path = 'interactive_trajectory_plotly_visualization.html'
        fig.write_html(default_path)
        print(f"交互式可视化已保存到: {default_path}")

    # 在浏览器中显示
    fig.show()

    return fig

def main():
    """主函数"""
    # 文件路径
    data_file = 'output/mr_pgeom_cumu.txt'

    print("开始并行轨迹可视化...")

    # 1. 并行读取轨迹数据
    trajectories_df = read_trajectories_parallel(data_file)

    if trajectories_df.empty:
        print("没有有效的轨迹数据")
        return

    # 显示数据统计信息
    print(f"\n数据统计:")
    print(f"轨迹数量: {len(trajectories_df)}")
    print(f"cumu_prob范围: {trajectories_df['cumu_prob'].min():.2e} 到 {trajectories_df['cumu_prob'].max():.2e}")
    print(f"平均cumu_prob: {trajectories_df['cumu_prob'].mean():.2e}")

    # 2. 创建Plotly交互式可视化
    fig = create_interactive_plotly_visualization(trajectories_df)

    print("\n可视化完成!")
    print("生成的HTML文件可以通过浏览器打开进行交互式查看")
    print("使用说明：")
    print("- 鼠标悬停在轨迹上可查看详细信息，包括cumu_prob值")
    print("- 轨迹颜色根据cumu_prob值设置：红色表示概率较低，蓝色表示概率较高")
    print("- 透明度效果使重叠轨迹更清晰")
    print("- 使用鼠标滚轮缩放")
    print("- 拖动鼠标平移视图")
    print("- 双击重置视图")

if __name__ == "__main__":
    main()