#!/usr/bin/env python3
"""
过滤后轨迹的简化可视化脚本
使用过滤后的轨迹数据，不需要并行处理，性能更好
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.wkt import loads
from shapely.geometry import LineString
import os


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


def visualize_filtered_trajectories(csv_file_path, output_image_path=None, max_trajectories=5000):
    """可视化过滤后的轨迹数据"""
    print(f"开始处理过滤后的轨迹数据: {csv_file_path}")

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path, sep=';')
        print(f"成功读取 {len(df)} 条过滤后的轨迹数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return False

    # 检查数据格式
    if 'id' not in df.columns or 'geom' not in df.columns:
        print("CSV文件格式错误，需要包含 'id' 和 'geom' 列")
        return False

    # 限制轨迹数量以提高性能
    if len(df) > max_trajectories:
        print(f"限制显示轨迹数量: 从 {len(df)} 减少到 {max_trajectories}")
        df = df.sample(n=max_trajectories, random_state=42)

    print(f"将显示 {len(df)} 条轨迹")

    # 创建图形
    plt.figure(figsize=(15, 12))

    # 设置背景色
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, alpha=0.3)

    # 收集所有坐标用于计算范围
    all_coords = []
    valid_trajectories = 0

    print("开始绘制轨迹...")
    for idx, row in df.iterrows():
        coords = parse_linestring(row['geom'])

        if coords and len(coords) >= 2:
            # 分离经纬度
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]

            # 为每个轨迹生成随机颜色
            color = np.random.rand(3,)

            # 绘制轨迹
            plt.plot(lons, lats, color=color, alpha=0.6, linewidth=1, zorder=2)

            # 收集坐标
            all_coords.extend(coords)
            valid_trajectories += 1

            if valid_trajectories % 1000 == 0:
                print(f"已绘制 {valid_trajectories} 条轨迹...")

    print(f"成功绘制 {valid_trajectories} 条轨迹")

    # 计算坐标范围
    if all_coords:
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]

        # 计算经纬度范围
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # 计算合适的边距
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat

        # 使用较小的边距比例
        lon_margin = max(lon_range * 0.05, 0.001)
        lat_margin = max(lat_range * 0.05, 0.001)

        plt.xlim(min_lon - lon_margin, max_lon + lon_margin)
        plt.ylim(min_lat - lat_margin, max_lat + lat_margin)

        # 显示坐标范围信息
        print(f"经度范围: {min_lon:.6f} 到 {max_lon:.6f}")
        print(f"纬度范围: {min_lat:.6f} 到 {max_lat:.6f}")
        print(f"经度跨度: {lon_range:.6f} 度")
        print(f"纬度跨度: {lat_range:.6f} 度")

    # 设置图形属性
    plt.title('Filtered Trajectory Visualization (Haikou Area)', fontsize=16, fontweight='bold')
    plt.xlabel('Longitude (°)', fontsize=12)
    plt.ylabel('Latitude (°)', fontsize=12)

    # 设置坐标轴格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

    # 设置图形样式
    plt.tight_layout()

    # 保存图像
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_image_path}")
    else:
        # 默认保存路径
        default_path = 'filtered_trajectory_visualization.png'
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {default_path}")

    # 显示图像
    plt.show()

    return True


def main():
    """主函数"""
    # 设置输入文件路径（使用过滤后的数据）
    csv_file_path = 'input/trajectory/all_2hour_data/all_2hour_data_Jan_parallel_filtered_distance.csv'

    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        print("请先运行 parallel_trajectory_filter.py 来过滤轨迹数据")
        return

    # 可选：设置输出图像路径
    output_image_path = None  # 设为None使用默认路径

    # 执行可视化
    success = visualize_filtered_trajectories(csv_file_path, output_image_path)

    if success:
        print("过滤后轨迹可视化完成！")
    else:
        print("轨迹可视化失败！")


if __name__ == "__main__":
    main()