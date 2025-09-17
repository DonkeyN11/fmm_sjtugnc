#!/usr/bin/env python3
"""
并行轨迹可视化脚本
使用所有可用的CPU核心绘制CSV中的轨迹数据
每个轨迹ID使用不同的颜色,不显示颜色图例
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.wkt import loads
from shapely.geometry import LineString
import multiprocessing as mp
from functools import partial
import os
import sys


def parse_linestring(wkt_str):
    """解析WKT格式的LineString为坐标点数组"""
    try:
        line = loads(wkt_str)
        if isinstance(line, LineString):
            return list(line.coords)
        return None
    except:
        return None


def plot_single_trajectory(trajectory_data, color_map):
    """绘制单个轨迹的函数，用于并行处理"""
    idx, row = trajectory_data
    coords = parse_linestring(row['geom'])

    if coords:
        # 分离经纬度
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]

        # 获取该轨迹的颜色
        trajectory_id = row['id']
        color = color_map[trajectory_id]

        # 绘制轨迹
        plt.plot(lons, lats, color=color, alpha=0.7, linewidth=1.5, zorder=2)

    return idx


def visualize_trajectories_parallel(csv_file_path, output_image_path=None, max_trajectories=5000):
    """并行可视化轨迹数据"""
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

    # 获取唯一的轨迹ID
    unique_ids = df['id'].unique()
    print(f"发现 {len(unique_ids)} 个唯一轨迹ID")

    # 为每个轨迹ID生成随机颜色
    color_map = {}
    np.random.seed(42)  # 设置随机种子以确保颜色一致

    for trajectory_id in unique_ids:
        # 生成随机颜色
        color_map[trajectory_id] = np.random.rand(3,)  # RGB值

    # 创建图形
    plt.figure(figsize=(15, 12))

    # 设置背景色
    ax = plt.gca()
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, alpha=0.3)

    # 获取CPU核心数
    num_cores = mp.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行并行处理")

    # 准备数据用于并行处理
    trajectory_data = list(df.iterrows())

    # 使用进程池并行处理
    with mp.Pool(processes=num_cores) as pool:
        # 使用partial函数传递color_map参数
        plot_func = partial(plot_single_trajectory, color_map=color_map)

        # 并行处理所有轨迹
        results = list(pool.imap(plot_func, trajectory_data))

    # 设置图形属性
    plt.title('Trajectory Visualization (Parallel Processing)', fontsize=16, fontweight='bold')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)

    # 自动调整坐标轴范围
    all_coords = []
    for _, row in df.iterrows():
        coords = parse_linestring(row['geom'])
        if coords:
            all_coords.extend(coords)

    if all_coords:
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        plt.xlim(min(lons) - 0.01, max(lons) + 0.01)
        plt.ylim(min(lats) - 0.01, max(lats) + 0.01)

    # 设置图形样式
    plt.tight_layout()

    # 保存图像
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_image_path}")
    else:
        # 默认保存路径
        default_path = 'trajectory_visualization_parallel.png'
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {default_path}")

    # 显示图像
    plt.show()

    return True


def main():
    """主函数"""
    # 设置输入文件路径
    csv_file_path = 'input/trajectory/all_2hour_data/all_2hour_data_Jan_parallel_filtered.csv'

    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        return

    # 可选：设置输出图像路径
    output_image_path = None  # 设为None使用默认路径

    # 执行可视化
    success = visualize_trajectories_parallel(csv_file_path, output_image_path)

    if success:
        print("轨迹可视化完成！")
    else:
        print("轨迹可视化失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()