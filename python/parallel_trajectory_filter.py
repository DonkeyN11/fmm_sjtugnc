#!/usr/bin/env python3
"""
并行轨迹过滤脚本
根据地图范围过滤轨迹中的不合理坐标点
保留轨迹中在地图范围内的正常点
并过滤相邻点距离过大的坐标点（最大距离0.005°，以限速200km/h，采样间隔10s计算）
"""

import pandas as pd
import numpy as np
from shapely.wkt import loads
from shapely.geometry import LineString
import multiprocessing as mp
from functools import partial
import json
import os
import sys


def load_map_bounds(bounds_file='haikou_map_bounds.json'):
    """加载地图边界范围"""
    try:
        with open(bounds_file, 'r') as f:
            bounds = json.load(f)
        return bounds
    except FileNotFoundError:
        print(f"警告: 找不到边界文件 {bounds_file}，使用默认海口范围")
        return {
            'min_lon': 110.0,
            'max_lon': 110.8,
            'min_lat': 19.7,
            'max_lat': 20.3
        }


def filter_trajectory_points(trajectory_row, map_bounds, max_distance=0.005):
    """
    过滤单个轨迹中的坐标点
    保留在地图范围内的正常点，并检查相邻点距离
    过滤相邻点距离超过max_distance的坐标点
    """
    idx, row = trajectory_row
    trajectory_id = row['id']
    original_id = row['original_id']
    wkt_str = row['geom']

    try:
        line = loads(wkt_str)
        if not isinstance(line, LineString):
            return idx, None, "不是LineString类型"

        original_coords = list(line.coords)

        # 第一步：过滤地图范围外的点
        range_filtered_coords = []
        for coord in original_coords:
            lon, lat = coord

            # 检查坐标是否在地图范围内
            if (map_bounds['min_lon'] <= lon <= map_bounds['max_lon'] and
                map_bounds['min_lat'] <= lat <= map_bounds['max_lat']):
                range_filtered_coords.append(coord)

        # 第二步：过滤相邻点距离过大的点
        filtered_coords = []
        if range_filtered_coords:
            # 保留第一个点
            filtered_coords.append(range_filtered_coords[0])

            # 检查后续点与前一個点的距离
            for i in range(1, len(range_filtered_coords)):
                prev_coord = range_filtered_coords[i-1]
                curr_coord = range_filtered_coords[i]

                # 计算经度和纬度的差距
                lon_diff = abs(curr_coord[0] - prev_coord[0])
                lat_diff = abs(curr_coord[1] - prev_coord[1])

                # 检查是否超过最大距离阈值
                if lon_diff <= max_distance and lat_diff <= max_distance:
                    filtered_coords.append(curr_coord)
                else:
                    # 跳过这个点，距离过大
                    pass

        # 统计过滤结果
        total_points = len(original_coords)
        range_valid_points = len(range_filtered_coords)
        final_valid_points = len(filtered_coords)
        range_filtered_points = total_points - range_valid_points
        distance_filtered_points = range_valid_points - final_valid_points

        # 如果过滤后没有点或只剩一个点，无法构成LineString
        if len(filtered_coords) < 2:
            return idx, None, f"距离过滤后只剩{len(filtered_coords)}个点，无法构成轨迹"

        # 创建新的LineString
        if filtered_coords:
            filtered_line = LineString(filtered_coords)
            filtered_wkt = filtered_line.wkt

            result = {
                'id': trajectory_id,
                'original_id': original_id,
                'original_geom': wkt_str,
                'filtered_geom': filtered_wkt,
                'total_points': total_points,
                'range_valid_points': range_valid_points,
                'final_valid_points': final_valid_points,
                'range_filtered_points': range_filtered_points,
                'distance_filtered_points': distance_filtered_points,
                'total_filtered_points': range_filtered_points + distance_filtered_points,
                'filter_rate': (range_filtered_points + distance_filtered_points) / total_points if total_points > 0 else 0
            }

            return idx, result, None

    except Exception as e:
        return idx, None, f"解析错误: {str(e)}"


def filter_trajectories_parallel(input_csv, output_csv, bounds_file='haikou_map_bounds.json', max_distance=0.0005):
    """并行过滤轨迹数据"""
    print(f"开始并行过滤轨迹数据: {input_csv}")
    print(f"相邻点最大距离阈值: {max_distance}°")

    # 加载地图边界
    map_bounds = load_map_bounds(bounds_file)
    print(f"地图边界范围:")
    print(f"  经度: {map_bounds['min_lon']:.6f} 到 {map_bounds['max_lon']:.6f}")
    print(f"  纬度: {map_bounds['min_lat']:.6f} 到 {map_bounds['max_lat']:.6f}")

    # 读取CSV文件
    try:
        df = pd.read_csv(input_csv, sep=';')
        print(f"成功读取 {len(df)} 条轨迹数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return False

    # 检查数据格式
    if 'id' not in df.columns or 'geom' not in df.columns:
        print("CSV文件格式错误，需要包含 'id' 和 'geom' 列")
        return False

    # 获取CPU核心数
    num_cores = mp.cpu_count()
    print(f"使用 {num_cores} 个CPU核心进行并行过滤")

    # 准备数据用于并行处理
    trajectory_data = list(df.iterrows())

    # 使用进程池并行处理
    filtered_results = []
    error_count = 0
    total_points = 0
    total_range_valid_points = 0
    total_final_valid_points = 0
    total_range_filtered_points = 0
    total_distance_filtered_points = 0

    with mp.Pool(processes=num_cores) as pool:
        # 使用partial函数传递map_bounds和max_distance参数
        filter_func = partial(filter_trajectory_points, map_bounds=map_bounds, max_distance=max_distance)

        # 并行处理所有轨迹
        results = pool.imap(filter_func, trajectory_data)

        # 收集结果
        for idx, result, error in results:
            if error:
                error_count += 1
                if error_count <= 10:  # 只显示前10个错误
                    print(f"轨迹 {idx} 错误: {error}")
                continue

            if result:
                filtered_results.append(result)
                total_points += result['total_points']
                total_range_valid_points += result['range_valid_points']
                total_final_valid_points += result['final_valid_points']
                total_range_filtered_points += result['range_filtered_points']
                total_distance_filtered_points += result['distance_filtered_points']

    # 统计结果
    total_trajectories = len(df)
    filtered_trajectories = len(filtered_results)
    removed_trajectories = total_trajectories - filtered_trajectories

    print(f"\n=== 过滤结果统计 ===")
    print(f"轨迹统计:")
    print(f"  总轨迹数: {total_trajectories}")
    print(f"  保留轨迹数: {filtered_trajectories}")
    print(f"  删除轨迹数: {removed_trajectories}")
    print(f"  轨迹保留率: {filtered_trajectories/total_trajectories*100:.2f}%")
    print(f"  错误轨迹数: {error_count}")
    print(f"\n坐标点统计:")
    print(f"  总坐标点数: {total_points}")
    print(f"  范围过滤后点数: {total_range_valid_points}")
    print(f"  最终保留点数: {total_final_valid_points}")
    print(f"  范围过滤点数: {total_range_filtered_points}")
    print(f"  距离过滤点数: {total_distance_filtered_points}")
    print(f"  总过滤点数: {total_range_filtered_points + total_distance_filtered_points}")
    print(f"  坐标点总保留率: {total_final_valid_points/total_points*100:.2f}%")
    print(f"  范围保留率: {total_range_valid_points/total_points*100:.2f}%")
    print(f"  距离保留率: {total_final_valid_points/total_range_valid_points*100:.2f}%")

    # 保存过滤后的数据
    if filtered_results:
        # 创建新的DataFrame
        filtered_df = pd.DataFrame(filtered_results)

        # 保存完整结果
        filtered_df.to_csv(output_csv.replace('.csv', '_detailed.csv'), index=False, sep=';')
        print(f"\n详细过滤结果已保存到: {output_csv.replace('.csv', '_detailed.csv')}")

        # 保存简化版结果（只包含原始字段）
        simple_df = filtered_df[['id', 'original_id', 'filtered_geom']]
        simple_df = simple_df.rename(columns={'filtered_geom': 'geom'})
        simple_df.to_csv(output_csv, index=False, sep=';')
        print(f"过滤后的轨迹数据已保存到: {output_csv}")

        return True
    else:
        print("警告: 没有有效的过滤结果")
        return False


def main():
    """主函数"""
    # 输入输出文件路径
    input_csv = 'input/trajectory/all_2hour_data/all_2hour_data_Jan_parallel.csv'
    output_csv = 'input/trajectory/all_2hour_data/all_2hour_data_Jan_parallel_filtered_distance.csv'

    # 检查输入文件
    if not os.path.exists(input_csv):
        print(f"错误: 输入文件 {input_csv} 不存在")
        return

    # 执行过滤
    success = filter_trajectories_parallel(input_csv, output_csv)

    if success:
        print("轨迹过滤完成！")
    else:
        print("轨迹过滤失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()