#!/usr/bin/env python3
"""
筛选mr_low.csv中的轨迹数据
1. 去掉位于海口地图之外的点
2. 去掉同一轨迹中相邻点距离超过0.005°的点
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
            'min_lon': 110.1,
            'max_lon': 110.6,
            'min_lat': 19.8,
            'max_lat': 20.2
        }


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
        return None


def filter_trajectory_points(trajectory_row, map_bounds, max_distance=0.005, cumu_prob_threshold=-183):
    """
    过滤单个轨迹中的坐标点
    1. 保留在地图范围内的正常点
    2. 过滤相邻点距离过大的点
    3. 只保留cumu_prob最后一个值大于阈值的轨迹
    """
    idx, row = trajectory_row
    trajectory_id = row['id']
    original_id = row['original_id'] if 'original_id' in row else trajectory_id
    wkt_str = row['mgeom']
    cumu_prob_str = row['cumu_prob']

    try:
        # 首先检查cumu_prob值
        last_cumu_prob = extract_last_cumu_prob(cumu_prob_str)
        if last_cumu_prob is None:
            return idx, None, "无法解析cumu_prob值"

        # 只保留cumu_prob大于阈值的轨迹
        if last_cumu_prob <= cumu_prob_threshold:
            return idx, None, f"cumu_prob值{last_cumu_prob:.6f}小于阈值{cumu_prob_threshold}"

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
            return idx, None, f"过滤后只剩{len(filtered_coords)}个点，无法构成轨迹"

        # 创建新的LineString
        if filtered_coords:
            filtered_line = LineString(filtered_coords)
            filtered_wkt = filtered_line.wkt

            result = {
                'id': trajectory_id,
                'original_id': original_id,
                'original_mgeom': wkt_str,
                'filtered_mgeom': filtered_wkt,
                'cumu_prob': cumu_prob_str,
                'last_cumu_prob': last_cumu_prob,
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


def filter_mr_low_parallel(input_csv, output_csv, bounds_file='haikou_map_bounds.json', max_distance=0.005, cumu_prob_threshold=-183):
    """并行过滤mr_low数据"""
    print(f"开始并行过滤mr_low数据: {input_csv}")
    print(f"相邻点最大距离阈值: {max_distance}°")
    print(f"cumu_prob阈值: 只保留大于 {cumu_prob_threshold} 的轨迹")

    # 加载地图边界
    map_bounds = load_map_bounds(bounds_file)
    print(f"地图边界范围:")
    print(f"  经度: {map_bounds['min_lon']:.6f} 到 {map_bounds['max_lon']:.6f}")
    print(f"  纬度: {map_bounds['min_lat']:.6f} 到 {map_bounds['max_lat']:.6f}")

    # 读取CSV文件
    try:
        df = pd.read_csv(input_csv, sep=';')
        print(f"成功读取 {len(df)} 条mr_low轨迹数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return False

    # 检查数据格式
    required_columns = ['id', 'mgeom', 'cumu_prob']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"CSV文件格式错误，缺少列: {missing_columns}")
        print(f"可用列: {list(df.columns)}")
        return False

    # 添加original_id列（如果不存在）
    if 'original_id' not in df.columns:
        df['original_id'] = df['id']

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
    cumu_prob_values = []

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
                if result['last_cumu_prob'] is not None:
                    cumu_prob_values.append(result['last_cumu_prob'])

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

    # cumu_prob统计
    if cumu_prob_values:
        print(f"\n=== cumu_prob统计 ===")
        print(f"有效cumu_prob数量: {len(cumu_prob_values)}")
        print(f"最小值: {min(cumu_prob_values):.6f}")
        print(f"最大值: {max(cumu_prob_values):.6f}")
        print(f"平均值: {np.mean(cumu_prob_values):.6f}")
        print(f"中位数: {np.median(cumu_prob_values):.6f}")

    # 保存过滤后的数据
    if filtered_results:
        # 创建输出目录
        os.makedirs('output/filtered', exist_ok=True)

        # 保存完整结果
        filtered_df = pd.DataFrame(filtered_results)
        detailed_output = output_csv.replace('.csv', '_detailed.csv')
        filtered_df.to_csv(detailed_output, index=False, sep=';')
        print(f"\n详细过滤结果已保存到: {detailed_output}")

        # 保存简化版结果（只包含原始字段）
        simple_df = filtered_df[['id', 'original_id', 'filtered_mgeom', 'cumu_prob']]
        simple_df = simple_df.rename(columns={'filtered_mgeom': 'mgeom'})
        simple_df.to_csv(output_csv, index=False, sep=';')
        print(f"过滤后的轨迹数据已保存到: {output_csv}")

        return True
    else:
        print("警告: 没有有效的过滤结果")
        return False


def main():
    """主函数"""
    # 输入输出文件路径
    input_csv = 'mr_low.csv'
    output_csv = 'output/filtered/mr_low_filtered.csv'

    # 检查输入文件
    if not os.path.exists(input_csv):
        print(f"错误: 输入文件 {input_csv} 不存在")
        return

    # 执行过滤
    success = filter_mr_low_parallel(input_csv, output_csv)

    if success:
        print("mr_low数据过滤完成！")
    else:
        print("mr_low数据过滤失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()