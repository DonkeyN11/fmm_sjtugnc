#!/usr/bin/env python3
"""
比较三个CSV文件中相同ID的几何长度
"""

import pandas as pd
from shapely.wkt import loads
from shapely.geometry import LineString
import numpy as np

def parse_linestring_length(geom_str):
    """解析LINESTRING并返回长度和点数"""
    if pd.isna(geom_str) or geom_str == '' or geom_str == 'EMPTY':
        return 0.0, 0, 0

    try:
        # 移除可能的空格
        geom_str = geom_str.strip()
        geom = loads(geom_str)

        if geom.is_empty:
            return 0.0, 0, 0

        if isinstance(geom, LineString):
            # 获取所有坐标点
            coords = list(geom.coords)
            num_points = len(coords)

            # 计算几何长度
            geom_length = geom.length

            # 计算点与点之间的距离（包含空点）
            distances = []
            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])
                dist = np.linalg.norm(p2 - p1)
                distances.append(dist)

            total_distance = sum(distances)

            return geom_length, num_points, total_distance
        else:
            return 0.0, 0, 0
    except Exception as e:
        print(f"解析错误: {e}")
        return 0.0, 0, 0

def read_csv_files():
    """读取三个CSV文件"""
    print("读取CSV文件...")

    cmm_df = pd.read_csv('dataset_hainan_06/mr/cmm_results.csv', sep=';')
    fmm_df = pd.read_csv('dataset_hainan_06/mr/fmm_results.csv', sep=';')
    merged_df = pd.read_csv('dataset_hainan_06/merged_cmm_trajectories.csv', sep=';')

    print(f"cmm_results.csv: {len(cmm_df)} 条记录")
    print(f"fmm_results.csv: {len(fmm_df)} 条记录")
    print(f"merged_cmm_trajectories.csv: {len(merged_df)} 条记录")

    return cmm_df, fmm_df, merged_df

def compare_geometries():
    """比较三个文件的几何长度"""
    cmm_df, fmm_df, merged_df = read_csv_files()

    # 创建字典，key为id，value为几何长度信息
    cmm_geoms = {}
    fmm_geoms = {}
    merged_geoms = {}

    print("\n处理 cmm_results.csv (pgeom字段)...")
    for _, row in cmm_df.iterrows():
        id_val = row['id']
        geom_length, num_points, total_distance = parse_linestring_length(row['pgeom'])
        cmm_geoms[id_val] = {
            'length': geom_length,
            'num_points': num_points,
            'total_distance': total_distance
        }

    print("处理 fmm_results.csv (pgeom字段)...")
    for _, row in fmm_df.iterrows():
        id_val = row['id']
        geom_length, num_points, total_distance = parse_linestring_length(row['pgeom'])
        fmm_geoms[id_val] = {
            'length': geom_length,
            'num_points': num_points,
            'total_distance': total_distance
        }

    print("处理 merged_cmm_trajectories.csv (geom字段)...")
    for _, row in merged_df.iterrows():
        id_val = row['id']
        geom_length, num_points, total_distance = parse_linestring_length(row['geom'])
        merged_geoms[id_val] = {
            'length': geom_length,
            'num_points': num_points,
            'total_distance': total_distance
        }

    # 找出所有文件中都存在的ID
    all_ids = set(cmm_geoms.keys()) | set(fmm_geoms.keys()) | set(merged_geoms.keys())
    common_ids = set(cmm_geoms.keys()) & set(fmm_geoms.keys()) & set(merged_geoms.keys())

    print(f"\n总共唯一ID数: {len(all_ids)}")
    print(f"三个文件共有的ID数: {len(common_ids)}")

    # 比较几何长度
    print("\n" + "="*80)
    print("几何长度比较结果")
    print("="*80)

    # 存储不一致的记录
    mismatches = []

    for id_val in sorted(common_ids):
        cmm_len = cmm_geoms[id_val]['length']
        fmm_len = fmm_geoms[id_val]['length']
        merged_len = merged_geoms[id_val]['length']

        cmm_pts = cmm_geoms[id_val]['num_points']
        fmm_pts = fmm_geoms[id_val]['num_points']
        merged_pts = merged_geoms[id_val]['num_points']

        # 使用较小的值作为容差（相对误差0.01%）
        tolerance = max(cmm_len, fmm_len, merged_len) * 0.0001

        if not (abs(cmm_len - fmm_len) < tolerance and
                abs(cmm_len - merged_len) < tolerance and
                abs(fmm_len - merged_len) < tolerance):
            mismatches.append({
                'id': id_val,
                'cmm_length': cmm_len,
                'fmm_length': fmm_len,
                'merged_length': merged_len,
                'cmm_points': cmm_pts,
                'fmm_points': fmm_pts,
                'merged_points': merged_pts
            })

    if mismatches:
        print(f"\n发现 {len(mismatches)} 条记录的几何长度不一致！\n")
        print(f"{'ID':<10} {'CMM长度':<15} {'FMM长度':<15} {'MERGED长度':<15} {'CMM点数':<10} {'FMM点数':<10} {'MERGED点数':<10}")
        print("-" * 95)

        for m in mismatches[:20]:  # 只显示前20条
            print(f"{m['id']:<10} {m['cmm_length']:<15.6f} {m['fmm_length']:<15.6f} {m['merged_length']:<15.6f} {m['cmm_points']:<10} {m['fmm_points']:<10} {m['merged_points']:<10}")

        if len(mismatches) > 20:
            print(f"\n... 还有 {len(mismatches) - 20} 条不一致记录")
    else:
        print("\n✓ 所有共有ID的几何长度都一致！")

    # 统计只在某些文件中存在的ID
    only_cmm = set(cmm_geoms.keys()) - set(fmm_geoms.keys()) - set(merged_geoms.keys())
    only_fmm = set(fmm_geoms.keys()) - set(cmm_geoms.keys()) - set(merged_geoms.keys())
    only_merged = set(merged_geoms.keys()) - set(cmm_geoms.keys()) - set(fmm_geoms.keys())

    cmm_fmm = set(cmm_geoms.keys()) & set(fmm_geoms.keys()) - set(merged_geoms.keys())
    cmm_merged = set(cmm_geoms.keys()) & set(merged_geoms.keys()) - set(fmm_geoms.keys())
    fmm_merged = set(fmm_geoms.keys()) & set(merged_geoms.keys()) - set(cmm_geoms.keys())

    print("\n" + "="*80)
    print("ID分布情况")
    print("="*80)
    print(f"只在 cmm_results.csv 中: {len(only_cmm)} 个")
    print(f"只在 fmm_results.csv 中: {len(only_fmm)} 个")
    print(f"只在 merged_cmm_trajectories.csv 中: {len(only_merged)} 个")
    print(f"在 cmm 和 fmm 中，但不在 merged 中: {len(cmm_fmm)} 个")
    print(f"在 cmm 和 merged 中，但不在 fmm 中: {len(cmm_merged)} 个")
    print(f"在 fmm 和 merged 中，但不在 cmm 中: {len(fmm_merged)} 个")
    print(f"在三个文件中都存在: {len(common_ids)} 个")

    # 详细统计
    print("\n" + "="*80)
    print("几何长度统计（三个文件共有的ID）")
    print("="*80)

    cmm_lengths = [cmm_geoms[id]['length'] for id in common_ids]
    fmm_lengths = [fmm_geoms[id]['length'] for id in common_ids]
    merged_lengths = [merged_geoms[id]['length'] for id in common_ids]

    print(f"\nCMM Results:")
    print(f"  平均长度: {np.mean(cmm_lengths):.2f}")
    print(f"  最小长度: {np.min(cmm_lengths):.2f}")
    print(f"  最大长度: {np.max(cmm_lengths):.2f}")
    print(f"  标准差: {np.std(cmm_lengths):.2f}")

    print(f"\nFMM Results:")
    print(f"  平均长度: {np.mean(fmm_lengths):.2f}")
    print(f"  最小长度: {np.min(fmm_lengths):.2f}")
    print(f"  最大长度: {np.max(fmm_lengths):.2f}")
    print(f"  标准差: {np.std(fmm_lengths):.2f}")

    print(f"\nMerged CMM Trajectories:")
    print(f"  平均长度: {np.mean(merged_lengths):.2f}")
    print(f"  最小长度: {np.min(merged_lengths):.2f}")
    print(f"  最大长度: {np.max(merged_lengths):.2f}")
    print(f"  标准差: {np.std(merged_lengths):.2f}")

if __name__ == '__main__':
    compare_geometries()
