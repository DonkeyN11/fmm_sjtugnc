#!/usr/bin/env python3
"""
详细分析几何长度不一致的原因
"""

import pandas as pd
from shapely.wkt import loads
from shapely.geometry import LineString
import numpy as np

def analyze_linestring(geom_str):
    """详细分析LINESTRING，包括空点、重复点等"""
    if pd.isna(geom_str) or geom_str == '' or geom_str == 'EMPTY':
        return None

    try:
        geom_str = geom_str.strip()
        geom = loads(geom_str)

        if geom.is_empty or not isinstance(geom, LineString):
            return None

        coords = list(geom.coords)

        # 统计信息
        total_points = len(coords)

        # 检查重复点（连续相同坐标）
        duplicate_count = 0
        duplicate_positions = []

        # 检查零长度线段（相邻点距离为0）
        zero_segments = 0
        zero_segment_positions = []

        # 计算所有线段长度
        segment_lengths = []
        for i in range(len(coords) - 1):
            p1 = np.array(coords[i][:2])  # 只取x,y坐标
            p2 = np.array(coords[i+1][:2])
            dist = np.linalg.norm(p2 - p1)
            segment_lengths.append(dist)

            if dist == 0:
                zero_segments += 1
                zero_segment_positions.append(i)

            # 检查是否是完全相同的点
            if np.array_equal(p1, p2):
                duplicate_count += 1
                duplicate_positions.append(i)

        # 统计
        geom_length = geom.length
        total_distance = sum(segment_lengths)

        # 检查是否所有点都相同
        unique_coords = set(coords)
        all_same = len(unique_coords) == 1

        return {
            'total_points': total_points,
            'unique_points': len(unique_coords),
            'geom_length': geom_length,
            'total_distance': total_distance,
            'duplicate_count': duplicate_count,
            'zero_segments': zero_segments,
            'all_same': all_same,
            'min_segment': min(segment_lengths) if segment_lengths else 0,
            'max_segment': max(segment_lengths) if segment_lengths else 0,
            'avg_segment': np.mean(segment_lengths) if segment_lengths else 0,
            'first_coord': coords[0] if coords else None,
            'last_coord': coords[-1] if coords else None,
        }
    except Exception as e:
        print(f"解析错误: {e}")
        return None

def detailed_comparison():
    """详细比较三个文件"""
    print("读取CSV文件...")
    cmm_df = pd.read_csv('dataset_hainan_06/mr/cmm_results.csv', sep=';')
    fmm_df = pd.read_csv('dataset_hainan_06/mr/fmm_results.csv', sep=';')
    merged_df = pd.read_csv('dataset_hainan_06/merged_cmm_trajectories.csv', sep=';')

    print("\n" + "="*120)
    print("详细几何分析")
    print("="*120)

    # 对于每个ID进行详细比较
    for _, cmm_row in cmm_df.iterrows():
        id_val = cmm_row['id']

        fmm_row = fmm_df[fmm_df['id'] == id_val]
        merged_row = merged_df[merged_df['id'] == id_val]

        if fmm_row.empty or merged_row.empty:
            continue

        cmm_geom = cmm_row['pgeom']
        fmm_geom = fmm_row.iloc[0]['pgeom']
        merged_geom = merged_row.iloc[0]['geom']

        cmm_stats = analyze_linestring(cmm_geom)
        fmm_stats = analyze_linestring(fmm_geom)
        merged_stats = analyze_linestring(merged_geom)

        if not all([cmm_stats, fmm_stats, merged_stats]):
            print(f"\nID {id_val}: 某些几何数据解析失败")
            continue

        print(f"\n{'='*120}")
        print(f"ID: {id_val}")
        print(f"{'='*120}")

        # 点数比较
        print(f"\n点数统计:")
        print(f"  {'文件':<30} {'总点数':<12} {'唯一点数':<12} {'重复点数':<12} {'零长度段':<12}")
        print(f"  {'-'*90}")
        print(f"  {'CMM (pgeom)':<30} {cmm_stats['total_points']:<12} {cmm_stats['unique_points']:<12} "
              f"{cmm_stats['duplicate_count']:<12} {cmm_stats['zero_segments']:<12}")
        print(f"  {'FMM (pgeom)':<30} {fmm_stats['total_points']:<12} {fmm_stats['unique_points']:<12} "
              f"{fmm_stats['duplicate_count']:<12} {fmm_stats['zero_segments']:<12}")
        print(f"  {'MERGED (geom)':<30} {merged_stats['total_points']:<12} {merged_stats['unique_points']:<12} "
              f"{merged_stats['duplicate_count']:<12} {merged_stats['zero_segments']:<12}")

        # 长度比较
        print(f"\n几何长度比较:")
        print(f"  {'文件':<30} {'几何长度':<20} {'累积距离':<20} {'平均段长':<15}")
        print(f"  {'-'*80}")
        print(f"  {'CMM (pgeom)':<30} {cmm_stats['geom_length']:<20.6f} {cmm_stats['total_distance']:<20.6f} "
              f"{cmm_stats['avg_segment']:<15.6f}")
        print(f"  {'FMM (pgeom)':<30} {fmm_stats['geom_length']:<20.6f} {fmm_stats['total_distance']:<20.6f} "
              f"{fmm_stats['avg_segment']:<15.6f}")
        print(f"  {'MERGED (geom)':<30} {merged_stats['geom_length']:<20.6f} {merged_stats['total_distance']:<20.6f} "
              f"{merged_stats['avg_segment']:<15.6f}")

        # 长度差异
        cmm_fmm_diff = abs(cmm_stats['geom_length'] - fmm_stats['geom_length'])
        cmm_merged_diff = abs(cmm_stats['geom_length'] - merged_stats['geom_length'])
        fmm_merged_diff = abs(fmm_stats['geom_length'] - merged_stats['geom_length'])

        print(f"\n长度差异:")
        print(f"  CMM vs FMM:     {cmm_fmm_diff:.6f} ({cmm_fmm_diff/max(cmm_stats['geom_length'], fmm_stats['geom_length'])*100:.4f}%)")
        print(f"  CMM vs MERGED:  {cmm_merged_diff:.6f} ({cmm_merged_diff/max(cmm_stats['geom_length'], merged_stats['geom_length'])*100:.4f}%)")
        print(f"  FMM vs MERGED:  {fmm_merged_diff:.6f} ({fmm_merged_diff/max(fmm_stats['geom_length'], merged_stats['geom_length'])*100:.4f}%)")

        # 坐标比较
        print(f"\n起止坐标比较:")
        print(f"  CMM   起点: {cmm_stats['first_coord']}")
        print(f"  CMM   终点: {cmm_stats['last_coord']}")
        print(f"  FMM   起点: {fmm_stats['first_coord']}")
        print(f"  FMM   终点: {fmm_stats['last_coord']}")
        print(f"  MERGED 起点: {merged_stats['first_coord']}")
        print(f"  MERGED 终点: {merged_stats['last_coord']}")

        # 检查起止点是否一致
        cmm_fmm_start_same = np.allclose(cmm_stats['first_coord'][:2], fmm_stats['first_coord'][:2], atol=1e-6)
        cmm_fmm_end_same = np.allclose(cmm_stats['last_coord'][:2], fmm_stats['last_coord'][:2], atol=1e-6)

        print(f"\n起止点一致性:")
        print(f"  CMM vs FMM:     起点{'相同' if cmm_fmm_start_same else '不同'}, 终点{'相同' if cmm_fmm_end_same else '不同'}")

if __name__ == '__main__':
    detailed_comparison()
