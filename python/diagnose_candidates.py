#!/usr/bin/env python3
"""
诊断候选点搜索问题
"""

import sys
sys.path.insert(0, 'build/python')
from fmm import *

def main():
    # 加载网络和图
    network = Network("input/map/hainan/edges.shp", "key", "u", "v")
    graph = NetworkGraph(network)

    # 创建测试点
    test_points = [
        (110.1975, 20.0145),
        (110.1995, 20.0165),
        (110.2015, 20.0185),
        (110.2035, 20.0205),
        (110.2055, 20.0225),
    ]

    print("候选点搜索测试:")
    print("=" * 80)

    for i, (x, y) in enumerate(test_points):
        print(f"\n点 {i}: ({x:.6f}, {y:.6f})")

        # 创建 LineString 对象（只包含一个点）
        geom = LineString()
        geom.add_point(x, y)

        # 搜索候选点 (k=16, radius=300)
        candidates_result = network.search_tr_cs_knn(geom, 16, 300.0)

        # 转换为列表处理
        try:
            candidates_list = list(candidates_result)
            num_candidates = len(candidates_list)
        except:
            num_candidates = 0
            candidates_list = []

        if num_candidates == 0:
            print(f"  ⚠️  没有找到候选点！")
            print(f"  提示: 可能需要增加搜索半径")
        else:
            print(f"  找到 {num_candidates} 个候选点:")
            print(f"  {'候选序号':<10} {'边ID':<15} {'源节点':<10} {'目标节点':<10} {'偏移(米)':<15} {'距离(米)':<15}")
            print("  " + "-" * 90)

            for j in range(min(5, num_candidates)):
                cand = candidates_list[j]
                edge_id = cand.edge_id
                source = cand.source
                target = cand.target
                offset = cand.offset
                dist = cand.dist

                print(f"  {j:<10} {edge_id:<15} {source:<10} {target:<10} {offset:<15.4f} {dist:<15.4f}")

            if num_candidates > 5:
                print(f"  ... (共 {num_candidates} 个候选点)")

    # 检查边 ID 0 的信息
    print(f"\n\n边 ID 0 的信息:")
    print("=" * 80)
    try:
        edge = network.get_edge(0)
        if edge:
            print(f"  边 ID: 0")
            print(f"  源节点: {edge.source}")
            print(f"  目标节点: {edge.target}")
            print(f"  长度: {edge.length:.4f} 米")
            print(f"  几何形状点数: {edge.geom.get_num_points()}")
            points = edge.geom.get_points()
            print(f"  起点坐标: ({points[0].x:.6f}, {points[0].y:.6f})")
            print(f"  终点坐标: ({points[-1].x:.6f}, {points[-1].y:.6f})")
        else:
            print(f"  ⚠️  边 ID 0 不存在")
    except Exception as e:
        print(f"  ✗ 错误: {e}")

    # 统计所有边的 ID 范围
    print(f"\n\n边 ID 统计:")
    print("=" * 80)
    edge_count = network.get_edge_count()
    print(f"  总边数: {edge_count}")

    # 尝试获取几条边的 ID
    print(f"\n前 10 条边的 ID:")
    for i in range(min(10, edge_count)):
        try:
            edge_id = network.get_edge_id(i)
            edge = network.get_edge(edge_id)
            print(f"  索引 {i} -> 边 ID {edge_id} (源:{edge.source}, 目标:{edge.target})")
        except Exception as e:
            print(f"  索引 {i} -> 错误: {e}")

    # 检查边 ID 的实际分布
    print(f"\n边 ID 范围采样:")
    print(f"  采样检查一些边 ID...")
    sample_indices = [0, 1000, 10000, 50000, 100000, 152546]
    for idx in sample_indices:
        if idx < edge_count:
            try:
                edge_id = network.get_edge_id(idx)
                print(f"  索引 {idx} -> 边 ID {edge_id}")
            except:
                pass

if __name__ == "__main__":
    main()
