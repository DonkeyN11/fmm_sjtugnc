#!/usr/bin/env python3
"""
CMM Python Debug Script with Internal Variable Output
演示如何在 Python 中查看 CMM match_traj 函数的内部变量
"""

import sys
sys.path.insert(0, '/home/dell/fmm_sjtugnc/build/python')

from fmm import *
import logging

# 方案1: 设置 C++ 日志级别为 DEBUG（需要在启动时设置）
# 注意：这需要在导入 fmm 模块之前设置环境变量
import os
os.environ['SPDLOG_LEVEL'] = 'debug'  # 可选: trace, debug, info, warn, error

# 重新导入模块以应用日志级别
import importlib
if 'fmm' in sys.modules:
    importlib.reload(sys.modules['fmm'])
else:
    from fmm import *


def print_match_debug_info(traj, config, cmm):
    """
    匹配前后的调试信息输出
    """
    print("\n" + "=" * 80)
    print("匹配前调试信息")
    print("=" * 80)

    # 输入轨迹信息
    print(f"\n轨迹 ID: {traj.id}")
    print(f"轨迹点数: {traj.geom.get_num_points()}")
    print(f"时间戳数量: {len(traj.timestamps)}")
    print(f"协方差矩阵数量: {len(traj.covariances)}")
    print(f"保护级别数量: {len(traj.protection_levels)}")
    print(f"轨迹有效性: {traj.is_valid()}")

    # 配置信息
    print(f"\n算法配置:")
    print(f"  k (候选点数): {config.k}")
    print(f"  min_candidates: {config.min_candidates}")
    print(f"  protection_level_multiplier: {config.protection_level_multiplier}")
    print(f"  reverse_tolerance: {config.reverse_tolerance}")
    print(f"  normalized: {config.normalized}")
    print(f"  use_mahalanobis_candidates: {config.use_mahalanobis_candidates}")
    print(f"  window_length: {config.window_length}")
    print(f"  filtered: {config.filtered}")

    # 协方差矩阵详情（前3个点）
    print(f"\n前3个点的协方差矩阵:")
    for i in range(min(3, len(traj.covariances))):
        cov = traj.covariances[i]
        print(f"  点 {i}:")
        print(f"    sde (East std): {cov.sde:.4f} 米")
        print(f"    sdn (North std): {cov.sdn:.4f} 米")
        print(f"    sdu (Up std): {cov.sdu:.4f} 米")
        print(f"    sdne (NE cov): {cov.sdne:.6f}")
        print(f"    2D uncertainty: {cov.get_2d_uncertainty():.6f}")

    # 保护级别详情（前3个点）
    print(f"\n前3个点的保护级别:")
    for i in range(min(3, len(traj.protection_levels))):
        pl = traj.protection_levels[i]
        print(f"  点 {i}: {pl:.4f} 米")

    print("\n" + "=" * 80)
    print("开始匹配...")
    print("=" * 80 + "\n")

    # 执行匹配
    result = cmm.match_traj(traj, config)

    print("\n" + "=" * 80)
    print("匹配结果调试信息")
    print("=" * 80)

    # 基本信息 (检查 cpath 是否非空来判断是否匹配成功)
    print(f"\n匹配状态: {'✓ 成功' if len(result.cpath) > 0 else '✗ 失败'}")
    print(f"匹配边数: {len(result.cpath)}")

    # 计算 sp_dist 和 eu_dist 的总和
    try:
        sp_dists = list(result.sp_distances) if hasattr(result.sp_distances, '__iter__') else []
        sp_dist_total = sum(sp_dists) if len(sp_dists) > 0 else 0.0
    except:
        sp_dist_total = 0.0

    try:
        eu_dists = list(result.eu_distances) if hasattr(result.eu_distances, '__iter__') else []
        eu_dist_total = sum(eu_dists) if len(eu_dists) > 0 else 0.0
    except:
        eu_dist_total = 0.0

    print(f"最短路径距离 (sp_dist): {sp_dist_total:.4f} 米")
    print(f"欧氏距离 (eu_dist): {eu_dist_total:.4f} 米")

    if sp_dist_total > 0:
        ratio = eu_dist_total / sp_dist_total
        print(f"匹配比率: {ratio:.6f}")
    else:
        print(f"匹配比率: N/A (sp_dist = 0)")

    # 完整路径
    if len(result.cpath) > 0:
        print(f"\n完整路径 (cpath, 边ID序列):")
        print(f"  {list(result.cpath)}")
    else:
        print(f"\n完整路径: 空 (匹配失败)")

    # 每个点的匹配详情
    print(f"\n每个点的匹配详情:")

    if len(result.opt_candidate_path) == 0:
        print("  ⚠️  opt_candidate_path 为空，无法显示详细信息")
        print("  提示: 检查 opath 和 cpath 来获取匹配结果")
    else:
        print(f"{'点序号':<8} {'边ID':<10} {'偏移(米)':<12} {'距离(米)':<12} {'发射概率':<15} {'转移概率':<15} {'可信度':<12}")
        print("-" * 100)

        for i, mc in enumerate(result.opt_candidate_path):
            # 使用 opath 来获取边 ID（因为 mc.c.edge.id 可能无法正常工作）
            edge_id = result.opath[i] if i < len(result.opath) else -1
            offset = mc.c.offset
            dist = mc.c.dist
            ep = mc.ep
            tp = mc.tp
            trust = mc.trustworthiness

            print(f"{i:<8} {edge_id:<10} {offset:<12.4f} {dist:<12.4f} {ep:<15.6e} {tp:<15.6e} {trust:<12.6f}")

    # 候选点详情（如果启用）
    # 注意: candidate_details 可能是空的或不可访问的，需要用 try-except 处理
    try:
        if hasattr(result, 'candidate_details') and result.candidate_details is not None:
            # 尝试获取长度
            cd_len = len(result.candidate_details) if hasattr(result.candidate_details, '__len__') else 0
            if cd_len > 0:
                print(f"\n候选点详情 (每点的所有候选):")
                for i in range(cd_len):
                    candidates = result.candidate_details[i]
                    cand_len = len(candidates) if hasattr(candidates, '__len__') else 0
                    if cand_len > 0:
                        print(f"\n点 {i}: {cand_len} 个候选")
                        print(f"  {'候选序号':<10} {'X坐标':<15} {'Y坐标':<15} {'发射概率':<15}")
                        print("  " + "-" * 60)
                        for j in range(cand_len):
                            cand = candidates[j]
                            print(f"  {j:<10} {cand.x:<15.6f} {cand.y:<15.6f} {cand.ep:<15.6e}")
                    else:
                        print(f"\n点 {i}: ❌ 无候选点")
    except Exception as e:
        print(f"\n候选点详情: 无法访问 ({e})")

    # 距离统计
    try:
        sp_dists = result.sp_distances if hasattr(result, 'sp_distances') else []
        sp_dists_list = list(sp_dists) if sp_dists and hasattr(sp_dists, '__iter__') else []
        if len(sp_dists_list) > 0:
            print(f"\n最短路径距离统计:")
            print(f"  总距离: {sum(sp_dists_list):.4f} 米")
            print(f"  平均距离: {sum(sp_dists_list)/len(sp_dists_list):.4f} 米")
            print(f"  最大距离: {max(sp_dists_list):.4f} 米")
            print(f"  最小距离: {min(sp_dists_list):.4f} 米")
    except Exception as e:
        print(f"\n最短路径距离统计: 无法访问 ({e})")

    try:
        eu_dists = result.eu_distances if hasattr(result, 'eu_distances') else []
        eu_dists_list = list(eu_dists) if eu_dists and hasattr(eu_dists, '__iter__') else []
        if len(eu_dists_list) > 0:
            print(f"\n欧氏距离统计:")
            print(f"  总距离: {sum(eu_dists_list):.4f} 米")
            print(f"  平均距离: {sum(eu_dists_list)/len(eu_dists_list):.4f} 米")
            print(f"  最大距离: {max(eu_dists_list):.4f} 米")
            print(f"  最小距离: {min(eu_dists_list):.4f} 米")
    except Exception as e:
        print(f"\n欧氏距离统计: 无法访问 ({e})")

    print("\n" + "=" * 80)

    return result


def main():
    print("\n" + "=" * 80)
    print("CMM Python 调试示例 - 查看内部变量")
    print("=" * 80 + "\n")

    try:
        # 1. 加载网络
        print("Step 1: 加载网络...")
        network_file = "input/map/hainan/edges.shp"

        # 检查文件是否存在
        import os
        if not os.path.exists(network_file):
            # 尝试其他可能的路径
            alternative_paths = [
                "input/map/haikou/edges.shp",
                "input/map/shanghai/edges.shp",
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    network_file = alt_path
                    print(f"  使用备用路径: {network_file}")
                    break

        print(f"  从 {network_file} 加载网络...")
        network = Network(network_file, "key", "u", "v")
        print(f"  ✓ 加载了 {network.get_edge_count()} 条边\n")

        # 2. 创建图
        print("Step 2: 创建网络图...")
        graph = NetworkGraph(network)
        print(f"  ✓ 图有 {graph.get_num_vertices()} 个顶点\n")

        # 3. 加载 UBODT
        print("Step 3: 加载 UBODT...")
        # 根据文件类型使用不同的读取方法：
        # - *_mmap.bin: 使用 read_ubodt_mmap_binary
        # - *_indexed.bin: 使用 read_ubodt_indexed_binary
        # - *.txt, *.csv: 使用 read_ubodt_file

        ubodt_file = "input/map/hainan_ubodt_indexed.bin"
        print(f"  从 {ubodt_file} 加载...")

        # 检测文件类型并使用正确的读取方法
        if ubodt_file.endswith("_indexed.bin"):
            ubodt = UBODT.read_ubodt_indexed_binary(ubodt_file)
            print(f"  ✓ 使用 read_ubodt_indexed_binary 加载了 {ubodt.get_num_rows()} 行 UBODT\n")
        elif ubodt_file.endswith("_mmap.bin"):
            ubodt = UBODT.read_ubodt_mmap_binary(ubodt_file)
            print(f"  ✓ 使用 read_ubodt_mmap_binary 加载了 {ubodt.get_num_rows()} 行 UBODT\n")
        else:
            # 尝试自动检测
            ubodt = UBODT.read_ubodt_file(ubodt_file)
            print(f"  ✓ 使用 read_ubodt_file 加载了 {ubodt.get_num_rows()} 行 UBODT\n")

        # 4. 创建 CMM 算法
        print("Step 4: 创建 CMM 算法...")
        cmm = CovarianceMapMatch(network, graph, ubodt)
        print("  ✓ CMM 算法初始化完成\n")

        # 5. 创建配置
        print("Step 5: 创建配置...")
        config = CovarianceMapMatchConfig(
            k_arg=16,
            min_candidates_arg=1,
            protection_level_multiplier_arg=10.0,
            reverse_tolerance=0.0001,
            normalized_arg=True,
            use_mahalanobis_candidates_arg=True,
            window_length_arg=100,
            margin_used_trustworthiness_arg=False
        )
        print(f"  ✓ 配置创建完成 (k={config.k}, pl_mult={config.protection_level_multiplier})\n")

        # 6. 创建测试轨迹
        print("Step 6: 创建测试轨迹...")

        # 从 CSV 读取或手动创建
        geom = LineString()
        # 海南的坐标示例
        geom.add_point(110.1975, 20.0145)
        geom.add_point(110.1995, 20.0165)
        geom.add_point(110.2015, 20.0185)
        geom.add_point(110.2035, 20.0205)
        geom.add_point(110.2055, 20.0225)

        timestamps = DoubleVector()
        for i in range(5):
            timestamps.append(float(i * 30))

        covariances = CovarianceMatrixVector()
        for i in range(5):
            cov = CovarianceMatrix()
            cov.sde = 2.0
            cov.sdn = 1.5
            cov.sdu = 3.0
            cov.sdne = 0.1
            cov.sdeu = 0.05
            cov.sdun = 0.08
            covariances.append(cov)

        protection_levels = DoubleVector()
        for i in range(5):
            protection_levels.append(5.0)

        traj = CMMTrajectory()
        traj.id = 1
        traj.geom = geom
        traj.timestamps = timestamps
        traj.covariances = covariances
        traj.protection_levels = protection_levels

        print(f"  ✓ 创建了 {traj.geom.get_num_points()} 个点的轨迹\n")

        # 7. 执行匹配并输出调试信息
        result = print_match_debug_info(traj, config, cmm)

        # 8. 诊断建议
        print("\n诊断建议:")
        if len(result.cpath) == 0:
            print("  ⚠️  匹配失败！可能原因:")
            print("    1. 轨迹点没有候选点 (检查上面的候选点详情)")
            print("    2. 候选点之间没有连通的路径")
            print("    3. protection_level_multiplier 太小")
            print("    4. 路网没有覆盖轨迹区域")
            print("    5. 坐标系不匹配")
        else:
            print("  ✓ 匹配成功！")
            try:
                sp_dists = list(result.sp_distances) if hasattr(result.sp_distances, '__iter__') else []
                sp_dist_total = sum(sp_dists) if len(sp_dists) > 0 else 0.0
                eu_dists = list(result.eu_distances) if hasattr(result.eu_distances, '__iter__') else []
                eu_dist_total = sum(eu_dists) if len(eu_dists) > 0 else 0.0
                if sp_dist_total > 0 and eu_dist_total / sp_dist_total < 0.8:
                    print("    ⚠️  匹配比率较低 (< 0.8)，可能匹配不准确")
                else:
                    print("    ✓ 匹配比率良好 (>= 0.8)")
            except Exception as e:
                print(f"    ℹ️  无法计算匹配比率: {e}")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
