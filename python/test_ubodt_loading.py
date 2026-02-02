#!/usr/bin/env python3
"""
快速测试脚本 - 验证 UBODT 加载是否正常
"""
import sys
sys.path.insert(0, '/home/dell/fmm_sjtugnc/build/python')

from fmm import *
import os

print("=" * 60)
print("UBODT 加载测试")
print("=" * 60)

# 可用的 UBODT 文件
ubodt_files = [
    "input/map/hainan_ubodt_indexed.bin",
    "input/map/haikou_ubodt_mmap.bin",
    "input/map/shanghai_ubodt_mmap.bin",
]

for ubodt_file in ubodt_files:
    if not os.path.exists(ubodt_file):
        print(f"\n✗ 文件不存在: {ubodt_file}")
        continue

    print(f"\n测试文件: {ubodt_file}")

    # 根据文件类型选择读取方法
    try:
        if ubodt_file.endswith("_indexed.bin"):
            print(f"  使用 read_ubodt_indexed_binary...")
            ubodt = UBODT.read_ubodt_indexed_binary(ubodt_file)
            print(f"  ✓ 成功! 行数: {ubodt.get_num_rows()}")
        elif ubodt_file.endswith("_mmap.bin"):
            print(f"  使用 read_ubodt_mmap_binary...")
            ubodt = UBODT.read_ubodt_mmap_binary(ubodt_file)
            print(f"  ✓ 成功! 行数: {ubodt.get_num_rows()}")
        else:
            print(f"  使用 read_ubodt_file...")
            ubodt = UBODT.read_ubodt_file(ubodt_file)
            print(f"  ✓ 成功! 行数: {ubodt.get_num_rows()}")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
