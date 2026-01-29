#!/usr/bin/env python3
"""
CMM Python 调试问题诊断报告
============================

问题: 所有测试点都匹配到边 ID 0

诊断结果:
---------

### 1. 根本原因：Shapefile 边 ID 字段全部为 0

检查 input/map/hainan/edges.shp 发现：
- 所有 152547 条边的 `key` 字段值都是 0
- 这导致所有匹配结果都指向边 ID 0

**影响**：
- `result.opath` 中所有元素都是 0
- 无法区分不同的边
- 匹配结果无意义

### 2. 次要问题：坐标系单位

- 路网坐标系: WGS84 (EPSG:4326) 经纬度
- 搜索半径计算: protection_level * protection_level_multiplier = 5.0 * 10.0 = 50.0
- 问题: 50.0 被当作"度"而不是"米"
- 在经纬度坐标系下，50 度是巨大的距离（约 5500 公里）

### 3. 候选点搜索测试

使用 GDAL 直接检查发现：
- 测试点 (110.1975, 20.0145) 附近有 159 条边
- 最近的边距离仅 69.63 米
- 但 Python API 的 `search_tr_cs_knn(geom, 16, 300.0)` 返回空

原因: 搜索半径 300.0 在经纬度坐标系下过大，可能导致 R-tree 搜索失败

### 4. 建议的解决方案

#### 方案 A: 修复 Shapefile 的边 ID 字段（推荐）

使用 `osmid` 字段或重新生成唯一的边 ID:

```python
from osgeo import ogr

shp_path = "input/map/hainan/edges.shp"
driver = ogr.GetDriverByName("ESRI Shapefile")
data_source = driver.Open(shp_path, 1)  # 1 = 可写
layer = data_source.GetLayer()

# 为每条边分配唯一的 key 值
layer.ResetReading()
for i, feat in enumerate(layer):
    feat.SetField("key", i)
    layer.SetFeature(feat)

data_source = None
```

#### 方案 B: 使用投影坐标系

将路网和轨迹转换到投影坐标系（如 UTM），使距离单位为米：

```python
# 使用 FMM 的 match_gps_file 而不是 match_traj
# 它支持 input_epsg 参数进行坐标转换
cmm.match_gps_file(gps_config, result_config, config, input_epsg=4326)
```

#### 方案 C: 使用路网中心附近的测试坐标

根据路网边界框计算的中心点：
- X 范围: [108.613800, 112.739354]
- Y 范围: [15.782399, 20.159777]
- 中心点: (110.676577, 17.971088)

### 5. 临时调试方案

如果要继续调试，可以：

1. 修复 shapefile 的 key 字段
2. 使用更合理的搜索半径（对于经纬度，使用 0.001 度约 100 米）
3. 使用路网实际覆盖范围内的坐标

示例测试坐标（海口市区，路网密集区域）：
```python
# 海口市国贸区域
geom = LineString()
geom.add_point(110.3130, 20.0370)  # 起点
geom.add_point(110.3170, 20.0390)  # 终点
```

"""

import sys
sys.path.insert(0, 'build/python')
from fmm import *
from osgeo import ogr, osr

def fix_shapefile_edge_ids(shp_path):
    """修复 shapefile 的边 ID 字段"""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(shp_path, 1)  # 1 = 可写
    if data_source is None:
        print(f"无法打开 {shp_path}")
        return False

    layer = data_source.GetLayer()
    layer.ResetReading()

    print(f"开始修复边 ID...")
    count = 0
    for i, feat in enumerate(layer):
        key = feat.GetField("key")
        if key == 0:
            feat.SetField("key", i)
            layer.SetFeature(feat)
            count += 1

    data_source = None
    print(f"✓ 修复完成！更新了 {count} 条边的 ID")
    return True

def get_network_center_coords(shp_path):
    """获取路网中心区域的坐标"""
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(shp_path, 0)
    layer = data_source.GetLayer()

    # 获取边界框
    min_x, max_x, min_y, max_y = layer.GetExtent()

    # 计算中心
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 在中心附近生成测试坐标
    test_coords = [
        (center_x - 0.1, center_y),      # 中心西侧约 10km
        (center_x, center_y),            # 中心点
        (center_x + 0.1, center_y),      # 中心东侧约 10km
        (center_x, center_y + 0.05),     # 中心北侧约 5km
    ]

    data_source = None
    return test_coords

if __name__ == "__main__":
    print(__doc__)

    print("\n" + "=" * 80)
    print("操作选项:")
    print("=" * 80)
    print("1. 修复 shapefile 边 ID")
    print("2. 获取路网中心区域坐标")
    print("3. 退出")

    choice = input("\n请选择操作 (1-3): ").strip()

    if choice == "1":
        shp_path = input("输入 shapefile 路径 [input/map/hainan/edges.shp]: ").strip()
        if not shp_path:
            shp_path = "input/map/hainan/edges.shp"
        fix_shapefile_edge_ids(shp_path)
    elif choice == "2":
        shp_path = input("输入 shapefile 路径 [input/map/hainan/edges.shp]: ").strip()
        if not shp_path:
            shp_path = "input/map/hainan/edges.shp"
        coords = get_network_center_coords(shp_path)
        print("\n建议的测试坐标:")
        for i, (x, y) in enumerate(coords):
            print(f"  {i}. ({x:.6f}, {y:.6f})")
