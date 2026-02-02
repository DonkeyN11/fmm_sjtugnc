# CMM Python 接口完整文档

本文档详细说明了 CMM (Covariance-based Map Matching) 的 Python API 接口，所有接口通过 SWIG 绑定暴露给 Python。

---

## 目录
1. [核心数据结构](#核心数据结构)
2. [配置类](#配置类)
3. [算法类](#算法类)
4. [结果类](#结果类)
5. [完整使用示例](#完整使用示例)
6. [接口说明](#接口说明)

---

## 核心数据结构

### 1. `CovarianceMatrix` - 协方差矩阵

表示 GNSS 观测的 3x3 协方差矩阵（对称矩阵，存储 6 个独立值）。

**属性：**
```python
cov = CovarianceMatrix()
cov.sde = 2.0      # East 标准差 (米)
cov.sdn = 1.5      # North 标准差 (米)
cov.sdu = 3.0      # Up 标准差 (米)
cov.sdne = 0.1     # North-East 协方差
cov.sdeu = 0.05    # East-Up 协方差
cov.sdun = 0.08    # Up-North 协方差
```

**方法：**
- `get_2d_uncertainty()` → `float`: 计算二维水平位置不确定性

**示例：**
```python
cov = CovarianceMatrix()
cov.sde = 2.0
cov.sdn = 1.5
cov.sdu = 3.0
cov.sdne = 0.1
cov.sdeu = 0.05
cov.sdun = 0.08

uncertainty = cov.get_2d_uncertainty()  # 二维不确定性
```

---

### 2. `CMMTrajectory` - CMM 轨迹

增强的轨迹结构，包含协方差矩阵和保护级别数据。

**属性：**
```python
traj = CMMTrajectory()
traj.id = 1                           # 轨迹 ID
traj.geom = LineString()              # 轨迹几何
traj.timestamps = DoubleVector()      # 时间戳列表 (秒)
traj.covariances = CovarianceMatrixVector()  # 每个点的协方差矩阵
traj.protection_levels = DoubleVector()      # 每个点的保护级别
```

**方法：**
- `is_valid()` → `bool`: 检查轨迹数据是否有效（协方差和保护级别数量是否匹配点数）

**创建轨迹示例：**
```python
# 1. 创建几何
geom = LineString()
geom.add_point(110.1975, 20.0145)
geom.add_point(110.1995, 20.0165)
geom.add_point(110.2015, 20.0185)

# 2. 创建时间戳
timestamps = DoubleVector()
timestamps.append(0.0)
timestamps.append(30.0)
timestamps.append(60.0)

# 3. 创建协方差矩阵
covariances = CovarianceMatrixVector()
for i in range(3):
    cov = CovarianceMatrix()
    cov.sde = 2.0
    cov.sdn = 1.5
    cov.sdu = 3.0
    cov.sdne = 0.1
    cov.sdeu = 0.05
    cov.sdun = 0.08
    covariances.append(cov)

# 4. 创建保护级别
protection_levels = DoubleVector()
protection_levels.append(5.0)
protection_levels.append(5.0)
protection_levels.append(5.0)

# 5. 组装轨迹
traj = CMMTrajectory()
traj.id = 1
traj.geom = geom
traj.timestamps = timestamps
traj.covariances = covariances
traj.protection_levels = protection_levels

# 6. 验证
if traj.is_valid():
    print("轨迹数据有效")
```

---

### 3. `LineString` - 线几何

表示轨迹或路径的几何形状。

**方法：**
- `add_point(x, y)`: 添加一个点 (x, y)
- `get_num_points()` → `int`: 获取点的数量

---

### 4. `DoubleVector` - 双精度浮点向量

STL vector<double> 的 Python 包装。

**方法：**
- `append(value)`: 添加元素
- `size()` → `int`: 获取元素数量
- `__len__()`: 支持 len() 函数
- `__getitem__(index)`: 支持索引访问

---

### 5. `CovarianceMatrixVector` - 协方差矩阵向量

`std::vector<CovarianceMatrix>` 的 Python 包装。

**方法：**
- `append(covariance_matrix)`: 添加协方差矩阵
- `size()` → `int`: 获取元素数量

---

## 配置类

### `CovarianceMapMatchConfig` - CMM 算法配置

**构造函数：**
```python
config = CovarianceMapMatchConfig(
    k_arg=8,                          # 候选点数量
    min_candidates_arg=3,             # 最少保留候选点数
    protection_level_multiplier_arg=1.0,  # 保护级别乘数
    reverse_tolerance=0.0,            # 反向移动容忍度 (米)
    normalized_arg=True,              # 是否归一化发射概率
    use_mahalanobis_candidates_arg=True,  # 是否使用马氏距离搜索候选点
    window_length_arg=10,             # 滑动窗口长度 (用于可信度)
    margin_used_trustworthiness_arg=True  # 是否使用边际计算可信度
)
```

**属性：**
```python
config.k                             # 候选点数量
config.min_candidates                # 最少保留候选点数
config.protection_level_multiplier   # 保护级别乘数
config.reverse_tolerance             # 反向移动容忍度
config.normalized                    # 是否归一化发射概率
config.use_mahalanobis_candidates    # 是否使用马氏距离
config.window_length                 # 滑动窗口长度
config.margin_used_trustworthiness   # 是否使用边际可信度
config.filtered                      # 是否过滤无候选点的点
```

**方法：**
- `validate()` → `bool`: 验证配置是否有效

**配置说明：**

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `k` | 每个点的最大候选数 | 8-16 | 值越大计算越慢，但可能更准确 |
| `min_candidates` | 最少保留的候选数 | 3 | 即使半径内候选不足也尝试保留 |
| `protection_level_multiplier` | 保护级别乘数 | 1.0-10.0 | 值越大搜索半径越大，候选越多 |
| `reverse_tolerance` | 允许的反向移动距离 (米) | 0.0-10.0 | 容忍GPS漂移或道路方向错误 |
| `normalized` | 归一化发射概率 | True | 归一化可以防止概率过小 |
| `use_mahalanobis_candidates` | 使用马氏距离搜索 | True | 考虑协方差形状，更准确 |
| `window_length` | 滑动窗口长度 | 10-100 | 计算可信度时考虑的历史点数 |
| `margin_used_trustworthiness` | 使用边际可信度 | True | True=用top1-top2, False=用top1 |
| `filtered` | 过滤无效点 | True | True=删除无候选点的点 |

---

## 算法类

### `CovarianceMapMatch` - CMM 算法

**构造函数：**
```python
cmm = CovarianceMapMatch(
    network: Network,           # 道路网络
    graph: NetworkGraph,        # 网络图
    ubodt: UBODT               # 预计算的最短路径表
)
```

**方法：**

#### 1. `match_traj(traj, config) -> MatchResult`

匹配单条轨迹。

**参数：**
- `traj`: `CMMTrajectory` - 待匹配的轨迹
- `config`: `CovarianceMapMatchConfig` - 算法配置

**返回：**
- `MatchResult`: 匹配结果

**示例：**
```python
result = cmm.match_traj(traj, config)
print(f"匹配成功: {result.is_matched()}")
print(f"匹配距离: {result.sp_dist} 米")
```

#### 2. `match_gps_file(gps_config, result_config, config, input_epsg, use_omp=True) -> str`

批量匹配GPS文件中的轨迹。

**参数：**
- `gps_config`: `GPSConfig` - GPS 输入配置
- `result_config`: `ResultConfig` - 结果输出配置
- `config`: `CovarianceMapMatchConfig` - 算法配置
- `input_epsg`: `int` - 输入轨迹的坐标系 EPSG 代码 (如 4326 for WGS84)
- `use_omp`: `bool` - 是否使用 OpenMP 并行

**返回：**
- `str`: 运行统计信息字符串

**示例：**
```python
gps_config = GPSConfig()
gps_config.file = "input_trajectories.csv"
gps_config.id = "id"
gps_config.geom = "geom"

result_config = ResultConfig()
result_config.file = "output_results.csv"

# 设置输出字段
result_config.output_config.write_candidates = True
result_config.output_config.write_cpath = True
result_config.output_config.write_mgeom = True

stats = cmm.match_gps_file(
    gps_config,
    result_config,
    config,
    input_epsg=4326,  # WGS84
    use_omp=True
)

print(stats)
```

---

## 结果类

### `MatchResult` - 匹配结果

包含地图匹配的完整结果。

**属性：**
```python
result.id                    # 轨迹 ID
result.opath                 # 每个点匹配的边ID列表
result.cpath                 # 完整路径（拓扑连通的边ID序列）
result.indices               # opath 在 cpath 中的索引
result.mgeom                 # 匹配路径的几何形状
result.opt_candidate_path    # 每个点匹配的候选点详细信息
result.candidate_details     # 所有候选点详细信息 (如果启用)
result.nbest_trustworthiness # Top-N 可信度分数
result.sp_distances          # 每个点的最短路径距离
result.eu_distances          # 每个点的欧氏距离
result.original_indices      # 过滤后点的原始索引
```

**方法：**

#### 1. `is_matched() -> bool`

检查是否成功匹配。

```python
if result.is_matched():
    print("匹配成功")
else:
    print("匹配失败")
```

#### 2. 访问匹配信息

```python
# 完整路径 (边ID列表)
print(f"匹配路径: {list(result.cpath)}")

# 每个点匹配的边
print(f"各点匹配边: {list(result.opath)}")

# 匹配距离
print(f"最短路径距离: {result.sp_dist} 米")
print(f"欧氏距离: {result.eu_dist} 米")

# 匹配比率
if result.sp_dist > 0:
    ratio = result.eu_dist / result.sp_dist
    print(f"匹配比率: {ratio:.4f}")
```

#### 3. 访问候选点详细信息

每个 `MatchedCandidate` 包含：
```python
for mc in result.opt_candidate_path:
    print(f"边ID: {mc.c.edge.id}")
    print(f"偏移: {mc.c.offset} 米")
    print(f"距离: {mc.c.dist} 米")
    print(f"发射概率: {mc.ep}")
    print(f"转移概率: {mc.tp}")
    print(f"累积概率: {mc.cumu_prob}")
    print(f"最短路径距离: {mc.sp_dist}")
    print(f"可信度: {mc.trustworthiness}")
```

---

## 辅助类

### `Network` - 道路网络

```python
network = Network(
    file: str,      # Shapefile 路径
    id_col: str,    # 边ID列名
    source_col: str, # 起点节点列名
    target_col: str  # 终点节点列名
)

# 方法
network.get_edge_count() -> int           # 获取边数量
network.get_node_count() -> int           # 获取节点数量
```

### `NetworkGraph` - 网络图

```python
graph = NetworkGraph(network)

# 方法
graph.get_num_vertices() -> int           # 获取顶点数
```

### `UBODT` - 上界起源目的地表

预计算的最短路径表，加速地图匹配。

```python
# 从内存映射文件加载
ubodt = UBODT.read_ubodt_mmap_binary(filename: str)

# 从普通二进制文件加载
ubodt = UBODT.read_ubodt_file(filename: str)

# 方法
ubodt.get_num_rows() -> int               # 获取行数
```

---

## 完整使用示例

### 示例 1: 基础地图匹配

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/path/to/build/python')

from fmm import *

# 1. 加载网络
network = Network("input/map/edges.shp", "id", "source", "target")
print(f"加载网络: {network.get_edge_count()} 条边")

# 2. 创建图
graph = NetworkGraph(network)
print(f"图顶点数: {graph.get_num_vertices()}")

# 3. 加载 UBODT
ubodt = UBODT.read_ubodt_mmap_binary("input/map/ubodt.bin")
print(f"加载 UBODT: {ubodt.get_num_rows()} 行")

# 4. 创建 CMM 算法
cmm = CovarianceMapMatch(network, graph, ubodt)

# 5. 配置算法
config = CovarianceMapMatchConfig(
    k_arg=8,
    min_candidates_arg=3,
    protection_level_multiplier_arg=1.0,
    reverse_tolerance=0.0,
    normalized_arg=True,
    use_mahalanobis_candidates_arg=True,
    window_length_arg=10,
    margin_used_trustworthiness_arg=True
)

# 6. 准备轨迹
geom = LineString()
geom.add_point(110.1975, 20.0145)
geom.add_point(110.1995, 20.0165)
geom.add_point(110.2015, 20.0185)

timestamps = DoubleVector()
for i in range(3):
    timestamps.append(float(i * 30))

covariances = CovarianceMatrixVector()
for i in range(3):
    cov = CovarianceMatrix()
    cov.sde = 2.0
    cov.sdn = 1.5
    cov.sdu = 3.0
    cov.sdne = 0.1
    cov.sdeu = 0.05
    cov.sdun = 0.08
    covariances.append(cov)

protection_levels = DoubleVector()
for i in range(3):
    protection_levels.append(5.0)

traj = CMMTrajectory()
traj.id = 1
traj.geom = geom
traj.timestamps = timestamps
traj.covariances = covariances
traj.protection_levels = protection_levels

# 7. 执行匹配
result = cmm.match_traj(traj, config)

# 8. 输出结果
print(f"\n匹配结果:")
print(f"  匹配成功: {result.is_matched()}")
print(f"  匹配距离: {result.sp_dist:.2f} 米")
print(f"  欧氏距离: {result.eu_dist:.2f} 米")
print(f"  匹配边数: {len(result.cpath)}")
print(f"  匹配路径: {list(result.cpath)}")

# 9. 详细候选信息
for i, mc in enumerate(result.opt_candidate_path):
    print(f"\n点 {i}:")
    print(f"  匹配边: {mc.c.edge.id}")
    print(f"  偏移: {mc.c.offset:.2f} 米")
    print(f"  距离: {mc.c.dist:.2f} 米")
    print(f"  发射概率: {mc.ep:.6e}")
    print(f"  转移概率: {mc.tp:.6e}")
    print(f"  可信度: {mc.trustworthiness:.6f}")
```

### 示例 2: 批量处理轨迹文件

```python
from fmm import *

# 初始化网络和算法
network = Network("input/map/edges.shp", "id", "source", "target")
graph = NetworkGraph(network)
ubodt = UBODT.read_ubodt_mmap_binary("input/map/ubodt.bin")
cmm = CovarianceMapMatch(network, graph, ubodt)

# 配置
config = CovarianceMapMatchConfig(
    k_arg=16,
    min_candidates_arg=1,
    protection_level_multiplier_arg=10.0,
    reverse_tolerance=0.0001,  # 约10米(度)
    normalized_arg=True,
    use_mahalanobis_candidates_arg=True,
    window_length_arg=100,
    margin_used_trustworthiness_arg=False
)

# GPS 输入配置
gps_config = GPSConfig()
gps_config.file = "input_trajectories.csv"
gps_config.id = "id"
gps_config.geom = "geom"
gps_config.timestamp = "timestamps"
gps_config.covariance = "covariances"
gps_config.protection_level = "protection_levels"

# 结果输出配置
result_config = ResultConfig()
result_config.file = "output_results.csv"

# 启用详细输出
result_config.output_config.write_candidates = True
result_config.output_config.write_cpath = True
result_config.output_config.write_opath = True
result_config.output_config.write_mgeom = True
result_config.output_config.write_pgeom = True
result_config.output_config.write_ep = True
result_config.output_config.write_tp = True
result_config.output_config.write_trustworthiness = True
result_config.output_config.write_sp_dist = True
result_config.output_config.write_eu_dist = True
result_config.output_config.write_timestamp = True

# 执行批量匹配
stats = cmm.match_gps_file(
    gps_config,
    result_config,
    config,
    input_epsg=4326,  # WGS84
    use_omp=True
)

print(stats)
```

---

## 接口限制

从 SWIG 接口文件 ([fmm.i](python/fmm.i)) 可以看出，以下方法**不会**暴露给 Python：

```cpp
// 这些方法被忽略，不在 Python API 中
- CovarianceMapMatchConfig::print()
- CovarianceMapMatchConfig::load_from_xml()
- CovarianceMapMatchConfig::load_from_arg()
- CovarianceMapMatchConfig::register_arg()
- CovarianceMapMatchConfig::register_help()
- Network::route2geometry()
- Network::get_edge()
```

---

## Python 模块构建

### 编译 Python 绑定

```bash
cd build
cmake .. -DBUILD_PYTHON=ON
make -j$(nproc)
```

### 使用 Python 绑定

```python
import sys
sys.path.insert(0, '/path/to/fmm_sjtugnc/build/python')

from fmm import *
```

---

## 参考文件

- SWIG 接口定义: [python/fmm.i](python/fmm.i)
- C++ 算法头文件: [src/mm/cmm/cmm_algorithm.hpp](src/mm/cmm/cmm_algorithm.hpp)
- Python 示例: [python/cmm_example.py](python/cmm_example.py)
- Python 测试: [python/test_cmm_python.py](python/test_cmm_python.py)

---

## 常见问题

### Q1: 如何调整候选点搜索半径？

A: 调整 `protection_level_multiplier` 参数：
```python
# 增大乘数，扩大搜索半径
config.protection_level_multiplier = 10.0

# 减小乘数，缩小搜索半径
config.protection_level_multiplier = 0.5
```

### Q2: 如何处理无候选点的轨迹？

A: 设置 `min_candidates` 和 `filtered` 参数：
```python
config.min_candidates = 1  # 至少保留1个候选
config.filtered = True      # 过滤无候选点的点
```

### Q3: 如何提高匹配准确率？

A: 调整以下参数：
```python
config.k = 16                           # 增加候选点数量
config.use_mahalanobis_candidates = True # 使用马氏距离
config.window_length = 100               # 增加可信度窗口
config.protection_level_multiplier = 10.0 # 扩大搜索范围
```

---

**文档版本:** 2025-01-26
**最后更新:** 2025-01-26
