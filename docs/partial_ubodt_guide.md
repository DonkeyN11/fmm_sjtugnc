# Partial UBODT 动态加载功能

## 概述

Partial UBODT 是一个新的优化功能，可以根据轨迹范围动态加载只相关的 UBODT 记录，显著减少：
- **内存占用**：只加载需要的记录
- **加载时间**：减少文件读取量
- **查询性能**：更小的哈希表，更好的缓存局部性

## 功能特点

### 1. 基于轨迹范围的自动加载
自动计算轨迹的边界框，只加载边界框内的网络节点相关的 UBODT 记录。

### 2. 可配置的缓冲区
支持在轨迹边界框周围添加缓冲区，确保匹配过程有足够的路径选择。

### 3. 内存映射加速
使用内存映射技术快速定位和读取相关记录。

### 4. 兼容现有代码
返回的 `PartialUBODT` 对象提供与 `UBODT` 相同的接口，可以无缝替换。

## 使用方法

### 方法一：从轨迹自动加载（推荐）

```cpp
#include "mm/fmm/ubodt_partial.hpp"
#include "network/network.hpp"
#include "io/gps_reader.hpp"

using namespace FMM;
using namespace FMM::MM;

// 1. 加载路网
CONFIG::NetworkConfig network_config("network.shp", "id", "source", "target");
NETWORK::Network network(network_config);

// 2. 读取轨迹
std::vector<CORE::Trajectory> trajectories;
IO::TrajectoryReader reader("trjectories.csv", "id", "geom");
while (reader.has_next_trajectory()) {
    trajectories.push_back(reader.read_next_trajectory());
}

// 3. 创建 PartialUBODT（自动计算边界框）
auto partial_ubodt = make_partial_ubodt_from_trajectories(
    "ubodt.bin",              // UBODT 文件路径
    network,                  // 路网对象
    trajectories,             // 轨迹向量
    0.1                       // 缓冲区比例（10%）
);

// 4. 使用（与普通 UBODT 相同）
const Record *rec = partial_ubodt->look_up(source_node, target_node);
```

### 方法二：从指定节点集合加载

```cpp
#include "mm/fmm/ubodt_partial.hpp"

using namespace FMM;
using namespace FMM::MM;

// 1. 定义需要的节点集合
std::unordered_set<NETWORK::NodeIndex> required_nodes;
required_nodes.insert(100);
required_nodes.insert(200);
required_nodes.insert(300);
// ... 添加更多节点

// 2. 创建 PartialUBODT
auto partial_ubodt = make_partial_ubodt_from_nodes(
    "ubodt.bin",
    network,
    required_nodes
);
```

### 方法三：在 FMM/CMM 算法中使用

```cpp
#include "mm/fmm/fmm_algorithm.hpp"
#include "mm/fmm/ubodt_partial.hpp"

// 创建 PartialUBODT
auto partial_ubodt = make_partial_ubodt_from_trajectories(
    "ubodt.bin", network, trajectories, 0.1
);

// 使用 PartialUBODT 初始化 FMM 算法
MM::FMMAlgorithm fmm_algo(network, partial_ubodt->get_ubodt());

// 执行地图匹配
auto result = fmm_algo.match_traj(trajectory, config);
```

## API 参考

### PartialUBODT 类

#### 构造函数

```cpp
// 从节点集合加载
PartialUBODT(
    const std::string &filename,                      // UBODT 文件路径
    const NETWORK::Network &network,                  // 路网对象
    const std::unordered_set<NodeIndex> &nodes        // 需要的节点集合
);

// 从轨迹向量加载
PartialUBODT(
    const std::string &filename,                      // UBODT 文件路径
    const NETWORK::Network &network,                  // 路网对象
    const std::vector<CORE::Trajectory> &trajectories, // 轨迹向量
    double buffer_ratio = 0.1                         // 缓冲区比例
);
```

#### 主要方法

```cpp
// 查找记录
const Record *look_up(NodeIndex source, NodeIndex target) const;

// 查找最短路径
std::vector<EdgeIndex> look_sp_path(NodeIndex source, NodeIndex target) const;

// 获取统计信息
size_t get_num_records() const;    // 已加载的记录数
size_t get_num_sources() const;    // 已加载的源节点数
bool is_valid() const;             // 是否成功加载

// 获取底层 UBODT 指针
std::shared_ptr<UBODT> get_ubodt() const;
```

#### 静态方法

```cpp
// 计算轨迹边界框
static boost::geometry::model::box<Point> calculate_trajectories_bbox(
    const std::vector<Trajectory> &trajectories
);

// 从边界框提取节点
static std::unordered_set<NodeIndex> extract_nodes_in_bbox(
    const Network &network,
    const boost::geometry::model::box<Point> &bbox,
    double buffer_ratio = 0.1
);
```

### 工厂函数

```cpp
// 从节点集合创建
std::shared_ptr<PartialUBODT> make_partial_ubodt_from_nodes(
    const std::string &filename,
    const Network &network,
    const std::unordered_set<NodeIndex> &required_nodes
);

// 从轨迹向量创建
std::shared_ptr<PartialUBODT> make_partial_ubodt_from_trajectories(
    const std::string &filename,
    const Network &network,
    const std::vector<Trajectory> &trajectories,
    double buffer_ratio = 0.1
);
```

## 性能优势

### 内存节省示例

假设：
- 路网总节点数：100,000
- 轨迹覆盖区域节点数：10,000
- UBODT 总记录数：50,000,000
- 平均每源节点 500 条记录

**传统方式：**
- 加载所有记录：50,000,000 条
- 内存占用：~1.8 GB

**Partial UBODT：**
- 加载 10,000 个源节点的记录：5,000,000 条
- 内存占用：~180 MB
- **节省 90% 内存**

### 加速比示例

| 场景 | 全量加载 | Partial加载 | 加速比 |
|------|---------|-------------|--------|
| 小范围轨迹 | 10s | 1s | 10x |
| 中等范围轨迹 | 10s | 3s | 3.3x |
| 大范围轨迹 | 10s | 8s | 1.25x |

## 最佳实践

### 1. 选择合适的缓冲区比例

```cpp
// 城市环境（路网密集）：较小缓冲区
double buffer_ratio = 0.05;  // 5%

// 高速公路环境（路网稀疏）：较大缓冲区
double buffer_ratio = 0.2;   // 20%

// 不确定时：使用默认值
double buffer_ratio = 0.1;   // 10%
```

### 2. 批量处理轨迹

如果有大量轨迹，建议分批处理：

```cpp
std::vector<Trajectory> batch;
for (int i = 0; i < all_trajectories.size(); i += 1000) {
    // 创建批次
    int end = std::min(i + 1000, (int)all_trajectories.size());
    batch.assign(all_trajectories.begin() + i,
                 all_trajectories.begin() + end);

    // 为该批次创建 PartialUBODT
    auto partial_ubodt = make_partial_ubodt_from_trajectories(
        ubodt_file, network, batch, 0.1
    );

    // 处理该批次
    process_batch(batch, partial_ubodt);
}
```

### 3. 复用 PartialUBODT

如果多个轨迹在同一区域，可以复用同一个 PartialUBODT：

```cpp
// 为一组轨迹创建 PartialUBODT
auto partial_ubodt = make_partial_ubodt_from_trajectories(
    ubodt_file, network, trajectories_group, 0.1
);

// 复用处理多条轨迹
for (const auto &traj : trajectories_group) {
    auto result = fmm_algo.match_traj(traj, config);
    // ...
}
```

## 注意事项

1. **边界框计算**：确保轨迹坐标与路网坐标系一致
2. **缓冲区大小**：太小可能导致路径不完整，太大影响性能
3. **文件格式**：支持所有 UBODT 二进制格式（.bin）
4. **线程安全**：PartialUBODT 对象不是线程安全的，多线程需各自创建

## 编译示例

```bash
# 编译示例程序
cd /home/dell/fmm_sjtugnc
g++ -std=c++17 -O3 \
    example/partial_ubodt_example.cpp \
    -o build/partial_ubodt_example \
    -I./src \
    -L./build \
    -lFMMLIB \
    $(pkg-config --cflags --libs gdal boost)

# 运行示例
./build/partial_ubodt_example \
    data/network.shp \
    data/ubodt.bin \
    data/trajectories.csv
```

## 未来改进

- [ ] 支持增量加载（动态扩展节点集合）
- [ ] 自动调整缓冲区大小
- [ ] 多线程加载优化
- [ ] LRU 缓存机制
- [ ] 压缩存储支持

## 反馈与贡献

如有问题或建议，请提交 Issue 或 Pull Request。
