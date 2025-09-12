# FMM (Fast Map Matching) 源代码架构说明

## 概述

FMM是一个基于隐马尔可夫模型(HMM)的快速地图匹配库，支持多种地图匹配算法，包括FMM、H3MM和STMATCH。本文档详细介绍了src目录下代码的组织架构、各模块功能以及输入输出关系。

## 目录结构

```
src/
├── algorithm/          # 几何算法工具
├── app/               # 应用程序入口
├── config/            # 配置管理
├── core/              # 核心数据类型
├── io/                # 输入输出处理
├── mm/                # 地图匹配算法
│   ├── fmm/           # FMM算法实现
│   ├── h3mm/          # H3MM算法实现
│   └── stmatch/       # STMATCH算法实现
├── network/           # 道路网络处理
├── python/            # Python API接口
└── util/              # 工具函数
```

## 核心模块详解

### 1. algorithm/ - 几何算法模块

**文件:**
- `geom_algorithm.hpp/cpp` - 几何计算算法

**功能:**
- 计算欧几里得距离
- 几何图形操作
- 投影计算

**输入输出:**
- 输入: 几何图形(点、线、多边形)
- 输出: 距离、投影点、几何变换结果

### 2. core/ - 核心数据类型

**文件:**
- `geometry.hpp` - 几何类型定义
- `gps.hpp` - GPS数据类型定义

**功能:**
- 定义基础几何类型：Point, LineString
- 定义GPS轨迹数据结构
- 提供几何操作和类型转换

**核心类型:**
```cpp
struct Point { double x, y; };
struct LineString { std::vector<Point> points; };
struct Trajectory { int id; LineString geom; std::vector<double> timestamps; };
```

### 3. network/ - 道路网络模块

**文件:**
- `network.hpp` - 道路网络核心类
- `network_graph.hpp` - 网络图结构
- `graph.hpp` - 图算法基础
- `bidirectional_network_graph.hpp` - 双向图
- `rtree.hpp` - 空间索引
- `type.hpp` - 网络类型定义

**功能:**
- 道路网络数据结构管理
- 空间索引构建(R-tree)
- 图算法支持(最短路径等)
- 候选点搜索

**输入输出:**
- 输入: Shapefile格式的道路网络文件
- 输出: 网络图结构、候选点、路径几何

**关键方法:**
```cpp
// 候选点搜索
Traj_Candidates search_tr_cs_knn(const LineString &geom, size_t k, double radius);

// 网络距离计算
double get_sp_dist(const Candidate *ca, const Candidate *cb);
```

### 4. mm/ - 地图匹配算法模块

#### 4.1 mm_type.hpp - 地图匹配类型定义

**核心类型:**
```cpp
struct Candidate {                    // 候选点
    NodeIndex index;                   // 候选点索引
    double offset;                     // 偏移距离
    double dist;                       // GPS误差距离
    Edge *edge;                        // 候选边
    Point point;                       // 投影点
};

struct MatchResult {                   // 匹配结果
    int id;                           // 轨迹ID
    MatchedCandidatePath opt_candidate_path; // 最优候选点路径
    O_Path opath;                     // 观察路径(边ID序列)
    C_Path cpath;                     // 完整路径(拓扑连接的边序列)
    std::vector<int> indices;         // 索引映射
    LineString mgeom;                 // 匹配路径几何
};
```

#### 4.2 transition_graph.hpp - 转移图实现

**功能:**
- 实现隐马尔可夫模型的转移图
- 计算观测概率和转移概率
- Viterbi算法最优路径推断

**关键方法:**
```cpp
// 观测概率计算
static double calc_ep(double dist, double error);

// 转移概率计算  
static double calc_tp(double sp_dist, double eu_dist);

// 最优路径回溯
TGOpath backtrack();
```

#### 4.3 fmm/ - FMM算法实现

**文件:**
- `fmm_algorithm.hpp/cpp` - FMM核心算法
- `ubodt.hpp` - 上界最短路径距离表
- `fmm_app.hpp` - FMM应用接口

**算法流程:**
1. **候选点搜索** - 为每个GPS点查找k个最近候选道路段
2. **转移图构建** - 基于候选点和GPS误差创建HMM模型
3. **概率更新** - 计算转移概率和观测概率，更新节点累积概率
4. **最优路径推断** - 使用Viterbi算法回溯找到最优路径
5. **路径补全** - 利用UBODT预计算表补全完整路径

**输入输出:**
- 输入: GPS轨迹、道路网络、算法配置
- 输出: 匹配结果(包含最优路径、几何形状、概率信息)

### 5. io/ - 输入输出模块

#### 5.1 gps_reader.hpp - GPS数据读取

**支持的格式:**
- **Shapefile**: GDAL支持的矢量格式
- **CSV轨迹**: 每行一个轨迹(WKT格式)
- **CSV点**: 每行一个GPS点，按ID和时间戳分组

**读取器类:**
- `GDALTrajectoryReader` - Shapefile轨迹读取
- `CSVTrajectoryReader` - CSV轨迹读取  
- `CSVPointReader` - CSV点数据读取
- `GPSReader` - 统一读取接口

#### 5.2 mm_writer.hpp - 匹配结果写入

**功能:**
- 将匹配结果写入CSV文件
- 支持可配置的输出字段

**输出字段:**
- 轨迹ID、原始几何、匹配几何
- 边ID序列、完整路径、候选点信息
- 概率值、距离误差等

### 6. config/ - 配置管理模块

**文件:**
- `gps_config.hpp` - GPS数据配置
- `network_config.hpp` - 网络数据配置  
- `result_config.hpp` - 结果输出配置

**功能:**
- 统一管理各种配置参数
- 支持命令行参数和XML配置文件
- 配置验证和默认值处理

### 7. app/ - 应用程序入口

**文件:**
- `fmm.cpp` - FMM主程序
- `h3mm.cpp` - H3MM主程序
- `stmatch.cpp` - STMATCH主程序
- `ubodt_gen_app.cpp` - UBODT生成程序

**功能:**
- 命令行接口实现
- 参数解析和验证
- 算法执行和结果输出

### 8. python/ - Python API接口

**文件:**
- `pyfmm.hpp` - Python绑定定义

**功能:**
- 提供Python API接口
- 支持轨迹匹配和结果查询

### 9. util/ - 工具函数

**文件:**
- `util.hpp` - 通用工具函数
- `debug.hpp` - 调试工具

**功能:**
- 时间测量、字符串处理
- 日志输出、调试信息

## 数据流和依赖关系

### 主要数据流

```
输入数据 → 读取器 → 网络处理 → 地图匹配算法 → 结果写入 → 输出文件
    ↓         ↓         ↓              ↓           ↓
  GPS文件 → Trajectory → Candidate → MatchResult → CSV/Shapefile
```

### 模块依赖关系

```
app/
  ↓
config/ ←→ io/
  ↓         ↓
mm/ ←→ network/
  ↓         ↓
core/ ←→ algorithm/
  ↓
util/
```

## 典型使用流程

### 1. 准备阶段
```cpp
// 1. 加载道路网络
Network network("network.shp", "id", "source", "target");

// 2. 构建网络图
NetworkGraph graph(network);

// 3. 生成UBODT(可选)
UBODT ubodt = UBODT::load("ubodt.txt");
```

### 2. 配置阶段
```cpp
// 1. 创建GPS读取器
GPSReader gps_reader(gps_config);

// 2. 配置匹配算法
FastMapMatchConfig config(8, 300.0, 50.0, 0.0);
```

### 3. 执行阶段
```cpp
// 1. 创建匹配器
FastMapMatch fmm(network, graph, ubodt);

// 2. 读取轨迹
Trajectory traj = gps_reader.read_next_trajectory();

// 3. 执行匹配
MatchResult result = fmm.match_traj(traj, config);
```

### 4. 输出阶段
```cpp
// 1. 创建写入器
CSVMatchResultWriter writer(output_file, output_config);

// 2. 写入结果
writer.write_result(traj, result);
```

## 输入输出格式

### 输入格式

1. **道路网络**: ESRI Shapefile
   - 必需字段: id, source, target
   - 几何类型: LineString

2. **GPS轨迹数据**:
   - **Shapefile**: 每个要素一个轨迹(LineString)
   - **CSV轨迹**: id;geom;timestamp（geom是经纬度坐标；timestamp）
   - **CSV点**: id;x;y;timestamp(需按id和时间排序)

### 输出格式

**CSV匹配结果**:
- `id`: 轨迹ID
- `opath`: 观察路径(边ID序列)
- `cpath`: 完整路径(边ID序列)  
- `mgeom`: 匹配路径几何(WKT)
- `error`: 匹配误差统计
- 其他可配置字段

## 完整输入输出流总结

### 输入数据格式和来源

**轨迹数据输入格式：**
- **ESRI Shapefile格式**：每个要素包含一个LineString几何体表示轨迹
- **CSV轨迹格式**：包含id;geom;timestamp字段，geom为WKT格式
- **CSV点格式**：包含id;x;y;timestamp字段，每个点一行，按id和timestamp排序

**网络数据输入格式：**
- **ESRI Shapefile**：包含道路网络的边信息，每条边有id、source、target字段
- **网络数据结构**：Edge(边)、Node(节点)、空间索引(R-tree)

### 数据结构和转换过程

**核心数据结构转换链：**

```
原始输入 → CORE::Trajectory → MM::Candidate → TransitionGraph → MatchResult → 输出
```

**详细转换过程：**

1. **输入解析阶段**：
   - Shapefile/CSV → `CORE::Trajectory` (包含id, LineString几何体, timestamps)
   - LineString使用Boost.Geometry存储，支持WKT和GeoJSON格式

2. **网络处理阶段**：
   - 道路网络加载到`NETWORK::Network`类
   - 构建R-tree空间索引用于快速查询
   - 边和节点映射：EdgeID ↔ EdgeIndex, NodeID ↔ NodeIndex

3. **候选点生成阶段**：
   - `search_tr_cs_knn()`函数为每个GPS点查找k个最近邻候选边
   - 生成`MM::Candidate`结构：包含edge指针、offset距离、dist误差、point几何体

4. **地图匹配算法阶段**：
   - 构建TransitionGraph实现隐马尔可夫模型
   - 计算发射概率(ep)：基于GPS误差和候选点距离
   - 计算转移概率(tp)：基于最短路径距离和欧氏距离比值
   - 使用Viterbi算法寻找最优路径

5. **结果生成阶段**：
   - 生成`MM::MatchResult`结构，包含：
     - `opath`：每个GPS点匹配的边ID序列
     - `cpath`：完整路径的边ID序列
     - `indices`：opath在cpath中的索引位置
     - `mgeom`：匹配路径的LineString几何体
     - `opt_candidate_path`：详细的匹配候选点信息

### 输出格式和内容

**输出配置选项**：
- `write_opath`：输出每个点匹配的边ID
- `write_cpath`：输出完整路径的边ID序列  
- `write_mgeom`：输出匹配路径的几何体(WKT格式)
- `write_offset`：输出匹配点到边起点的偏移距离
- `write_error`：输出GPS点到匹配点的误差距离
- `write_spdist`：输出点间最短路径距离
- `write_ep`：输出发射概率
- `write_tp`：输出转移概率

**输出格式**：
- **CSV格式**：分隔符为';'，包含配置的字段
- **几何体输出**：WKT格式(如LINESTRING(1 0,1 1,2 1))

### 关键算法处理

**距离计算改进**：
- 原始欧氏距离已改为大圆距离(Haversine公式)
- 支持WGS84坐标系下的精确地理距离计算

**空间索引优化**：
- 使用Boost.Geometry的R-tree进行快速空间查询
- 支持KNN查询和半径查询

**并行处理**：
- 支持OpenMP多线程并行处理多条轨迹
- UBODT预计算加速最短路径查询

### 数据流示例

```
输入：CSV轨迹文件 → GPSReader → Trajectory(id=1, geom=LineString, timestamps=[1,2,3])
     ↓
网络：Shapefile道路网 → Network(带R-tree索引)
     ↓  
候选点：search_tr_cs_knn() → Traj_Candidates[[Candidate1,Candidate2],...]
     ↓
地图匹配：TransitionGraph + Viterbi → MatchResult(opath, cpath, mgeom)
     ↓
输出：CSVMatchResultWriter → 输出文件(id;cpath;mgeom;...)
```

## 扩展和定制

### 添加新算法
1. 在`mm/`下创建新目录
2. 实现算法类和配置类
3. 在`app/`添加对应的主程序
4. 更新CMake构建配置

### 添加新数据格式
1. 在`io/`中扩展对应的读写器
2. 在`config/`中添加配置支持
3. 更新GPSReader统一接口

## 性能优化

- **空间索引**: 使用R-tree加速候选点搜索
- **并行处理**: 支持OpenMP多线程处理
- **预计算**: UBODT表预计算最短路径距离
- **内存管理**: 智能指针和对象池优化

## 总结

FMM采用模块化设计，各模块职责明确，接口清晰。核心算法与数据存储、输入输出分离，便于扩展和维护。支持多种数据格式和算法实现，是一个功能完整、性能优秀的地图匹配库。