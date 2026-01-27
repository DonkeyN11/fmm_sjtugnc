# CMM/FMM 坐标系重构使用指南

## 概述

已完成从 `convert_to_projected` 到 `input_epsg` 的重构，用户现在需要显式指定输入轨迹的坐标系（EPSG代码），系统会自动判断是否需要转换。

## 主要变化

### 旧方式（自动判断）
```xml
<other>
  <convert_to_projected>false</convert_to_projected>
  <!-- 程序自动检测轨迹坐标系并决定是否转换 -->
</other>
```

### 新方式（显式指定）
```xml
<other>
  <input_epsg>4326</input_epsg>  <!-- 用户显式指定输入轨迹的EPSG代码 -->
  <!-- 程序比较 input_epsg 和网络 EPSG，自动判断是否需要转换 -->
</other>
```

## 已修改的文件

### 1. ✅ CMM配置
- `src/mm/cmm/cmm_app_config.hpp` - 头文件
- `src/mm/cmm/cmm_app_config.cpp` - 实现文件

### 2. ✅ FMM配置
- `src/mm/fmm/fmm_app_config.hpp` - 头文件
- `src/mm/fmm/fmm_app_config.cpp` - 实现文件

### 3. ✅ GPS读取工具
- `src/io/gps_reader.hpp` - 函数签名
- `src/io/gps_reader.cpp` - 实现

### 4. ✅ CMM算法
- `src/mm/cmm/cmm_algorithm.hpp` - 函数签名
- `src/mm/cmm/cmm_algorithm.cpp` - 实现
  - `match_gps_file()`
  - `maybe_reproject_trajectories()`

### 5. ✅ CMM应用
- `src/mm/cmm/cmm_app.cpp`

## 使用示例

### 示例1：WGS84 经纬度坐标系（最常用）

#### XML配置
```xml
<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network>
      <file>input/map/hainan/edges.shp</file>
      <!-- 网络数据应为 WGS84 坐标系 -->
    </network>

    <gps>
      <file>dataset/trajectory_wgs84.csv</file>
      <!-- 输入轨迹：WGS84 经纬度（度） -->
    </gps>
  </input>

  <other>
    <input_epsg>4326</input_epsg>  <!-- 关键参数：指定输入为WGS84 -->
  </other>
</config>
```

#### Python脚本
```python
from fmm import CovarianceMapMatch, CovarianceMapMatchConfig, NetworkGraph, Network, UBODT

network = Network("input/map/hainan/edges.shp", "key", "u", "v", False)
graph = NetworkGraph(network)
ubodt = UBODT.read_ubodt_file("input/map/hainan/hainan_ubodt.bin")

cmm_config = CovarianceMapMatchConfig(
    k_arg=16,
    min_candidates_arg=1,
    reverse_tolerance=0.1,  # 比例值，与单位无关
    # ... 其他参数
)

cmm = CovarianceMapMatch(network, graph, ubodt)

# 注意：Python API也需要添加 input_epsg 参数（待修改）
result = cmm.match_traj_file(
    "dataset/trajectory_wgs84.csv",
    "output/cmm_result.csv",
    cmm_config,
    input_epsg=4326,  # WGS84
    use_omp=True
)
```

### 示例2：UTM投影坐标系

#### XML配置
```xml
<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network>
      <file>input/map/edges_utm.shp</file>
      <!-- 网络数据：UTM Zone 49N -->
    </network>

    <gps>
      <file>dataset/trajectory_utm.csv</file>
      <!-- 输入轨迹：UTM Zone 49N (米) -->
    </gps>
  </input>

  <other>
    <input_epsg>32649</input_epsg>  <!-- UTM Zone 49N -->
  </other>
</config>
```

## 常用EPSG代码

| EPSG代码 | 坐标系 | 单位 | 适用场景 |
|---------|--------|------|---------|
| **4326** | **WGS84** | **度** | **GPS原始数据、经纬度坐标** |
| 32649 | UTM Zone 49N | 米 | 中国海南（约108-114°E） |
| 32650 | UTM Zone 50N | 米 | 中国东部（约114-120°E） |
| 32601-32660 | UTM Zone 1N-60N | 米 | 北半球UTM |
| 32701-32760 | UTM Zone 1S-60S | 米 | 南半球UTM |

## 参数说明

### input_epsg
- **类型**: 整数
- **默认值**: 4326 (WGS84)
- **说明**: 输入轨迹文件的EPSG代码
- **影响**: 决定是否需要将轨迹数据转换到网络坐标系

### reverse_tolerance
- **类型**: 浮点数
- **推荐值**: 0.1
- **说明**: 允许的反向移动比例（边长的百分比）
- **注意**: 这是比例值，与坐标系单位无关！

## 转换逻辑

### 自动判断流程
1. 读取输入轨迹的EPSG（通过 `input_epsg` 参数）
2. 读取网络数据的EPSG（从shapefile的.prj文件）
3. 比较两者：
   - **相同**: 不转换，直接使用原始坐标
   - **不同**: 自动转换轨迹数据到网络坐标系
4. 协方差矩阵通过雅可比变换自动转换
5. 保护等级按比例缩放

### 示例场景

#### 场景1：输入和网络都是WGS84
```xml
<input_epsg>4326</input_epsg>  <!-- 输入: WGS84 -->
<!-- 网络: WGS84 (从shapefile读取) -->
<!-- 结果：不转换，直接使用 -->
```

#### 场景2：输入是WGS84，网络是UTM
```xml
<input_epsg>4326</input_epsg>  <!-- 输入: WGS84 (度) -->
<!-- 网络: UTM Zone 49N (米) -->
<!-- 结果：自动转换度→米，并转换协方差 -->
```

#### 场景3：输入和网络都是UTM
```xml
<input_epsg>32649</input_epsg>  <!-- 输入: UTM Zone 49N -->
<!-- 网络: UTM Zone 49N -->
<!-- 结果：不转换，直接使用 -->
```

## 编译和测试

### 1. 清理旧构建
```bash
cd /home/dell/fmm_sjtugnc/build
make clean
```

### 2. 重新编译
```bash
cmake ..
make -j4
```

### 3. 测试WGS84输入
```bash
./cmm --config ../input/config/cmm_config_wgs84.xml
```

预期输出：
```
[info] Input EPSG: 4326, Network EPSG: 4326, Reprojection needed: 0
[info] Input trajectory CRS set to EPSG:4326
```

### 4. 查看结果
```bash
head -20 dataset_hainan_06/1.3/mr/cmm_results_wgs84.csv
```

## 注意事项

### ⚠️ 重要提示
1. **网络和轨迹必须匹配**: 如果网络是WGS84，输入也必须是WGS84（或兼容的坐标系）
2. **UTM区域要正确**: 中国大部分地区在UTM Zone 48-50N
3. **reverse_tolerance是比例值**: 0.1表示10%，无论坐标系单位

### 🔧 调试技巧
启用详细日志查看坐标转换：
```xml
<log_level>1</log_level>  <!-- debug级别 -->
```

## 迁移指南

### 旧配置
```xml
<convert_to_projected>false</convert_to_projected>
```

### 新配置
```xml
<input_epsg>4326</input_epsg>
```

### 确定你的EPSG代码
1. 查看shapefile的.prj文件
2. 使用 `ogrinfo` 命令:
   ```bash
   ogrinfo your_network.shp
   ```
3. WGS84经纬度: EPSG:4326
4. 中国UTM: EPSG:32648-32650

## 常见问题

### Q: 如何判断应该使用什么EPSG？
**A**:
- GPS原始数据（度）：4326
- 已投影的米单位数据：查看数据源文档
- 中国UTM：根据经度确定 Zone

### Q: reverse_tolerance 应该如何设置？
**A**: 推荐使用 0.1（10%），这是比例值，与单位无关。

### Q: 程序会自动转换网络数据吗？
**A**: 不会。网络数据保持原始坐标系，只转换轨迹数据。

### Q: 如何验证配置是否正确？
**A**: 查看日志输出中的 "Input EPSG" 和 "Network EPSG"。

## 下一步

### 待完成的工作
1. ⏳ FMM应用需要类似修改
2. ⏳ Python绑定需要更新
3. ⏳ 单元测试需要更新

### 需要帮助？
查看完整文档：`REFACTORING_SUMMARY.md`
