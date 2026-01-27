# 坐标系重构 - 编译和测试指南

## 修改完成情况

### ✅ 已完成的所有修改

#### 1. CMM (Covariance Map Matching)
- ✅ `src/mm/cmm/cmm_app_config.hpp/cpp` - 配置类
- ✅ `src/mm/cmm/cmm_algorithm.hpp/cpp` - 算法类
- ✅ `src/mm/cmm/cmm_app.cpp` - 应用类
- ✅ `src/io/gps_reader.hpp/cpp` - GPS读取工具

#### 2. FMM (Fast Map Matching)
- ✅ `src/mm/fmm/fmm_app_config.hpp/cpp` - 配置类
- ✅ `src/mm/fmm/fmm_app.cpp` - 应用类
- ✅ `src/network/network.hpp/cpp` - 网络类（保持不变）

#### 3. 配置文件示例
- ✅ `input/config/cmm_config_wgs84.xml` - CMM WGS84示例
- ✅ `input/config/fmm_config_wgs84.xml` - FMM WGS84示例

#### 4. 文档
- ✅ `REFACTORING_SUMMARY.md` - 技术摘要
- ✅ `EPSG_MIGRATION_GUIDE.md` - 用户迁移指南

## 编译步骤

### 1. 清理旧构建
```bash
cd /home/dell/fmm_sjtugnc/build
make clean
```

### 2. 重新配置和编译
```bash
cmake ..
make -j4
```

**预期结果**: 编译应该成功，只有一些警告（关于deprecated函数）

### 3. 检查生成的文件
```bash
ls -lh python/fmm  # Python绑定应该重新生成
```

## 测试步骤

### 测试1：验证配置加载
```bash
# 测试CMM配置加载
./cmm --config ../input/config/cmm_config_wgs84.xml --help
```

**预期输出**：
```
---- CMM Application Configuration ----
Network config: ...
GPS config: ...
...
Input EPSG 4326
...
```

### 测试2：运行CMM匹配（WGS84输入）
```bash
./cmm --config ../input/config/cmm_config_wgs84.xml
```

**预期日志输出**：
```
[info] Input EPSG: 4326, Network EPSG: 4326, Reprojection needed: 0
[info] Input trajectory CRS set to EPSG:4326
```

### 测试3：运行FMM匹配（WGS84输入）
```bash
./fmm --config ../input/config/fmm_config_wgs84.xml
```

**预期日志输出**：
```
[info] Input EPSG: 4326, Network EPSG: 4326; no reprojection needed
```

### 测试4：检查结果
```bash
# 查看CMM结果
head -20 dataset_hainan_06/1.3/mr/cmm_results_wgs84.csv

# 查看FMM结果
head -20 dataset_hainan_06/1.3/mr/fmm_results_wgs84.csv
```

## 常见错误和解决方案

### 错误1：编译失败 - 'convert_to_projected' was not declared
**原因**: 某个文件还在使用旧的参数名
**解决**: 检查错误提示的文件，将 `convert_to_projected` 改为使用 `input_epsg`

### 错误2：运行时警告 - "convert_to_projected enabled but network CRS is not projected"
**原因**: 这是我们已经移除的旧逻辑
**解决**: 这个警告不应该再出现，因为新的逻辑不再使用这个标志

### 错误3：坐标系不匹配
**原因**: input_epsg设置错误
**解决**:
- WGS84经纬度: `<input_epsg>4326</input_epsg>`
- UTM Zone 49N: `<input_epsg>32649</input_epsg>`

## Python API使用

### CMM Python API（待自动更新）
由于我们修改了C++类，Python API会自动通过SWIG重新生成。

```python
from fmm import CovarianceMapMatch, CovarianceMapMatchConfig, NetworkGraph, Network, UBODT

# 注意：当前Python API使用旧的配置方式
# 需要等待 SWIG 重新生成绑定

# 网络和图
network = Network("input/map/hainan/edges.shp", "key", "u", "v", False)
graph = NetworkGraph(network)

# UBODT
ubodt = UBODT.read_ubodt_file("input/map/hainan/hainan_ubodt_indexed.bin")

# CMM配置
cmm_config = CovarianceMapMatchConfig(
    k_arg=16,
    min_candidates_arg=1,
    reverse_tolerance=0.1,  # 比例值，与单位无关！
    protection_level_multiplier_arg=10.0,
    normalized_arg=True,
    use_mahalanobis_candidates_arg=True,
    window_length_arg=100
)

# CMM算法
cmm = CovarianceMapMatch(network, graph, ubodt)

# 注意：当前没有input_epsg参数的Python API
# 需要等待SWIG重新生成绑定，或使用CLI方式
```

## 参数对照表

| 旧参数 | 新参数 | 说明 |
|--------|--------|------|
| `convert_to_projected=false` | `input_epsg=4326` | WGS84经纬度（不转换）|
| `convert_to_projected=true` | `input_epsg=32649` | UTM Zone 49N（可能转换）|
| - | `reverse_tolerance=0.1` | **重要**：现在是比例值，与单位无关！|

## 验证清单

使用以下命令验证修改：

```bash
# 1. 编译检查
cd /home/dell/fmm_sjtugnc/build
cmake .. 2>&1 | grep -i "error\|warn"
make -j4 2>&1 | grep -i "error"

# 2. 检查日志输出
./cmm --config ../input/config/cmm_config_wgs84.xml 2>&1 | grep -i "EPSG"

# 3. 对比新旧结果（如果有的话）
# 使用旧配置的结果 vs 新配置的结果应该相同
```

## 回退方案

如果需要回退到旧版本：

```bash
git diff HEAD~1 src/mm/cmm/cmm_app_config.hpp
git diff HEAD~1 src/mm/cmm/cmm_app_config.cpp
# ... 其他修改

git checkout HEAD~1 -- src/mm/cmm/
git checkout HEAD~1 -- src/mm/fmm/
make -j4
```

## 总结

✅ **所有C++代码修改已完成**
✅ **配置文件示例已创建**
✅ **文档已完善**

⏳ **需要**: 重新编译和测试
⏳ **可选**: Python绑定会自动更新（通过SWIG）

关键改进：**显式优于隐式** - 用户现在必须明确指定输入坐标系，程序自动判断是否需要转换。
