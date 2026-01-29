# 坐标系重构总结：从 convert_to_projected 到 input_epsg

## 已完成的修改

### 1. ✅ 配置头文件
- **src/mm/cmm/cmm_app_config.hpp**: `bool convert_to_projected` → `int input_epsg`
- **src/mm/fmm/fmm_app_config.hpp**: `bool convert_to_projected` → `int input_epsg` (默认4326)

### 2. ✅ 配置实现文件
- **src/mm/cmm/cmm_app_config.cpp**:
  - 构造函数初始化
  - `load_from_xml()`: 读取 `<input_epsg>` 参数
  - `load_from_arg()`: 读取 `--input_epsg` 参数
  - `register_arg()`: 注册命令行参数
  - `register_help()`: 更新帮助信息
  - `print()`: 输出EPSG代码

- **src/mm/fmm/fmm_app_config.cpp**: 同上

### 3. ✅ GPS读取工具
- **src/io/gps_reader.hpp**: 函数签名修改
- **src/io/gps_reader.cpp**: `compute_gps_bounds_in_network_crs()` 实现
  - 移除自动检测逻辑
  - 使用显式 `input_epsg` 参数
  - 简化坐标转换逻辑

### 4. ✅ CMM应用
- **src/mm/cmm/cmm_app.cpp**: Network构造函数使用 `false`（不转换网络）

## ⚠️ 剩余需要修改的部分

### 5. ⏳ CMM算法函数签名（重要）
需要修改 **src/mm/cmm/cmm_algorithm.hpp** 和 **src/mm/cmm/cmm_algorithm.cpp**:

```cpp
// 旧签名
std::string match_gps_file(
    const GPSConfig &gps_config,
    const ResultConfig &result_config,
    const CovarianceMapMatchConfig &config,
    bool convert_to_projected,  // ❌ 删除
    bool use_omp = true);

// 新签名
std::string match_gps_file(
    const GPSConfig &gps_config,
    const ResultConfig &result_config,
    const CovarianceMapMatchConfig &config,
    int input_epsg,  // ✅ 添加：输入轨迹的EPSG代码
    bool use_omp = true);
```

修改 `match_gps_file` 函数内部：
- 将所有 `convert_to_projected` 替换为使用 `input_epsg` 的逻辑
- `maybe_reproject_trajectories(&trajectories, network_, input_epsg)`

### 6. ⏳ CMM App run() 方法
**src/mm/cmm/cmm_app.cpp** 中的 `run()` 函数：

```cpp
// 旧代码
cmm_->match_gps_file(
    config_.gps_config,
    config_.result_config,
    config_.cmm_config,
    config_.convert_to_projected,  // ❌ 修改
    config_.use_omp);

// 新代码
cmm_->match_gps_file(
    config_.gps_config,
    config_.result_config,
    config_.cmm_config,
    config_.input_epsg,  // ✅ 修改
    config_.use_omp);
```

### 7. ⏳ FMM类似的修改
对 FMM 应用做类似的修改。

### 8. ⏳ XML配置文件
更新所有示例配置文件：
```xml
<!-- 旧配置 -->
<other>
  <convert_to_projected>false</convert_to_projected>
  ...
</other>

<!-- 新配置 -->
<other>
  <input_epsg>4326</input_epsg>  <!-- WGS84 经纬度 -->
  <!-- 或者 -->
  <input_epsg>32649</input_epsg>  <!-- UTM Zone 49N -->
  ...
</other>
```

### 9. ⏳ Python绑定
更新 Python API，添加 `input_epsg` 参数。

## 使用示例

### WGS84 经纬度坐标系
```xml
<config>
  ...
  <other>
    <input_epsg>4326</input_epsg>  <!-- WGS84 -->
    ...
  </other>
</config>
```

### UTM投影坐标系
```xml
<config>
  ...
  <other>
    <input_epsg>32649</input_epsg>  <!-- UTM Zone 49N -->
    ...
  </other>
</config>
```

## 编译和测试

```bash
cd build
cmake ..
make -j4

# 测试 WGS84 输入
./cmm --config ../input/config/cmm_config_omp_wgs84.xml

# 测试 UTM 输入
./cmm --config ../input/config/cmm_config_omp_utm.xml
```

## 重要说明

1. **网络数据保持原始坐标系**：不再自动转换网络数据到投影坐标系
2. **轨迹数据按需转换**：根据 `input_epsg` 和网络 EPSG，自动转换轨迹数据
3. **显式优于隐式**：用户必须明确指定输入轨迹的坐标系

## 常用EPSG代码

| EPSG代码 | 坐标系 | 说明 |
|---------|--------|------|
| 4326 | WGS84 | 经纬度（度）|
| 32601-32660 | UTM Zone 1N-60N | 北半球UTM |
| 32701-32760 | UTM Zone 1S-60S | 南半球UTM |
| 3851 | GCJ02 | 中国火星坐标 |
| 4490 | CGCS2000 | 中国大地坐标 |
