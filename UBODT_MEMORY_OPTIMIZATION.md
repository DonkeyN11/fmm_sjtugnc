# UBODT内存优化方案

## 概述

本优化方案为FMM (Fast Map Matching) 添加了UBODT (Upperbounded Origin Destination Table) 内存管理功能，避免重复读取大型UBODT文件，提高性能。

## 功能特性

### 1. UBODT内存缓存
- **单例模式**：全局UBODT内存管理器
- **LRU缓存**：最近最少使用淘汰策略
- **内存限制**：可设置最大内存使用量
- **线程安全**：多线程环境下安全访问

### 2. 智能范围检查
- **节点范围**：基于节点ID的范围检查
- **地理范围**：基于坐标的范围检查
- **动态范围**：支持范围扩展和更新

### 3. 内存安全保护
- **系统内存检查**：检查可用系统内存
- **进程限制检查**：检查进程内存限制
- **安全边距**：保留10%安全内存空间
- **溢出保护**：防止内存溢出错误

## 使用方法

### 1. 预加载UBODT到内存

```bash
# 基本用法
./ubodt_read --ubodt data/ubodt.bin

# 限制内存使用为2GB
./ubodt_read --ubodt data/ubodt.bin --max_memory 2048

# 查看缓存状态
./ubodt_read --status

# 清空缓存
./ubodt_read --clear

# 卸载特定UBODT文件
./ubodt_read --unload data/ubodt.bin
```

### 2. 运行FMM匹配

FMM会自动检查内存缓存中的UBODT：

```bash
# 正常运行FMM，自动使用内存缓存
./fmm --config config/fmm_config.xml

# 禁用内存缓存
./fmm --config config/fmm_config.xml --disable_cache
```

### 3. 编程接口

#### C++接口

```cpp
#include "mm/fmm/ubodt_memory_manager.hpp"
#include "mm/fmm/fmm_app.hpp"

// 获取内存管理器实例
auto& manager = FMM::MM::UBODTMemoryManager::get_instance();

// 预加载UBODT
bool success = manager.load_ubodt("data/ubodt.bin", 50000, 2048);

// 检查特定范围的UBODT
auto cached_ubodt = manager.get_ubodt_for_range(start_node, end_node);

// 创建FMM应用（自动使用缓存）
FMM::MM::FMMApp app(config);

// 或者使用预加载的UBODT
FMM::MM::FMMApp app(config, preloaded_ubodt);
```

## 配置选项

### ubodt_read命令选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--ubodt FILE` | UBODT文件路径 | 必需 |
| `--max_memory MB` | 最大内存使用量(MB) | 无限制 |
| `--multiplier N` | UBODT乘数 | 50000 |
| `--status` | 显示缓存状态 | false |
| `--clear` | 清空缓存 | false |
| `--unload FILE` | 卸载特定文件 | - |
| `--help` | 显示帮助信息 | false |

### FMM配置选项

在FMM配置中添加了以下选项：

```xml
<config>
    <use_memory_cache>true</use_memory_cache>  <!-- 启用内存缓存 -->
    <!-- 其他配置选项... -->
</config>
```

## 性能优化效果

### 1. 加载时间优化
- **首次加载**：与原版相同
- **后续加载**：几乎为零（内存缓存命中）

### 2. 内存使用优化
- **智能缓存**：只保留必要的UBODT文件
- **自动清理**：超过内存限制时自动清理
- **范围检查**：只加载匹配范围内的数据

### 3. 系统稳定性
- **内存保护**：防止内存溢出
- **安全边距**：保留系统运行所需内存
- **监控报告**：实时内存使用统计

## 实现细节

### 内存管理器架构

```cpp
class UBODTMemoryManager {
    // 单例实例
    static UBODTMemoryManager& get_instance();

    // 核心功能
    bool load_ubodt(const std::string& filename, ...);
    std::shared_ptr<CachedUBODT> get_ubodt_for_range(...);
    std::shared_ptr<CachedUBODT> get_ubodt_for_point(...);

    // 内存管理
    void cleanup_if_needed();
    bool check_memory_availability(size_t required_memory);

    // 缓存管理
    void clear_cache();
    bool unload_ubodt(const std::string& filename);
};
```

### 数据结构

```cpp
struct UBODTRange {
    double min_x, max_x;  // 地理范围
    double min_y, max_y;
    NodeIndex min_node, max_node;  // 节点范围
};

struct CachedUBODT {
    std::shared_ptr<UBODT> ubodt;  // UBODT数据
    UBODTRange range;              // 范围信息
    size_t memory_usage;           // 内存使用量
    std::chrono::system_clock::time_point last_access;  // 最后访问时间
};
```

## 注意事项

1. **内存限制**：设置合理的内存限制，避免系统内存不足
2. **文件路径**：确保UBODT文件路径正确，避免重复加载
3. **并发访问**：多线程环境下已保证线程安全
4. **监控**：定期检查内存使用情况，避免内存泄漏

## 故障排除

### 1. 内存不足错误
```
ERROR: Insufficient memory to load UBODT file
```
**解决方案**：
- 增加系统内存
- 调整内存限制参数
- 清理其他缓存数据

### 2. 范围检查失败
```
WARN: Cannot load range info for file
```
**解决方案**：
- 检查UBODT文件格式
- 确保文件完整性
- 重新生成UBODT文件

### 3. 缓存未命中
```
INFO: UBODT not found in cache, loading from file
```
**解决方案**：
- 确保已运行ubodt_read命令
- 检查文件路径是否匹配
- 查看缓存状态

## 扩展功能

### 1. 分布式缓存
- 支持多节点间共享UBODT缓存
- 网络传输优化

### 2. 持久化缓存
- 将缓存数据保存到磁盘
- 快速恢复缓存状态

### 3. 动态加载
- 按需加载UBODT数据块
- 智能预加载机制

---

**注意**：本优化方案需要重新编译FMM项目，请确保已正确更新CMakeLists.txt文件。