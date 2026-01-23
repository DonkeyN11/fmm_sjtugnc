# UBODT Manager 使用指南

## 功能概述

UBODT Manager 是一个新增的命令行工具，用于在内存中持久化存储UBODT（Upper Bounded Origin-Destination Table）数据，以便在多次FMM/CMM操作中重复使用，显著减少加载时间。

## 主要特性

- **持久化存储**：UBODT文件加载后将一直保存在内存中，直到显式释放
- **自动复用**：FMM和CMM应用会自动检查内存中是否有所需的UBODT，如果有则直接使用
- **智能管理**：如果内存中没有所需的UBODT，应用会自动加载并在运行结束后释放
- **多格式支持**：支持.csv、.txt、.bin等格式的UBODT文件

## 命令行工具使用

### 1. 加载UBODT到内存

```bash
# 基本用法
./build/ubodt_manager load data/ubodt.bin

# 使用更大的哈希表（提高查询性能）
./build/ubodt_manager load data/ubodt.bin --multiplier 2

# 详细输出
./build/ubodt_manager load data/ubodt.bin --verbose
```

**输出示例：**
```
✓ UBODT loaded successfully!
  File: data/ubodt.bin
  Rows: 1234567
  Load time: 2.34s
  Status: Kept in memory for reuse

The UBODT will now be automatically used by FMM and CMM applications.
Use 'ubodt_manager status' to view loaded UBODTs.
Use 'ubodt_manager release data/ubodt.bin' to release it from memory.
```

### 2. 查看已加载的UBODT状态

```bash
./build/ubodt_manager status
```

**输出示例：**
```
========== UBODT Manager Status ==========
Total UBODTs loaded: 1
Total references: 1
Estimated memory: 456.78 MB
Auto-release: disabled
==========================================

Loaded UBODTs will be automatically used by FMM and CMM.
Use 'ubodt_manager release <file>' or 'release_all' to free memory.
```

### 3. 释放特定的UBODT

```bash
./build/ubodt_manager release data/ubodt.bin
```

**输出示例：**
```
✓ Released 1 UBODT instance(s)
  File: data/ubodt.bin
```

### 4. 释放所有UBODT

```bash
./build/ubodt_manager release_all
```

**输出示例：**
```
✓ Released 2 UBODT instance(s) from memory
```

## 与FMM/CMM应用集成

### 使用场景1：预先加载UBODT（推荐用于批量处理）

```bash
# 1. 先加载UBODT到内存
./build/ubodt_manager load data/ubodt.bin

# 2. 运行多次FMM/CMM，都会自动使用内存中的UBODT
./build/fmm --config config1.xml
./build/fmm --config config2.xml
./build/fmm --config config3.xml

# 3. 处理完成后释放UBODT
./build/ubodt_manager release data/ubodt.bin
```

**优势：**
- 只需加载一次UBODT
- 后续运行FMM/CMM时无需等待加载时间
- 适合批量处理多个轨迹文件

### 使用场景2：让应用自动管理UBODT（默认行为）

```bash
# 直接运行FMM/CMM，应用会自动加载和释放UBODT
./build/fmm --config config.xml
```

**行为：**
- 如果内存中已有UBODT，会直接使用（即使之前通过ubodt_manager加载）
- 如果内存中没有，会加载UBODT，运行结束后自动释放
- 这种方式适合单次运行或不需要内存缓存的情况

### 禁用内存缓存

如果不想使用内存缓存功能（即使内存中有UBODT），可以通过修改配置来禁用。

**对于FMM：**
在FMMAppConfig中设置：
```cpp
bool use_memory_cache = false;
```

**对于CMM：**
在CMMAppConfig中设置：
```cpp
bool use_memory_cache = false;
```

## 实现细节

### 核心组件

1. **UBODTManager** (src/mm/fmm/ubodt_manager.hpp)
   - 单例模式管理所有UBODT实例
   - 线程安全
   - 支持Full UBODT、PartialUBODT和CachedUBODT

2. **ubodt_manager工具** (src/app/ubodt_manager_app.cpp)
   - 命令行接口
   - 提供load、release、release_all、status命令

3. **FMM/CMM集成**
   - FMMApp和CMMApp构造函数中检查UBODT管理器
   - 自动使用缓存的UBODT（如果存在）

### 关键文件

```
src/mm/fmm/ubodt_manager.hpp          - UBODT管理器头文件
src/mm/fmm/ubodt_manager.cpp          - UBODT管理器实现
src/mm/fmm/ubodt_manage_app.hpp       - 应用类头文件
src/mm/fmm/ubodt_manage_app.cpp       - 应用类实现
src/mm/fmm/ubodt_manage_app_config.hpp - 配置类头文件
src/mm/fmm/ubodt_manage_app_config.cpp - 配置类实现
src/app/ubodt_manager_app.cpp         - 命令行工具主程序
src/mm/fmm/fmm_app.cpp                - FMM应用（已修改）
src/mm/cmm/cmm_app.cpp                - CMM应用（已修改）
src/mm/fmm/fmm_app_config.hpp         - FMM配置（已包含use_memory_cache）
src/mm/cmm/cmm_app_config.hpp         - CMM配置（已添加use_memory_cache）
```

## 性能优势

### 场景：处理100个轨迹文件

**传统方式：**
```
每次运行: 加载UBODT(5s) + 处理(10s) = 15s
100次总计: 100 × 15s = 1500s (25分钟)
```

**使用UBODT Manager：**
```
第一次: 加载UBODT(5s) + 处理(10s) = 15s
后续99次: 处理(10s) = 990s
总计: 15s + 990s = 1005s (16.75分钟)
```

**节省时间：495秒 (8.25分钟)** - 约33%的性能提升！

## 注意事项

1. **内存占用**：UBODT文件会占用内存，大型UBODT（如几GB）需要确保有足够的系统内存
2. **持久性**：使用`ubodt_manager load`加载的UBODT会一直保留在内存中，直到：
   - 程序退出
   - 显式使用`release`或`release_all`命令
3. **线程安全**：UBODTManager是线程安全的，可以在多线程环境中使用
4. **自动释放**：当FMM/CMM使用缓存的UBODT时，会启用自动释放模式

## 编译

UBODT Manager已集成到CMake构建系统中：

```bash
cd build
cmake ..
make ubodt_manager
```

编译成功后，可执行文件位于：`build/ubodt_manager`

## 示例工作流

```bash
# 1. 查看当前状态（应该为空）
./build/ubodt_manager status

# 2. 加载UBODT
./build/ubodt_manager load data/ubodt.bin

# 3. 再次查看状态
./build/ubodt_manager status

# 4. 运行FMM匹配（会自动使用内存中的UBODT）
./build/fmm --network data/network.shp \
            --gps data/trjectories.csv \
            --ubodt data/ubodt.bin \
            --result output.csv

# 5. 运行更多FMM匹配（无需重新加载UBODT）
./build/fmm --network data/network.shp \
            --gps data/trjectories2.csv \
            --ubodt data/ubodt.bin \
            --result output2.csv

# 6. 完成后释放UBODT
./build/ubodt_manager release data/ubodt.bin

# 7. 或者释放所有UBODT
./build/ubodt_manager release_all
```

## 技术支持

如有问题或建议，请查看项目文档或提交issue。
