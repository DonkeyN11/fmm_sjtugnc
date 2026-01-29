# UBODT 管理器使用指南

## 概述

UBODT Manager 是一个单例类，用于缓存和管理 UBODT 实例，避免重复加载，显著提升批量处理的效率。

---

## 核心功能

### 1. 持久化UBODT存储
- 加载一次，多次使用
- 自动引用计数
- 线程安全操作
- 手动释放控制

### 2. 支持多种UBODT类型
- **Full UBODT** - 完整加载
- **PartialUBODT** - 部分加载（基于轨迹范围）
- **CachedUBODT** - 带LRU缓存的UBODT

---

## 快速开始

### 方法1：使用UBODTHelper（推荐）

```cpp
#include "mm/fmm/ubodt_manager.hpp"

using namespace FMM::MM;

// 1. 加载UBODT（会自动缓存）
auto ubodt = UBODTHelper::load_ubodt("data/ubodt.bin", 1, true);

// 2. 使用UBODT进行匹配
FMMAlgorithm fmm_algo(network, ubodt);
auto result = fmm_algo.match_traj(trajectory, config);

// 3. 再次调用load_ubodt会直接返回缓存的版本（极快）
auto ubodt2 = UBODTHelper::load_ubodt("data/ubodt.bin");

// 4. 完成后手动释放
UBODTHelper::release_all_ubodts();
```

### 方法2：使用UBODTManager

```cpp
#include "mm/fmm/ubodt_manager.hpp"

using namespace FMM::MM;

// 获取管理器实例
auto &manager = UBODTManager::getInstance();

// 加载UBODT
auto ubodt = manager.get_ubodt("data/ubodt.bin", 1);

// 检查是否已加载
if (manager.is_loaded("data/ubodt.bin")) {
    std::cout << "UBODT is loaded!\n";
}

// 查看状态
manager.print_status();

// 释放UBODT
manager.release_ubodt("data/ubodt.bin");
```

---

## 详细API

### UBODTHelper 便捷函数

#### 加载Full UBODT
```cpp
auto ubodt = UBODTHelper::load_ubodt(
    "data/ubodt.bin",  // 文件路径
    1,                 // multiplier（默认1）
    true               // keep in memory（默认true）
);
```

#### 加载PartialUBODT
```cpp
auto partial_ubodt = UBODTHelper::load_partial_ubodt(
    "data/ubodt.bin",     // 文件路径
    network,               // 路网
    trajectories,          // 轨迹向量
    0.1,                   // 缓冲区比例（默认0.1）
    true                   // keep in memory（默认true）
);
```

#### 加载CachedUBODT
```cpp
auto cached_ubodt = UBODTHelper::load_cached_ubodt(
    "data/ubodt.bin",  // 文件路径
    10000,             // 缓存大小（默认10000）
    1,                 // multiplier（默认1）
    true               // keep in memory（默认true）
);
```

#### 释放UBODT
```cpp
// 释放特定UBODT
UBODTHelper::release_ubodt("data/ubodt.bin");

// 释放所有UBODT
UBODTHelper::release_all_ubodts();

// 检查是否已加载
bool loaded = UBODTHelper::is_ubodt_loaded("data/ubodt.bin");

// 打印状态
UBODTHelper::print_ubodt_status();
```

---

## 使用场景

### 场景1：批量处理多个轨迹文件

**问题**：需要处理100个轨迹文件，每个文件都重新加载UBODT太慢

**解决方案**：
```cpp
#include "mm/fmm/ubodt_manager.hpp"
#include "mm/fmm/fmm_algorithm.hpp"

using namespace FMM::MM;

int main() {
    // 1. 一次性加载UBODT
    auto ubodt = UBODTHelper::load_ubodt("data/ubodt.bin");
    std::cout << "UBODT loaded once\n";

    // 2. 处理多个文件
    std::vector<std::string> files = {
        "data/traj1.csv",
        "data/traj2.csv",
        "data/traj3.csv",
        // ... 100个文件
    };

    for (const auto &file : files) {
        // UBODT已在内存中，无需重新加载
        auto trajectories = read_trajectories(file);
        auto results = match_trajectories(trajectories, ubodt);
        save_results(file + ".matched", results);
    }

    // 3. 完成后释放
    UBODTHelper::release_all_ubodt();
    return 0;
}
```

**性能对比**：
| 方法 | 100个文件总时间 | 说明 |
|------|---------------|------|
| 每次重新加载 | 1000秒 (100×10s) | 每次加载10秒 |
| **使用UBODTManager** | **110秒 (10s+100×1s)** | 只加载一次 |

---

### 场景2：交互式匹配会话

```cpp
int main() {
    auto &manager = UBODTManager::getInstance();

    while (true) {
        std::cout << "\n1. Load UBODT\n";
        std::cout << "2. Match trajectories\n";
        std::cout << "3. Show status\n";
        std::cout << "4. Release UBODT\n";
        std::cout << "5. Exit\n";
        std::cout << "Choose: ";

        int choice;
        std::cin >> choice;

        switch (choice) {
            case 1: {
                auto ubodt = manager.get_ubodt("data/ubodt.bin");
                std::cout << "UBODT loaded!\n";
                break;
            }
            case 2: {
                std::string file;
                std::cout << "Enter trajectory file: ";
                std::cin >> file;
                auto ubodt = manager.get_ubodt("data/ubodt.bin");
                // 执行匹配...
                break;
            }
            case 3:
                manager.print_status();
                break;
            case 4:
                manager.release_all();
                std::cout << "UBODT released!\n";
                break;
            case 5:
                manager.release_all();
                return 0;
        }
    }
}
```

---

### 场景3：PartialUBODT + 缓存加速

```cpp
int main() {
    Network network("network.shp", "id", "source", "target");

    // 1. 加载PartialUBODT（只加载相关区域）
    auto trajectories = read_trajectories("data/traj_group1.csv");
    auto partial_ubodt = UBODTHelper::load_partial_ubodt(
        "data/ubodt.bin", network, trajectories, 0.1
    );

    // 2. 包装为CachedUBODT（查询加速）
    CachedUBODT cached_ubodt(partial_ubodt->get_ubodt(), 10000);

    // 3. 执行匹配
    FMMAlgorithm fmm_algo(network, cached_ubodt.get_ubodt());
    auto results = fmm_algo.match_traj(trajectory, config);

    // 4. 查看缓存统计
    auto stats = cached_ubodt.get_stats();
    std::cout << "Cache hit rate: " << stats.hit_rate() * 100 << "%\n";

    UBODTHelper::release_all_ubodt();
    return 0;
}
```

---

## 高级功能

### 1. 强制重新加载

```cpp
// 第一次加载
auto ubodt1 = UBODTHelper::load_ubodt("data/ubodt.bin");

// 强制重新加载（忽略缓存）
auto ubodt2 = UBODTHelper::load_ubodt("data/ubodt.bin", 1, true);
// 或者
auto &manager = UBODTManager::getInstance();
auto ubodt3 = manager.get_ubodt("data/ubodt.bin", 1, true);  // force_reload=true
```

### 2. 统计信息

```cpp
auto &manager = UBODTManager::getInstance();
auto stats = manager.get_stats();

std::cout << "Total UBODTs: " << stats.total_ubodts << "\n";
std::cout << "Total references: " << stats.total_references << "\n";
std::cout << "Estimated memory: " << stats.memory_estimated / (1024*1024) << " MB\n";
```

### 3. 自动释放控制

```cpp
auto &manager = UBODTManager::getInstance();

// 禁用自动释放（默认）
manager.set_auto_release(false);

// 启用自动释放
manager.set_auto_release(true);
```

---

## 完整示例：批处理工具

```cpp
/**
 * Batch matching tool with UBODT Manager
 */
#include "mm/fmm/ubodt_manager.hpp"
#include "mm/fmm/fmm_algorithm.hpp"
#include "io/gps_reader.hpp"

using namespace FMM;
using namespace FMM::MM;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <network.shp> <ubodt.bin> <traj_file1> [traj_file2 ...]\n";
        return 1;
    }

    std::string network_file = argv[1];
    std::string ubodt_file = argv[2];

    // Load network
    CONFIG::NetworkConfig network_config(network_file, "id", "source", "target");
    NETWORK::Network network(network_config);

    // Load UBODT once
    std::cout << "Loading UBODT from " << ubodt_file << "...\n";
    auto ubodt = UBODTHelper::load_ubodt(ubodt_file);
    if (!ubodt) {
        std::cerr << "Failed to load UBODT\n";
        return 1;
    }
    std::cout << "UBODT loaded: " << ubodt->get_num_rows() << " rows\n\n";

    // Process each trajectory file
    for (int i = 3; i < argc; ++i) {
        std::string traj_file = argv[i];
        std::cout << "[" << (i-2) << "] Processing " << traj_file << "\n";

        try {
            // Read trajectories
            IO::TrajectoryReader reader(traj_file, "id", "geom");
            std::vector<CORE::Trajectory> trajectories;
            while (reader.has_next_trajectory()) {
                trajectories.push_back(reader.read_next_trajectory());
            }

            // Match
            MM::FMMAlgorithm fmm_algo(network, ubodt);
            MM::CONFIG::FMMConfig config;
            config.radius = 300;

            int matched = 0;
            for (const auto &traj : trajectories) {
                auto result = fmm_algo.match_traj(traj, config);
                if (!result.cpath.empty()) ++matched;
            }

            std::cout << "  Matched " << matched << "/" << trajectories.size()
                      << " trajectories\n";

        } catch (const std::exception &e) {
            std::cerr << "  Error: " << e.what() << "\n";
        }
    }

    // Show status before releasing
    std::cout << "\n";
    UBODTHelper::print_ubodt_status();

    // Release UBODT
    std::cout << "\nReleasing UBODT...\n";
    UBODTHelper::release_all_ubodt();
    std::cout << "Done!\n";

    return 0;
}
```

**编译**：
```bash
g++ -std=c++17 -O3 \
    batch_match.cpp \
    -o batch_match \
    -I./src \
    -L./build \
    -lFMMLIB \
    $(pkg-config --cflags --libs gdal boost)
```

**使用**：
```bash
./batch_match \
    data/network.shp \
    data/ubodt.bin \
    data/traj1.csv \
    data/traj2.csv \
    data/traj3.csv
```

---

## 性能优势

### 传统方式 vs UBODTManager

| 场景 | 传统方式 | UBODTManager | 加速比 |
|------|---------|--------------|--------|
| 10个文件 | 100秒 | 19秒 | **5.3x** |
| 100个文件 | 1000秒 | 109秒 | **9.2x** |
| 1000个文件 | 10000秒 | 1009秒 | **9.9x** |

**说明**：
- 传统方式：每个文件都重新加载UBODT（10秒）
- UBODTManager：只加载一次（10秒），后续文件直接使用缓存

### 内存占用

| UBODT大小 | 传统方式 | UBODTManager | 说明 |
|-----------|---------|--------------|------|
| 1.8 GB | 1.8 GB | 1.8 GB | 单个UBODT |
| 1.8 GB × 3 | 5.4 GB | 1.8 GB | 多个UBODT复用 |

---

## 最佳实践

### 1. 及时释放
```cpp
// 处理完成后立即释放
UBODTHelper::release_all_ubodt();
```

### 2. 检查状态
```cpp
// 使用前检查是否已加载
if (!UBODTHelper::is_ubodt_loaded("data/ubodt.bin")) {
    auto ubodt = UBODTHelper::load_ubodt("data/ubodt.bin");
}
```

### 3. 批量处理
```cpp
// 一次性加载，批量处理
auto ubodt = UBODTHelper::load_ubodt("data/ubodt.bin");

for (const auto &file : trajectory_files) {
    process_file(file, ubodt);  // UBODT已在内存中
}
```

### 4. 错误处理
```cpp
auto ubodt = UBODTHelper::load_ubodt("data/ubodt.bin");
if (!ubodt) {
    std::cerr << "Failed to load UBODT\n";
    return 1;
}
```

---

## 故障排查

### 问题1：UBODT未自动释放

**原因**：默认 `keep=true`

**解决**：
```cpp
// 方法1：手动释放
UBODTHelper::release_all_ubodt();

// 方法2：启用自动释放
auto &manager = UBODTManager::getInstance();
manager.set_auto_release(true);
```

### 问题2：内存占用过高

**原因**：加载了太多不同UBODT

**解决**：
```cpp
// 查看状态
UBODTHelper::print_ubodt_status();

// 释放不需要的UBODT
UBODTHelper::release_ubodt("data/ubodt.bin");
```

### 问题3：UBODT未更新

**原因**：使用了缓存的旧版本

**解决**：
```cpp
// 强制重新加载
auto ubodt = UBODTHelper::load_ubodt("data/ubodt.bin", 1, true);
```

---

## API参考

### UBODTManager类

```cpp
class UBODTManager {
public:
    static UBODTManager& getInstance();

    // 加载UBODT
    std::shared_ptr<UBODT> get_ubodt(const std::string &filename,
                                      int multiplier = 1,
                                      bool force_reload = false);

    std::shared_ptr<PartialUBODT> get_partial_ubodt(
        const std::string &filename,
        const NETWORK::Network &network,
        const std::vector<CORE::Trajectory> &trajectories,
        double buffer_ratio = 0.1,
        bool force_reload = false);

    std::shared_ptr<CachedUBODT> get_cached_ubodt(
        const std::string &filename,
        size_t cache_size = 10000,
        int multiplier = 1,
        bool force_reload = false);

    // 状态查询
    bool is_loaded(const std::string &filename) const;
    ManagerStats get_stats() const;
    void print_status() const;

    // 释放
    size_t release_ubodt(const std::string &filename);
    size_t release_all();

    // 配置
    void set_auto_release(bool enable);
    bool get_auto_release() const;
};
```

### UBODTHelper命名空间

```cpp
namespace UBODTHelper {
    std::shared_ptr<UBODT> load_ubodt(const std::string &filename,
                                       int multiplier = 1,
                                       bool keep = true);

    std::shared_ptr<PartialUBODT> load_partial_ubodt(...);

    std::shared_ptr<CachedUBODT> load_cached_ubodt(...);

    size_t release_ubodt(const std::string &filename);
    size_t release_all_ubodts();
    bool is_ubodt_loaded(const std::string &filename);
    void print_ubodt_status();
}
```

---

## 总结

UBODT Manager 是提升批量地图匹配效率的关键工具：

✅ **避免重复加载** - 加载一次，多次使用
✅ **显著加速** - 批量处理场景加速5-10x
✅ **简单易用** - 几行代码即可集成
✅ **线程安全** - 支持多线程环境
✅ **灵活控制** - 手动释放或自动释放

**适用场景**：
- ✅ 批量处理多个轨迹文件
- ✅ 交互式匹配会话
- ✅ 重复匹配实验
- ✅ 长时间运行的匹配服务

**不适用场景**：
- ❌ 单次匹配（没有重复加载）
- ❌ 内存极度受限（无法保持UBODT在内存中）
