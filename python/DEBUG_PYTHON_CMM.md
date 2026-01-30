# CMM Python 调试指南 - 查看 match_traj 内部变量

本文档介绍如何在 Python 中调用 `match_traj` 时查看函数内部的变量值。

---

## 方案对比

| 方案 | 难度 | 信息详细程度 | 性能影响 | 推荐度 |
|------|------|-------------|---------|--------|
| **方案1: Python 包装函数** | ⭐ 简单 | ⭐⭐⭐ 中等 | 无 | ⭐⭐⭐⭐⭐ 最推荐 |
| **方案2: 设置 C++ 日志级别** | ⭐⭐ 中等 | ⭐⭐⭐⭐ 详细 | 小 | ⭐⭐⭐⭐ 推荐 |
| **方案3: 修改 C++ 代码** | ⭐⭐⭐ 复杂 | ⭐⭐⭐⭐⭐ 最详细 | 小 | ⭐⭐⭐ 可选 |

---

## 方案 1: Python 包装函数（最简单，推荐）

直接在 Python 中创建一个包装函数，在匹配前后输出详细信息。

### 使用方法

我已经为你创建了完整的调试脚本：[python/debug_cmm_python.py](python/debug_cmm_python.py)

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/dell/fmm_sjtugnc/build/python')
from fmm import *

def debug_match_traj(traj, config, cmm):
    """带调试输出的匹配函数"""

    # 输入信息
    print(f"\n=== 匹配前信息 ===")
    print(f"轨迹点数: {traj.geom.get_num_points()}")
    print(f"配置: k={config.k}, pl_mult={config.protection_level_multiplier}")

    # 前3个点的协方差
    for i in range(min(3, len(traj.covariances))):
        cov = traj.covariances[i]
        print(f"点 {i}: sde={cov.sde:.2f}, sdn={cov.sdn:.2f}, pl={traj.protection_levels[i]:.2f}")

    # 执行匹配
    result = cmm.match_traj(traj, config)

    # 输出信息
    print(f"\n=== 匹配结果 ===")
    print(f"匹配状态: {result.is_matched()}")
    print(f"匹配边数: {len(result.cpath)}")
    print(f"sp_dist: {result.sp_dist:.2f} 米")
    print(f"eu_dist: {result.eu_dist:.2f} 米")

    # 每个点的详细信息
    print(f"\n=== 每点匹配详情 ===")
    for i, mc in enumerate(result.opt_candidate_path):
        print(f"点 {i}: 边ID={mc.c.edge.id}, 偏移={mc.c.offset:.2f}m, "
              f"距离={mc.c.dist:.2f}m, ep={mc.ep:.2e}, tp={mc.tp:.2e}")

    # 候选点详情
    if hasattr(result, 'candidate_details'):
        print(f"\n=== 候选点详情 ===")
        for i, candidates in enumerate(result.candidate_details):
            print(f"点 {i}: {len(candidates)} 个候选")
            for j, cand in enumerate(candidates[:3]):  # 只显示前3个
                print(f"  候选{j}: ({cand.x:.6f}, {cand.y:.6f}), ep={cand.ep:.2e}")

    return result

# 使用示例
result = debug_match_traj(traj, config, cmm)
```

### 运行

```bash
cd /home/dell/fmm_sjtugnc
python python/debug_cmm_python.py
```

---

## 方案 2: 设置 C++ 日志级别（推荐）

利用现有的 `SPDLOG_INFO` 输出，只需调整日志级别即可。

### 方法 A: 修改 XML 配置文件

在 `cmm_config_omp_wgs84.xml` 中设置：

```xml
<config>
  <other>
    <!-- 0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical, 6=off -->
    <log_level>1</log_level>  <!-- 设置为 debug 级别 -->
  </other>
</config>
```

### 方法 B: 在 Python 中设置环境变量

```python
import os
os.environ['SPDLOG_LEVEL'] = 'debug'  # 或 'info'

import sys
sys.path.insert(0, '/home/dell/fmm_sjtugnc/build/python')
from fmm import *

# 之后的所有操作都会输出 debug 日志
result = cmm.match_traj(traj, config)
```

### 可用的日志级别

```cpp
0 - trace   // 最详细，包含所有函数调用
1 - debug   // 调试信息，包含关键变量
2 - info    // 一般信息（默认级别）
3 - warn    // 警告信息
4 - error   // 错误信息
```

### 现有的日志输出

代码中已有的 `SPDLOG_DEBUG` 和 `SPDLOG_INFO` 输出包括：

- **候选点搜索**：每个点的候选数、发射概率
- **过滤结果**：删除的空点和不连通点
- **转移图**：层数、节点数
- **最优路径**：路径大小、边ID
- **匹配结果**：opath, indices, cpath

---

## 方案 3: 修改 C++ 代码添加更多输出

如果需要更详细的内部变量信息，可以直接修改 C++ 代码。

### 我已为你准备了补丁文件

文件位置：[src/mm/cmm/cmm_algorithm_debug_patch.cpp](src/mm/cmm/cmm_algorithm_debug_patch.cpp)

### 应用步骤

#### 1. 备份原文件

```bash
cd /home/dell/fmm_sjtugnc
cp src/mm/cmm/cmm_algorithm.cpp src/mm/cmm/cmm_algorithm.cpp.bak
```

#### 2. 查看补丁文件

```bash
cat src/mm/cmm/cmm_algorithm_debug_patch.cpp
```

这个文件包含了需要在 `match_traj` 函数中添加的调试代码。

#### 3. 手动应用补丁（关键位置）

在 `src/mm/cmm/cmm_algorithm.cpp` 的 `match_traj` 函数中添加：

```cpp
// 在函数开头添加
SPDLOG_INFO("=== CMM match_traj 开始 ===");
SPDLOG_INFO("轨迹 ID: {}", traj.id);
SPDLOG_INFO("轨迹点数: {}", traj.geom.get_num_points());
SPDLOG_INFO("配置: k={}, pl_mult={}", config.k, config.protection_level_multiplier);

// 在候选点搜索后添加
SPDLOG_INFO("✓ 候选点搜索完成");
for (size_t i = 0; i < candidates.size(); ++i) {
    SPDLOG_INFO("  点 {}: {} 个候选", i, candidates[i].size());
}

// 在最优路径推断后添加
SPDLOG_INFO("✓ 最优路径推断完成，{} 个节点", tg_opath.size());
SPDLOG_INFO("  总最短路径距离: {:.2f} 米", total_sp_dist);
SPDLOG_INFO("  总欧氏距离: {:.2f} 米", total_eu_dist);
```

#### 4. 重新编译

```bash
cd build
cmake --build . --target cmm -j$(nproc)
```

#### 5. 运行并查看输出

```bash
# 命令行方式
./build/app/cmm input/config/cmm_config_omp_wgs84.xml

# Python 方式（确保 log_level 设置为 info 或 debug）
python your_script.py
```

---

## 快速诊断检查清单

使用上述方案后，检查以下关键指标：

### ✓ 正常情况的输出示例

```
[INFO] 轨迹点数: 5
[INFO] 点 0: 8 个候选
[INFO] 点 1: 12 个候选
[INFO] 点 2: 10 个候选
[INFO] 点 3: 9 个候选
[INFO] 点 4: 8 个候选
[INFO] ✓ 候选点搜索完成
[INFO]   有候选点的点数: 5/5
[INFO] ✓ 最优路径推断完成，5 个节点
[INFO]   总最短路径距离: 125.43 米
[INFO]   总欧氏距离: 118.76 米
[INFO]   匹配比率: 0.946812
[INFO] ✓ CMM match_traj 完成
```

### ✗ 异常情况的输出示例

```
[WARN] 点 0: 无候选点!
[WARN] 点 1: 无候选点!
[ERROR] ✗ 所有点都没有候选点，匹配失败
[ERROR]   可能原因:
[ERROR]     1. 轨迹坐标与路网不在同一区域
[ERROR]     2. protection_level_multiplier 太小 (当前: 1.0)
[ERROR]     3. 路网文件未正确加载
[ERROR]     4. 坐标系不匹配
```

### 关键诊断指标

1. **候选点数量**：每个点应该至少有 1-3 个候选
2. **发射概率**：不应该全部是 0 或 NaN
3. **最短路径距离**：应该大于 0
4. **匹配比率**：通常应该在 0.7-1.0 之间
5. **连通性**：候选点之间应该有可达的路径

---

## 常见问题诊断

### 问题 1: 所有点都没有候选点

**症状：**
```
[WARN] 点 0: 无候选点!
[WARN] 点 1: 无候选点!
[ERROR] ✗ 所有点都没有候选点
```

**解决方案：**
1. 增大 `protection_level_multiplier`（如从 1.0 改为 10.0）
2. 检查路网是否正确加载
3. 检查坐标系是否匹配（WGS84 vs UTM）
4. 检查轨迹坐标是否在路网范围内

### 问题 2: 有候选点但匹配失败

**症状：**
```
[INFO] 点 0: 5 个候选
[INFO] 点 1: 3 个候选
[INFO] ✓ 最优路径推断完成，0 个节点
[WARN] ⚠️ 完整路径为空
```

**解决方案：**
1. 增大 `reverse_tolerance` 允许反向移动
2. 检查 UBODT 是否正确生成
3. 检查路网连通性

### 问题 3: 匹配比率过低

**症状：**
```
[INFO]   匹配比率: 0.345678
[WARN]   ⚠️ 匹配比率过低 (< 0.5)
```

**解决方案：**
1. 增加候选点数量 `k`
2. 调整 `protection_level_multiplier`
3. 检查协方差矩阵是否合理

---

## 推荐工作流程

1. **第一步：使用方案 1（Python 包装）**
   - 运行 `python/debug_cmm_python.py`
   - 查看基本的输入输出信息

2. **第二步：启用方案 2（日志级别）**
   - 修改 XML 配置文件，设置 `log_level=1`
   - 重新运行，查看 C++ 层面的日志

3. **第三步（可选）：应用方案 3（修改代码）**
   - 如果前两步不够，添加更多自定义日志
   - 重新编译

4. **第四步：分析结果**
   - 使用 `python/debug_candidates.py` 分析输出文件
   - 根据诊断指标调整参数

---

## 相关文件

- **Python 调试脚本**：[python/debug_cmm_python.py](python/debug_cmm_python.py)
- **候选点分析脚本**：[python/debug_candidates.py](python/debug_candidates.py)
- **C++ 调试补丁**：[src/mm/cmm/cmm_algorithm_debug_patch.cpp](src/mm/cmm/cmm_algorithm_debug_patch.cpp)
- **Python API 文档**：[python/CMM_PYTHON_API.md](python/CMM_PYTHON_API.md)

---

**最后更新：** 2025-01-26
