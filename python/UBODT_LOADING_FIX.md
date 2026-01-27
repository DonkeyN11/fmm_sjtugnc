# UBODT 加载问题修复说明

## 问题描述

运行 `debug_cmm_python.py` 时遇到错误：
```
RuntimeError: input stream error
```

**原因：**
使用了错误的 UBODT 读取方法。`hainan_ubodt_indexed.bin` 是 **indexed binary** 格式，应该使用 `read_ubodt_indexed_binary()` 而不是 `read_ubodt_file()`。

---

## 解决方案

### UBODT 文件类型和对应的读取方法

| 文件后缀 | 读取方法 | 示例文件 |
|---------|---------|---------|
| `*_indexed.bin` | `UBODT.read_ubodt_indexed_binary()` | `hainan_ubodt_indexed.bin` |
| `*_mmap.bin` | `UBODT.read_ubodt_mmap_binary()` | `haikou_ubodt_mmap.bin` |
| `*.txt`, `*.csv` | `UBODT.read_ubodt_file()` | `hainan_ubodt.txt` |

### 修复代码

```python
# ❌ 错误（会导致 input stream error）
ubodt = UBODT.read_ubodt_file("hainan_ubodt_indexed.bin")

# ✓ 正确（indexed binary 格式）
ubodt = UBODT.read_ubodt_indexed_binary("hainan_ubodt_indexed.bin")

# ✓ 正确（mmap binary 格式）
ubodt = UBODT.read_ubodt_mmap_binary("haikou_ubodt_mmap.bin")
```

### 自动检测方法

```python
ubodt_file = "input/map/hainan_ubodt_indexed.bin"

if ubodt_file.endswith("_indexed.bin"):
    ubodt = UBODT.read_ubodt_indexed_binary(ubodt_file)
elif ubodt_file.endswith("_mmap.bin"):
    ubodt = UBODT.read_ubodt_mmap_binary(ubodt_file)
else:
    ubodt = UBODT.read_ubodt_file(ubodt_file)
```

---

## 使用修复后的脚本

### 方法 1: 快速测试（推荐）

首先测试 UBODT 能否正常加载：

```bash
cd /home/dell/fmm_sjtugnc
python python/test_ubodt_loading.py
```

这会测试所有可用的 UBODT 文件并显示结果。

### 方法 2: 运行完整的调试脚本

```bash
python python/debug_cmm_python.py
```

修复后的脚本会：
1. 自动检测文件类型
2. 使用正确的读取方法
3. 显示详细的调试信息

---

## 可用的 Python UBODT 读取方法

通过检查 Python 绑定，确认以下方法可用：

```python
from fmm import *

# 1. Indexed binary 格式
ubodt = UBODT.read_ubodt_indexed_binary(filename)

# 2. Memory-mapped binary 格式
ubodt = UBODT.read_ubodt_mmap_binary(filename)

# 3. 普通二进制格式
ubodt = UBODT.read_ubodt_binary(filename)

# 4. CSV 格式
ubodt = UBODT.read_ubodt_csv(filename)

# 5. 自动检测（推荐用于 .txt 文件）
ubodt = UBODT.read_ubodt_file(filename)
```

---

## 常见问题

### Q: 为什么 `read_ubodt_file` 不能读取 `_indexed.bin`？

A: `read_ubodt_file` 在 Python 绑定中的自动检测功能可能不完整，对于 indexed binary 格式需要明确调用 `read_ubodt_indexed_binary`。

### Q: 如何知道我的文件是什么格式？

A: 查看文件名：
- `*_indexed.bin` → indexed binary
- `*_mmap.bin` → memory-mapped binary
- `*.bin` (无特殊后缀) → 普通二进制
- `*.txt` → 文本格式

### Q: 我的 UBODT 文件不存在怎么办？

A: 检查以下位置：
- `input/map/hainan/edges.shp` (路网)
- `input/map/hainan_ubodt_indexed.bin` (UBODT)

如果不存在，需要先生成 UBODT：
```bash
# 使用命令行工具生成
./build/app/ubodt_gen
```

---

## 相关文件

- 修复后的脚本: [python/debug_cmm_python.py](python/debug_cmm_python.py)
- 测试脚本: [python/test_ubodt_loading.py](python/test_ubodt_loading.py)
- Python API 文档: [python/CMM_PYTHON_API.md](python/CMM_PYTHON_API.md)

---

**修复日期:** 2025-01-26
