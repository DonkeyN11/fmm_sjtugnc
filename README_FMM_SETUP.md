# FMM模块安装和使用指南

## 问题解决总结

成功解决了 `No module named 'fmm'` 错误，主要问题是 **GLIBCXX版本不兼容**。

## 使用方法

### 1. 使用系统Python（推荐）

```bash
# 使用系统Python 3.10
/usr/bin/python3 -c "import sys; sys.path.insert(0, '/home/dell/Czhang/fmm_sjtugnc/build/python'); import fmm; print('fmm模块导入成功！')"
```

### 2. 使用测试脚本

```bash
cd /home/dell/Czhang/fmm_sjtugnc
/usr/bin/python3 test_fmm.py
```

### 3. 在Python代码中使用

```python
import sys
import os

# 添加fmm模块路径
sys.path.insert(0, '/home/dell/Czhang/fmm_sjtugnc/build/python')

# 导入fmm模块
import fmm
import osmnx

# 现在可以使用fmm和osmnx了
print("fmm模块功能:", [attr for attr in dir(fmm) if not attr.startswith('_')])
```

## 可用的fmm功能

fmm模块提供了以下主要类和方法：
- **FastMapMatch**: 快速地图匹配
- **STMATCH**: ST-匹配算法
- **H3MM**: H3网格匹配
- **Network**: 网络图操作
- **Trajectory**: 轨迹处理
- **UBODT**: UBODT算法
- 各种配置类和结果处理类

## 注意事项

1. **推荐使用系统Python** (`/usr/bin/python3`) 而不是conda环境
2. **osmnx已经安装**，可以在系统Python和conda环境中使用
3. **库兼容性**: fmm模块使用系统GDAL/GEOS库编译，确保最佳兼容性

## 恢复conda环境

如果需要重新使用conda环境：

```bash
source /home/dell/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name
```

## 故障排除

如果仍然遇到导入问题：

1. 确保使用系统Python: `/usr/bin/python3`
2. 检查库路径是否正确: `ls /home/dell/Czhang/fmm_sjtugnc/build/python/`
3. 运行测试脚本: `python3 test_fmm.py`

---
**编译日期**: 2025-10-13
**Python版本**: 3.10.12 (系统)
**GDAL版本**: 3.8.4 (系统)