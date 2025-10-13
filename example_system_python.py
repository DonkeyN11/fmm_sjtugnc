#!/usr/bin/python3
"""
示例：在系统Python中使用fmm
"""
import sys
import os

# 确保使用系统Python环境
os.environ['PATH'] = '/usr/bin:/bin:/usr/local/bin'

# 添加fmm模块路径
sys.path.insert(0, '/home/dell/Czhang/fmm_sjtugnc/build_system_backup/python')

try:
    import fmm
    print("✅ fmm模块导入成功")
    print(f"fmm主要功能: {[attr for attr in dir(fmm) if not attr.startswith('_') and attr[0].isupper()][:5]}")

    # 简单测试
    config = fmm.FastMapMatchConfig()
    print("✅ fmm配置创建成功")

except ImportError as e:
    print(f"❌ fmm导入失败: {e}")
    sys.exit(1)

print("🎉 系统Python环境正常工作！")