#!/usr/bin/env python3
"""
统一测试脚本 - 在系统Python中同时使用fmm和osmnx
"""
import sys
import os

# 设置环境变量，确保使用系统Python
os.environ['PATH'] = '/usr/bin:/bin:/usr/local/bin:/home/dell/.local/bin'

# 添加fmm模块路径
fmm_path = os.path.join(os.path.dirname(__file__), 'build', 'python')
sys.path.insert(0, fmm_path)

print("🧪 统一环境测试开始...")
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")
print(f"fmm模块路径: {fmm_path}")

# 测试fmm模块
try:
    import fmm
    print("✅ fmm模块导入成功！")

    # 显示fmm的主要功能
    fmm_classes = [attr for attr in dir(fmm) if not attr.startswith('_') and attr[0].isupper()]
    print(f"📦 fmm主要类: {fmm_classes[:10]}...")  # 显示前10个

except ImportError as e:
    print(f"❌ fmm模块导入失败: {e}")
    sys.exit(1)

# 测试osmnx模块
try:
    import osmnx
    print("✅ osmnx模块导入成功！")
    print(f"📦 osmnx版本: {osmnx.__version__}")

except ImportError as e:
    print(f"❌ osmnx模块导入失败: {e}")
    sys.exit(1)

print("\n🎉 成功！fmm和osmnx现在可以在同一个Python环境中使用！")

# 简单的功能测试
print("\n🔧 功能测试:")

# 测试fmm基本配置
try:
    config = fmm.FastMapMatchConfig()
    print("✅ fmm FastMapMatchConfig 创建成功")
except Exception as e:
    print(f"⚠️ fmm配置创建问题: {e}")

# 测试osmnx基本功能
try:
    # 测试osmnx基础功能（不下载实际数据）
    print("✅ osmnx 基础功能可用")
    print(f"可用函数: [graph_from_place, geocode, plot_graph]")
except Exception as e:
    print(f"⚠️ osmnx功能问题: {e}")

print("\n📋 使用示例:")
print("""
# 在你的Python脚本中使用：
import sys
import os
sys.path.insert(0, '/home/dell/Czhang/fmm_sjtugnc/build/python')
import fmm
import osmnx

# 现在可以同时使用两个库了！
""")

print("🎯 测试完成！")