#!/usr/bin/env python3
"""
Test script for fmm module import
"""
import sys
import os

# Add the build directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build', 'python'))

try:
    import fmm
    print("✅ fmm模块导入成功！")
    print(f"fmm模块版本信息: {fmm}")

    # Test basic functionality
    print("\n📋 测试基本功能:")
    print(f"可用的类和方法: {[attr for attr in dir(fmm) if not attr.startswith('_')]}")

except ImportError as e:
    print(f"❌ fmm模块导入失败: {e}")
    print("\n🔍 可能的解决方案:")
    print("1. 确保使用系统Python: /usr/bin/python3")
    print("2. 设置LD_LIBRARY_PATH指向系统库路径")
    print("3. 检查GDAL和GEOS库兼容性")
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    sys.exit(1)

# Test osmnx import
try:
    import osmnx
    print("✅ osmnx模块导入成功！")
except ImportError as e:
    print(f"❌ osmnx模块导入失败: {e}")
    print("请使用: pip install --user osmnx")

print("\n🎉 测试完成！")