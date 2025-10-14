#!/usr/bin/env bash
"""
FMM启动器 - 自动使用正确的Python环境
"""
import subprocess
import sys
import os

def run_with_system_python():
    """使用系统Python运行脚本"""
    # 确保使用系统Python
    system_python = "/usr/bin/python3"

    # 设置环境变量
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin:/usr/local/bin:/home/dell/.local/bin"
    env["PYTHONPATH"] = "/home/dell/Czhang/fmm_sjtugnc/build/python"

    # 运行命令
    if len(sys.argv) > 1:
        # 如果有参数，运行指定脚本
        result = subprocess.run([system_python] + sys.argv[1:], env=env)
    else:
        # 如果没有参数，运行测试
        test_code = '''
import sys
import fmm
print("✅ fmm模块导入成功！")
print(f"Python版本: {sys.version}")
print(f"可用类: {[attr for attr in dir(fmm) if not attr.startswith('_') and attr[0].isupper()][:5]}")
'''
        result = subprocess.run([system_python, "-c", test_code], env=env)

    return result.returncode

if __name__ == "__main__":
    sys.exit(run_with_system_python())