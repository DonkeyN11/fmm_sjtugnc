#!/bin/bash

echo "🚀 FMM系统Python启动器"
echo "================================"

# 完全隔离系统环境
export PATH="/usr/bin:/bin:/usr/local/bin:/home/dell/.local/bin"
unset PYTHONPATH
unset LD_LIBRARY_PATH

# 添加fmm路径
export PYTHONPATH="$(pwd)/build/python:$PYTHONPATH"

echo "📍 当前Python: $(which python3)"
echo "📍 Python版本: $(python3 --version)"
echo "📍 FMM路径: $(pwd)/build/python"

# 测试fmm模块
echo "🧪 测试fmm模块..."
if python3 -c "import fmm; print('✅ fmm模块可用'); print(f'可用类数量: {len([attr for attr in dir(fmm) if not attr.startswith(\"_\") and attr[0].isupper()])}')" 2>/dev/null; then
    echo "🎉 fmm模块测试成功！"

    if [ $# -eq 0 ]; then
        echo "💡 用法: ./fmm_starter.sh <你的Python脚本>"
        echo "💡 或直接运行: python3 your_script.py"
        echo "💡 交互式Python: python3"
    else
        echo "📝 运行脚本: $1"
        python3 "$@"
    fi
else
    echo "❌ fmm模块导入失败"
    echo "🔍 请检查fmm模块路径和依赖库"
    exit 1
fi
