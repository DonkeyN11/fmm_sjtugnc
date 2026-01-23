#!/bin/bash
# UBODT Daemon 演示脚本

echo "=========================================="
echo "UBODT Daemon 演示"
echo "=========================================="
echo ""

# 1. 检查初始状态
echo "1. 检查daemon状态 (应该未运行):"
ubodt_daemon status
echo ""

# 2. 启动daemon (需要有效的UBODT文件)
echo "2. 启动daemon..."
echo "注意: 这需要有效的UBODT文件"
echo ""
echo "使用方法:"
echo "  ubodt_daemon start --ubodt <你的ubodt文件路径>"
echo ""
echo "例如:"
echo "  ubodt_daemon start --ubodt data/ubodt.bin"
echo ""

# 3. 在后台运行演示
echo "3. Daemon启动后，你可以在任何终端运行:"
echo "   ubodt_daemon status"
echo ""
echo "   这会显示daemon的状态，包括:"
echo "   - PID (进程ID)"
echo "   - UBODT文件路径"
echo "   - 加载的行数"
echo ""

# 4. 停止daemon
echo "4. 停止daemon:"
echo "   ubodt_daemon stop"
echo ""

echo "=========================================="
echo "主要区别:"
echo "=========================================="
echo ""
echo "ubodt_manager load:"
echo "  - 只在当前进程中加载UBODT"
echo "  - 命令执行完，进程退出，UBODT释放"
echo "  - 其他终端看不到"
echo ""
echo "ubodt_daemon start:"
echo "  - 在后台持续运行"
echo "  - UBODT持久化在内存中"
echo "  - 任何终端都能查询状态"
echo "  - 可以被多个进程使用"
echo ""

echo "=========================================="
echo "推荐工作流程:"
echo "=========================================="
echo ""
echo "# 终端1: 启动daemon"
echo "ubodt_daemon start --ubodt data/ubodt.bin"
echo ""
echo "# 终端2,3,4...: 查看状态或运行FMM/CMM"
echo "ubodt_daemon status"
echo "fmm --config config.xml --ubodt data/ubodt.bin"
echo "cmm --config config2.xml --ubodt data/ubodt.bin"
echo ""
echo "# 所有终端都能看到daemon在运行!"
echo ""
echo "# 完成后停止daemon"
echo "ubodt_daemon stop"
echo ""
