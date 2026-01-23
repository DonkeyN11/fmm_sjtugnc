#!/bin/bash
# UBODT Daemon 集成演示脚本

set -e

echo "=========================================="
echo "UBODT Daemon 集成演示"
echo "=========================================="
echo ""

# 检查是否有UBODT文件
UBODT_FILE=""
if [ -f "dataset_hainan_06/1.1/od/ubodt.bin" ]; then
    UBODT_FILE="dataset_hainan_06/1.1/od/ubodt.bin"
elif [ -f "data/ubodt.bin" ]; then
    UBODT_FILE="data/ubodt.bin"
else
    echo "错误：找不到UBODT文件"
    echo "请确保以下文件之一存在："
    echo "  - dataset_hainan_06/1.1/od/ubodt.bin"
    echo "  - data/ubodt.bin"
    echo ""
    echo "你可以指定自己的UBODT文件路径来运行此演示。"
    exit 1
fi

echo "找到UBODT文件: $UBODT_FILE"
echo ""

# ========================================
# 步骤1：检查daemon状态
# ========================================
echo "步骤1：检查daemon初始状态"
echo "----------------------------"
ubodt_daemon status || echo "Daemon未运行（预期）"
echo ""

# ========================================
# 步骤2：启动daemon
# ========================================
echo "步骤2：启动daemon并加载UBODT"
echo "----------------------------"
echo "命令: ubodt_daemon start --ubodt $UBODT_FILE"
echo ""
echo "注意：这会加载UBODT到daemon的内存中"
echo "进程将在后台持续运行..."
echo ""

# 询问用户是否继续
read -p "是否继续启动daemon？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "演示已取消"
    exit 0
fi

# 启动daemon（后台运行）
ubodt_daemon start --ubodt "$UBODT_FILE" > /tmp/ubodt_daemon_output.log 2>&1 &
DAEMON_PID=$!

# 等待daemon启动
sleep 3

echo ""
echo "Daemon已启动 (PID: $DAEMON_PID)"
echo ""

# ========================================
# 步骤3：验证daemon状态
# ========================================
echo "步骤3：验证daemon状态"
echo "----------------------------"
ubodt_daemon status
echo ""

# ========================================
# 步骤4：演示FMM如何检测daemon
# ========================================
echo "步骤4：演示FMM/CMM如何检测daemon"
echo "----------------------------"
echo "现在运行FMM或CMM时，会检测到daemon已加载UBODT"
echo ""
echo "示例命令："
echo "  fmm --ubodt $UBODT_FILE --network data/network.shp --gps data/traj.csv --result output.csv"
echo "  cmm --ubodt $UBODT_FILE --config config.xml"
echo ""
echo "预期日志输出："
echo "  [info] UBODT is preloaded by ubodt_daemon. Using fast loading from OS cache."
echo "  [info] UBODT loaded in 0.15s  # 比首次加载快很多！"
echo ""

# ========================================
# 步骤5：查看daemon日志
# ========================================
echo "步骤5：查看daemon启动日志"
echo "----------------------------"
if [ -f "/tmp/ubodt_daemon_output.log" ]; then
    tail -10 /tmp/ubodt_daemon_output.log
else
    echo "日志文件未找到"
fi
echo ""

# ========================================
# 步骤6：测试不同场景
# ========================================
echo "步骤6：测试不同场景"
echo "----------------------------"
echo ""
echo "场景1：Daemon已加载相同UBODT"
echo "  检测结果：✅ 利用页缓存，快速加载"
echo ""
echo "场景2：使用不同的UBODT文件"
echo "  检测结果：⚠️  正常从磁盘加载"
echo ""
echo "场景3：Daemon未运行"
echo "  检测结果：⚠️  正常从磁盘加载"
echo ""

# ========================================
# 步骤7：停止daemon
# ========================================
echo "步骤7：停止daemon"
echo "----------------------------"
read -p "是否停止daemon？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ubodt_daemon stop
    echo ""
    echo "Daemon已停止"
else
    echo ""
    echo "Daemon仍在后台运行"
    echo "你可以稍后手动停止："
    echo "  ubodt_daemon stop"
fi

echo ""
echo "=========================================="
echo "演示完成"
echo "=========================================="
echo ""
echo "关键点："
echo "1. Daemon预加载UBODT到内存"
echo "2. FMM/CMM自动检测daemon"
echo "3. 利用操作系统页缓存加速加载"
echo "4. 无需修改任何代码或配置"
echo ""
echo "更多信息请参考：UBODT_DAEMON_INTEGRATION.md"
