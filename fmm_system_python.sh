#!/bin/bash

# FMM + OSMNX 系统Python启动脚本
export PATH="/usr/bin:/bin:/usr/local/bin"

# 添加fmm模块路径
export PYTHONPATH="/home/dell/Czhang/fmm_sjtugnc/build_system_backup/python:$PYTHONPATH"

# 使用系统Python运行
/usr/bin/python3 "$@"