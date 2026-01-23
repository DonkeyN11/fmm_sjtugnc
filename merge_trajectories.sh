#!/bin/bash
# 合并dataset_hainan_06下所有版本的cmm_trajectory.csv文件

set -e

OUTPUT_FILE="/home/dell/fmm_sjtugnc/dataset_hainan_06/merged_cmm_trajectories.csv"
BASE_DIR="/home/dell/fmm_sjtugnc/dataset_hainan_06"

echo "开始合并cmm_trajectory.csv文件..."
echo "输出文件: $OUTPUT_FILE"
echo ""

# 清空或创建输出文件
> "$OUTPUT_FILE"

# 查找所有cmm_trajectory.csv文件并按版本号排序
FILES=$(ls "$BASE_DIR"/*/mr/cmm_trajectory.csv 2>/dev/null | sort -V)

if [ -z "$FILES" ]; then
    echo "错误：未找到任何cmm_trajectory.csv文件"
    exit 1
fi

# 统计文件数量
TOTAL_FILES=$(echo "$FILES" | wc -l)
echo "找到 $TOTAL_FILES 个文件"
echo ""

# 合并文件
COUNT=0
FIRST_FILE=true

for FILE in $FILES; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL_FILES] 处理: $FILE"

    if [ "$FIRST_FILE" = true ]; then
        # 第一个文件，保留表头
        cat "$FILE" >> "$OUTPUT_FILE"
        FIRST_FILE=false
    else
        # 后续文件，跳过表头（跳过第一行）
        tail -n +2 "$FILE" >> "$OUTPUT_FILE"
    fi
done

echo ""
echo "✓ 合并完成！"
echo ""
echo "统计信息："
echo "  - 输入文件数: $TOTAL_FILES"
echo "  - 输出文件: $OUTPUT_FILE"

# 统计输出文件的行数
TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
DATA_LINES=$((TOTAL_LINES - 1))  # 减去表头
echo "  - 总行数: $TOTAL_LINES (包含表头)"
echo "  - 数据行数: $DATA_LINES"
echo ""

# 显示文件大小
FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "  - 文件大小: $FILE_SIZE"
echo ""

# 显示前几行和后几行
echo "输出文件预览（前5行）："
head -5 "$OUTPUT_FILE"
echo ""
echo "输出文件预览（最后5行）："
tail -5 "$OUTPUT_FILE"
echo ""

echo "✓ 完成！合并后的文件已保存到："
echo "  $OUTPUT_FILE"
