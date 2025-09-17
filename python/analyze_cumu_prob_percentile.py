#!/usr/bin/env python3
"""
统计mr_fields_all_filtered.csv中cumu_prob大于0和小于0的数量，并计算分位点
"""

import pandas as pd
import sys
import numpy as np


def analyze_cumu_prob(csv_file_path):
    """分析cumu_prob列的统计信息"""
    print(f"正在分析文件: {csv_file_path}")

    try:
        # 读取CSV文件，使用分号分隔
        df = pd.read_csv(csv_file_path, sep=';')
        print(f"成功读取 {len(df)} 条记录")

        # 检查cumu_prob列是否存在
        if 'cumu_prob' not in df.columns:
            print("错误: 文件中找不到 'cumu_prob' 列")
            print(f"可用列: {list(df.columns)}")
            return False

        # 处理cumu_prob列 - 提取每个逗号分隔字符串的最后一个值
        print("正在提取cumu_prob的最后一个值...")
        last_cumu_prob_values = []

        for idx, value in enumerate(df['cumu_prob']):
            if pd.isna(value):
                last_cumu_prob_values.append(None)
            else:
                # 分割字符串并获取最后一个值
                str_values = str(value).split(',')
                if str_values and str_values[-1].strip():
                    try:
                        last_value = float(str_values[-1].strip())
                        last_cumu_prob_values.append(last_value)
                    except ValueError:
                        print(f"警告: 第{idx+1}行无法解析值: {str_values[-1]}")
                        last_cumu_prob_values.append(None)
                else:
                    last_cumu_prob_values.append(None)

        # 转换为pandas Series
        last_cumu_prob_series = pd.Series(last_cumu_prob_values)

        # 统计大于0的数量
        greater_than_zero = (last_cumu_prob_series > 0).sum()

        # 统计小于0的数量
        less_than_zero = (last_cumu_prob_series < 0).sum()

        # 统计等于0的数量
        equal_to_zero = (last_cumu_prob_series == 0).sum()

        # 统计空值数量
        null_values = last_cumu_prob_series.isnull().sum()

        # 计算总数验证
        total_records = len(df)
        verified_total = greater_than_zero + less_than_zero + equal_to_zero + null_values

        print(f"\n=== cumu_prob最后一个值统计结果 ===")
        print(f"总记录数: {total_records}")
        print(f"大于0的数量: {greater_than_zero}")
        print(f"小于0的数量: {less_than_zero}")
        print(f"等于0的数量: {equal_to_zero}")
        print(f"空值数量: {null_values}")
        print(f"验证总数: {verified_total}")

        # 计算百分比
        if total_records > 0:
            print(f"\n=== 百分比分布 ===")
            print(f"大于0的比例: {greater_than_zero/total_records*100:.2f}%")
            print(f"小于0的比例: {less_than_zero/total_records*100:.2f}%")
            print(f"等于0的比例: {equal_to_zero/total_records*100:.2f}%")
            print(f"空值比例: {null_values/total_records*100:.2f}%")

        # 显示基本统计信息
        print(f"\n=== 基本统计信息 ===")
        valid_values = last_cumu_prob_series.dropna()
        if len(valid_values) > 0:
            print(f"最大值: {valid_values.max():.6f}")
            print(f"最小值: {valid_values.min():.6f}")
            print(f"平均值: {valid_values.mean():.6f}")
            print(f"中位数: {valid_values.median():.6f}")
            print(f"标准差: {valid_values.std():.6f}")

        # 计算分位点
        print(f"\n=== 分位点统计 ===")
        if len(valid_values) > 0:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                value = valid_values.quantile(p/100)
                print(f"{p}%分位点: {value:.6f}")

        # 显示一些示例
        print(f"\n=== 示例数据 ===")
        print("前5个cumu_prob的最后一个值:")
        for i in range(min(5, len(last_cumu_prob_values))):
            print(f"  记录{i+1}: {last_cumu_prob_values[i]}")

        # 额外分析：小于不同阈值的数量
        print(f"\n=== 额外分析 ===")
        thresholds = [-1, -5, -10, -20, -50, -100]
        for threshold in thresholds:
            count = (last_cumu_prob_series < threshold).sum()
            percentage = count / total_records * 100
            print(f"小于{threshold}的数量: {count} ({percentage:.2f}%)")

        return True

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return False


def main():
    """主函数"""
    # 设置输入文件路径
    csv_file_path = '../output/mr_fields_all_filtered.csv'

    # 检查文件是否存在
    import os
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        return

    # 执行分析
    success = analyze_cumu_prob(csv_file_path)

    if success:
        print("\n分析完成！")
    else:
        print("\n分析失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()