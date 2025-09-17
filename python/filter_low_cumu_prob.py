#!/usr/bin/env python3
"""
筛选mr_fields_all_filtered.csv中最后一个cumu_prob小于10%分位点（-81.826588）的行
保存为mr_low.csv
"""

import pandas as pd
import sys


def filter_low_cumu_prob(input_csv_path, output_csv_path, threshold=-81.826588):
    """筛选最后一个cumu_prob小于阈值的行"""
    print(f"正在读取文件: {input_csv_path}")
    print(f"筛选阈值: cumu_prob < {threshold}")

    try:
        # 读取CSV文件，使用分号分隔
        df = pd.read_csv(input_csv_path, sep=';')
        print(f"成功读取 {len(df)} 条记录")

        # 检查cumu_prob列是否存在
        if 'cumu_prob' not in df.columns:
            print("错误: 文件中找不到 'cumu_prob' 列")
            print(f"可用列: {list(df.columns)}")
            return False

        # 处理cumu_prob列 - 提取每个逗号分隔字符串的最后一个值
        print("正在提取cumu_prob的最后一个值...")
        last_cumu_prob_values = []
        valid_indices = []

        for idx, value in enumerate(df['cumu_prob']):
            if pd.isna(value):
                last_cumu_prob_values.append(None)
                valid_indices.append(False)
            else:
                # 分割字符串并获取最后一个值
                str_values = str(value).split(',')
                if str_values and str_values[-1].strip():
                    try:
                        last_value = float(str_values[-1].strip())
                        last_cumu_prob_values.append(last_value)
                        valid_indices.append(True)
                    except ValueError:
                        print(f"警告: 第{idx+1}行无法解析值: {str_values[-1]}")
                        last_cumu_prob_values.append(None)
                        valid_indices.append(False)
                else:
                    last_cumu_prob_values.append(None)
                    valid_indices.append(False)

        # 转换为pandas Series
        last_cumu_prob_series = pd.Series(last_cumu_prob_values)

        # 筛选小于阈值的行
        print(f"正在筛选 cumu_prob < {threshold} 的行...")
        mask = (last_cumu_prob_series < threshold) & pd.Series(valid_indices)
        filtered_df = df[mask].copy()

        # 统计结果
        total_records = len(df)
        filtered_records = len(filtered_df)
        filter_percentage = filtered_records / total_records * 100

        print(f"\n=== 筛选结果 ===")
        print(f"总记录数: {total_records}")
        print(f"筛选后记录数: {filtered_records}")
        print(f"筛选比例: {filter_percentage:.2f}%")

        if filtered_records == 0:
            print("警告: 没有找到符合条件的记录")
            return False

        # 显示筛选后数据的基本统计
        valid_filtered_values = last_cumu_prob_series[mask]
        print(f"\n=== 筛选后cumu_prob统计 ===")
        print(f"最大值: {valid_filtered_values.max():.6f}")
        print(f"最小值: {valid_filtered_values.min():.6f}")
        print(f"平均值: {valid_filtered_values.mean():.6f}")
        print(f"中位数: {valid_filtered_values.median():.6f}")

        # 保存筛选结果
        print(f"\n正在保存筛选结果到: {output_csv_path}")
        filtered_df.to_csv(output_csv_path, index=False, sep=';')
        print(f"成功保存 {len(filtered_df)} 条记录")

        # 显示一些示例数据
        print(f"\n=== 筛选后示例数据 ===")
        print("前5个筛选出的记录的cumu_prob最后一个值:")
        filtered_values = last_cumu_prob_series[mask].head(5)
        for i, (idx, value) in enumerate(filtered_values.items()):
            print(f"  记录{i+1}: {value:.6f}")

        return True

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return False


def main():
    """主函数"""
    # 设置文件路径
    input_csv_path = '../output/mr_fields_all_filtered.csv'
    output_csv_path = 'mr_low.csv'

    # 检查输入文件是否存在
    import os
    if not os.path.exists(input_csv_path):
        print(f"错误: 输入文件 {input_csv_path} 不存在")
        return

    # 10%分位点阈值
    threshold = -81.826588

    # 执行筛选
    success = filter_low_cumu_prob(input_csv_path, output_csv_path, threshold)

    if success:
        print("\n筛选完成！")
        print(f"筛选结果已保存到: {output_csv_path}")
    else:
        print("\n筛选失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()