#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo 2D 空间分布绘图脚本 (带比例尺版)
功能：
1. 绘制定位误差的六边形热力图 (Hexbin Heatmap)。
2. 叠加显示 95% 协方差椭圆。
3. 每个子图左下角添加自适应比例尺 (Scale Bar)，解决视觉比例误导问题。
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rcParams

# -----------------------------------------------------------------------------
# 1. 配置与辅助函数
# -----------------------------------------------------------------------------

def _configure_chinese_font():
    """配置 Matplotlib 以支持中文显示 (Linux 优先)"""
    rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC',
        'Noto Serif CJK SC',
        'AR PL UMing TW MBE',
        'AR PL UKai CN',
        'WenQuanYi Micro Hei',
        'SimHei',
        'Microsoft YaHei',
        'sans-serif'
    ]
    rcParams['axes.unicode_minus'] = False


def _calc_covariance_ellipse(points_2d, n_std=2.0):
    """计算 2D 点集的经验协方差椭圆参数"""
    if len(points_2d) < 2:
        return 0, 0, 0, 0, 0
    center = points_2d.mean(axis=0)
    cov = np.cov(points_2d, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    vals = np.maximum(vals, 0)
    width, height = 2 * n_std * np.sqrt(vals)
    return width, height, theta, center[0], center[1]


def _draw_scale_bar(ax, limit):
    """
    在子图左下角绘制自适应比例尺
    Args:
        limit: 当前坐标轴的单侧最大范围 (axis limit)
    """
    # 视窗总宽度约为 limit * 2
    # 我们希望比例尺长度大约占视窗宽度的 1/4 到 1/5
    target_len = (limit * 2) * 0.2

    # 候选的比例尺长度 (米)
    candidates = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # 找到最接近 target_len 的候选值
    bar_len = min(candidates, key=lambda x: abs(x - target_len))

    # 确定绘制位置 (左下角，留出 10% 的边距)
    # 坐标轴范围是 [-limit, limit]
    start_x = -limit * 0.85
    start_y = -limit * 0.85

    # 绘制比例尺线条 (黑色前景，白色描边以增强对比度)
    # 白色描边 (稍微粗一点)
    ax.plot([start_x, start_x + bar_len], [start_y, start_y],
            color='white', linewidth=4, solid_capstyle='butt', zorder=19)
    # 黑色实线
    ax.plot([start_x, start_x + bar_len], [start_y, start_y],
            color='black', linewidth=2, solid_capstyle='butt', zorder=20)

    # 绘制文字标签
    label = f"{int(bar_len)} m" if bar_len >= 1 else f"{bar_len} m"
    ax.text(start_x + bar_len / 2, start_y + limit * 0.05, label,
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='black', zorder=20)


# -----------------------------------------------------------------------------
# 2. 绘图逻辑
# -----------------------------------------------------------------------------

def plot_hexbin_grid(df_merged: pd.DataFrame, output_path: Path):
    """绘制多子图 Hexbin + 椭圆 + 比例尺"""
    _configure_chinese_font()

    sigmas = sorted(df_merged['sigma_m'].unique())
    n_plots = len(sigmas)

    # 布局计算
    cols = 4
    rows = math.ceil(n_plots / cols)

    # 调整 figsize，保证每个子图足够大且接近正方形
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), constrained_layout=True)

    axes_flat = axes.flatten() if n_plots > 1 else [axes]
    cmap = plt.cm.Spectral_r

    for i, sigma in enumerate(sigmas):
        ax = axes_flat[i]

        # 数据提取
        data = df_merged[df_merged['sigma_m'] == sigma]
        x_err = data['error_x'].values
        y_err = data['error_y'].values
        points_2d = np.column_stack((x_err, y_err))

        # 1. 计算协方差椭圆参数 & 确定动态范围
        w, h, ang, cx, cy = _calc_covariance_ellipse(points_2d, n_std=2.0)

        # 智能计算显示范围：
        # 基础范围是椭圆的大小
        base_limit = max(w, h) * 0.75
        # 同时要包含绝大多数散点 (99分位数)，防止极少数离群点把图拉得太远
        scatter_limit = max(np.percentile(np.abs(x_err), 99), np.percentile(np.abs(y_err), 99))

        # 最终范围取两者的较大值（保证椭圆和大部分点可见）
        limit = max(base_limit, scatter_limit)
        limit = max(limit, 1.0)  # 防止范围太小（比如全0）

        # 2. 绘制 Hexbin
        # gridsize 设为 25，mincnt=1 隐藏空白背景
        # extent 参数确保 hexbin 铺满我们计算好的范围
        hb = ax.hexbin(x_err, y_err, gridsize=25, cmap=cmap, mincnt=1,
                       extent=(-limit, limit, -limit, limit),
                       linewidths=0.2, edgecolors='gray', alpha=0.85)

        # 3. 绘制椭圆
        if w > 0 and h > 0:
            ell = Ellipse(xy=(cx, cy), width=w, height=h, angle=ang,
                          edgecolor='black', facecolor='none',
                          linewidth=1.5, linestyle='--', label='95% Cov')
            ax.add_patch(ell)
            ax.scatter(cx, cy, c='black', marker='+', s=30, zorder=10)

        # 4. 样式设置
        ax.set_title(rf"$\sigma$ = {sigma}m", fontsize=11, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')  # 关键：强制正方形
        ax.grid(True, linestyle=':', alpha=0.3)

        # 应用计算好的范围
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        # 5. 绘制比例尺 (关键修改)
        _draw_scale_bar(ax, limit)

        # 坐标轴标签简化
        if i % cols == 0:
            ax.set_ylabel('北向误差 (m)', fontsize=9)
        if i >= (rows - 1) * cols:
            ax.set_xlabel('东向误差 (m)', fontsize=9)

        # 刻度字体设小一点
        ax.tick_params(axis='both', which='major', labelsize=8)

    # 隐藏多余子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle('定位误差空间分布 (Hexbin热力图 + 95%协方差椭圆)', fontsize=14, y=1.02)

    # 共享 Colorbar
    cbar = fig.colorbar(axes_flat[0].collections[0], ax=axes, location='right', shrink=0.5, aspect=30)
    cbar.ax.set_title('Count', fontsize=9)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {output_path}")
    plt.close()


# -----------------------------------------------------------------------------
# 3. 主程序
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="绘制 Hexbin + 协方差椭圆 + 比例尺图")
    parser.add_argument("--points", default="monte_carlo_points.csv", help="观测结果 CSV")
    parser.add_argument("--truth", default="monte_carlo_truth.csv", help="真值 CSV")
    parser.add_argument("--output", default="error_hexbin_scaled.png", help="输出图片路径")
    args = parser.parse_args()

    points_path = Path(args.points)
    truth_path = Path(args.truth)

    if not points_path.exists() or not truth_path.exists():
        print("错误: 找不到输入 CSV 文件。")
        return

    print("读取数据...")
    df_points = pd.read_csv(points_path)
    df_truth = pd.read_csv(truth_path)

    print("关联并计算误差...")
    df_merged = pd.merge(df_points, df_truth, on=['traj_id', 'point_idx'], suffixes=('_est', '_true'))

    if 'sigma_m' not in df_merged.columns:
        print("错误: 缺少 'sigma_m' 列")
        return

    df_merged['error_x'] = df_merged['x_m_est'] - df_merged['x_m_true']
    df_merged['error_y'] = df_merged['y_m_est'] - df_merged['y_m_true']

    # 检查是否为经纬度坐标 (如果是，则转换为米)
    if df_merged['x_m_true'].abs().max() < 181 and df_merged['y_m_true'].abs().max() < 91:
        print("检测到经纬度坐标，正在转换为米...")
        avg_lat = df_merged['y_m_true'].mean()
        lat_to_m = 111320.0
        lon_to_m = 111320.0 * math.cos(math.radians(avg_lat))
        df_merged['error_x'] *= lon_to_m
        df_merged['error_y'] *= lat_to_m

    print(f"绘图 (共 {len(df_merged['sigma_m'].unique())} 组)...")
    plot_hexbin_grid(df_merged, Path(args.output))


if __name__ == "__main__":
    main()
