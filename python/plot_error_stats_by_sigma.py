#!/usr/bin/env python3
"""
高级统计绘图脚本 (中文版)：
展示不同伪距噪声(Sigma)下的定位误差。
- 上半部分：误差模长的统计分布 (箱线图)。
- 下半部分：误差的空间协方差椭圆，展示误差的方向性。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from matplotlib import rcParams

# 引入原有的数据读取逻辑
try:
    from plot_error_vectors import (
        _compute_errors_for_trajs,
        _load_sigma_map,
        _read_csv_auto,
    )
except ImportError:
    print("Error: Could not import from plot_error_vectors.py. Ensure it is available.")
    exit(1)


def _configure_chinese_font():
    """配置 Matplotlib 以支持中文显示"""
    # Linux 系统优先使用 Noto Sans CJK SC，回退到其他可用字体
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
    rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


def _calc_covariance_ellipse(points_2d, n_std=2.0):
    """
    计算2D点集的协方差椭圆参数。
    Args:
        points_2d: (N, 2) numpy array.
        n_std: 标准差倍数 (2.0 约等于 95% 置信区间).
    Returns:
        width, height, angle_degrees
    """
    if len(points_2d) < 2:
        return 0, 0, 0

    cov = np.cov(points_2d, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)

    # 排序特征值
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # 计算旋转角度 (最大特征向量与 x 轴的夹角)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # 计算轴长
    vals = np.maximum(vals, 0)
    width, height = 2 * n_std * np.sqrt(vals)
    return width, height, theta


def plot_split_stats(groups_2d: dict, output_path: Path, title: str, limit_percentile: float = 99.0):
    """
    绘制上下分屏图：上图箱线图，下图协方差椭圆。
    """
    _configure_chinese_font()  # 启用中文支持

    # 1. 数据准备
    # 按数值大小对 Sigma Key 进行排序
    sorted_sigmas_keys = sorted(groups_2d.keys(), key=lambda x: float(x))

    plot_data_list = []  # 用于箱线图
    ellipse_params = []  # 用于椭圆 [(w, h, angle), ...]

    # 全局最大尺寸，用于计算缩放比例
    max_ellipse_dim_m = 0.0

    for sigma in sorted_sigmas_keys:
        errors_2d = groups_2d[sigma]

        # 提取模长用于箱线图
        if errors_2d.ndim > 1 and errors_2d.shape[1] >= 2:
            vec = errors_2d[:, :2]  # 取前两列 (North, East)
            norm = np.linalg.norm(vec, axis=1)

            # 计算椭圆
            w, h, ang = _calc_covariance_ellipse(vec, n_std=2.0)
            ellipse_params.append((w, h, ang))
            max_ellipse_dim_m = max(max_ellipse_dim_m, w, h)
        else:
            norm = errors_2d
            ellipse_params.append((0, 0, 0))  # 无效数据

        for val in norm:
            plot_data_list.append({'Sigma': str(sigma), 'Error': val})

    df = pd.DataFrame(plot_data_list)

    # [关键修复] 计算均值，并强制按照 sorted_sigmas_keys 的顺序重排
    # 默认 groupby 会按字符串排序 (如 "10.0" 排在 "5.0" 前)，导致与图表不对应
    means = df.groupby('Sigma')['Error'].mean().reindex([str(s) for s in sorted_sigmas_keys])

    # 2. 创建画布 (2行1列)
    # hspace=0.05 大幅减少上下图之间的空白
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={'height_ratios': [1.8, 1], 'hspace': 0.05}
    )

    # 设置颜色板
    palette = sns.color_palette("viridis", len(sorted_sigmas_keys))

    # ---------------------------------------------------------
    # 上半部分：箱线图 (Boxplot)
    # ---------------------------------------------------------
    sns.boxplot(x="Sigma", y="Error", data=df, ax=ax_top,
                order=[str(s) for s in sorted_sigmas_keys],
                palette=palette, linewidth=1.2, fliersize=1, width=0.6)

    # 叠加均值线
    x_coords = np.arange(len(sorted_sigmas_keys))
    ax_top.plot(x_coords, means.values, color='red', marker='o', linestyle='--',
                linewidth=2, markersize=6, label='平均误差', zorder=10)

    # 设置上图 Y 轴限制
    y_limit = np.percentile(df['Error'], limit_percentile) * 1.2
    ax_top.set_ylim(0, y_limit)
    ax_top.set_ylabel('定位误差模长 (m)', fontsize=12)
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_top.legend(loc='upper left', frameon=True)
    ax_top.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # ---------------------------------------------------------
    # 下半部分：协方差椭圆 (Covariance Ellipses)
    # ---------------------------------------------------------
    # 计算缩放因子：我们需要把椭圆放入 X 轴的间隔中 (间隔为 1.0)
    # 设定最大椭圆占据 0.8 的宽度
    target_max_width_units = 0.8
    scale_factor = target_max_width_units / max_ellipse_dim_m if max_ellipse_dim_m > 0 else 1.0

    ax_bottom.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.5)

    for i, (w_m, h_m, angle) in enumerate(ellipse_params):
        if w_m == 0:
            continue

        # 缩放后的尺寸 (Data Units)
        w_plot = w_m * scale_factor
        h_plot = h_m * scale_factor

        # 绘制椭圆，中心在 (i, 0)
        ell = Ellipse(xy=(i, 0), width=w_plot, height=h_plot, angle=angle,
                      facecolor=palette[i], edgecolor='black', linewidth=1, alpha=0.6)
        ax_bottom.add_patch(ell)

        # 绘制中心点
        ax_bottom.scatter(i, 0, color='black', s=10, zorder=5)

    # 美化下图
    ax_bottom.set_ylabel('空间分布形态\n(经缩放)', fontsize=12)
    ax_bottom.set_xlabel(r'伪距噪声 $\sigma_{pr}$ (m)', fontsize=12)

    # 设置下图 Y 轴范围，保持对称
    # 计算最大的绘图高度
    max_plot_height = (max_ellipse_dim_m * scale_factor) / 2.0
    y_margin = max_plot_height * 1.2
    ax_bottom.set_ylim(-y_margin, y_margin)

    # 隐藏下图的 Y 刻度，因为它们被缩放过了，直接读数没有物理意义
    ax_bottom.set_yticks([])

    # 添加比例尺 (Scale Bar)
    # 我们画一个代表 N 米的线段
    ref_meters = 5.0  # 参考长度 5米
    ref_len_units = ref_meters * scale_factor

    # 在左下角添加比例尺
    rect_x = -0.4
    rect_y = -y_margin * 0.8
    ax_bottom.plot([rect_x, rect_x + ref_len_units], [rect_y, rect_y],
                   color='black', linewidth=3)
    ax_bottom.text(rect_x + ref_len_units / 2, rect_y - y_margin * 0.1, f'{ref_meters}米 参考尺',
                   ha='center', va='top', fontsize=10)

    # 强制下图保持 1:1 的视觉比例 (对于椭圆很重要)
    ax_bottom.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()


def _plot_dataset(dataset_root: Path, output_dir: Path | None, limit_percentile: float) -> Path:
    points_path = dataset_root / "raw_data" / "observations.csv"
    truth_path = dataset_root / "raw_data" / "ground_truth_points.csv"
    metadata_path = dataset_root / "raw_data" / "metadata.json"

    if not points_path.exists() or not truth_path.exists():
        print(f"Warning: Raw data files not found in {dataset_root}, skipping...")
        return Path()

    points_all = _read_csv_auto(points_path)
    truth_all = _read_csv_auto(truth_path)

    try:
        sigma_map = _load_sigma_map(metadata_path)
    except Exception:
        print(f"Warning: Could not load sigma map for {dataset_root}")
        return Path()

    # 构建 sigma 分组：映射从 sigma 值到轨迹 ID 列表
    sigma_groups: dict[float, list[int]] = {}
    for traj_id, sigma in sigma_map.items():
        # 如果 sigma_map 中的 traj_id 是字符串，转换为整数
        traj_id_int = int(traj_id) if isinstance(traj_id, str) else traj_id
        sigma_groups.setdefault(float(sigma), []).append(traj_id_int)

    # 计算每个 sigma 组的 2D 误差向量
    groups_2d = {}
    for sigma, traj_ids in sigma_groups.items():
        errors = _compute_errors_for_trajs(points_all, truth_all, traj_ids)
        if errors.size > 0:
            groups_2d[sigma] = errors

    if not groups_2d:
        return Path()

    title = f"定位误差-伪距误差分析"
    out_dir = output_dir if output_dir else (dataset_root / "statistic")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{dataset_root.name}_error_stats_split.png"

    plot_split_stats(groups_2d, output_path, title, limit_percentile)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制分屏误差统计图 (箱线图 + 椭圆)")
    parser.add_argument(
        "--dataset_roots",
        nargs="+",
        required=True,
        help="数据集根目录 (每个目录需包含 raw_data/observations.csv 和 metadata.json)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录 (默认: <dataset_root>/statistic)",
    )
    parser.add_argument(
        "--limit_percentile",
        type=float,
        default=98.0,
        help="Y轴限制的百分位数",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    for dataset_root in [Path(p) for p in args.dataset_roots]:
        _plot_dataset(dataset_root, output_dir, args.limit_percentile)


if __name__ == "__main__":
    main()
