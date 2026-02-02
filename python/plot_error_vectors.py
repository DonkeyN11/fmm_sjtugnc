#!/usr/bin/env python3
"""
Plot positioning error vectors for a trajectory and overlay covariance ellipse.

Usage example:
  python python/plot_error_vectors.py \
    --points simulation/raw_data/observations.csv \
    --truth simulation/raw_data/ground_truth_points.csv \
    --traj_id 0 \
    --output simulation/statistic/error_plot_traj0.png \
    --style 2d


If points and truth coincide (zero error), an optional --sample_noise flag will
draw one noise sample per point from the provided covariance (cov_xx, cov_yy,
cov_xy) to visualize plausible errors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _read_csv_auto(path: Path) -> pd.DataFrame:
    # 自动识别分隔符：优先尝试 ';'，如果只有一列则回退到 ','。
    df = pd.read_csv(path, sep=';')
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=',')
    return df


def _pick_id_seq_columns(df: pd.DataFrame) -> tuple[str, str]:
    # 兼容不同数据格式：仿真数据使用 id/seq，蒙特卡洛使用 traj_id/point_idx。
    if {"id", "seq"}.issubset(df.columns):
        return "id", "seq"
    if {"traj_id", "point_idx"}.issubset(df.columns):
        return "traj_id", "point_idx"
    raise ValueError("Could not find trajectory/id and sequence columns in input data")


def _pick_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    # 同时支持以米为单位的 x_m/y_m 或原始 x/y 字段。
    if {"x_m", "y_m"}.issubset(df.columns):
        return "x_m", "y_m"
    if {"x", "y"}.issubset(df.columns):
        return "x", "y"
    raise ValueError("Could not find x/y columns in input data")


def load_data(points_path: Path, truth_path: Path, traj_id: int):
    # 读取观测与真值，并按轨迹 ID 过滤、以序号对齐。
    points = _read_csv_auto(points_path)
    truth = _read_csv_auto(truth_path)
    id_col_p, seq_col_p = _pick_id_seq_columns(points)
    id_col_t, seq_col_t = _pick_id_seq_columns(truth)
    p = points[points[id_col_p] == traj_id].set_index(seq_col_p)
    t = truth[truth[id_col_t] == traj_id].set_index(seq_col_t)
    if p.empty or t.empty:
        raise ValueError(f"Trajectory {traj_id} not found in provided files")
    return p, t


def compute_errors(p: pd.DataFrame, t: pd.DataFrame, sample_noise: bool):
    # 对齐真值与观测；生成每个点的二维误差向量 (x_error, y_error)。
    x_p, y_p = _pick_xy_columns(p)
    x_t, y_t = _pick_xy_columns(t)
    aligned = p.join(t[[x_t, y_t]], rsuffix="_truth")
    x_truth = f"{x_t}_truth" if x_t in p.columns else x_t
    y_truth = f"{y_t}_truth" if y_t in p.columns else y_t
    err_x = aligned[x_p] - aligned[x_truth]
    err_y = aligned[y_p] - aligned[y_truth]
    errors = np.stack([err_x.values, err_y.values], axis=1)

    # 有些数据直接提供协方差，有些只提供标准差；二者都可用于采样。
    has_cov = {"cov_xx", "cov_yy", "cov_xy"}.issubset(aligned.columns)
    has_sd = {"sde", "sdn"}.issubset(aligned.columns)
    if sample_noise and np.allclose(errors, 0):
        if not (has_cov or has_sd):
            raise ValueError("sample_noise requires cov_xx/cov_yy/cov_xy or sde/sdn columns")
        # 当误差全为 0 时，使用协方差为每个点生成合成误差。
        covs = []
        for _, row in aligned.iterrows():
            if has_cov:
                cov = np.array(
                    [[row["cov_xx"], row["cov_xy"]], [row["cov_xy"], row["cov_yy"]]]
                )
            else:
                cov_xy = row["sdne"] if "sdne" in aligned.columns else 0.0
                cov = np.array([[row["sde"] ** 2, cov_xy], [cov_xy, row["sdn"] ** 2]])
            covs.append(cov)
        errors = np.stack(
            [np.random.multivariate_normal(mean=[0, 0], cov=c) for c in covs],
            axis=0,
        )
    return errors


def _compute_errors_for_trajs(points: pd.DataFrame, truth: pd.DataFrame, traj_ids: list[int]) -> np.ndarray:
    if not traj_ids:
        return np.zeros((0, 2))
    id_col_p, seq_col_p = _pick_id_seq_columns(points)
    id_col_t, seq_col_t = _pick_id_seq_columns(truth)
    x_p, y_p = _pick_xy_columns(points)
    x_t, y_t = _pick_xy_columns(truth)
    p = points[points[id_col_p].isin(traj_ids)].set_index([id_col_p, seq_col_p])
    t = truth[truth[id_col_t].isin(traj_ids)].set_index([id_col_t, seq_col_t])
    aligned = p.join(t[[x_t, y_t]], rsuffix="_truth")
    x_truth = f"{x_t}_truth" if x_t in p.columns else x_t
    y_truth = f"{y_t}_truth" if y_t in p.columns else y_t
    err_x = aligned[x_p] - aligned[x_truth]
    err_y = aligned[y_p] - aligned[y_truth]
    mask = np.isfinite(err_x.values) & np.isfinite(err_y.values)
    return np.stack([err_x.values[mask], err_y.values[mask]], axis=1)


def _mean_cov_from_points(points: pd.DataFrame, errors: np.ndarray) -> np.ndarray:
    if {"cov_xx", "cov_yy", "cov_xy"}.issubset(points.columns):
        return np.array(
            [
                [points["cov_xx"].mean(), points["cov_xy"].mean()],
                [points["cov_xy"].mean(), points["cov_yy"].mean()],
            ]
        )
    if {"sde", "sdn"}.issubset(points.columns):
        cov_xy = points["sdne"].mean() if "sdne" in points.columns else 0.0
        return np.array(
            [
                [(points["sde"] ** 2).mean(), cov_xy],
                [cov_xy, (points["sdn"] ** 2).mean()],
            ]
        )
    return np.cov(errors.T) if errors.shape[0] > 1 else np.zeros((2, 2))


def _load_sigma_map(metadata_path: Path) -> dict[int, float]:
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    sigma_map: dict[int, float] = {}
    for traj in meta.get("trajectories", []):
        traj_id = traj.get("id")
        if traj_id is None:
            continue
        sigma_val = (
            traj.get("sigma_pr")
            if "sigma_pr" in traj
            else traj.get("sigma_m", traj.get("sigma"))
        )
        if sigma_val is None:
            continue
        sigma_map[int(traj_id)] = float(sigma_val)
    return sigma_map


def _plot_sigma_group_grid(
    groups: list[tuple[float, np.ndarray, np.ndarray]],
    output: Path,
    title: str,
    limit_percentile: float,
    lims: tuple[float, float] | None,
) -> None:
    count = len(groups)
    if count == 0:
        raise ValueError("No sigma groups to plot")
    ncols = min(4, max(2, math.ceil(math.sqrt(count))))
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 3.4), squeeze=False)

    for ax, (sigma, errors, mean_cov) in zip(axes.flat, groups):
        if errors.size == 0:
            ax.axis("off")
            ax.set_title(f"sigma_pr={sigma:g} m", fontsize=9)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            continue
        mu = errors.mean(axis=0)
        errors_c = errors - mu
        if errors_c.shape[0] >= 2000:
            ax.hexbin(errors_c[:, 0], errors_c[:, 1], gridsize=35, cmap="Blues", mincnt=1)
        else:
            ax.scatter(
                errors_c[:, 0],
                errors_c[:, 1],
                s=4,
                alpha=0.2,
                color="tab:blue",
                edgecolors="none",
            )
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
        w, h, ang = covariance_ellipse(mean_cov, n_std=2.0)
        ax.add_patch(
            Ellipse(
                (0, 0),
                width=w,
                height=h,
                angle=ang,
                fill=False,
                lw=1.5,
                color="black",
                linestyle="--",
            )
        )
        if lims is None:
            lim_x, lim_y = _limits_from_errors(errors_c, limit_percentile)
        else:
            lim_x, lim_y = lims
        lim = max(lim_x, lim_y)
        ax.set_xlim(-lim * 1.2, lim * 1.2)
        ax.set_ylim(-lim * 1.2, lim * 1.2)
        ax.set_aspect("equal", adjustable="box")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        stats = (
            f"N={errors.shape[0]}\n"
            f"mu=({mu[0]:.2f},{mu[1]:.2f})\n"
            f"std=({errors[:,0].std():.2f},{errors[:,1].std():.2f})"
        )
        ax.text(
            0.02,
            0.98,
            stats,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
        )
        ax.set_title(f"sigma_pr={sigma:g} m", fontsize=9)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes.flat[len(groups) :]:
        ax.axis("off")

    fig.suptitle(title, y=0.98, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def covariance_ellipse(cov: np.ndarray, n_std: float = 2.0):
    # 将 2x2 协方差矩阵转为椭圆的宽、高、旋转角。
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    width, height = 2 * n_std * np.sqrt(eigvals)
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle


def _kde_1d(data: np.ndarray, xs: np.ndarray) -> np.ndarray:
    # 轻量 KDE（高斯核 + Silverman 带宽），避免额外依赖 SciPy。
    n = data.size
    if n == 0:
        return np.zeros_like(xs)
    std = np.std(data)
    bw = 1.06 * std * (n ** (-1 / 5)) if std > 0 else 1.0
    bw = max(bw, 1e-6)
    diffs = (xs[:, None] - data[None, :]) / bw
    kernel = np.exp(-0.5 * diffs ** 2) / math.sqrt(2 * math.pi)
    return kernel.mean(axis=1) / bw


def _limits_from_errors(errors: np.ndarray, percentile: float) -> tuple[float, float]:
    # 使用误差的指定分位数确定可视范围，抑制异常值。
    lim_x = np.percentile(np.abs(errors[:, 0]), percentile)
    lim_y = np.percentile(np.abs(errors[:, 1]), percentile)
    lim_x = max(lim_x, 1e-6)
    lim_y = max(lim_y, 1e-6)
    return lim_x, lim_y


def plot_2d(
    errors: np.ndarray,
    mean_cov: np.ndarray,
    traj_id: int,
    output: Path,
    corr_emp: float,
    cov_corr: float,
    lims: tuple[float, float],
    constellation: tuple[np.ndarray, np.ndarray] | None,
):
    # 2D 视图：主散点 + 2 个边缘 KDE + 右上角统计信息。
    fig = plt.figure(figsize=(8, 8))
    # Use explicit axes to guarantee a square main plot area.
    left = 0.1
    bottom = 0.1
    size = 0.6
    gap = 0.02
    hist_thickness = 0.18
    stats_size = 0.18

    ax_scatter = fig.add_axes([left, bottom, size, size])
    ax_histx = fig.add_axes([left, bottom + size + gap, size, hist_thickness], sharex=ax_scatter)
    ax_histy = fig.add_axes([left + size + gap, bottom, hist_thickness, size], sharey=ax_scatter)
    ax_stats = None
    ax_sky = None
    if constellation is None:
        ax_stats = fig.add_axes([left + size + gap, bottom + size + gap, stats_size, stats_size])
    else:
        ax_sky = fig.add_axes(
            [left + size + gap, bottom + size + gap, stats_size, stats_size],
            projection="polar",
        )

    mu = errors.mean(axis=0)
    errors_c = errors - mu

    # 若样本数较大，用 hexbin 更清晰；否则用轻量散点。
    if errors_c.shape[0] >= 1000:
        ax_scatter.hexbin(errors_c[:, 0], errors_c[:, 1], gridsize=40, cmap="Blues", mincnt=1)
    else:
        ax_scatter.scatter(
            errors_c[:, 0],
            errors_c[:, 1],
            s=6,
            alpha=0.2,
            color="tab:blue",
            edgecolors="none",
        )
    ax_scatter.axhline(0, color="0.7", lw=0.8)
    ax_scatter.axvline(0, color="0.7", lw=0.8)

    # 协方差椭圆绘制在零点（已按均值居中），突出模型预测的相关结构。
    for n_std, linestyle, label in [
        (1.0, "-", "1σ (WLS covariance)"),
        (2.0, "--", "2σ (WLS covariance)"),
    ]:
        w, h, ang = covariance_ellipse(mean_cov, n_std=n_std)
        ell = Ellipse(
            (0, 0),
            width=w,
            height=h,
            angle=ang,
            fill=False,
            lw=2.5,
            color="black",
            linestyle=linestyle,
            label=label,
        )
        ax_scatter.add_patch(ell)

    # 使用统一轴范围，确保 X/Y 等比例，避免视觉拉伸。
    lim_x, lim_y = lims
    lim = max(lim_x, lim_y)
    ax_scatter.set_xlim(-lim * 1.2, lim * 1.2)
    ax_scatter.set_ylim(-lim * 1.2, lim * 1.2)
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_scatter.set_aspect("equal", adjustable="box")
    if hasattr(ax_scatter, "set_box_aspect"):
        ax_scatter.set_box_aspect(1)

    # 边缘 KDE 曲线，提供平滑的一维误差分布。
    xs = np.linspace(*ax_scatter.get_xlim(), 300)
    ys = np.linspace(*ax_scatter.get_ylim(), 300)
    kde_x = _kde_1d(errors_c[:, 0], xs)
    kde_y = _kde_1d(errors_c[:, 1], ys)
    ax_histx.plot(xs, kde_x, color="tab:blue", lw=2.2)
    ax_histy.plot(kde_y, ys, color="tab:orange", lw=2.2)
    ax_histx.set_ylabel("KDE (X)")
    ax_histy.set_xlabel("KDE (Y)")
    ax_histx.margins(x=0)
    ax_histy.margins(y=0)

    # 统计信息移动到右上角空白格，避免遮挡主图。
    stats = (
        f"N={errors.shape[0]}\n"
        f"mu=({mu[0]:.2f}, {mu[1]:.2f}) m\n"
        f"sigma=({errors[:,0].std():.2f}, {errors[:,1].std():.2f}) m\n"
        f"rho_emp={corr_emp:.3f}\n"
        f"rho_cov={cov_corr:.3f}"
    )
    if ax_stats is not None:
        ax_stats.axis("off")
        ax_stats.text(0.0, 1.0, stats, va="top", ha="left", fontsize=9)
    else:
        ax_scatter.text(
            0.02,
            0.98,
            stats,
            transform=ax_scatter.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    if ax_sky is not None and constellation is not None:
        az_deg, el_deg = constellation
        theta = np.deg2rad(az_deg)
        r = 90.0 - el_deg
        ax_sky.scatter(theta, r, c="tab:purple", s=14, alpha=0.8)
        ax_sky.set_theta_zero_location("N")
        ax_sky.set_theta_direction(-1)
        ax_sky.set_rlim(90, 0)
        ax_sky.set_yticks([0, 30, 60, 90])
        ax_sky.set_yticklabels(["90°", "60°", "30°", "0°"])
        ax_sky.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax_sky.set_xticklabels(["N", "E", "S", "W"])
        ax_sky.set_title("Skyplot", fontsize=9, pad=6)

    ax_scatter.set_xlabel("Error X (m)")
    ax_scatter.set_ylabel("Error Y (m)")
    ax_histx.tick_params(labelbottom=False)
    ax_histy.tick_params(labelleft=False)
    fig.suptitle(f"Trajectory {traj_id} errors", y=0.98)
    ax_scatter.legend(loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_3d(errors: np.ndarray, mean_cov: np.ndarray, traj_id: int, output: Path, corr_emp: float, cov_corr: float):
    # 3D 视图仅用于调试展示，默认不会用于论文图。
    # 3D layout: XY plane on bottom; hist Y on left wall; hist X on back wall.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Bottom: error points and covariance ellipse
    ax.scatter(errors[:, 0], errors[:, 1], zs=0, zdir="z", c="tab:blue", s=12, alpha=0.6, label="Errors")
    w, h, ang = covariance_ellipse(mean_cov, n_std=2.0)
    theta = np.linspace(0, 2 * np.pi, 200)
    ellipse = np.stack([0.5 * w * np.cos(theta), 0.5 * h * np.sin(theta)], axis=1)
    rot = np.array(
        [
            [np.cos(np.radians(ang)), -np.sin(np.radians(ang))],
            [np.sin(np.radians(ang)), np.cos(np.radians(ang))],
        ]
    )
    ellipse_rot = ellipse @ rot.T
    ax.plot(ellipse_rot[:, 0], ellipse_rot[:, 1], zs=0, zdir="z", color="tab:red", lw=2, label="Cov ellipse (2σ)")

    # Histograms on walls
    # X hist on back (y at max)
    max_y = max(errors[:, 1].max(), abs(errors[:, 1].min()))
    max_x = max(errors[:, 0].max(), abs(errors[:, 0].min()))
    hist_x, bins_x = np.histogram(errors[:, 0], bins=40)
    bin_centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
    ax.bar(bin_centers_x, hist_x, zs=max_y * 1.1, zdir="y", alpha=0.5, width=bins_x[1]-bins_x[0], color="tab:green")
    ax.text(0, max_y * 1.2, hist_x.max() * 0.6, f"X mean={errors[:,0].mean():.2f}\nstd={errors[:,0].std():.2f}", color="tab:green")

    # Y hist on left (x at min)
    hist_y, bins_y = np.histogram(errors[:, 1], bins=40)
    bin_centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
    ax.bar(bin_centers_y, hist_y, zs=-max_x * 1.1, zdir="x", alpha=0.5, width=bins_y[1]-bins_y[0], color="tab:orange")
    ax.text(-max_x * 1.2, 0, hist_y.max() * 0.6, f"Y mean={errors[:,1].mean():.2f}\nstd={errors[:,1].std():.2f}", color="tab:orange")

    ax.set_xlabel("Error X (m)")
    ax.set_ylabel("Error Y (m)")
    ax.set_zlabel("Counts (hist walls)")
    ax.set_title(f"Trajectory {traj_id} errors\ncorr_emp={corr_emp:.3f}, corr_cov={cov_corr:.3f}")
    ax.legend(loc="upper right")
    # Adjust limits to show walls
    ax.set_xlim(-max_x * 1.5, max_x * 1.5)
    ax.set_ylim(-max_y * 1.5, max_y * 1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main():
    # 命令行入口：加载数据、计算误差与协方差，绘制指定风格图像。
    parser = argparse.ArgumentParser(description="Plot error vectors with covariance ellipse")
    parser.add_argument("--points", required=True, help="Path to monte_carlo_points.csv")
    parser.add_argument("--truth", required=True, help="Path to monte_carlo_truth.csv")
    parser.add_argument("--traj_id", type=int, required=True, help="Trajectory ID to plot")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--style", choices=["3d", "2d"], default="2d", help="Plot style (3d is debug-only)")
    parser.add_argument(
        "--constellation",
        type=Path,
        default=None,
        help="Optional constellation CSV (from generate_data_cmm.py --export-constellation).",
    )
    parser.add_argument(
        "--global_limits",
        action="store_true",
        help="Use percentile limits computed across all trajectories for consistent scales",
    )
    parser.add_argument(
        "--limit_percentile",
        type=float,
        default=99.0,
        help="Percentile for axis limits (used with and without --global_limits)",
    )
    parser.add_argument("--sample_noise", action="store_true", help="Sample noise if errors are zero")
    args = parser.parse_args()

    p, t = load_data(Path(args.points), Path(args.truth), args.traj_id)
    errors = compute_errors(p, t, args.sample_noise)

    # Aggregate covariance (mean of per-point covariances); fall back to empirical.
    if {"cov_xx", "cov_yy", "cov_xy"}.issubset(p.columns):
        mean_cov = np.array(
            [
                [p["cov_xx"].mean(), p["cov_xy"].mean()],
                [p["cov_xy"].mean(), p["cov_yy"].mean()],
            ]
        )
    elif {"sde", "sdn"}.issubset(p.columns):
        cov_xy = p["sdne"].mean() if "sdne" in p.columns else 0.0
        mean_cov = np.array(
            [
                [(p["sde"] ** 2).mean(), cov_xy],
                [cov_xy, (p["sdn"] ** 2).mean()],
            ]
        )
    else:
        mean_cov = np.cov(errors.T) if errors.shape[0] > 1 else np.zeros((2, 2))

    # Stats
    corr_emp = np.corrcoef(errors[:, 0], errors[:, 1])[0, 1]
    denom = math.sqrt(mean_cov[0, 0] * mean_cov[1, 1])
    cov_corr = mean_cov[0, 1] / denom if denom > 0 else float("nan")

    errors_centered = errors - errors.mean(axis=0)
    if args.global_limits:
        points_all = _read_csv_auto(Path(args.points))
        truth_all = _read_csv_auto(Path(args.truth))
        id_col_p, seq_col_p = _pick_id_seq_columns(points_all)
        id_col_t, seq_col_t = _pick_id_seq_columns(truth_all)
        x_p, y_p = _pick_xy_columns(points_all)
        x_t, y_t = _pick_xy_columns(truth_all)
        p_all = points_all.set_index([id_col_p, seq_col_p])
        t_all = truth_all.set_index([id_col_t, seq_col_t])
        aligned_all = p_all.join(t_all[[x_t, y_t]], rsuffix="_truth")
        x_truth = f"{x_t}_truth" if x_t in p_all.columns else x_t
        y_truth = f"{y_t}_truth" if y_t in p_all.columns else y_t
        err_x = aligned_all[x_p] - aligned_all[x_truth]
        err_y = aligned_all[y_p] - aligned_all[y_truth]
        mask = np.isfinite(err_x.values) & np.isfinite(err_y.values)
        errors_all = np.stack([err_x.values[mask], err_y.values[mask]], axis=1)
        if errors_all.size:
            errors_all_centered = errors_all - errors_all.mean(axis=0)
            lims = _limits_from_errors(errors_all_centered, args.limit_percentile)
        else:
            lims = _limits_from_errors(errors_centered, args.limit_percentile)
    else:
        lims = _limits_from_errors(errors_centered, args.limit_percentile)

    if args.style == "2d":
        constellation = None
        if args.constellation is not None:
            df_const = _read_csv_auto(args.constellation)
            if {"id", "azimuth_deg", "elevation_deg"}.issubset(df_const.columns):
                row = df_const[df_const["id"] == args.traj_id]
                if not row.empty:
                    az = np.array(json.loads(row.iloc[0]["azimuth_deg"]), dtype=float)
                    el = np.array(json.loads(row.iloc[0]["elevation_deg"]), dtype=float)
                    constellation = (az, el)
        plot_2d(errors, mean_cov, args.traj_id, Path(args.output), corr_emp, cov_corr, lims, constellation)
    else:
        plot_3d(errors, mean_cov, args.traj_id, Path(args.output), corr_emp, cov_corr)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
