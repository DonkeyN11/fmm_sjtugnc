#!/usr/bin/env python3
"""
Plot positioning error vectors for a trajectory and overlay covariance ellipse.

Usage example:
  python python/plot_error_vectors.py \
    --points python/output/monte_carlo/monte_carlo_points.csv \
    --truth python/output/monte_carlo/monte_carlo_truth.csv \
    --traj_id 0 \
    --output output/error_plot_traj0.png

If points and truth coincide (zero error), an optional --sample_noise flag will
draw one noise sample per point from the provided covariance (cov_xx, cov_yy,
cov_xy) to visualize plausible errors.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_data(points_path: Path, truth_path: Path, traj_id: int):
    points = pd.read_csv(points_path)
    truth = pd.read_csv(truth_path)
    p = points[points["traj_id"] == traj_id].set_index("point_idx")
    t = truth[truth["traj_id"] == traj_id].set_index("point_idx")
    if p.empty or t.empty:
        raise ValueError(f"Trajectory {traj_id} not found in provided files")
    return p, t


def compute_errors(p: pd.DataFrame, t: pd.DataFrame, sample_noise: bool):
    # Align on point_idx
    aligned = p.join(t[["x_m", "y_m"]], rsuffix="_truth")
    err_x = aligned["x_m"] - aligned["x_m_truth"]
    err_y = aligned["y_m"] - aligned["y_m_truth"]
    errors = np.stack([err_x.values, err_y.values], axis=1)

    if sample_noise and np.allclose(errors, 0):
        # Draw synthetic errors from covariance per point
        covs = []
        for _, row in aligned.iterrows():
            cov = np.array([[row["cov_xx"], row["cov_xy"]], [row["cov_xy"], row["cov_yy"]]])
            covs.append(cov)
        errors = np.stack(
            [np.random.multivariate_normal(mean=[0, 0], cov=c) for c in covs],
            axis=0,
        )
    return errors


def covariance_ellipse(cov: np.ndarray, n_std: float = 2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    width, height = 2 * n_std * np.sqrt(eigvals)
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    return width, height, angle


def main():
    parser = argparse.ArgumentParser(description="Plot error vectors with covariance ellipse")
    parser.add_argument("--points", required=True, help="Path to monte_carlo_points.csv")
    parser.add_argument("--truth", required=True, help="Path to monte_carlo_truth.csv")
    parser.add_argument("--traj_id", type=int, required=True, help="Trajectory ID to plot")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--sample_noise", action="store_true", help="Sample noise if errors are zero")
    args = parser.parse_args()

    p, t = load_data(Path(args.points), Path(args.truth), args.traj_id)
    errors = compute_errors(p, t, args.sample_noise)

    # Aggregate covariance (mean of per-point covariances)
    mean_cov = np.array(
        [
            [p["cov_xx"].mean(), p["cov_xy"].mean()],
            [p["cov_xy"].mean(), p["cov_yy"].mean()],
        ]
    )

    # Stats
    corr_emp = np.corrcoef(errors[:, 0], errors[:, 1])[0, 1]
    cov_corr = mean_cov[0, 1] / math.sqrt(mean_cov[0, 0] * mean_cov[1, 1])

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
    ax.set_title(f"Trajectory {args.traj_id} errors\ncorr_emp={corr_emp:.3f}, corr_cov={cov_corr:.3f}")
    ax.legend(loc="upper right")
    # Adjust limits to show walls
    ax.set_xlim(-max_x * 1.5, max_x * 1.5)
    ax.set_ylim(-max_y * 1.5, max_y * 1.5)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
