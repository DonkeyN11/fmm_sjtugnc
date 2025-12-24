#!/usr/bin/env python3
"""
Hypothesis test for whether positioning error vectors follow the Gaussian
distribution defined by their covariance matrices.

Method:
  - Compute Mahalanobis distance squared for each error vector:
      d^2 = e^T Sigma^{-1} e
    Under the null hypothesis (Gaussian with given Sigma), d^2 ~ chi2(df=2).
  - Aggregate across points/trajectories, plot histogram of d^2 vs chi-square PDF.
  - Report Kolmogorov–Smirnov statistic against chi2(df=2).

Usage:
  python python/check_error_gaussian.py \
    --points python/output/monte_carlo/monte_carlo_points.csv \
    --truth python/output/monte_carlo/monte_carlo_truth.csv \
    --output_hist output/mahalanobis_hist.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest


def load_data(points_path: Path, truth_path: Path):
    p = pd.read_csv(points_path)
    t = pd.read_csv(truth_path)
    merged = p.merge(t, on=["traj_id", "point_idx"], suffixes=("", "_truth"))
    return merged


def compute_mahalanobis(merged: pd.DataFrame) -> np.ndarray:
    errs = merged[["x_m", "y_m"]].values - merged[["x_m_truth", "y_m_truth"]].values
    cov_xx = merged["cov_xx"].values
    cov_yy = merged["cov_yy"].values
    cov_xy = merged["cov_xy"].values

    d2 = []
    for e, cxx, cyy, cxy in zip(errs, cov_xx, cov_yy, cov_xy):
        cov = np.array([[cxx, cxy], [cxy, cyy]])
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        d2_val = float(e.T @ inv_cov @ e)
        if np.isfinite(d2_val):
            d2.append(d2_val)
    return np.array(d2)


def main():
    parser = argparse.ArgumentParser(description="Gaussianity test for error vectors vs covariance")
    parser.add_argument("--points", required=True, help="Path to points CSV (estimated positions, covariances)")
    parser.add_argument("--truth", required=True, help="Path to truth CSV")
    parser.add_argument("--output_hist", required=True, help="Output histogram PNG")
    args = parser.parse_args()

    merged = load_data(Path(args.points), Path(args.truth))
    d2 = compute_mahalanobis(merged)

    ks_stat, ks_p = kstest(d2, chi2(df=2).cdf)

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(d2, 99), 60)
    plt.hist(d2, bins=bins, density=True, alpha=0.7, label="Empirical d^2")
    x = np.linspace(0, bins[-1], 400)
    plt.plot(x, chi2(df=2).pdf(x), "r--", label="chi2 df=2 PDF")
    plt.title(f"Mahalanobis d^2 vs chi2(df=2)\nKS stat={ks_stat:.3f}, p={ks_p:.3g}, n={len(d2)}")
    plt.xlabel("Mahalanobis distance squared")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    Path(args.output_hist).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_hist, dpi=150)
    plt.close()

    print(f"Saved histogram to {args.output_hist}")
    print(f"KS statistic: {ks_stat:.4f}, p-value: {ks_p:.4g}, samples: {len(d2)}")


if __name__ == "__main__":
    main()
