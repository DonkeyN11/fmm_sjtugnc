#!/usr/bin/env python3
"""
Experiment 1: GNSS Covariance Model Validation

Validates that empirically estimated covariance matrices correctly characterize
the actual positioning error distribution. For each sigma level:

Plots:
  1. Chi-square histogram: d^2 distribution vs chi^2(df=2) PDF
  2. P-P plot: empirical CDF vs theoretical chi^2 CDF
  3. Whitened error scatter (should be isotropic N(0,I))
  4. Rayleigh CDF of error magnitude vs theoretical

Output: summary_table.csv with KS statistics across all sigma levels.

Usage:
  python experiments/scripts/exp1_covariance_validation.py \
    --data-root experiments/data \
    --output-dir experiments/output/1_covariance_validation
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    DPI, COLOR_CMM, COLOR_FMM,
    _read_csv_auto, compute_mahalanobis_d2, _whiten_error, _rayleigh_cdf,
    load_dataset,
)

CHI2_2_CDF = lambda x: 1.0 - math.exp(-0.5 * max(x, 0.0))


def analyze_single_dataset(dataset_path: Path, output_dir: Path) -> dict:
    """Analyze one dataset: compute d^2, produce 4-panel figure, return KS stats."""
    # Use the sigma directory name as label (e.g. "sigma_10")
    parts = dataset_path.parts
    label = next((p for p in parts if p.startswith("sigma_")), dataset_path.name)
    obs, truth_pts, meta = load_dataset(dataset_path)

    if obs.empty or truth_pts is None or truth_pts.empty:
        print(f"  SKIP {label}: missing data")
        return {}

    # Match observations to truth by id/seq
    if "id" in obs.columns and "seq" in obs.columns:
        merged = obs.merge(truth_pts, on=["id", "seq"], suffixes=("", "_truth"))
    else:
        # Simple alignment by index
        merged = obs.join(truth_pts, rsuffix="_truth").iloc[:min(len(obs), len(truth_pts))]
    if merged.empty:
        return {}

    # Compute error vectors
    err_x = merged["x"].values - merged["x_truth"].values
    err_y = merged["y"].values - merged["y_truth"].values
    errors = np.stack([err_x, err_y], axis=1)

    # Read covariance from sde, sdn, sdne columns
    covs = np.zeros((len(merged), 2, 2))
    for i, (_, row) in enumerate(merged.iterrows()):
        sde = float(row.get("sde", 0))
        sdn = float(row.get("sdn", 0))
        sdne = float(row.get("sdne", 0))
        covs[i] = np.array([[sde * sde, sdne], [sdne, sdn * sdn]])

    # Compute squared Mahalanobis distances for cov-based and iso models
    d2_cov = compute_mahalanobis_d2(errors, covs)
    d2_cov = d2_cov[np.isfinite(d2_cov)]

    # Isotropic: use mean equivalent sigma
    sigma_iso = np.sqrt(np.mean(np.trace(covs, axis1=1, axis2=2) / 2.0))
    d2_iso = (err_x ** 2 + err_y ** 2) / (sigma_iso ** 2)
    d2_iso = d2_iso[np.isfinite(d2_iso)]

    if len(d2_cov) == 0 or len(d2_iso) == 0:
        return {}

    # Sigma from metadata or directory name
    sigma = 0.0
    if meta:
        args = meta.get("arguments", {})
        sigma = args.get("min_sigma_pr", args.get("max_sigma_pr", 0.0))
    if sigma == 0.0:
        for part in dataset_path.parts:
            if "sigma_" in part:
                try: sigma = float(part.replace("sigma_", "").split("_")[0]); break
                except: pass

    # Whitened errors
    whitened = []
    for i in range(min(3000, len(errors))):
        w = _whiten_error(errors[i], covs[i])
        if w is not None and np.all(np.isfinite(w)):
            whitened.append(w)

    # Radial error
    radial = np.sqrt(err_x ** 2 + err_y ** 2)
    radial = radial[np.isfinite(radial)]
    sigma_eq = np.sqrt(np.maximum(np.trace(covs, axis1=1, axis2=2) / 2.0, 1e-9))

    # KS statistic for covariance-based
    d2_sorted = np.sort(d2_cov)
    n = len(d2_sorted)
    emp_cdf = np.arange(1, n + 1) / n
    theo_cdf = np.array([CHI2_2_CDF(x) for x in d2_sorted])
    ks_cov = float(np.max(np.abs(emp_cdf - theo_cdf)))

    # KS for isotropic
    d2_sorted_iso = np.sort(d2_iso)
    n_iso = len(d2_sorted_iso)
    emp_cdf_iso = np.arange(1, n_iso + 1) / n_iso
    theo_cdf_iso = np.array([CHI2_2_CDF(x) for x in d2_sorted_iso])
    ks_iso = float(np.max(np.abs(emp_cdf_iso - theo_cdf_iso)))

    # ── 4-panel figure ──
    hist_max = min(20.0, np.percentile(np.concatenate([d2_cov, d2_iso]), 99))
    xs = np.linspace(0, hist_max, 400)
    chi_pdf = 0.5 * np.exp(-0.5 * xs)
    bins = min(80, max(30, int(hist_max * 8)))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    ax_hist, ax_pp, ax_white, ax_rad = axes.flat

    # (a) Chi-square histogram
    ax_hist.hist(d2_cov, bins=bins, density=True, alpha=0.6, color=COLOR_CMM, label="Covariance-based")
    ax_hist.hist(d2_iso, bins=bins, density=True, alpha=0.6, color=COLOR_FMM, label="Isotropic")
    ax_hist.plot(xs, chi_pdf, "k-", lw=1.5, label=r"$\chi^2(2)$ PDF")
    ax_hist.set_xlabel(r"Mahalanobis $d^2$")
    ax_hist.set_ylabel("PDF")
    ax_hist.set_title(rf"(a) Normalized innovation ($\sigma$={sigma:.0f}m)")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(True, alpha=0.3)

    # (b) P-P plot
    ax_pp.plot([0, 1], [0, 1], color="0.5", lw=1.0, linestyle="--")
    for d2, color, plot_label in [(d2_cov, COLOR_CMM, "Covariance-based"), (d2_iso, COLOR_FMM, "Isotropic")]:
        sorted_vals = np.sort(d2)
        n_v = len(sorted_vals)
        emp = np.arange(1, n_v + 1) / n_v
        theo = np.array([CHI2_2_CDF(x) for x in sorted_vals])
        ax_pp.plot(theo, emp, color=color, lw=1.2, label=f"{plot_label} (KS={_ks(d2):.4f})")
    ax_pp.set_xlabel("Theoretical CDF")
    ax_pp.set_ylabel("Empirical CDF")
    ax_pp.set_title(r"(b) P-P plot ($\chi^2$)")
    ax_pp.legend(loc="lower right")
    ax_pp.grid(True, alpha=0.3)

    # (c) Whitened errors
    if whitened:
        w = np.asarray(whitened)
        ax_white.scatter(w[:, 0], w[:, 1], s=4, alpha=0.25, color=COLOR_CMM, edgecolors="none")
        theta = np.linspace(0, 2 * math.pi, 200)
        ax_white.plot(np.cos(theta), np.sin(theta), "k-", lw=1.2, label=r"1$\sigma$")
        ax_white.plot(2 * np.cos(theta), 2 * np.sin(theta), "k--", lw=0.8, label=r"2$\sigma$")
        ax_white.set_aspect("equal", adjustable="box")
        ax_white.legend(loc="upper right", fontsize=7)
    ax_white.set_xlabel("Whitened X")
    ax_white.set_ylabel("Whitened Y")
    ax_white.set_title("(c) Whitened errors")
    ax_white.grid(True, alpha=0.3)

    # (d) Rayleigh CDF
    r_sorted = np.sort(radial)
    emp_cdf_r = np.arange(1, r_sorted.size + 1) / r_sorted.size
    r_grid = np.linspace(0, np.percentile(radial, 99.5), 300)
    cov_cdf = _rayleigh_cdf(r_grid[:, None], sigma_eq[None, :]).mean(axis=1)
    iso_cdf_vals = _rayleigh_cdf(r_grid, np.full_like(r_grid, sigma_iso))
    ax_rad.step(r_sorted, emp_cdf_r, where="post", color=COLOR_CMM, lw=1.2, label="MC radial CDF")
    ax_rad.plot(r_grid, cov_cdf, color=COLOR_CMM, lw=1.0, linestyle="--", label="Rayleigh (cov)")
    ax_rad.plot(r_grid, iso_cdf_vals, color=COLOR_FMM, lw=1.0, linestyle="--", label="Rayleigh (iso)")
    ax_rad.set_xlabel("$|e|$ (m)")
    ax_rad.set_ylabel("CDF")
    ax_rad.set_title("(d) Radial error distribution")
    ax_rad.legend(loc="lower right", fontsize=7)
    ax_rad.grid(True, alpha=0.3)

    fig.suptitle(rf"Covariance Model Validation ($\sigma$ = {sigma:.0f} m)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = output_dir / f"validation_sigma_{sigma:02.0f}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    print(f"  {label}: KS_cov={ks_cov:.4f}, KS_iso={ks_iso:.4f}, n={n}")
    return {"dataset": label, "sigma": sigma, "ks_cov": ks_cov, "ks_iso": ks_iso,
            "n_d2_cov": n, "n_d2_iso": n_iso, "fig": str(out.name)}


def _ks(d2):
    sorted_vals = np.sort(d2)
    n_v = len(sorted_vals)
    emp = np.arange(1, n_v + 1) / n_v
    theo = np.array([CHI2_2_CDF(x) for x in sorted_vals])
    return float(np.max(np.abs(emp - theo)))


def main():
    parser = argparse.ArgumentParser(description="Covariance model validation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/output/1_covariance_validation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(args.data_root.glob("sigma_*/no_occlusion/no_fault"))
    if not dataset_dirs:
        dataset_dirs = sorted([d for d in args.data_root.glob("sigma_*")
                               if d.is_dir() and (d / "observations.csv").exists()])

    if not dataset_dirs:
        raise SystemExit(f"No datasets found under {args.data_root}")

    print(f"Analyzing {len(dataset_dirs)} datasets:")
    rows = []
    for d in dataset_dirs:
        print(f"  {d.relative_to(args.data_root)}")
        row = analyze_single_dataset(d, args.output_dir)
        if row:
            rows.append(row)

    if rows:
        out = args.output_dir / "summary_table.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Summary table: {out}")

    print("Done.")


if __name__ == "__main__":
    main()
