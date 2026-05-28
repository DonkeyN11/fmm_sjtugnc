#!/usr/bin/env python3
"""
Experiment 2: Protection Level Coverage / Stanford Plot

Evaluates whether the Protection Level (PL) provides the claimed integrity
bound. Includes fault injection scenarios.

Plots:
  1. Stanford plot: error vs PL scatter with diagonal integrity line
  2. Summary table: P_md, P_fa across all conditions

Usage:
  python experiments/scripts/exp2_stanford_pl.py \
    --data-root experiments/data \
    --output-dir experiments/output/2_stanford
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    DPI, COLOR_CMM, COLOR_FMM,
    load_dataset, compute_stanford_metrics,
)


def stanford_single(dataset_path: Path, output_dir: Path) -> dict:
    """Generate Stanford plot and compute P_md/P_fa for one dataset."""
    # Use sigma directory name as label
    parts = dataset_path.parts
    label = next((p for p in parts if p.startswith("sigma_")), dataset_path.name)
    # Append condition info
    if "with_occlusion" in parts:
        label += "_occ"
    if "with_fault" in parts:
        label += "_fault"

    obs, truth_pts, meta = load_dataset(dataset_path)

    if obs.empty or truth_pts is None or truth_pts.empty:
        print(f"  SKIP {label}: missing data")
        return {}

    # Match observations to truth
    if "id" in obs.columns and "seq" in obs.columns:
        merged = obs.merge(truth_pts, on=["id", "seq"], suffixes=("", "_truth"))
    else:
        merged = obs.join(truth_pts, rsuffix="_truth").iloc[:min(len(obs), len(truth_pts))]
    if merged.empty:
        return {}

    # Compute horizontal error
    err_x = merged["x"].values - merged["x_truth"].values
    err_y = merged["y"].values - merged["y_truth"].values
    errors = np.sqrt(err_x ** 2 + err_y ** 2)

    # Protection level
    if "protection_level" in merged.columns:
        pl = merged["protection_level"].values.astype(float)
    else:
        pl = np.full_like(errors, np.nan)

    # Sigma
    sigma = 0.0
    if meta:
        args = meta.get("arguments", {})
        sigma = args.get("min_sigma_pr", args.get("max_sigma_pr", 0.0))
    if sigma == 0.0:
        for p in parts:
            if p.startswith("sigma_"):
                try: sigma = float(p.replace("sigma_", "").split("_")[0]); break
                except: pass

    # Compute Stanford metrics
    metrics = compute_stanford_metrics(errors, pl)
    metrics["dataset"] = label
    metrics["sigma"] = sigma

    # ── Stanford plot ──
    mask = np.isfinite(errors) & np.isfinite(pl)
    err = errors[mask]
    pl_val = pl[mask]
    if len(err) == 0:
        return metrics

    max_val = max(np.percentile(err, 99), np.percentile(pl_val, 99), 1.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Color by HMI status
    hmi_mask = err > pl_val
    ax.scatter(err[~hmi_mask], pl_val[~hmi_mask], s=6, alpha=0.25,
               color=COLOR_CMM, edgecolors="none", label=r"Safe (err $\leq$ PL)")
    ax.scatter(err[hmi_mask], pl_val[hmi_mask], s=8, alpha=0.5,
               color=COLOR_FMM, edgecolors="none", marker="x", label="HMI (err > PL)")

    ax.plot([0, max_val], [0, max_val], "k--", lw=1.2, label="PL = Error")

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Protection level (m)")
    ax.set_title(rf"Stanford Plot ($\sigma$ = {sigma:.0f} m, {metrics['n_HMI']}/{metrics['n_total']} HMI)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / f"stanford_{label}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)

    print(f"  {label}: P_md={metrics['P_md']:.6f}, n_HMI={metrics['n_HMI']}/{metrics['n_total']}, "
          f"mean_PL={metrics['mean_PL']:.1f}m, mean_err={metrics['mean_error']:.1f}m")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stanford PL analysis.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/output/2_stanford"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find datasets across conditions
    dataset_dirs = []
    for pattern in ["sigma_*/*/*", "sigma_*"]:
        dataset_dirs.extend(sorted(args.data_root.glob(pattern)))

    # Filter to dirs with observations.csv
    dataset_dirs = [d for d in dataset_dirs
                    if d.is_dir() and (d / "observations.csv").exists()]

    if not dataset_dirs:
        raise SystemExit(f"No datasets found under {args.data_root}")

    print(f"Analyzing {len(dataset_dirs)} datasets:")
    rows = []
    for d in dataset_dirs:
        print(f"  {d.relative_to(args.data_root)}")
        row = stanford_single(d, args.output_dir)
        if row:
            rows.append(row)

    if rows:
        out = args.output_dir / "summary_table.csv"
        fieldnames = ["dataset", "sigma", "P_md", "P_fa", "n_HMI", "n_FA",
                      "n_total", "mean_PL", "mean_error"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  Summary table: {out}")

    print("Done.")


if __name__ == "__main__":
    main()
