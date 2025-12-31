#!/usr/bin/env python3
"""
Compute |sp_dist - eu_dist| distribution from CMM result CSV and plot histogram.

Example:
    python python/plot_sp_eu_abs_diff.py \
        --input simulation/mm_result/cmm_result.csv \
        --output output/sp_eu_abs_diff_hist.png \
        --bins 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb


def parse_list(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        return np.array([], dtype=float)
    return np.array([float(v) for v in value.split(",") if v.strip() != ""], dtype=float)


def compute_abs_diffs(sp_values: Iterable[str], eu_values: Iterable[str]) -> np.ndarray:
    diffs: list[float] = []
    mismatch_rows = 0
    for sp_str, eu_str in zip(sp_values, eu_values):
        sp = parse_list(sp_str)
        eu = parse_list(eu_str)
        if sp.size != eu.size:
            mismatch_rows += 1
        count = min(sp.size, eu.size)
        if count == 0:
            continue
        diffs.extend(np.abs(sp[:count] - eu[:count]).tolist())
    if mismatch_rows:
        print(f"Warning: {mismatch_rows} rows have length mismatch; used min length per row.")
    return np.array(diffs, dtype=float)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distribution of |sp_dist - eu_dist| from a cmm_result.csv file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("simulation/mm_result/cmm_result.csv"),
        help="Path to cmm_result.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sp_eu_abs_diff_hist.png"),
        help="Output PNG path",
    )
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument(
        "--range",
        dest="hist_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional x-axis range for histogram",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, sep=";")
    if "sp_dist" not in df.columns or "eu_dist" not in df.columns:
        raise SystemExit("Expected columns 'sp_dist' and 'eu_dist' in input CSV.")

    diffs = compute_abs_diffs(df["sp_dist"], df["eu_dist"])
    if diffs.size == 0:
        raise SystemExit("No valid sp_dist/eu_dist pairs found.")

    if args.hist_range is not None:
        range_min, range_max = sorted(args.hist_range)
        diffs = diffs[(diffs >= range_min) & (diffs <= range_max)]
        if diffs.size == 0:
            raise SystemExit("No data points within the specified range.")

    stats_summary = {
        "count": diffs.size,
        "mean": float(np.mean(diffs)),
        "std": float(np.std(diffs)),
        "min": float(np.min(diffs)),
        "median": float(np.median(diffs)),
        "max": float(np.max(diffs)),
    }

    fit_loc = float(stats_summary["min"])
    fit_scale = stats_summary["mean"] if stats_summary["mean"] > 0 else 1.0
    try:
        from scipy import stats  # type: ignore

        fit_loc, fit_scale = stats.expon.fit(diffs)
        expon_dist = stats.expon(loc=fit_loc, scale=fit_scale)
        pdb.set_trace()
    except Exception:
        expon_dist = None

    fig, ax = plt.subplots(figsize=(7, 4))
    counts, bin_edges, _ = ax.hist(
        diffs,
        bins=args.bins,
        range=args.hist_range,
        color="#4C78A8",
        edgecolor="white",
        alpha=0.7,
        density=False,
        label="Histogram",
    )

    x_min = 0.0 if args.hist_range is None else min(args.hist_range)
    x_max = stats_summary["max"] if args.hist_range is None else max(args.hist_range)
    x_vals = np.linspace(x_min, x_max, 400)
    if expon_dist is not None:
        pdf_vals = expon_dist.pdf(x_vals)
    else:
        rate = 1.0 / fit_scale if fit_scale > 0 else 1.0
        pdf_vals = rate * np.exp(-rate * (x_vals - fit_loc))
        pdf_vals[x_vals < fit_loc] = 0.0
    bin_width = float(np.mean(np.diff(bin_edges))) if len(bin_edges) > 1 else 1.0
    pdf_scaled = pdf_vals * diffs.size * bin_width
    ax.plot(x_vals, pdf_scaled, color="#F58518", linewidth=2.0, label="Exponential fit")

    ax.set_title("Distribution of |sp_dist - eu_dist|")
    ax.set_xlabel("|sp_dist - eu_dist|")
    ax.set_ylabel("frequent")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    summary_text = (
        f"count: {stats_summary['count']}\n"
        f"mean: {stats_summary['mean']:.4f}\n"
        f"std: {stats_summary['std']:.4f}\n"
        f"min: {stats_summary['min']:.4f}\n"
        f"median: {stats_summary['median']:.4f}\n"
        f"max: {stats_summary['max']:.4f}"
    )
    ax.text(
        0.98,
        0.5,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved histogram -> {args.output}")


if __name__ == "__main__":
    main()
