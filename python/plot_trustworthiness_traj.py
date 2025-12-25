#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_list(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        return np.array([], dtype=float)
    return np.array([float(v) for v in value.split(",") if v.strip() != ""], dtype=float)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def find_plateau_boundaries(values: np.ndarray, flat_tol: float, min_len: int) -> list:
    if values.size < 2:
        return []
    dy = np.diff(values)
    flat = np.abs(dy) <= flat_tol
    segments = []
    start = None
    for idx, is_flat in enumerate(flat):
        if is_flat and start is None:
            start = idx
        elif not is_flat and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx - 1))
            start = None
    if start is not None and len(flat) - start >= min_len:
        segments.append((start, len(flat) - 1))
    boundaries = []
    for start, end in segments:
        boundaries.extend([start, end + 1])
    return sorted(set(boundaries))


def find_turning_points(values: np.ndarray, flat_tol: float) -> list:
    if values.size < 3:
        return []
    dy = np.diff(values)
    turning = []
    for idx in range(1, len(dy)):
        prev_dy = dy[idx - 1]
        next_dy = dy[idx]
        if prev_dy > flat_tol and next_dy < -flat_tol:
            turning.append(idx)
        elif prev_dy < -flat_tol and next_dy > flat_tol:
            turning.append(idx)
    return sorted(set(turning))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot trustworthiness for a trajectory and annotate plateau boundaries."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to cmm_result.csv")
    parser.add_argument("--traj-id", type=int, required=True, help="Trajectory id to plot")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--flat-tol-rel", type=float, default=0.02, help="Relative flat tolerance")
    parser.add_argument("--flat-tol-abs", type=float, default=1e-4, help="Absolute flat tolerance")
    parser.add_argument("--min-plateau", type=int, default=3, help="Minimum dy length for plateau")
    parser.add_argument("--smooth-window", type=int, default=3, help="Smoothing window for detection")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")
    row = df.loc[df["id"] == args.traj_id]
    if row.empty:
        raise SystemExit(f"Trajectory id {args.traj_id} not found in {args.input}")
    trust_str = row.iloc[0]["trustworthiness"]
    trust = parse_list(trust_str)
    if trust.size == 0:
        raise SystemExit("Empty trustworthiness sequence.")

    x = np.arange(trust.size)
    y_smooth = smooth_series(trust, args.smooth_window)
    y_range = float(np.nanmax(y_smooth) - np.nanmin(y_smooth))
    flat_tol = max(args.flat_tol_abs, args.flat_tol_rel * y_range)

    mark_points = find_plateau_boundaries(y_smooth, flat_tol, args.min_plateau)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, trust, color="tab:blue", linewidth=1.5)
    ax.set_title(f"Trustworthiness Trajectory {args.traj_id}")
    ax.set_xlabel("Point Index")
    ax.set_ylabel("Trustworthiness")

    if mark_points:
        ax.scatter(
            np.array(mark_points),
            trust[mark_points],
            color="tab:red",
            s=24,
            zorder=3,
        )
        y_offset = (np.nanmax(trust) - np.nanmin(trust)) * 0.02
        for idx in mark_points:
            ax.text(
                idx,
                trust[idx] + y_offset,
                f"{idx}",
                fontsize=8,
                ha="center",
                va="bottom",
                color="tab:red",
            )

    output = args.output
    if output is None:
        output = Path("output") / f"trustworthiness_traj{args.traj_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
