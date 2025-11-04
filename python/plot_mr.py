#!/usr/bin/env python3
"""
Plot matched routes with cumulative probability gradient.

The script visualizes the `pgeom` column from a CMM result CSV on top of the
road network shapefile, using a colour gradient driven by per-point
`cumu_prob` values.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely import wkt
from shapely.geometry import LineString


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CMM matched routes with cumulative probability gradient.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/cmm_result_positive_determ.csv"),
        help="Path to the CMM result CSV with a `pgeom` column (default: %(default)s).",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("input/map/haikou/edges.shp"),
        help="Path to the road network shapefile (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the rendered figure (PNG). If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--title",
        default="CMM Matched Trajectories",
        help="Plot title (default: %(default)s).",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap to use for the probability gradient (default: %(default)s).",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=2.0,
        help="Line width for matched trajectory segments (default: %(default)s).",
    )
    return parser.parse_args()


def parse_floats(value: str) -> List[float]:
    if not isinstance(value, str):
        return []
    items = [item.strip() for item in value.split(",")]
    return [float(item) for item in items if item]


def build_segments(geometry: LineString, weights: Sequence[float]) -> Tuple[List[np.ndarray], List[float]]:
    coords = list(geometry.coords)
    if len(coords) < 2:
        return [], []

    limit = min(len(coords), len(weights))
    if limit < 2:
        return [], []

    segments: List[np.ndarray] = []
    segment_weights: List[float] = []
    for idx in range(limit - 1):
        segment = np.array([coords[idx], coords[idx + 1]])
        weight = (weights[idx] + weights[idx + 1]) / 2.0
        segments.append(segment)
        segment_weights.append(weight)
    return segments, segment_weights


def collect_matched_segments(data: pd.DataFrame) -> Tuple[List[np.ndarray], List[float]]:
    segments: List[np.ndarray] = []
    weights: List[float] = []
    for _, row in data.iterrows():
        wkt_text = row.get("pgeom")
        probability_string = row.get("cumu_prob")
        if not isinstance(wkt_text, str) or not isinstance(probability_string, str):
            continue

        try:
            geometry = wkt.loads(wkt_text)
        except Exception:
            continue

        if not isinstance(geometry, LineString):
            continue

        prob_values = parse_floats(probability_string)
        segs, seg_weights = build_segments(geometry, prob_values)
        if segs:
            segments.extend(segs)
            weights.extend(seg_weights)
    return segments, weights


def plot_matched_routes(
    network_path: Path,
    segments: Sequence[np.ndarray],
    weights: Sequence[float],
    cmap: str,
    line_width: float,
    title: str,
    output_path: Path | None,
) -> None:
    if not segments:
        raise ValueError("No trajectory segments available to plot.")

    network = gpd.read_file(network_path)

    fig, ax = plt.subplots(figsize=(12, 10))
    network.plot(ax=ax, color="#dddddd", linewidth=0.4, edgecolor="#aaaaaa")

    weights_array = np.array(weights)
    norm = Normalize(vmin=weights_array.min(), vmax=weights_array.max())
    line_collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=line_width,
    )
    line_collection.set_array(weights_array)
    ax.add_collection(line_collection)

    colorbar = fig.colorbar(line_collection, ax=ax, shrink=0.7, pad=0.01)
    colorbar.set_label("Cumulative Probability")

    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def main() -> None:
    args = parse_arguments()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    if not args.network.exists():
        raise FileNotFoundError(f"Network shapefile not found: {args.network}")

    data = pd.read_csv(args.csv, sep=";")
    segments, weights = collect_matched_segments(data)
    if not segments:
        raise RuntimeError("No valid matched trajectory segments found in the CSV.")

    plot_matched_routes(
        network_path=args.network,
        segments=segments,
        weights=weights,
        cmap=args.cmap,
        line_width=args.line_width,
        title=args.title,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
