#!/usr/bin/env python3
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.geometry
import shapely.ops
import shapely.wkt
from matplotlib.patches import ConnectionPatch
from pyproj import CRS

import math


def parse_list(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        return np.array([], dtype=float)
    return np.array([float(v) for v in value.split(",") if v.strip() != ""], dtype=float)


def parse_n_best_first(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        return np.array([], dtype=float)
    entries = [v.strip() for v in value.split("|")]
    first_vals = []
    for entry in entries:
        stripped = entry.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            stripped = stripped[1:-1]
        parts = [p.strip() for p in stripped.split(",") if p.strip() != ""]
        if not parts:
            continue
        first_vals.append(float(parts[0]))
    return np.array(first_vals, dtype=float)


def load_traj_geometry(row: pd.Series):
    if "pgeom" not in row or not isinstance(row["pgeom"], str) or not row["pgeom"].strip():
        raise SystemExit("Missing 'pgeom' column for trajectory geometry.")
    geom = shapely.wkt.loads(row["pgeom"])
    if geom.geom_type == "MultiLineString":
        geom = shapely.ops.linemerge(geom)
    if geom.geom_type != "LineString":
        raise SystemExit(f"Unsupported geometry type: {geom.geom_type}")
    return geom


def determine_utm_epsg(lon_deg: float, lat_deg: float) -> int:
    if not (np.isfinite(lon_deg) and np.isfinite(lat_deg)):
        raise ValueError("Invalid lon/lat for UTM zone.")
    if lat_deg <= -80.0 or lat_deg >= 84.0:
        raise ValueError("Lat outside UTM supported range.")
    zone = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    base = 32600 if lat_deg >= 0.0 else 32700
    return base + zone


def project_network_to_cps(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if network.crs is None:
        network = network.set_crs("EPSG:4326", allow_override=True)
    if network.crs.is_projected:
        return network
    lonlat = network.to_crs("EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = lonlat.total_bounds
    lon0 = float((min_lon + max_lon) * 0.5)
    lat0 = float((min_lat + max_lat) * 0.5)
    epsg = determine_utm_epsg(lon0, lat0)
    return lonlat.to_crs(CRS.from_epsg(epsg))


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
    parser.add_argument(
        "--network",
        type=Path,
        required=True,
        help="Road network shapefile for context map.",
    )
    parser.add_argument("--flat-tol-rel", type=float, default=0.02, help="Relative flat tolerance")
    parser.add_argument("--flat-tol-abs", type=float, default=1e-4, help="Absolute flat tolerance")
    parser.add_argument("--min-plateau", type=int, default=3, help="Minimum dy length for plateau")
    parser.add_argument("--smooth-window", type=int, default=3, help="Smoothing window for detection")
    parser.add_argument(
        "--map-height-ratio",
        type=float,
        default=1.8,
        help="Relative height ratio for the map subplot (main plot is fixed).",
    )
    parser.add_argument(
        "--map-start-index",
        type=int,
        default=0,
        help="Start index of the trajectory segment for the map subplot.",
    )
    parser.add_argument(
        "--map-pad-y",
        type=float,
        default=0.4,
        help="Vertical padding ratio for map extent (fraction of trajectory height).",
    )
    parser.add_argument(
        "--source",
        choices=["trustworthiness", "n_best"],
        default="trustworthiness",
        help="Choose trustworthiness or the first value in n_best_trustworthiness.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=";")
    row = df.loc[df["id"] == args.traj_id]
    if row.empty:
        raise SystemExit(f"Trajectory id {args.traj_id} not found in {args.input}")
    if args.source == "trustworthiness":
        trust_str = row.iloc[0]["trustworthiness"]
        trust = parse_list(trust_str)
    else:
        trust_str = row.iloc[0]["n_best_trustworthiness"]
        trust = parse_n_best_first(trust_str)
    if trust.size == 0:
        raise SystemExit("Empty trustworthiness sequence.")

    x = np.arange(trust.size)
    y_smooth = smooth_series(trust, args.smooth_window)
    y_range = float(np.nanmax(y_smooth) - np.nanmin(y_smooth))
    flat_tol = max(args.flat_tol_abs, args.flat_tol_rel * y_range)

    mark_points = find_plateau_boundaries(y_smooth, flat_tol, args.min_plateau)

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, args.map_height_ratio], hspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[1, 0])
    ax.plot(x, trust, color="#9bbcff", linewidth=1.2, alpha=0.7, label="Raw")
    ax.plot(x, y_smooth, color="#1f2a44", linewidth=2.0, label="Smoothed")
    ax.set_title(f"Trustworthiness Trajectory {args.traj_id}")
    ax.set_xlabel("Point Index")
    ax.set_ylabel("Trustworthiness")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    points = np.array(mark_points, dtype=int)
    if points.size:
        ax.scatter(
            points,
            y_smooth[points],
            color="#e63946",
            s=28,
            zorder=3,
            label="Plateau boundary",
        )
        for idx in points:
            ax.axvline(idx, color="#e63946", linewidth=0.8, alpha=0.25)

    y_min = float(np.nanmin(trust))
    y_max = float(np.nanmax(trust))
    pad = max((y_max - y_min) * 0.06, 1e-4)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend(loc="upper right", frameon=False)

    traj_geom = load_traj_geometry(row.iloc[0])
    coords = np.array(traj_geom.coords)
    network = gpd.read_file(args.network)
    try:
        network = project_network_to_cps(network)
    except Exception as exc:
        raise SystemExit(f"Failed to project network to CPS: {exc}")
    start = coords[0]
    end = coords[-1]
    angle_deg = -np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
    pivot = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)

    traj_geom_rot = shapely.affinity.rotate(traj_geom, angle_deg, origin=pivot, use_radians=False)
    network["geometry"] = network["geometry"].apply(
        lambda geom: shapely.affinity.rotate(geom, angle_deg, origin=pivot, use_radians=False)
    )
    coords_rot = np.array(traj_geom_rot.coords)
    start_idx = max(0, min(args.map_start_index, coords_rot.shape[0] - 1))
    coords_view = coords_rot[start_idx:]

    if coords_view.size == 0:
        raise SystemExit("No trajectory points available for the map subplot.")
    min_x, min_y = coords_view.min(axis=0)
    max_x, max_y = coords_view.max(axis=0)
    bounds = (min_x, min_y, max_x, max_y)
    pad_x = (bounds[2] - bounds[0]) * 0.1 or 1e-5
    pad_y = (bounds[3] - bounds[1]) * args.map_pad_y or 1e-5
    roi = shapely.geometry.box(
        bounds[0] - pad_x,
        bounds[1] - pad_y,
        bounds[2] + pad_x,
        bounds[3] + pad_y,
    )
    try:
        if not network.empty:
            idx = list(network.sindex.intersection(roi.bounds))
            network = network.iloc[idx]
            network = network[network.intersects(roi)]
    except Exception:
        pass

    if not network.empty and network.geometry.notna().any():
        try:
            network.plot(
                ax=ax_map,
                color="#b0b0b0",
                linewidth=0.6,
                alpha=0.6,
                aspect="equal",
            )
        except Exception:
            pass
    ax_map.plot(coords_view[:, 0], coords_view[:, 1], color="#1f2a44", linewidth=2.0, zorder=2)
    ax_map.set_xlabel("")
    ax_map.set_ylabel("")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title("Trajectory on Network")

    ax_map.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax_map.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    if points.size:
        points = points[(points >= start_idx) & (points < coords_rot.shape[0])]
        ax_map.scatter(
            coords_rot[points, 0],
            coords_rot[points, 1],
            color="#e63946",
            s=22,
            zorder=3,
        )
        for idx in points:
            con = ConnectionPatch(
                xyA=(idx, y_smooth[idx]),
                coordsA=ax.transData,
                xyB=(coords_rot[idx, 0], coords_rot[idx, 1]),
                coordsB=ax_map.transData,
                linestyle="--",
                linewidth=0.8,
                color="#e63946",
                alpha=0.6,
            )
            fig.add_artist(con)

    output = args.output
    if output is None:
        output = Path("output") / f"trustworthiness_traj{args.traj_id}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, hspace=0.25)
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
