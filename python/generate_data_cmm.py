#!/usr/bin/env python3
"""
Synthetic data generator for the CMM command-line interface.

The script samples random road segments from a shapefile, assumes uniform motion
with a user-provided speed, and creates noisy GNSS observations. Each observation
is associated with a unique covariance matrix and protection level so that the
resulting CSV can be consumed directly by the `cmm` CLI (as a point-based GPS file).

Generated artefacts (placed in the output directory):

  - `observations.csv`    : per-point observations with covariance/protection level
  - `ground_truth.csv`    : trajectories in WKT form with timestamp lists
  - `ground_truth_points.csv` (optional, for inspection)
  - `metadata.json`       : parameters used for the run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "geopandas and shapely are required to run this script. "
        "Install them with `pip install geopandas shapely`."
    ) from exc

try:
    from pyproj import CRS
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "pyproj is required to build the local ENU projection. "
        "Install it with `pip install pyproj`."
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
WGS84_CRS = CRS.from_epsg(4326)
CHI_SQUARE_QUANTILE_9999_DF2 = 23.025850929940457  # chi2.ppf(0.9999, df=2)
# MIN_SIGMA_M = 10.0            # metres (horizontal)
# MAX_SIGMA_M = 50.0           # metres (horizontal)
# MIN_SIGMA_UP = 10.0           # metres (vertical)
# MAX_SIGMA_UP = 50.0           # metres (vertical)
MIN_SIGMA_PR = 1.0
MAX_SIGMA_PR = 10.0
MAX_COVARIANCE_SAMPLING_ATTEMPTS = 64
MIN_TRAJECTORY_SPAN_M = 11_000.0  # ≈ 0.1 degrees at the equator
ORBIT_RADIUS_M = 20_200_000.0


# Shared between worker processes after initialisation.
ROAD_SEGMENTS: List[Tuple[np.ndarray, str, str, str]] = []
OUTGOING_SEGMENTS: Dict[str, List[int]] = {}
INCOMING_SEGMENTS: Dict[str, List[int]] = {}


@dataclass(frozen=True)
class TrajectoryConfig:
    traj_id: int
    num_points: int
    speed_mps: float
    sample_rate_hz: float
    base_epoch: float
    seed: int
    sigma_pr: float
    num_sats: int


@dataclass
class Observation:
    seq: int
    timestamp: float
    x: float
    y: float
    sde: float
    sdn: float
    sdu: float
    sdne: float
    sdeu: float
    sdun: float
    protection_level: float


@dataclass
class TrajectoryResult:
    traj_id: int
    sigma_pr: float
    timestamps: List[float]
    truth_coords: List[Tuple[float, float]]
    observations: List[Observation]
    edge_ids: List[str]

    @property
    def truth_wkt(self) -> str:
        return LineString(self.truth_coords).wkt


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _initialize_worker(segments: Sequence[Tuple[Sequence[Sequence[float]], str, str, str]]) -> None:
    """Store road segments and adjacency in worker-local state."""
    global ROAD_SEGMENTS, OUTGOING_SEGMENTS, INCOMING_SEGMENTS
    ROAD_SEGMENTS = []
    OUTGOING_SEGMENTS = {}
    INCOMING_SEGMENTS = {}
    for idx, (coords, u, v, edge_id) in enumerate(segments):
        arr = np.asarray(coords, dtype=float)
        src = str(u)
        dst = str(v)
        ROAD_SEGMENTS.append((arr, src, dst, edge_id))
        OUTGOING_SEGMENTS.setdefault(src, []).append(idx)
        INCOMING_SEGMENTS.setdefault(dst, []).append(idx)


def _determine_utm_epsg(lon_deg: float, lat_deg: float) -> Optional[int]:
    """Determine the EPSG code for the UTM zone covering the provided lon/lat."""
    if not (np.isfinite(lon_deg) and np.isfinite(lat_deg)):
        return None
    if lat_deg <= -80.0 or lat_deg >= 84.0:
        return None
    zone = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    base = 32600 if lat_deg >= 0.0 else 32700
    return base + zone


def _project_to_local_utm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project the road network to the appropriate UTM zone (metres)."""
    working = gdf
    if working.crs is None:
        working = working.set_crs(WGS84_CRS, allow_override=True)

    lonlat_gdf = working.to_crs(WGS84_CRS) if working.crs != WGS84_CRS else working
    min_lon, min_lat, max_lon, max_lat = lonlat_gdf.total_bounds
    bounds = np.array([min_lon, min_lat, max_lon, max_lat], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Shapefile bounds contain invalid values, cannot derive UTM zone.")
    if abs(max_lon - min_lon) < 1e-12 and abs(max_lat - min_lat) < 1e-12:
        raise ValueError("Shapefile bounds too small to derive a UTM projection.")

    lon0 = float((min_lon + max_lon) * 0.5)
    lat0 = float((min_lat + max_lat) * 0.5)
    epsg = _determine_utm_epsg(lon0, lat0)
    if epsg is None:
        raise ValueError(
            f"Unable to determine UTM zone for map centre ({lon0:.6f}, {lat0:.6f})."
        )
    utm_crs = CRS.from_epsg(epsg)
    projected = lonlat_gdf.to_crs(utm_crs)
    print(f"Projected road network to UTM EPSG:{epsg} (lat0={lat0:.6f}, lon0={lon0:.6f}).")
    return projected


def _load_roads(shapefile: Path) -> List[Tuple[List[List[float]], str, str, str]]:
    """Load road segments with node connectivity information."""
    gdf = gpd.read_file(shapefile)
    gdf = _project_to_local_utm(gdf)
    if "u" not in gdf.columns or "v" not in gdf.columns:
        raise ValueError("Shapefile must contain 'u' and 'v' columns for connectivity.")

    id_columns = [col for col in ("fid", "id", "edge_id") if col in gdf.columns]

    segments: List[Tuple[List[List[float]], str, str, str]] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            parts = [geom]
        elif isinstance(geom, MultiLineString):
            parts = list(geom.geoms)
        else:
            continue

        if id_columns:
            edge_identifier = row[id_columns[0]]
        else:
            edge_identifier = idx
        edge_identifier_str = str(edge_identifier)

        u = str(row["u"])
        v = str(row["v"])
        for part in parts:
            coords = [list(coord) for coord in part.coords]
            if len(coords) >= 2:
                segments.append((coords, u, v, edge_identifier_str))
    if not segments:
        raise ValueError(f"No valid road segments found in {shapefile}")
    return segments


@dataclass
class CovarianceParams:
    sde: float
    sdn: float
    sdu: float
    sdne: float
    sdeu: float
    sdun: float
    noise_cov: np.ndarray  # covariance matrix aligned with (x=east, y=north)
    pl_matrix: np.ndarray  # covariance matrix aligned with (east, north)


def _is_positive_definite(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True if the symmetric matrix is strictly positive-definite."""
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(eigenvalues > atol))


# def _random_covariance_params(rng: np.random.Generator) -> CovarianceParams:
    """Sample a positive-definite covariance description for one observation."""
    for _ in range(MAX_COVARIANCE_SAMPLING_ATTEMPTS):
        sdn = rng.uniform(MIN_SIGMA_M, MAX_SIGMA_M)
        sde = rng.uniform(MIN_SIGMA_M, MAX_SIGMA_M)
        rho = rng.uniform(-0.6, 0.6)
        sdne = rho * sdn * sde
        sdu = rng.uniform(MIN_SIGMA_UP, MAX_SIGMA_UP)
        # Keep vertical cross-covariances small but non-zero for realism.
        sdeu = rng.uniform(-0.2, 0.2) * sde * sdu * 0.05
        sdun = rng.uniform(-0.2, 0.2) * sdn * sdu * 0.05

        full_cov = np.array(
            [
                [sde * sde, sdne, sdeu],
                [sdne, sdn * sdn, sdun],
                [sdeu, sdun, sdu * sdu],
            ],
            dtype=float,
        )
        if not _is_positive_definite(full_cov):
            continue

        noise_cov = np.array(
            [
                [sde * sde, sdne],
                [sdne, sdn * sdn],
            ],
            dtype=float,
        )
        if not _is_positive_definite(noise_cov):
            continue

        pl_matrix = np.array(
            [
                [sde * sde, sdne],
                [sdne, sdn * sdn],
            ],
            dtype=float,
        )
        if not _is_positive_definite(pl_matrix):
            continue

        return CovarianceParams(sde, sdn, sdu, sdne, sdeu, sdun, noise_cov, pl_matrix)

    raise RuntimeError("Failed to sample a positive-definite covariance matrix.")


def _segment_lengths_m(coords: np.ndarray) -> np.ndarray:
    """Return Euclidean segment lengths (metres) for a polyline in projected space."""
    if coords.shape[0] < 2:
        return np.empty(0, dtype=float)
    diffs = np.diff(coords, axis=0)
    return np.linalg.norm(diffs, axis=1)


def _interpolate_along(coords: np.ndarray, target_dist_m: float,
                       cumulative: np.ndarray, seg_lengths: np.ndarray) -> Tuple[float, float]:
    """Interpolate a point at the specified cumulative distance (metres)."""
    if target_dist_m <= 0:
        return tuple(coords[0])
    if target_dist_m >= cumulative[-1]:
        return tuple(coords[-1])
    seg_idx = int(np.searchsorted(cumulative, target_dist_m, side="right") - 1)
    seg_len = seg_lengths[seg_idx]
    if seg_len <= 0:
        return tuple(coords[seg_idx])
    local_offset = target_dist_m - cumulative[seg_idx]
    fraction = local_offset / seg_len
    x0, y0 = coords[seg_idx]
    x1, y1 = coords[seg_idx + 1]
    x = x0 + fraction * (x1 - x0)
    y = y0 + fraction * (y1 - y0)
    return float(x), float(y)


# def _compute_protection_level(params: CovarianceParams) -> float:
    eigenvalues = np.linalg.eigvalsh(params.pl_matrix)
    max_eig = float(np.max(eigenvalues))
    return math.sqrt(max_eig * CHI_SQUARE_QUANTILE_9999_DF2)


def _compute_protection_level_from_cov(cov_h: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(cov_h)
    max_eig = float(np.max(eigenvalues))
    return math.sqrt(max_eig * CHI_SQUARE_QUANTILE_9999_DF2)


def _random_satellite_unit_vector(
    rng: np.random.Generator, min_el_deg: float = 45.0
) -> np.ndarray:
    az = rng.uniform(0.0, 2.0 * math.pi)
    el = rng.uniform(math.radians(min_el_deg), math.radians(80.0))
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    return np.array([x, y, z], dtype=float)


def _generate_satellite_geometry(num_sats: int, rng: np.random.Generator) -> np.ndarray:
    return np.stack(
        [_random_satellite_unit_vector(rng) for _ in range(num_sats)],
        axis=0,
    )


def _wls_position(
    truth_pos: np.ndarray,
    sat_positions: np.ndarray,
    sigma_pr: float,
    rng: np.random.Generator,
    max_iter: int = 8,
) -> Tuple[np.ndarray, float, np.ndarray]:
    num_sats = sat_positions.shape[0]
    if num_sats < 4:
        raise ValueError("Need at least 4 satellites for WLS")

    true_ranges = np.linalg.norm(sat_positions - truth_pos, axis=1)
    noise = rng.normal(scale=sigma_pr, size=num_sats)
    rho = true_ranges + noise

    x = truth_pos.copy()
    b = 0.0
    Q = None
    for _ in range(max_iter):
        ranges = np.linalg.norm(sat_positions - x, axis=1)
        los = (sat_positions - x) / ranges[:, None]
        H = np.hstack((-los, np.ones((num_sats, 1))))
        r = rho - ranges - b

        try:
            Q = np.linalg.inv(H.T @ H)
        except np.linalg.LinAlgError:
            break
        dx = Q @ H.T @ r
        x += dx[:3]
        b += dx[3]
        if np.linalg.norm(dx[:3]) < 1e-4:
            break

    cov_full = (sigma_pr ** 2) * Q if Q is not None else np.full((4, 4), np.nan)
    cov_xyz = cov_full[:3, :3]
    return x, b, cov_xyz


def _generate_single(config: TrajectoryConfig) -> TrajectoryResult:
    if not ROAD_SEGMENTS:
        raise RuntimeError("ROAD_SEGMENTS has not been initialised")
    if not OUTGOING_SEGMENTS or not INCOMING_SEGMENTS:
        raise RuntimeError("Road segment adjacency has not been initialised")

    rng = np.random.default_rng(config.seed)
    sats_unit = _generate_satellite_geometry(config.num_sats, rng)
    sat_positions = sats_unit * ORBIT_RADIUS_M
    sample_rate = max(config.sample_rate_hz, 1e-6)
    step_m = config.speed_mps / sample_rate
    required_length = step_m * max(config.num_points - 1, 1)

    min_span_m = MIN_TRAJECTORY_SPAN_M
    max_attempts = 200
    max_extension_steps = 500
    max_global_attempts = 50
    options_count = len(ROAD_SEGMENTS)

    def segment_span(coords_array: np.ndarray) -> float:
        lon_span = float(coords_array[:, 0].max() - coords_array[:, 0].min())
        lat_span = float(coords_array[:, 1].max() - coords_array[:, 1].min())
        return lon_span + lat_span

    def try_extend_path(path: List[List[float]],
                        start: str,
                        end: str,
                        visited_edges: set[int]) -> Tuple[str, str, bool, Optional[str], Optional[str]]:
        choices = ["forward", "backward"]
        order = list(rng.permutation(len(choices)))
        for idx in order:
            direction = choices[idx]
            if direction == "forward":
                candidate_edges = [
                    edge_idx for edge_idx in OUTGOING_SEGMENTS.get(end, [])
                    if edge_idx not in visited_edges
                ]
                if not candidate_edges:
                    continue
                edge_idx = int(candidate_edges[int(rng.integers(0, len(candidate_edges)))])
                coords_edge, u, v, edge_identifier = ROAD_SEGMENTS[edge_idx]
                if u != end or coords_edge.shape[0] < 2:
                    continue
                new_points = coords_edge[1:]
                if new_points.size == 0:
                    continue
                path.extend(new_points.tolist())
                visited_edges.add(edge_idx)
                return start, v, True, edge_identifier, "append"
            else:
                candidate_edges = [
                    edge_idx for edge_idx in INCOMING_SEGMENTS.get(start, [])
                    if edge_idx not in visited_edges
                ]
                if not candidate_edges:
                    continue
                edge_idx = int(candidate_edges[int(rng.integers(0, len(candidate_edges)))])
                coords_edge, u, v, edge_identifier = ROAD_SEGMENTS[edge_idx]
                if v != start or coords_edge.shape[0] < 2:
                    continue
                new_points = coords_edge[:-1]
                if new_points.size == 0:
                    continue
                path[0:0] = new_points.tolist()
                visited_edges.add(edge_idx)
                return u, end, True, edge_identifier, "prepend"
        return start, end, False, None, None

    for _ in range(max_global_attempts):
        chosen_coords: np.ndarray | None = None
        chosen_edges: List[str] | None = None
        seg_lengths = None
        cumulative = None

        for _ in range(max_attempts):
            base_idx = int(rng.integers(0, options_count))
            base_coords, start_node, end_node, base_edge_id = ROAD_SEGMENTS[base_idx]
            if base_coords.shape[0] < 2:
                continue

            path_coords = base_coords.tolist()
            visited_edges = {base_idx}
            edge_sequence = [base_edge_id]

            current_span = segment_span(base_coords)

            extension_counter = 0
            while current_span < min_span_m and extension_counter < max_extension_steps:
                start_node, end_node, extended, new_edge_id, mode = try_extend_path(
                    path_coords, start_node, end_node, visited_edges
                )
                if not extended:
                    break
                extension_counter += 1
                current_span = segment_span(np.asarray(path_coords, dtype=float))
                if new_edge_id is not None and mode:
                    if mode == "append":
                        edge_sequence.append(new_edge_id)
                    elif mode == "prepend":
                        edge_sequence.insert(0, new_edge_id)

            if current_span < min_span_m:
                continue

            candidate_arr = np.asarray(path_coords, dtype=float)
            seg_lengths_candidate = _segment_lengths_m(candidate_arr)
            if seg_lengths_candidate.size == 0:
                continue
            cumulative_candidate = np.concatenate(([0.0], np.cumsum(seg_lengths_candidate)))
            if cumulative_candidate[-1] < required_length:
                continue
            chosen_coords = candidate_arr
            chosen_edges = edge_sequence
            seg_lengths = seg_lengths_candidate
            cumulative = cumulative_candidate
            break

        if chosen_coords is None or chosen_edges is None or seg_lengths is None or cumulative is None:
            continue

        sample_distances = np.linspace(0.0, required_length, num=config.num_points, dtype=float)

        truth_coords: List[Tuple[float, float]] = []
        timestamps: List[float] = []
        observations: List[Observation] = []

        for seq, dist_m in enumerate(sample_distances):
            base_x, base_y = _interpolate_along(chosen_coords, float(dist_m), cumulative, seg_lengths)
            truth_coords.append((base_x, base_y))
            truth_pos = np.array([base_x, base_y, 0.0], dtype=float)
            est_pos, clock_bias, cov_xyz = _wls_position(
                truth_pos, sat_positions, config.sigma_pr, rng
            )
            if not np.all(np.isfinite(cov_xyz)):
                cov_xyz = np.eye(3, dtype=float) * (config.sigma_pr ** 2)
            cov_h = cov_xyz[:2, :2]
            sde = math.sqrt(max(cov_h[0, 0], 0.0))
            sdn = math.sqrt(max(cov_h[1, 1], 0.0))
            sdu = math.sqrt(max(cov_xyz[2, 2], 0.0))
            sdne = float(cov_h[0, 1])
            sdeu = float(cov_xyz[0, 2])
            sdun = float(cov_xyz[1, 2])
            obs_x = float(est_pos[0])
            obs_y = float(est_pos[1])
            timestamp = float(config.base_epoch + seq / sample_rate)

            observations.append(
                Observation(
                    seq=seq,
                    timestamp=timestamp,
                    x=obs_x,
                    y=obs_y,
                    sdn=sdn,
                    sde=sde,
                    sdu=sdu,
                    sdne=sdne,
                    sdeu=sdeu,
                    sdun=sdun,
                    protection_level=_compute_protection_level_from_cov(cov_h),
                )
            )
            timestamps.append(timestamp)

        arr_truth = np.asarray(truth_coords, dtype=float)
        if arr_truth.shape[0] < 2:
            continue
        span_total = segment_span(arr_truth)
        if span_total < min_span_m:
            continue

        return TrajectoryResult(
            traj_id=config.traj_id,
            sigma_pr=config.sigma_pr,
            timestamps=timestamps,
            truth_coords=truth_coords,
            observations=observations,
            edge_ids=chosen_edges,
        )

    raise RuntimeError("Unable to generate a trajectory with ≥11 km combined span in projected space.")


def _make_tasks(
    count: int,
    num_points: int,
    speed: float,
    sample_rate_hz: float,
    base_epoch: float,
    seed: int | None,
    start_id: int,
    sigma_min: float,
    sigma_max: float,
    num_sats: int,
) -> List[TrajectoryConfig]:
    master_rng = np.random.default_rng(seed)
    tasks: List[TrajectoryConfig] = []
    group_count = max(1, int(math.ceil(count / 10)))
    sigma_values = np.linspace(sigma_min, sigma_max, num=group_count)
    for offset in range(count):
        task_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        sigma_idx = min(offset // 10, group_count - 1)
        tasks.append(
            TrajectoryConfig(
                traj_id=start_id + offset,
                num_points=num_points,
                speed_mps=speed,
                sample_rate_hz=sample_rate_hz,
                base_epoch=base_epoch,
                seed=task_seed,
                sigma_pr=float(sigma_values[sigma_idx]),
                num_sats=num_sats,
            )
        )
    return tasks


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic observation files for the CMM CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=100, help="Number of trajectories to generate.")
    parser.add_argument("--points", type=int, default=1000, help="Number of points per trajectory.")
    parser.add_argument("--speed", type=float, default=12.0, help="Vehicle speed (m/s).")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Sampling rate (Hz).")
    parser.add_argument("--num-sats", type=int, default=8, help="Satellites per trajectory.")
    parser.add_argument("--calib-ratio", type=float, default=0.2, help="Calibration split ratio.")
    parser.add_argument("--min-sigma-pr", type=float, default=MIN_SIGMA_PR, help="Min PR sigma (m).")
    parser.add_argument("--max-sigma-pr", type=float, default=MAX_SIGMA_PR, help="Max PR sigma (m).")
    parser.add_argument("--output-dir", type=Path, default=Path("python/cmm_data"),
                        help="Directory where generated files will be stored.")
    parser.add_argument("--shapefile", type=Path, default=Path("input/map/haikou/edges.shp"),
                        help="Road shapefile used as the sampling base.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="Maximum number of worker processes.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Master random seed for reproducibility.")
    parser.add_argument("--start-id", type=int, default=1,
                        help="Identifier assigned to the first trajectory.")
    parser.add_argument("--base-epoch", type=float, default=None,
                        help="Starting UNIX epoch (seconds). Defaults to current time.")
    parser.add_argument("--export-points", action="store_true",
                        help="Emit ground truth sample points alongside trajectories.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def _write_observation_csv(results: List[TrajectoryResult], destination: Path) -> None:
    header = [
        "id", "seq", "timestamp", "x", "y",
        "sde", "sdn", "sdne", "protection_level",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(header)
        for traj in sorted(results, key=lambda item: item.traj_id):
            for obs in traj.observations:
                writer.writerow(
                    [
                        traj.traj_id,
                        obs.seq,
                        f"{obs.timestamp:.6f}",
                        f"{obs.x:.8f}",
                        f"{obs.y:.8f}",
                        f"{obs.sde:.8f}",
                        f"{obs.sdn:.8f}",
                        f"{obs.sdne:.10f}",
                        f"{obs.protection_level:.8f}",
                    ]
                )


def _write_cmm_trajectory_csv(results: List[TrajectoryResult], destination: Path) -> None:
    header = ["id", "geom", "timestamps", "covariances", "protection_levels"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(header)
        for traj in sorted(results, key=lambda item: item.traj_id):
            timestamps = [round(value, 6) for value in traj.timestamps]
            # must follow the sequence of sde, sdn, sdu, sdne, sdeu, sdun
            covariances = [
                [
                    round(obs.sde, 8),
                    round(obs.sdn, 8),
                    round(obs.sdu, 8),
                    round(obs.sdne, 10),
                    round(obs.sdeu, 10),
                    round(obs.sdun, 10),
                ]
                for obs in traj.observations
            ]
            protection_levels = [round(obs.protection_level, 8) for obs in traj.observations]
            observation_coords = [
                (round(obs.x, 8), round(obs.y, 8)) for obs in traj.observations
            ]
            observation_wkt = "LINESTRING (" + ", ".join(
                f"{x:.8f} {y:.8f}" for x, y in observation_coords
            ) + ")"

            writer.writerow(
                [
                    traj.traj_id,
                    observation_wkt,
                    json.dumps(timestamps, separators=(",", ":"), ensure_ascii=False),
                    json.dumps(covariances, separators=(",", ":"), ensure_ascii=False),
                    json.dumps(protection_levels, separators=(",", ":"), ensure_ascii=False),
                ]
            )


def _write_ground_truth_csv(results: List[TrajectoryResult], destination: Path) -> None:
    header = ["id", "geom", "timestamp", "edge_ids"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(header)
        for traj in sorted(results, key=lambda item: item.traj_id):
            timestamp_str = ",".join(f"{ts:.6f}" for ts in traj.timestamps)
            edge_str = ",".join(traj.edge_ids)
            writer.writerow([traj.traj_id, traj.truth_wkt, timestamp_str, edge_str])


def _write_ground_truth_points(results: List[TrajectoryResult], destination: Path) -> None:
    header = ["id", "seq", "timestamp", "x", "y"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(header)
        for traj in sorted(results, key=lambda item: item.traj_id):
            for seq, (coord, ts) in enumerate(zip(traj.truth_coords, traj.timestamps)):
                writer.writerow(
                    [
                        traj.traj_id,
                        seq,
                        f"{ts:.6f}",
                        f"{coord[0]:.8f}",
                        f"{coord[1]:.8f}",
                    ]
                )


def _write_metadata(results: List[TrajectoryResult], args: argparse.Namespace, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.time(),
        "arguments": {
            "count": args.count,
            "points": args.points,
            "speed": args.speed,
            "sample_rate": args.sample_rate,
            "num_sats": args.num_sats,
            "min_sigma_pr": args.min_sigma_pr,
            "max_sigma_pr": args.max_sigma_pr,
            "calib_ratio": args.calib_ratio,
            "shapefile": str(Path(args.shapefile).resolve()),
            "seed": args.seed,
            "start_id": args.start_id,
            "base_epoch": args.base_epoch,
        },
        "trajectories": [
            {
                "id": traj.traj_id,
                "sigma_pr": traj.sigma_pr,
                "num_points": len(traj.truth_coords),
                "duration": traj.timestamps[-1] - traj.timestamps[0] if len(traj.timestamps) > 1 else 0.0,
                "edge_ids": traj.edge_ids,
            }
            for traj in sorted(results, key=lambda item: item.traj_id)
        ],
    }
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_split_csvs(calib_ids: List[int], test_ids: List[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_path = output_dir / "calibration_trajs.csv"
    test_path = output_dir / "test_trajs.csv"
    with calib_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["traj_id"])
        for traj_id in calib_ids:
            writer.writerow([traj_id])
    with test_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["traj_id"])
        for traj_id in test_ids:
            writer.writerow([traj_id])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    shapefile = args.shapefile.resolve()
    if not shapefile.exists():
        raise SystemExit(f"Shapefile not found: {shapefile}")

    road_geoms = _load_roads(shapefile)
    base_epoch = float(args.base_epoch) if args.base_epoch is not None else time.time()
    tasks = _make_tasks(
        args.count,
        args.points,
        args.speed,
        args.sample_rate,
        base_epoch,
        args.seed,
        args.start_id,
        args.min_sigma_pr,
        args.max_sigma_pr,
        args.num_sats,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs <= 1:
        _initialize_worker(road_geoms)
        results = [_generate_single(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_initialize_worker,
            initargs=(road_geoms,),
        ) as executor:
            results = list(executor.map(_generate_single, tasks))

    sigma_to_trajs: Dict[float, List[int]] = {}
    for traj in results:
        sigma_to_trajs.setdefault(traj.sigma_pr, []).append(traj.traj_id)
    calib_ids: List[int] = []
    test_ids: List[int] = []
    split_rng = np.random.default_rng(args.seed)
    for sigma, trajs in sorted(sigma_to_trajs.items(), key=lambda item: item[0]):
        shuffled = list(trajs)
        split_rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * args.calib_ratio))
        calib_ids.extend(shuffled[:split_idx])
        test_ids.extend(shuffled[split_idx:])
    calib_ids = sorted(set(calib_ids))
    test_ids = sorted(set(test_ids))

    _write_observation_csv(results, output_dir / "observations.csv")
    _write_ground_truth_csv(results, output_dir / "ground_truth.csv")
    if args.export_points:
        _write_ground_truth_points(results, output_dir / "ground_truth_points.csv")
    _write_metadata(results, args, output_dir / "metadata.json")
    _write_cmm_trajectory_csv(results, output_dir / "cmm_trajectory.csv")
    _write_split_csvs(calib_ids, test_ids, output_dir)

    print(f"Generated {len(results)} trajectories in {output_dir}")
    print("  - observations.csv       (input for `cmm` with --gps_point)")
    print("  - ground_truth.csv       (WKT trajectories for reference)")
    if args.export_points:
        print("  - ground_truth_points.csv (sampled ground-truth points)")
    print("  - metadata.json          (generation summary)")
    print("  - cmm_trajectory.csv     (per-trajectory data for CMMTrajectory)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
