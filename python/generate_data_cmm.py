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
    from pyproj import Geod
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "pyproj is required to compute geodesic distances. "
        "Install it with `pip install pyproj`."
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
GEOD = Geod(ellps="WGS84")
CHI_SQUARE_QUANTILE_9999_DF2 = 23.025850929940457  # chi2.ppf(0.9999, df=2)
MIN_SIGMA_DEG = 1.0e-5       # ≈ 1.01 metres
MAX_SIGMA_DEG = 1.4e-4       # ≈ 15.05 metres
MIN_SIGMA_UP = 1.0           # metres (vertical)
MAX_SIGMA_UP = 10.0           # metres (vertical)
MAX_COVARIANCE_SAMPLING_ATTEMPTS = 64


# Shared between worker processes after initialisation.
ROAD_SEGMENTS: List[Tuple[np.ndarray, str, str, str]] = []
OUTGOING_SEGMENTS: Dict[str, List[int]] = {}
INCOMING_SEGMENTS: Dict[str, List[int]] = {}


@dataclass(frozen=True)
class TrajectoryConfig:
    traj_id: int
    num_points: int
    speed_mps: float
    base_epoch: float
    seed: int


@dataclass
class Observation:
    seq: int
    timestamp: float
    x: float
    y: float
    sdn: float
    sde: float
    sdu: float
    sdne: float
    sdeu: float
    sdun: float
    protection_level: float


@dataclass
class TrajectoryResult:
    traj_id: int
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


def _load_roads(shapefile: Path) -> List[Tuple[List[List[float]], str, str, str]]:
    """Load road segments with node connectivity information."""
    gdf = gpd.read_file(shapefile)
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
    sdn: float
    sde: float
    sdu: float
    sdne: float
    sdeu: float
    sdun: float
    noise_cov: np.ndarray  # covariance matrix aligned with (x=east, y=north)
    pl_matrix: np.ndarray  # covariance matrix aligned with (north, east)


def _is_positive_definite(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True if the symmetric matrix is strictly positive-definite."""
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(eigenvalues > atol))


def _random_covariance_params(rng: np.random.Generator) -> CovarianceParams:
    """Sample a positive-definite covariance description for one observation."""
    for _ in range(MAX_COVARIANCE_SAMPLING_ATTEMPTS):
        sdn = rng.uniform(MIN_SIGMA_DEG, MAX_SIGMA_DEG)
        sde = rng.uniform(MIN_SIGMA_DEG, MAX_SIGMA_DEG)
        rho = rng.uniform(-0.6, 0.6)
        sdne = rho * sdn * sde
        sdu = rng.uniform(MIN_SIGMA_UP, MAX_SIGMA_UP)
        # Keep vertical cross-covariances small but non-zero for realism.
        sdeu = rng.uniform(-0.2, 0.2) * sde * sdu * 0.05
        sdun = rng.uniform(-0.2, 0.2) * sdn * sdu * 0.05

        full_cov = np.array(
            [
                [sdn * sdn, sdne, sdun],
                [sdne, sde * sde, sdeu],
                [sdun, sdeu, sdu * sdu],
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
                [sdn * sdn, sdne],
                [sdne, sde * sde],
            ],
            dtype=float,
        )
        if not _is_positive_definite(pl_matrix):
            continue

        return CovarianceParams(sdn, sde, sdu, sdne, sdeu, sdun, noise_cov, pl_matrix)

    raise RuntimeError("Failed to sample a positive-definite covariance matrix.")


def _segment_lengths_m(coords: np.ndarray) -> np.ndarray:
    """Return geodesic segment lengths (metres) for a polyline."""
    lon0 = coords[:-1, 0]
    lat0 = coords[:-1, 1]
    lon1 = coords[1:, 0]
    lat1 = coords[1:, 1]
    _, _, dist = GEOD.inv(lon0, lat0, lon1, lat1)
    return np.asarray(dist, dtype=float)


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


def _compute_protection_level(params: CovarianceParams) -> float:
    eigenvalues = np.linalg.eigvalsh(params.pl_matrix)
    max_eig = float(np.max(eigenvalues))
    return math.sqrt(max_eig * CHI_SQUARE_QUANTILE_9999_DF2)


def _generate_single(config: TrajectoryConfig) -> TrajectoryResult:
    if not ROAD_SEGMENTS:
        raise RuntimeError("ROAD_SEGMENTS has not been initialised")
    if not OUTGOING_SEGMENTS or not INCOMING_SEGMENTS:
        raise RuntimeError("Road segment adjacency has not been initialised")

    rng = np.random.default_rng(config.seed)

    min_span_deg = 0.1
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
            while current_span < min_span_deg and extension_counter < max_extension_steps:
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

            if current_span < min_span_deg:
                continue

            candidate_arr = np.asarray(path_coords, dtype=float)
            seg_lengths_candidate = _segment_lengths_m(candidate_arr)
            if seg_lengths_candidate.size == 0:
                continue
            chosen_coords = candidate_arr
            chosen_edges = edge_sequence
            seg_lengths = seg_lengths_candidate
            cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths_candidate)))
            break

        if chosen_coords is None or chosen_edges is None or seg_lengths is None or cumulative is None:
            continue

        sample_distances = np.linspace(0.0, cumulative[-1], num=config.num_points, dtype=float)

        truth_coords: List[Tuple[float, float]] = []
        timestamps: List[float] = []
        observations: List[Observation] = []
        prev_signature: Tuple[float, float, float] | None = None

        for seq, dist_m in enumerate(sample_distances):
            base_x, base_y = _interpolate_along(chosen_coords, float(dist_m), cumulative, seg_lengths)
            truth_coords.append((base_x, base_y))

            params = _random_covariance_params(rng)
            signature = (params.sdn, params.sde, params.sdne)
            attempts = 0
            while prev_signature is not None and np.allclose(signature, prev_signature) and attempts < 5:
                params = _random_covariance_params(rng)
                signature = (params.sdn, params.sde, params.sdne)
                attempts += 1
            prev_signature = signature

            noise = rng.multivariate_normal(mean=np.zeros(2, dtype=float), cov=params.noise_cov)
            obs_x = float(base_x + noise[0])
            obs_y = float(base_y + noise[1])
            timestamp = float(config.base_epoch + dist_m / max(config.speed_mps, 1e-3))

            observations.append(
                Observation(
                    seq=seq,
                    timestamp=timestamp,
                    x=obs_x,
                    y=obs_y,
                    sdn=params.sdn,
                    sde=params.sde,
                    sdu=params.sdu,
                    sdne=params.sdne,
                    sdeu=params.sdeu,
                    sdun=params.sdun,
                    protection_level=_compute_protection_level(params),
                )
            )
            timestamps.append(timestamp)

        arr_truth = np.asarray(truth_coords, dtype=float)
        if arr_truth.shape[0] < 2:
            continue
        span_total = segment_span(arr_truth)
        if span_total < min_span_deg:
            continue

        return TrajectoryResult(
            traj_id=config.traj_id,
            timestamps=timestamps,
            truth_coords=truth_coords,
            observations=observations,
            edge_ids=chosen_edges,
        )

    raise RuntimeError("Unable to generate a trajectory with ≥0.1° combined longitude/latitude span.")


def _make_tasks(count: int, num_points: int, speed: float,
                base_epoch: float, seed: int | None, start_id: int) -> List[TrajectoryConfig]:
    master_rng = np.random.default_rng(seed)
    tasks: List[TrajectoryConfig] = []
    for offset in range(count):
        task_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        tasks.append(
            TrajectoryConfig(
                traj_id=start_id + offset,
                num_points=num_points,
                speed_mps=speed,
                base_epoch=base_epoch,
                seed=task_seed,
            )
        )
    return tasks


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic observation files for the CMM CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--points", type=int, default=60, help="Number of points per trajectory.")
    parser.add_argument("--speed", type=float, default=12.0, help="Vehicle speed (m/s).")
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
        "sdn", "sde", "sdne", "protection_level",
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
                        f"{obs.sdn:.8f}",
                        f"{obs.sde:.8f}",
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
            covariances = [
                [
                    round(obs.sdn, 8),
                    round(obs.sde, 8),
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
            "shapefile": str(Path(args.shapefile).resolve()),
            "seed": args.seed,
            "start_id": args.start_id,
            "base_epoch": args.base_epoch,
        },
        "trajectories": [
            {
                "id": traj.traj_id,
                "num_points": len(traj.truth_coords),
                "duration": traj.timestamps[-1] - traj.timestamps[0] if len(traj.timestamps) > 1 else 0.0,
                "edge_ids": traj.edge_ids,
            }
            for traj in sorted(results, key=lambda item: item.traj_id)
        ],
    }
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


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
    tasks = _make_tasks(args.count, args.points, args.speed, base_epoch, args.seed, args.start_id)

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

    _write_observation_csv(results, output_dir / "observations.csv")
    _write_ground_truth_csv(results, output_dir / "ground_truth.csv")
    if args.export_points:
        _write_ground_truth_points(results, output_dir / "ground_truth_points.csv")
    _write_metadata(results, args, output_dir / "metadata.json")
    _write_cmm_trajectory_csv(results, output_dir / "cmm_trajectory.csv")

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
