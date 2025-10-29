#!/usr/bin/env python3
"""
Utility for generating synthetic CMM trajectory observations with per-point covariance.

The script samples a random road segment from the given shapefile, assumes uniform motion,
and creates noisy observations drawn from multivariate normal distributions governed by
random positive-definite covariance matrices. Protection levels are computed from the
99.99% quantile of the resulting positional error distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "geopandas and shapely are required to run this script. "
        "Install them with `pip install geopandas shapely`."
    ) from exc


# Shared between worker processes after initialization.
ROAD_GEOMETRIES: List[LineString] = []
CHI_SQUARE_QUANTILE_9999_DF2 = 23.025850929940457


@dataclass(frozen=True)
class TrajectoryConfig:
    traj_id: int
    num_points: int
    speed_mps: float
    base_epoch: float
    seed: int


def _initialize_worker(geometries: Sequence[LineString]) -> None:
    global ROAD_GEOMETRIES
    ROAD_GEOMETRIES = list(geometries)


def _load_roads(shapefile: Path) -> List[LineString]:
    gdf = gpd.read_file(shapefile)
    lines: List[LineString] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            if geom.length > 0:
                lines.append(geom)
        elif isinstance(geom, MultiLineString):
            lines.extend([part for part in geom.geoms if part.length > 0])
    if not lines:
        raise ValueError(f"No valid LineString geometries found in {shapefile}")
    return lines


def _random_covariance(rng: np.random.Generator) -> np.ndarray:
    # 1 degree in hainan is about 111 km, so 15 m is about 0.000135 degree.
    sigma_x = rng.uniform(0.00001, 0.000135)
    sigma_y = rng.uniform(0.00001, 0.000135)
    rho = rng.uniform(-0.6, 0.6)
    cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ],
        dtype=float,
    )
    # Numerical safeguard: make sure smallest eigenvalue stays positive.
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = eigvals.min()
    if min_eig <= 1e-6:
        cov += np.eye(2) * (1e-6 - min_eig)
    return cov


def _generate_single(config: TrajectoryConfig) -> dict:
    if not ROAD_GEOMETRIES:
        raise RuntimeError("ROAD_GEOMETRIES is not initialized")
    rng = np.random.default_rng(config.seed)

    line: LineString
    for _ in range(10):
        line = ROAD_GEOMETRIES[int(rng.integers(0, len(ROAD_GEOMETRIES)))]
        if line.length > 0:
            break
    else:
        raise RuntimeError("Failed to select a non-empty LineString")

    distances = np.linspace(0.0, line.length, num=config.num_points)
    coords: List[Tuple[float, float]] = []
    noisy_coords: List[Tuple[float, float]] = []
    timestamps: List[str] = []
    covariances: List[List[List[float]]] = []
    protection_levels: List[float] = []

    prev_cov: np.ndarray | None = None

    for d in distances:
        point = line.interpolate(float(d))
        coords.append((point.x, point.y))

        cov = _random_covariance(rng)
        if prev_cov is not None:
            attempts = 0
            while np.allclose(cov, prev_cov, atol=1e-9) and attempts < 5:
                cov = _random_covariance(rng)
                attempts += 1
        prev_cov = cov

        observation = rng.multivariate_normal([point.x, point.y], cov)
        noisy_coords.append((float(observation[0]), float(observation[1])))

        pl = math.sqrt(float(np.linalg.eigvalsh(cov).max()) * CHI_SQUARE_QUANTILE_9999_DF2)
        protection_levels.append(pl)
        covariances.append(cov.tolist())

        t_seconds = d / max(config.speed_mps, 1e-3)
        timestamp = datetime.fromtimestamp(config.base_epoch, tz=timezone.utc) + timedelta(seconds=float(t_seconds))
        timestamps.append(timestamp.isoformat())

    ground_truth = {
        "geometry": [list(coord) for coord in coords],
        "timestamps": timestamps,
    }
    observations = []
    for idx in range(config.num_points):
        observations.append(
            {
                "timestamp": timestamps[idx],
                "position": list(noisy_coords[idx]),
                "covariance": covariances[idx],
                "protection_level": protection_levels[idx],
            }
        )

    return {
        "id": config.traj_id,
        "speed_m_per_s": config.speed_mps,
        "ground_truth": ground_truth,
        "observations": observations,
    }


def _make_tasks(count: int, num_points: int, speed: float, base_epoch: float, seed: int | None, start_id: int) -> List[TrajectoryConfig]:
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
    parser = argparse.ArgumentParser(description="Generate synthetic CMM trajectory observations.")
    parser.add_argument("--count", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--points", type=int, default=50, help="Number of points per trajectory.")
    parser.add_argument("--speed", type=float, default=12.0, help="Assumed constant speed (m/s).")
    parser.add_argument("--output", type=Path, default=Path("python/generated_cmm_data.json"), help="Output JSON file.")
    parser.add_argument("--shapefile", type=Path, default=Path("input/map/haikou/haikou.shp"), help="Road shapefile to sample.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="Maximum parallel workers.")
    parser.add_argument("--seed", type=int, default=None, help="Master random seed for reproducibility.")
    parser.add_argument("--start-id", type=int, default=1, help="Starting trajectory identifier.")
    parser.add_argument("--base-epoch", type=float, default=None, help="Base UNIX epoch (seconds). Defaults to current time.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    shapefile = args.shapefile.resolve()
    if not shapefile.exists():
        raise SystemExit(f"Shapefile not found: {shapefile}")

    road_geoms = _load_roads(shapefile)
    base_epoch = args.base_epoch if args.base_epoch is not None else time.time()
    tasks = _make_tasks(args.count, args.points, args.speed, base_epoch, args.seed, args.start_id)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.jobs <= 1:
        _initialize_worker(road_geoms)
        results = [_generate_single(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_initialize_worker, initargs=(road_geoms,)) as executor:
            results = list(executor.map(_generate_single, tasks))

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "shapefile": str(shapefile),
        "count": args.count,
        "points_per_trajectory": args.points,
        "speed_m_per_s": args.speed,
        "trajectories": results,
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Generated {args.count} trajectories -> {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
