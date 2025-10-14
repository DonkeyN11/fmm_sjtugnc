#!/usr/bin/env bash
# 在脚本开头添加：
# export PATH="/usr/bin:/bin:/usr/local/bin"
# export PYTHONPATH="/home/dell/Czhang/fmm_sjtugnc/build/python:$PYTHONPATH"
# 使用:./fmm_starter.sh ./python/Monte_Carlo.py 
# --network input/map/haikou/edges.shp 
# --trials 1000 
# --num-paths 5 
# --noise-std 0.00005 0.0001 units in degrees, 5m 10m
# --dump-ground-truth 
# --ubodt input/map/haikou_ubodt_mmap.bin


"""Monte Carlo simulation for GNSS noise impact on map matching credibility.

The script automates the following workflow for one or multiple road networks:

1. Select a configurable number of random ground-truth paths on the network.
2. Densify each path to obtain reference trajectory points.
3. Generate noisy observations by perturbing the reference trajectories with
   Gaussian noise drawn from a supplied (isotropic) covariance.
4. Persist the simulated observations to CSV files (one file per path).
5. Match every noisy trajectory with FMM and collect cpath/pgeom/cumulative
   probability information.
6. Evaluate the matching accuracy, derive a trustworthiness score from the
   cumulative probability, and compute their convergence behaviour.
7. Export per-trial statistics, aggregated summaries, and diagnostic plots that
   show accuracy and confidence versus the number of trials.

The implementation focuses on configurability and traceability: command line
arguments define the simulation scope, intermediate artefacts are organised in
per-scenario folders, and summary files make it easy to inspect outcomes or
reuse the data for further analysis.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, '/home/dell/Czhang/fmm_sjtugnc/build/python')
os.environ.setdefault("SPDLOG_LEVEL", "err")

import argparse
import csv
import json
import math
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # GDAL Python bindings for field introspection
    from osgeo import ogr
except ImportError:  # pragma: no cover - optional dependency
    ogr = None

try:  # Progress bar for long-running loops
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


try:  # Matplotlib is only needed for the reporting stage.
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency guard.
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when matplotlib is available.
    _MATPLOTLIB_IMPORT_ERROR = None


try:  # FMM bindings provide the core map matching capabilities.
    from fmm import (
        Network,
        NetworkGraph,
        FastMapMatch,
        FastMapMatchConfig,
        Trajectory,
        wkt2linestring,
        UBODT,
    )
except ImportError as exc:  # pragma: no cover - import failure surfaced later.
    Network = NetworkGraph = FastMapMatch = FastMapMatchConfig = UBODT = None  # type: ignore
    _FMM_IMPORT_ERROR = exc
else:
    _FMM_IMPORT_ERROR = None


DEFAULT_NETWORK = "input/map/haikou/edges.shp"
DEFAULT_NOISE_STD = (5.0,)  # metres


@dataclass
class GroundTruthPath:
    """Container for ground-truth path metadata and reference coordinates."""

    identifier: str
    edge_indices: Tuple[int, ...]
    edge_ids: Tuple[int, ...]
    reference_coords: np.ndarray  # shape (N, 2)
    length: float
    wkt: str


@dataclass
class TrialOutcome:
    """Per-trial evaluation outcome."""

    index: int
    matched: bool
    accuracy: float
    credibility: float
    cumulative_log_prob: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Monte Carlo simulations to assess the correlation between "
            "map matching credibility and correctness."
        )
    )
    parser.add_argument(
        "--network",
        dest="networks",
        action="append",
        default=[],
        help=(
            "Path to a network shapefile. Provide multiple times to evaluate "
            "several networks. Defaults to input/map/haikou/edges.shp."
        ),
    )
    parser.add_argument(
        "--network-id-field",
        type=str,
        default=None,
        help="Field name storing edge IDs (e.g. id, fid, key).",
    )
    parser.add_argument(
        "--network-source-field",
        type=str,
        default=None,
        help="Field name storing source node IDs (e.g. source, u).",
    )
    parser.add_argument(
        "--network-target-field",
        type=str,
        default=None,
        help="Field name storing target node IDs (e.g. target, v).",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=10,
        help="Number of random ground-truth paths to sample per scenario.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10000,
        help="Number of noisy trajectories (trials) to generate per path.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs="*",
        default=list(DEFAULT_NOISE_STD),
        help=(
            "Standard deviation(s) in metres for isotropic GNSS noise. Each "
            "value defines one simulation scenario (covariance = sigma^2 I)."
        ),
    )
    parser.add_argument(
        "--point-spacing",
        type=float,
        default=25.0,
        help=(
            "Spacing in coordinate units when densifying ground-truth paths. "
            "The default of 25 roughly corresponds to 25 metres if the "
            "network is stored in a metric CRS."
        ),
    )
    parser.add_argument(
        "--min-path-length",
        type=float,
        default=0.003,
        help=(
            "Minimum acceptable ground-truth path length (in coordinate "
            "units). Paths shorter than this threshold are skipped."
        ),
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=5,
        help=("Minimum number of edges a ground-truth path must include."),
    )
    parser.add_argument(
        "--max-path-attempts",
        type=int,
        default=500,
        help="Maximum number of sampling attempts per path before aborting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/monte_carlo"),
        help="Directory used for all intermediate and final artefacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator (numpy default).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel worker processes (default: CPU count).",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="Number of OpenMP threads FMM should use internally (sets OMP_NUM_THREADS).",
    )
    parser.add_argument(
        "--config-k",
        type=int,
        default=8,
        help="FastMapMatch k parameter (number of candidates per observation).",
    )
    parser.add_argument(
        "--config-radius",
        type=float,
        default=120.0,
        help="FastMapMatch search radius in metres (or CRS units).",
    )
    parser.add_argument(
        "--reverse-tolerance",
        type=float,
        default=0.0,
        help="FastMapMatch reverse tolerance (metres).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (useful on headless systems without GUI).",
    )
    parser.add_argument(
        "--ground-truth-cache",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with precomputed ground-truth paths to reuse. "
            "When provided, the sampling stage is skipped and the cached "
            "paths guide the simulations."
        ),
    )
    parser.add_argument(
        "--dump-ground-truth",
        action="store_true",
        help="Persist sampled ground-truth paths to ground_truth.json files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during simulation loops.",
    )
    parser.add_argument(
        "--ubodt",
        dest="ubodt_paths",
        action="append",
        default=[],
        help=(
            "Path to a UBODT file corresponding to the network(s). Provide once "
            "per network or a single value to reuse across all networks. When "
            "omitted, the script attempts to infer the path from the network "
            "location (e.g. input/map/<name>_ubodt_mmap.bin)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.networks:
        args.networks = [DEFAULT_NETWORK]
    if args.ubodt_paths and len(args.ubodt_paths) not in (1, len(args.networks)):
        parser.error(
            "Number of --ubodt values must match the number of networks or be a single shared path"
        )
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.num_paths < 1:
        parser.error("--num-paths must be at least 1")
    if any(std <= 0 for std in args.noise_std):
        parser.error("--noise-std values must be positive")
    if args.min_edges < 1:
        parser.error("--min-edges must be at least 1")
    if args.point_spacing <= 0:
        parser.error("--point-spacing must be positive")
    if args.min_path_length <= 0:
        parser.error("--min-path-length must be positive")
    return args


def ensure_dependencies(no_plots: bool) -> None:
    """Validate that required third-party modules are available."""

    if _FMM_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Unable to import the `fmm` Python bindings. Ensure the module is "
            "installed and that all shared library dependencies are resolvable "
            f"(original error: {_FMM_IMPORT_ERROR})."
        )
    if not no_plots and _MATPLOTLIB_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Matplotlib is required for plotting but could not be imported. "
            f"Install matplotlib or pass --no-plots. Original error: "
            f"{_MATPLOTLIB_IMPORT_ERROR}"
        )


def detect_network_fields(
    path: Path,
    id_hint: Optional[str],
    source_hint: Optional[str],
    target_hint: Optional[str],
) -> Tuple[str, str, str]:
    """Determine field names for edge ID, source, and target."""

    if id_hint and source_hint and target_hint:
        return id_hint, source_hint, target_hint

    if ogr is None:
        raise RuntimeError(
            "GDAL Python bindings are required to infer network field names. "
            "Install the `gdal` package or specify --network-*-field arguments."
        )

    dataset = ogr.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Failed to open network shapefile {path} for inspection")
    layer = dataset.GetLayer(0)
    definition = layer.GetLayerDefn()
    field_lookup = {}
    for idx in range(definition.GetFieldCount()):
        original = definition.GetFieldDefn(idx).GetName()
        field_lookup[original.lower()] = original

    def resolve(hint: Optional[str], candidates: Sequence[str]) -> Optional[str]:
        if hint:
            name = hint.lower()
            if name in field_lookup:
                return field_lookup[name]
        for candidate in candidates:
            name = candidate.lower()
            if name in field_lookup:
                return field_lookup[name]
        return None

    id_field = resolve(id_hint, ("id", "fid", "edge_id", "key"))
    source_field = resolve(source_hint, ("source", "from", "src", "u"))
    target_field = resolve(target_hint, ("target", "to", "dst", "v"))

    missing = [label for label, value in (("id", id_field), ("source", source_field), ("target", target_field)) if value is None]
    if missing:
        raise RuntimeError(
            "Unable to determine required network fields: "
            + ", ".join(missing)
            + ". Specify them via --network-id-field, --network-source-field, --network-target-field."
        )
    return id_field, source_field, target_field


def load_network(
    path: Path,
    id_field: Optional[str],
    source_field: Optional[str],
    target_field: Optional[str],
) -> Tuple[Network, NetworkGraph, str, str, str]:
    """Load the network shapefile and construct its graph representation."""

    print(f"Loading network from {path} ...")
    resolved_id, resolved_source, resolved_target = detect_network_fields(path, id_field, source_field, target_field)
    network = Network(str(path), resolved_id, resolved_source, resolved_target)
    graph = NetworkGraph(network)
    print(
        f"  > nodes: {network.get_node_count():,}, "
        f"edges: {network.get_edge_count():,}"
    )
    print(
        "  > using fields id=%s, source=%s, target=%s"
        % (resolved_id, resolved_source, resolved_target)
    )
    return network, graph, resolved_id, resolved_source, resolved_target


def infer_ubodt_path(network_path: Path) -> Optional[Path]:
    """Guess a UBODT path based on the network location."""

    base_names = {network_path.stem, network_path.parent.name}
    base_names.discard("")
    candidate_dirs = {network_path.parent, network_path.parent.parent}
    candidate_paths: List[Path] = []
    for directory in candidate_dirs:
        if directory is None or not directory.exists():
            continue
        for name in base_names:
            candidate_paths.extend(
                [
                    directory / f"{name}_ubodt_mmap.bin",
                    directory / f"{name}_ubodt.bin",
                    directory / f"{name}_ubodt.txt",
                    directory / f"{name}_ubodt.csv",
                ]
            )
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def load_ubodt(path: Path, quiet: bool = False) -> UBODT:
    """Load a UBODT file using the appropriate reader based on the extension."""

    if not path.exists():
        raise FileNotFoundError(f"UBODT file not found: {path}")
    if not quiet:
        print(f"  > Loading UBODT from {path} ...")
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".txt":
        return UBODT.read_ubodt_file(str(path))
    if suffix == ".csv":
        return UBODT.read_ubodt_csv(str(path))
    if suffix == ".bin":
        if name.endswith("_mmap.bin"):
            return UBODT.read_ubodt_mmap_binary(str(path))
        return UBODT.read_ubodt_binary(str(path))
    raise ValueError(
        f"Unsupported UBODT format for {path}. Use .txt, .csv, or .bin (optionally *_mmap.bin)."
    )


def generate_observations_worker(
    reference_coords: List[List[float]],
    trials: int,
    covariance: List[List[float]],
    seed: int,
    destination: str,
) -> str:
    """Generate noisy observations for a single ground-truth path."""

    coords = np.asarray(reference_coords, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    rng = np.random.default_rng(seed)
    mean = np.zeros(2, dtype=float)

    dest_path = Path(destination)
    with dest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trajectory_id", "wkt"])
        for trial_idx in range(trials):
            noise = rng.multivariate_normal(mean=mean, cov=cov, size=len(coords))
            noisy_coords = ensure_linestring_valid(coords + noise)
            writer.writerow([trial_idx, coords_to_wkt(noisy_coords)])
    return destination


_MATCHER: Optional[FastMapMatch] = None
_MATCH_CONFIG: Optional[FastMapMatchConfig] = None
_NETWORK_HOLDER: Optional[Network] = None
_GRAPH_HOLDER: Optional[NetworkGraph] = None
_UBODT_HOLDER: Optional[UBODT] = None


def prepare_matcher(
    network: Network,
    graph: NetworkGraph,
    ubodt: UBODT,
    config_params: Dict[str, float],
) -> None:
    """Load matcher resources once; forked workers reuse them."""

    global _MATCHER, _MATCH_CONFIG, _NETWORK_HOLDER, _GRAPH_HOLDER, _UBODT_HOLDER

    _NETWORK_HOLDER = network
    _GRAPH_HOLDER = graph
    _UBODT_HOLDER = ubodt
    _MATCHER = FastMapMatch(_NETWORK_HOLDER, _GRAPH_HOLDER, _UBODT_HOLDER)
    _MATCH_CONFIG = FastMapMatchConfig(
        k_arg=int(config_params["k"]),
        r_arg=float(config_params["radius"]),
        gps_error=float(config_params["gps_error"]),
        reverse_tolerance=float(config_params["reverse_tolerance"]),  
    )


def match_single_trajectory(
    path_id: str,
    trial_idx: int,
    wkt: str,
    ground_truth_edges: Tuple[int, ...],
) -> Tuple[str, TrialOutcome, Optional[str]]:
    """Match one trajectory and return outcome with optional warning."""

    if _MATCHER is None or _MATCH_CONFIG is None:
        raise RuntimeError("Match worker not initialised correctly")

    traj = Trajectory(trial_idx, wkt2linestring(wkt), [])

    try:
        match_result = _MATCHER.match_traj(traj, _MATCH_CONFIG)
    except RuntimeError as exc:
        warning = f"{path_id},trial={trial_idx}: runtime error {exc}"
        outcome = TrialOutcome(
            index=trial_idx,
            matched=False,
            accuracy=0.0,
            credibility=float("nan"),
            cumulative_log_prob=float("nan"),
        )
        return path_id, outcome, warning

    matched_edges = tuple(match_result.cpath)
    is_matched = bool(matched_edges)
    accuracy = 1.0 if matched_edges == ground_truth_edges and is_matched else 0.0

    cumu_prob = float("nan")
    opt_path = getattr(match_result, "opt_candidate_path", None)
    if opt_path is not None:
        last_mc = None
        try:
            for mc in opt_path:
                last_mc = mc
        except TypeError:
            last_mc = None
        if last_mc is not None:
            cumu_prob = last_mc.cumu_prob
    credibility = credibility_from_cumu_prob(cumu_prob)

    warning = None
    if not is_matched:
        warning = f"{path_id},trial={trial_idx}: unmatched trajectory"

    outcome = TrialOutcome(
        index=trial_idx,
        matched=is_matched,
        accuracy=accuracy,
        credibility=credibility,
        cumulative_log_prob=cumu_prob,
    )
    return path_id, outcome, warning


def create_process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create a process pool with fork context when available."""

    try:
        ctx = mp.get_context("fork")
        try:
            return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        except TypeError:
            pass
    except (ValueError, AttributeError):
        pass
    return ProcessPoolExecutor(max_workers=max_workers)


def linestring_to_coords(line) -> List[Tuple[float, float]]:
    """Convert an FMM LineString to a list of (x, y) coordinates."""

    num_points = line.get_num_points()
    return [(line.get_x(i), line.get_y(i)) for i in range(num_points)]


def deduplicate_coords(coords: Sequence[Tuple[float, float]], tol: float = 1e-6) -> List[Tuple[float, float]]:
    """Remove consecutive duplicates within a tolerance."""

    if not coords:
        return []
    cleaned: List[Tuple[float, float]] = [coords[0]]
    for x, y in coords[1:]:
        px, py = cleaned[-1]
        if math.hypot(x - px, y - py) > tol:
            cleaned.append((x, y))
    if len(cleaned) == 1:
        cleaned.append(coords[-1])
    return cleaned


def densify_coords(
    coords: Sequence[Tuple[float, float]],
    spacing: float,
    tol: float = 1e-6,
) -> np.ndarray:
    """Densify a polyline to approximately equal spacing between points."""

    if len(coords) < 2:
        raise ValueError("A ground-truth path must contain at least two points")
    densified: List[Tuple[float, float]] = [coords[0]]
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len <= tol:
            continue
        steps = max(int(seg_len // spacing), 0)
        for step in range(1, steps + 1):
            ratio = (step * spacing) / seg_len
            if ratio >= 1:
                break
            densified.append((x0 + ratio * dx, y0 + ratio * dy))
        densified.append((x1, y1))
    cleaned = deduplicate_coords(densified, tol)
    if len(cleaned) < 2:
        raise ValueError("Densification collapsed the path to a single point")
    return np.asarray(cleaned, dtype=float)


def compute_covariance(std: float) -> np.ndarray:
    """Return a 2x2 isotropic covariance matrix for the provided std."""

    variance = float(std) ** 2
    return np.array([[variance, 0.0], [0.0, variance]], dtype=float)


def coords_to_wkt(coords: np.ndarray) -> str:
    """Convert a coordinate array of shape (N, 2) into a WKT LineString."""

    parts = [f"{x:.6f} {y:.6f}" for x, y in coords]
    return "LINESTRING(" + ",".join(parts) + ")"


def sample_ground_truth_paths(
    network: Network,
    graph: NetworkGraph,
    num_paths: int,
    rng: np.random.Generator,
    point_spacing: float,
    min_edges: int,
    min_length: float,
    max_attempts: int,
) -> List[GroundTruthPath]:
    """Randomly sample ground-truth paths that meet the configured criteria."""

    results: List[GroundTruthPath] = []
    vertex_count = graph.get_num_vertices()
    attempts = 0
    while len(results) < num_paths and attempts < max_attempts:
        attempts += 1
        source_idx = rng.integers(0, vertex_count)
        target_idx = rng.integers(0, vertex_count)
        if source_idx == target_idx:
            continue
        try:
            path_indices = list(graph.shortest_path_astar(int(source_idx), int(target_idx)))
        except RuntimeError:
            continue
        if len(path_indices) < min_edges:
            continue
        line = network.route2geometry(path_indices)
        length = float(line.get_length())
        if not math.isfinite(length) or length < min_length:
            continue
        raw_coords = deduplicate_coords(linestring_to_coords(line))
        try:
            reference_coords = densify_coords(raw_coords, point_spacing)
        except ValueError:
            continue
        edge_ids = tuple(network.get_edge_id(index) for index in path_indices)
        identifier = f"path_{len(results):02d}"
        results.append(
            GroundTruthPath(
                identifier=identifier,
                edge_indices=tuple(path_indices),
                edge_ids=edge_ids,
                reference_coords=reference_coords,
                length=length,
                wkt=line.export_wkt(),
            )
        )
    if len(results) < num_paths:
        raise RuntimeError(
            "Unable to sample enough ground-truth paths with the specified "
            "constraints. Consider relaxing --min-path-length or --min-edges."
        )
    return results


def load_ground_truth_cache(cache_path: Path) -> List[GroundTruthPath]:
    """Load cached ground-truth paths from a JSON file."""

    with cache_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    result: List[GroundTruthPath] = []
    for item in data:
        coords = np.asarray(item["reference_coords"], dtype=float)
        result.append(
            GroundTruthPath(
                identifier=item["identifier"],
                edge_indices=tuple(item["edge_indices"]),
                edge_ids=tuple(item["edge_ids"]),
                reference_coords=coords,
                length=float(item["length"]),
                wkt=item["wkt"],
            )
        )
    return result


def dump_ground_truth_cache(paths: Sequence[GroundTruthPath], destination: Path) -> None:
    """Persist ground-truth paths to JSON for reproducibility."""

    payload = [
        {
            "identifier": p.identifier,
            "edge_indices": list(p.edge_indices),
            "edge_ids": list(p.edge_ids),
            "reference_coords": p.reference_coords.tolist(),
            "length": p.length,
            "wkt": p.wkt,
        }
        for p in paths
    ]
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"  > Ground truth cache written to {destination}")


def write_truth_file(
    paths: Sequence[GroundTruthPath],
    truth_file: Path,
    network_label: str,
    scenario_tag: str,
) -> None:
    """Write ground-truth trajectories to a shared CSV file."""

    truth_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = truth_file.exists()
    with truth_file.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow([
                "network",
                "scenario",
                "path_id",
                "length",
                "edge_ids",
                "reference_wkt",
                "network_wkt",
            ])
        for path in paths:
            writer.writerow(
                [
                    network_label,
                    scenario_tag,
                    path.identifier,
                    f"{path.length:.6f}",
                    "|".join(str(edge) for edge in path.edge_ids),
                    coords_to_wkt(path.reference_coords),
                    path.wkt,
                ]
            )


def credibility_from_cumu_prob(value: float) -> float:
    """Map the cumulative log probability to a bounded credibility score."""

    if not math.isfinite(value):
        return float("nan")
    if value >= 0:
        return 1.0
    # Clamp to avoid underflow when the likelihood is extremely small.
    value = max(value, -50.0)
    return math.exp(value)


def ensure_linestring_valid(coords: np.ndarray) -> np.ndarray:
    """Guarantee a trajectory contains at least two distinct points."""

    if len(coords) < 2:
        raise ValueError("Observation has fewer than two points")
    filtered = [coords[0]]
    for point in coords[1:]:
        if np.linalg.norm(point - filtered[-1]) > 1e-6:
            filtered.append(point)
    if len(filtered) < 2:
        filtered.append(coords[-1])
    return np.asarray(filtered, dtype=float)


def running_mean(series: Sequence[float]) -> List[float]:
    """Compute running means while gracefully handling NaN values."""

    means: List[float] = []
    total = 0.0
    count = 0
    for value in series:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            means.append(total / count if count else float("nan"))
            continue
        total += value
        count += 1
        means.append(total / count)
    return means


def export_trial_metrics(destination: Path, outcomes: Sequence[TrialOutcome]) -> None:
    """Write per-trial metrics to CSV for downstream analysis."""

    with destination.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trial", "accuracy", "credibility", "cumu_log_prob"])
        for outcome in outcomes:
            writer.writerow(
                [
                    outcome.index,
                    f"{outcome.accuracy:.6f}",
                    "" if math.isnan(outcome.credibility) else f"{outcome.credibility:.6f}",
                    "" if math.isnan(outcome.cumulative_log_prob) else f"{outcome.cumulative_log_prob:.6f}",
                ]
            )


def aggregate_outcomes(paths: Sequence[List[TrialOutcome]]) -> Tuple[np.ndarray, np.ndarray]:
    """Transform outcomes into matrices for accuracy and credibility."""

    accuracy_rows = [[outcome.accuracy for outcome in seq] for seq in paths]
    credibility_rows = [[outcome.credibility for outcome in seq] for seq in paths]
    return np.asarray(accuracy_rows, dtype=float), np.asarray(credibility_rows, dtype=float)


def compute_running_statistics(matrix: np.ndarray) -> np.ndarray:
    """Return running means for each row of a matrix."""

    running = []
    for row in matrix:
        running.append(running_mean(row))
    return np.asarray(running, dtype=float)


def nanmean_axis0(data: np.ndarray) -> np.ndarray:
    """Compute nan-aware mean along axis 0."""

    with np.errstate(invalid="ignore"):
        return np.nanmean(data, axis=0)


def plot_scenario(
    destination: Path,
    accuracies: np.ndarray,
    credibilities: np.ndarray,
    scenario_title: str,
) -> None:
    """Plot running accuracy and credibility trends for a scenario."""

    if plt is None:
        return

    accuracy_running = compute_running_statistics(accuracies)
    credibility_running = compute_running_statistics(credibilities)
    x_axis = np.arange(1, accuracies.shape[1] + 1)

    fig, (ax_acc, ax_cred) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for idx, row in enumerate(accuracy_running):
        ax_acc.plot(x_axis, row, alpha=0.3, label=f"Path {idx:02d}" if idx < 10 else None)
    ax_acc.plot(x_axis, nanmean_axis0(accuracy_running), color="black", linewidth=2.0, label="Scenario mean")
    ax_acc.set_ylabel("Running accuracy")
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.grid(True, linestyle="--", alpha=0.4)
    ax_acc.legend(loc="lower right", fontsize="small", ncol=2)

    for idx, row in enumerate(credibility_running):
        ax_cred.plot(x_axis, row, alpha=0.3)
    ax_cred.plot(x_axis, nanmean_axis0(credibility_running), color="black", linewidth=2.0)
    ax_cred.set_xlabel("Number of trials")
    ax_cred.set_ylabel("Running credibility")
    ax_cred.set_ylim(0.0, 1.05)
    ax_cred.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(scenario_title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    print(f"  > Plot saved to {destination}")


def summarise_scenario(
    destination: Path,
    scenario_name: str,
    noise_std: float,
    accuracies: np.ndarray,
    credibilities: np.ndarray,
) -> None:
    """Persist scenario-level summary statistics."""

    accuracy_mean = float(np.mean(accuracies))
    credibility_mean = float(np.nanmean(credibilities))
    summary = {
        "scenario": scenario_name,
        "noise_std": noise_std,
        "average_accuracy": accuracy_mean,
        "average_credibility": credibility_mean,
    }
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"  > Scenario summary: accuracy={accuracy_mean:.4f}, "
        f"credibility={credibility_mean:.4f}"
    )


def run_scenario(
    network_path: Path,
    network: Network,
    graph: NetworkGraph,
    ubodt: UBODT,
    ground_truth_paths: Sequence[GroundTruthPath],
    noise_std: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    """Execute the full Monte Carlo workflow for one network/noise pair."""

    scenario_tag = f"noise_{noise_std:.2f}"
    scenario_root = args.output_dir / network_path.stem / scenario_tag
    scenario_root.mkdir(parents=True, exist_ok=True)

    covariance_list = compute_covariance(noise_std).tolist()
    truth_file = args.output_dir.parent / "monte_carlo_truth.csv"
    write_truth_file(ground_truth_paths, truth_file, network_path.stem, scenario_tag)

    max_workers = args.max_workers or (os.cpu_count() or 1)

    # --- Generate observations in parallel ---
    observation_files: Dict[str, Path] = {}
    seeds = rng.integers(0, 2**32 - 1, size=len(ground_truth_paths))
    obs_bar = tqdm(total=len(ground_truth_paths), desc="Observations", unit="path") if (not args.no_progress and tqdm is not None) else None

    with create_process_pool(max_workers) as executor:
        futures = {
            executor.submit(
                generate_observations_worker,
                path.reference_coords.tolist(),
                args.trials,
                covariance_list,
                int(seed),
                str(scenario_root / f"{path.identifier}_observations.csv"),
            ): path.identifier
            for path, seed in zip(ground_truth_paths, seeds)
        }

        for future in as_completed(futures):
            path_id = futures[future]
            destination = Path(future.result())
            observation_files[path_id] = destination
            if obs_bar is not None:
                obs_bar.update(1)

    if obs_bar is not None:
        obs_bar.close()

    # --- Prepare matching tasks ---
    metrics_by_path: Dict[str, List[TrialOutcome]] = defaultdict(list)
    warnings: List[str] = []
    tasks: List[Tuple[str, int, str, Tuple[int, ...]]] = []
    edge_lookup = {path.identifier: tuple(path.edge_ids) for path in ground_truth_paths}

    for path in ground_truth_paths:
        obs_file = observation_files[path.identifier]
        with obs_file.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                tasks.append((path.identifier, int(row[0]), row[1], edge_lookup[path.identifier]))

    config_params = {
        "k": args.config_k,
        "radius": args.config_radius,
        "gps_error": noise_std,
        "reverse_tolerance": args.reverse_tolerance,
    }

    try:
        prepare_matcher(network, graph, ubodt, config_params)
    except Exception as exc:
        print(f"  ! Failed to initialise matcher: {exc}", file=sys.stderr)
        return

    match_bar = tqdm(total=len(tasks), desc="Matching", unit="trial") if (not args.no_progress and tqdm is not None) else None

    if tasks:
        path_ids = [t[0] for t in tasks]
        trial_indices = [t[1] for t in tasks]
        wkts = [t[2] for t in tasks]
        edges_seq = [t[3] for t in tasks]
    else:
        path_ids = trial_indices = wkts = edges_seq = []

    with create_process_pool(max_workers) as executor:
        results_iter = executor.map(match_single_trajectory, path_ids, trial_indices, wkts, edges_seq)
        for path_id, outcome, warning in results_iter:
            metrics_by_path[path_id].append(outcome)
            if warning:
                warnings.append(warning)
            if match_bar is not None:
                match_bar.update(1)

    if match_bar is not None:
        match_bar.close()

    # Sort outcomes per path and write metrics
    ordered_outcomes: List[List[TrialOutcome]] = []
    for path in ground_truth_paths:
        outcomes = metrics_by_path[path.identifier]
        outcomes.sort(key=lambda item: item.index)
        ordered_outcomes.append(outcomes)
        metrics_file = scenario_root / f"{path.identifier}_metrics.csv"
        export_trial_metrics(metrics_file, outcomes)

    accuracies, credibilities = aggregate_outcomes(ordered_outcomes)
    summary_file = scenario_root / "scenario_summary.json"
    summarise_scenario(summary_file, scenario_tag, noise_std, accuracies, credibilities)

    if warnings:
        warnings_file = scenario_root / "warnings.log"
        with warnings_file.open("w", encoding="utf-8") as fh:
            for entry in warnings:
                fh.write(entry + "\n")
        print(f"  > Warnings recorded in {warnings_file}")

    if not args.no_plots:
        plot_file = scenario_root / "convergence.png"
        title = f"{network_path.stem} – σ={noise_std:.2f}"
        plot_scenario(plot_file, accuracies, credibilities, title)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.omp_threads is not None:
        if args.omp_threads < 1:
            raise ValueError("--omp-threads must be positive")
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    ensure_dependencies(args.no_plots)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for net_idx, network_str in enumerate(args.networks):
        network_path = Path(network_str)
        if not network_path.exists():
            print(f"Skipping {network_path}: file not found", file=sys.stderr)
            continue

        print("=" * 72)
        print(f"Preparing network: {network_path}")
        try:
            network, graph, resolved_id, resolved_source, resolved_target = load_network(
                network_path,
                args.network_id_field,
                args.network_source_field,
                args.network_target_field,
            )
        except RuntimeError as exc:
            print(f"Skipping {network_path}: {exc}", file=sys.stderr)
            continue

        network_root = args.output_dir / network_path.stem
        network_root.mkdir(parents=True, exist_ok=True)

        ubodt_arg: Optional[Path]
        if args.ubodt_paths:
            if len(args.ubodt_paths) == 1:
                ubodt_arg = Path(args.ubodt_paths[0])
            else:
                ubodt_arg = Path(args.ubodt_paths[net_idx])
        else:
            ubodt_arg = None

        if ubodt_arg is not None:
            ubodt_path = ubodt_arg
        else:
            inferred = infer_ubodt_path(network_path)
            if inferred is None:
                print(
                    f"Skipping {network_path}: unable to infer UBODT path. "
                    "Provide --ubodt explicitly.",
                    file=sys.stderr,
                )
                continue
            ubodt_path = inferred

        try:
            ubodt_obj = load_ubodt(ubodt_path)
        except Exception as exc:
            print(f"Skipping {network_path}: failed to load UBODT ({exc})", file=sys.stderr)
            continue

        if args.ground_truth_cache:
            cache_path = Path(args.ground_truth_cache)
            if cache_path.exists():
                ground_truth_paths = load_ground_truth_cache(cache_path)
                print(
                    f"  > Loaded {len(ground_truth_paths)} ground-truth paths "
                    f"from cache {cache_path}"
                )
            else:
                network_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                ground_truth_paths = sample_ground_truth_paths(
                    network=network,
                    graph=graph,
                    num_paths=args.num_paths,
                    rng=network_rng,
                    point_spacing=args.point_spacing,
                    min_edges=args.min_edges,
                    min_length=args.min_path_length,
                    max_attempts=args.max_path_attempts,
                )
                if args.dump_ground_truth:
                    dump_ground_truth_cache(ground_truth_paths, cache_path)
        else:
            network_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
            ground_truth_paths = sample_ground_truth_paths(
                network=network,
                graph=graph,
                num_paths=args.num_paths,
                rng=network_rng,
                point_spacing=args.point_spacing,
                min_edges=args.min_edges,
                min_length=args.min_path_length,
                max_attempts=args.max_path_attempts,
            )
            if args.dump_ground_truth:
                dump_ground_truth_cache(ground_truth_paths, network_root / "ground_truth.json")

        for noise_std in args.noise_std:
            print("-" * 72)
            print(f"Scenario: network={network_path}, noise_std={noise_std:.2f}")
            run_scenario(
                network_path=network_path,
                network=network,
                graph=graph,
                ubodt=ubodt_obj,
                ground_truth_paths=ground_truth_paths,
                noise_std=noise_std,
                args=args,
                rng=rng,
            )


if __name__ == "__main__":  # pragma: no cover - CL entry point.
    try:
        main()
    except Exception as exc:  # Surface errors with context.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
