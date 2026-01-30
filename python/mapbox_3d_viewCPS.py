#!/usr/bin/env python3
"""
Interactive Mapbox GL viewer for comparing CMM/FMM results with ground truth.

The script renders three datasets for the requested trajectory IDs:
  1. Ground truth trajectories (as lines)
  2. CMM match results (as points)
  3. FMM match results (as points)

Hovering over CMM/FMM points displays detailed attributes such as timestamp,
error, ep, tp, and trustworthiness, cumulative probability.
"""

from __future__ import annotations

import argparse
import csv

csv.field_size_limit(10 ** 7)

import json
import math
import os
import warnings
from pathlib import Path
from string import Template
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np

try:
    from pyproj import CRS, Transformer, Geod
except ImportError:  # pragma: no cover - optional dependency
    CRS = None
    Transformer = None
    Geod = None

try:
    import fiona
except ImportError:  # pragma: no cover - optional dependency
    fiona = None

MAPBOX_DEFAULT_TOKEN = (
    "pk.eyJ1IjoiZG9ua2V5LW5pbmciLCJhIjoiY21kenJ5OTY5MGc5azJqb25hdTVtc2tvNiJ9.CgMc9ZNXaZ1HDrC4Zl2aMQ"
)

DEFAULT_EDGES_SHP = "input/map/haikou/edges.shp"
DEFAULT_CMM_TRAJECTORY_CSV = "input_cmm/cmm_trajectory.csv"
ELLIPSE_SCALE = 2.0  # number of standard deviations for covariance ellipses
ELLIPSE_SEGMENTS = 64
CIRCLE_SEGMENTS = 64
GEOD = Geod(ellps="WGS84") if Geod is not None else None
WGS84_CRS = CRS.from_epsg(4326) if CRS is not None else None


class CoordinateTransformer:
    """Helper for converting projected coordinates to lon/lat."""

    def __init__(self, source_crs: Optional["CRS"]) -> None:
        self.source_crs = source_crs
        self.transformer: Optional["Transformer"] = None
        if CRS is None or Transformer is None or source_crs is None or WGS84_CRS is None:
            return
        try:
            crs_obj = CRS.from_user_input(source_crs)
        except Exception:
            return
        if crs_obj.equals(WGS84_CRS):
            return
        self.transformer = Transformer.from_crs(crs_obj, WGS84_CRS, always_xy=True)

    def is_active(self) -> bool:
        return self.transformer is not None

    @staticmethod
    def _looks_like_lonlat(coord: Coordinate) -> bool:
        lon, lat = coord
        return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0

    def convert_point(self, coord: Coordinate) -> Coordinate:
        converted = self.convert_coords([coord])
        return converted[0] if converted else coord

    def convert_coords(self, coords: Sequence[Coordinate]) -> List[Coordinate]:
        if not coords:
            return []
        if self.transformer is None:
            return list(coords)
        sample = coords[0]
        if self._looks_like_lonlat(sample):
            return list(coords)
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        lon_vals, lat_vals = self.transformer.transform(xs, ys)
        return [(float(lon), float(lat)) for lon, lat in zip(lon_vals, lat_vals)]


def read_crs_from_shapefile(shapefile: Path) -> Optional["CRS"]:
    if fiona is None or not shapefile or not shapefile.exists():
        return None
    try:
        with fiona.open(shapefile, "r") as src:
            if src.crs_wkt:
                return CRS.from_wkt(src.crs_wkt) if CRS is not None else None
            if src.crs:
                return CRS.from_user_input(src.crs) if CRS is not None else None
    except Exception as exc:
        warnings.warn(f"Unable to read CRS from shapefile {shapefile}: {exc}")
    return None


def infer_utm_epsg_from_bounds(shapefile: Path) -> Optional[int]:
    if fiona is None or not shapefile or not shapefile.exists():
        return None
    try:
        with fiona.open(shapefile, "r") as src:
            min_x, min_y, max_x, max_y = src.bounds
    except Exception as exc:
        warnings.warn(f"Unable to read bounds from shapefile {shapefile}: {exc}")
        return None
    bounds = (min_x, min_y, max_x, max_y)
    if not all(math.isfinite(value) for value in bounds):
        return None
    lon0 = (min_x + max_x) * 0.5
    lat0 = (min_y + max_y) * 0.5
    if not (-90.0 <= lat0 <= 90.0):
        return None
    zone = int(math.floor((lon0 + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    base = 32600 if lat0 >= 0 else 32700
    return base + zone


def determine_input_crs(
    crs_arg: Optional[str],
    shapefile: Path,
    sample_coord: Optional[Coordinate],
) -> Optional["CRS"]:
    if CRS is None:
        if crs_arg and crs_arg.lower() not in {"", "auto"}:
            raise SystemExit(
                "--input-crs requires pyproj to be installed. Please install pyproj and retry."
            )
        return None
    if crs_arg and crs_arg.lower() not in {"", "auto"}:
        try:
            return CRS.from_user_input(crs_arg)
        except Exception as exc:
            raise SystemExit(f"Unable to parse --input-crs '{crs_arg}': {exc}")
    shapefile_crs = read_crs_from_shapefile(shapefile)
    if shapefile_crs is None:
        return None
    try:
        if shapefile_crs.is_projected:
            return shapefile_crs
    except Exception:
        pass
    if sample_coord and CoordinateTransformer._looks_like_lonlat(sample_coord):
        return None
    epsg = infer_utm_epsg_from_bounds(shapefile)
    if epsg is not None:
        try:
            return CRS.from_epsg(epsg)
        except Exception as exc:
            warnings.warn(f"Failed to build CRS from EPSG:{epsg}: {exc}")
    return None

Coordinate = Tuple[float, float]
Bounds = Tuple[float, float, float, float]


def parse_linestring(wkt: str) -> List[Coordinate]:
    """Parse a WKT LINESTRING into a list of [lon, lat] coordinate pairs."""
    text = (wkt or "").strip()
    if not text or not text.upper().startswith("LINESTRING"):
        return []
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        return []
    body = text[open_idx + 1 : close_idx]
    coords: List[Coordinate] = []
    for token in body.split(","):
        parts = token.strip().split()
        if len(parts) != 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        coords.append((lon, lat))
    return coords


def sample_coordinate_from_csv(path: Path, geom_column: str) -> Optional[Coordinate]:
    if not path or not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=";")
            for row in reader:
                coords = parse_linestring(row.get(geom_column, ""))
                if coords:
                    return coords[0]
    except Exception:
        return None
    return None


def sample_coordinate_from_inputs(candidates: Sequence[Tuple[Path, str]]) -> Optional[Coordinate]:
    for path, column in candidates:
        coord = sample_coordinate_from_csv(path, column)
        if coord:
            return coord
    return None


def parse_value_list(cell: Optional[str]) -> List[str]:
    """Split a semicolon cell containing comma separated values."""
    if not cell:
        return []
    return [part.strip() for part in cell.split(",") if part.strip()]


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        if value.lower() in {"nan", "inf", "-inf"}:  # type: ignore[arg-type]
            return None
    except AttributeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


CandidatePoint = Tuple[Coordinate, Optional[float]]
CandidateGroup = List[CandidatePoint]


def parse_candidate_groups(cell: Optional[str]) -> List[CandidateGroup]:
    """Parse the CMM 'candidates' cell into grouped coordinate/probability triples."""
    if not cell:
        return []
    groups: List[CandidateGroup] = []
    for group_text in cell.split("|"):
        group_text = group_text.strip()
        if not group_text:
            groups.append([])
            continue
        # Remove a single wrapping set of parentheses if present.
        if group_text.startswith("(") and group_text.endswith(")"):
            group_text = group_text[1:-1]
        entries: CandidateGroup = []
        for entry in group_text.split("),("):
            token = entry.strip().strip("()")
            if not token:
                continue
            parts = [part.strip() for part in token.split(",")]
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            prob: Optional[float] = None
            if len(parts) >= 3:
                try:
                    prob = float(parts[2])
                except ValueError:
                    prob = None
            entries.append(((x, y), prob))
        groups.append(entries)
    return groups


def convert_candidate_groups(
    groups: List[CandidateGroup],
    coord_transformer: Optional[CoordinateTransformer],
) -> List[CandidateGroup]:
    """Convert candidate coordinates to lon/lat if a transformer is available."""
    if not groups:
        return []
    converted: List[CandidateGroup] = []
    for group in groups:
        if not group:
            converted.append([])
            continue
        coords_only = [pt for pt, _ in group]
        coords = (
            coord_transformer.convert_coords(coords_only)
            if coord_transformer
            else coords_only
        )
        converted_group: CandidateGroup = []
        for (orig_pt, prob), coord in zip(group, coords):
            converted_group.append((coord, prob))
        converted.append(converted_group)
    return converted


def initial_bounds() -> Bounds:
    return (math.inf, math.inf, -math.inf, -math.inf)


def bounds_valid(bounds: Bounds) -> bool:
    min_lon, min_lat, max_lon, max_lat = bounds
    return (
      math.isfinite(min_lon)
      and math.isfinite(min_lat)
      and math.isfinite(max_lon)
      and math.isfinite(max_lat)
      and (max_lon > min_lon)
      and (max_lat > min_lat)
    )


def update_bounds(bounds: Bounds, coords: Iterable[Coordinate]) -> Bounds:
    min_lon, min_lat, max_lon, max_lat = bounds
    for lon, lat in coords:
        if lon < min_lon:
            min_lon = lon
        if lat < min_lat:
            min_lat = lat
        if lon > max_lon:
            max_lon = lon
        if lat > max_lat:
            max_lat = lat
    return min_lon, min_lat, max_lon, max_lat


def load_edge_geometries(
    shapefile: Path,
    id_field: Optional[str],
    coord_transformer: Optional[CoordinateTransformer],
) -> Dict[str, List[Coordinate]]:
    """Load edges from a shapefile keyed by an ID field."""
    if not shapefile:
        return {}
    if not shapefile.exists():
        warnings.warn(f"Edge shapefile not found: {shapefile}")
        return {}
    if fiona is None:
        warnings.warn("Fiona is required to read the edge shapefile but is not installed.")
        return {}

    mapping: Dict[str, List[Coordinate]] = {}
    with fiona.open(shapefile, "r") as src:
        schema_props = src.schema.get("properties", {})
        candidate_fields: List[str]
        if id_field:
            if id_field not in schema_props:
                warnings.warn(
                    f"Specified edge id field '{id_field}' not present in shapefile. "
                    "Attempting to auto-detect instead."
                )
                candidate_fields = []
            else:
                candidate_fields = [id_field]
        else:
            candidate_fields = []

        if not candidate_fields:
            for field in ("fid", "id", "edge_id", "edgeid", "ID"):
                if field in schema_props:
                    candidate_fields.append(field)
                    break
            if not candidate_fields:
                candidate_fields.append(next(iter(schema_props.keys()), "id"))

        chosen_field = candidate_fields[0]

        for feature in src:
            geom = feature.get("geometry")
            if not geom:
                continue
            geom_type = geom.get("type")
            coords: List[Coordinate] = []
            if geom_type == "LineString":
                coords = [(float(x), float(y)) for x, y in geom.get("coordinates", [])]
            elif geom_type == "MultiLineString":
                for part in geom.get("coordinates", []):
                    if part:
                        coords = [(float(x), float(y)) for x, y in part]
                        break
            if len(coords) < 2:
                continue

            raw_id = feature["properties"].get(chosen_field)
            if raw_id is None:
                continue
            if coord_transformer:
                coords = coord_transformer.convert_coords(coords)
            mapping[str(raw_id)] = coords
    return mapping


def _covariance_ellipse_geod(
    center: Coordinate,
    covariance_m: np.ndarray,
    scale: float,
    segments: int,
) -> List[Coordinate]:
    if covariance_m.shape != (2, 2):
        return []
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_m)
    except np.linalg.LinAlgError:
        return []
    eigenvalues = np.maximum(eigenvalues, 0.0)
    radii = np.sqrt(eigenvalues) * float(scale)
    transform = eigenvectors @ np.diag(radii)

    lon0, lat0 = center
    coords: List[Coordinate] = []
    for idx in range(segments):
        angle = 2.0 * math.pi * idx / segments
        unit = np.array([math.cos(angle), math.sin(angle)])
        offset = transform @ unit
        dx = float(offset[0])  # metres east
        dy = float(offset[1])  # metres north
        if dx == 0 and dy == 0:
            coords.append(center)
            continue
        distance = math.hypot(dx, dy)
        if distance == 0:
            coords.append(center)
            continue
        azimuth = math.degrees(math.atan2(dx, dy))
        if GEOD is not None:
            lon, lat, _ = GEOD.fwd(lon0, lat0, azimuth, distance)
        else:
            deg_lat = dy / 111_320.0
            cos_lat = max(math.cos(math.radians(lat0)), 1e-6)
            deg_lon = dx / (cos_lat * 111_320.0)
            lon = lon0 + deg_lon
            lat = lat0 + deg_lat
        coords.append((lon, lat))
    if coords:
        coords.append(coords[0])
    return coords


def covariance_ellipse_coords(
    center_wgs: Coordinate,
    center_proj: Optional[Coordinate],
    covariance_m: np.ndarray,
    coord_transformer: Optional[CoordinateTransformer],
    scale: float = ELLIPSE_SCALE,
    segments: int = ELLIPSE_SEGMENTS,
) -> List[Coordinate]:
    if covariance_m.shape != (2, 2):
        return []
    use_planar = (
        coord_transformer is not None
        and coord_transformer.is_active()
        and center_proj is not None
    )
    if use_planar:
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_m)
        except np.linalg.LinAlgError:
            return []
        eigenvalues = np.maximum(eigenvalues, 0.0)
        radii = np.sqrt(eigenvalues) * float(scale)
        transform = eigenvectors @ np.diag(radii)
        planar_points: List[Coordinate] = []
        for idx in range(segments):
            angle = 2.0 * math.pi * idx / segments
            unit = np.array([math.cos(angle), math.sin(angle)])
            offset = transform @ unit
            planar_points.append(
                (
                    center_proj[0] + float(offset[0]),
                    center_proj[1] + float(offset[1]),
                )
            )
        if planar_points:
            planar_points.append(planar_points[0])
        coords = coord_transformer.convert_coords(planar_points)
        return coords
    else:
        return _covariance_ellipse_geod(center_wgs, covariance_m, scale, segments)


def protection_level_circle(
    center_wgs: Coordinate,
    center_proj: Optional[Coordinate],
    radius_m: float,
    coord_transformer: Optional[CoordinateTransformer],
    segments: int = CIRCLE_SEGMENTS,
) -> List[Coordinate]:
    """Approximate a PL circle in metres and convert to lon/lat."""
    if radius_m is None or radius_m <= 0:
        return []
    use_planar = (
        coord_transformer is not None
        and coord_transformer.is_active()
        and center_proj is not None
    )
    if use_planar:
        planar_points: List[Coordinate] = []
        for idx in range(segments):
            angle = 2 * math.pi * idx / segments
            dx = math.cos(angle) * radius_m
            dy = math.sin(angle) * radius_m
            planar_points.append((center_proj[0] + dx, center_proj[1] + dy))
        if planar_points:
            planar_points.append(planar_points[0])
        return coord_transformer.convert_coords(planar_points)
    lon0, lat0 = center_wgs
    coords: List[Coordinate] = []
    if GEOD is None:
        deg_lat = radius_m / 111_320.0
        deg_lon = radius_m / max(math.cos(math.radians(lat0)) * 111_320.0, 1e-6)
        for idx in range(segments):
            angle = 2 * math.pi * idx / segments
            dx = math.cos(angle) * deg_lon
            dy = math.sin(angle) * deg_lat
            coords.append((lon0 + dx, lat0 + dy))
    else:
        for idx in range(segments):
            azimuth = 360.0 * idx / segments
            lon, lat, _ = GEOD.fwd(lon0, lat0, azimuth, radius_m)
            coords.append((lon, lat))
    if coords:
        coords.append(coords[0])
    return coords


def build_ground_truth_edge_features(
    edge_sequences: Dict[str, List[str]],
    edge_geometries: Dict[str, List[Coordinate]],
    bounds: Bounds,
) -> Tuple[List[Dict], Bounds]:
    features: List[Dict] = []
    updated_bounds = bounds
    for traj_id, ids in edge_sequences.items():
        for seq, edge_id in enumerate(ids):
            coords = edge_geometries.get(edge_id)
            if not coords:
                continue
            updated_bounds = update_bounds(updated_bounds, coords)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": traj_id,
                        "edge_id": edge_id,
                        "seq": seq,
                        "kind": "ground_edge",
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                }
            )
    return features, updated_bounds


def collect_observation_features_from_cmm(
    path: Path,
    selected_ids: Sequence[str],
    bounds: Bounds,
    highlight_map: Optional[Dict[str, Optional[Set[int]]]] = None,
    coord_transformer: Optional[CoordinateTransformer] = None,
) -> Tuple[List[Dict], List[Dict], List[Dict], Bounds]:
    point_features: List[Dict] = []
    ellipse_features: List[Dict] = []
    pl_circle_features: List[Dict] = []
    updated_bounds = bounds
    id_set = {str(_id) for _id in selected_ids}
    highlight_map = highlight_map or {}
    highlight_active = bool(highlight_map)

    if not path or not path.exists():
        return point_features, ellipse_features, pl_circle_features, updated_bounds

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id")
            if traj_id is None:
                continue
            traj_id_str = traj_id.strip()
            if traj_id_str not in id_set:
                continue

            coords_proj = parse_linestring(row.get("geom", ""))
            if not coords_proj:
                continue
            coords = (
                coord_transformer.convert_coords(coords_proj)
                if coord_transformer
                else coords_proj
            )
            if not coords:
                continue

            try:
                timestamps = json.loads(row.get("timestamps", "[]"))
            except json.JSONDecodeError:
                timestamps = []
            try:
                covariance_list = json.loads(row.get("covariances", "[]"))
            except json.JSONDecodeError:
                covariance_list = []
            try:
                pl_list = json.loads(row.get("protection_levels", "[]"))
            except json.JSONDecodeError:
                pl_list = []

            count = min(len(coords), len(covariance_list))
            if count == 0:
                continue

            highlight_entry = highlight_map.get(traj_id_str) if highlight_active else None
            has_highlight_for_id = highlight_active and (traj_id_str in highlight_map)

            for seq in range(count):
                lon, lat = coords[seq]
                center_proj = coords_proj[seq] if seq < len(coords_proj) else None
                cov_entry = covariance_list[seq] if seq < len(covariance_list) else None
                pl = pl_list[seq] if seq < len(pl_list) else None
                timestamp_val = None
                if seq < len(timestamps):
                    try:
                        timestamp_val = float(timestamps[seq])
                    except (TypeError, ValueError):
                        timestamp_val = None

                if cov_entry is None or len(cov_entry) < 4:
                    continue
                sde = to_float(cov_entry[0])
                sdn = to_float(cov_entry[1])
                sdne = to_float(cov_entry[3])
                if sdn is None or sde is None or sdne is None:
                    continue

                center = (lon, lat)
                updated_bounds = update_bounds(updated_bounds, [center])

                point_features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "id": traj_id_str,
                            "kind": "observation_point",
                            "seq": seq,
                            "lon": lon,
                            "lat": lat,
                            "timestamp": timestamp_val,
                            "timestamp_raw": timestamps[seq] if seq < len(timestamps) else None,
                            "pl": to_float(pl),
                            "sde": sde,
                            "sdn": sdn,
                            "sdne": sdne,
                        },
                        "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    }
                )

                cov_matrix_m = np.array(
                    [[sde * sde, sdne], [sdne, sdn * sdn]], dtype=float
                )
                active_transformer = (
                    coord_transformer if coord_transformer and coord_transformer.is_active() else None
                )

                show_envelope = True
                if highlight_active:
                    seq_set = highlight_entry
                    if not has_highlight_for_id:
                        show_envelope = False
                    elif seq_set is not None:
                        show_envelope = seq in seq_set
                if show_envelope:
                    ellipse_coords = covariance_ellipse_coords(
                        center,
                        center_proj,
                        cov_matrix_m,
                        active_transformer,
                    )
                    if ellipse_coords:
                        updated_bounds = update_bounds(updated_bounds, ellipse_coords)
                        ellipse_features.append(
                            {
                                "type": "Feature",
                                "properties": {
                                    "id": traj_id_str,
                                    "kind": "observation_cov",
                                    "seq": seq,
                                    "timestamp": timestamp_val,
                                    "timestamp_raw": timestamps[seq] if seq < len(timestamps) else None,
                                    "pl": to_float(pl),
                                    "sde": sde,
                                    "sdn": sdn,
                                    "sdne": sdne,
                                },
                                "geometry": {"type": "Polygon", "coordinates": [ellipse_coords]},
                                }
                            )

                    pl_val = to_float(pl)
                    if pl_val is not None and pl_val > 0:
                        circle_coords = protection_level_circle(
                            center,
                            center_proj,
                            pl_val,
                            active_transformer,
                        )
                        if circle_coords:
                            

                            
                            updated_bounds = update_bounds(updated_bounds, circle_coords)
                            pl_circle_features.append(
                                {
                                    "type": "Feature",
                                    "properties": {
                                        "id": traj_id_str,
                                        "kind": "pl_circle",
                                        "seq": seq,
                                        "timestamp": timestamp_val,
                                        "timestamp_raw": timestamps[seq] if seq < len(timestamps) else None,
                                        "pl": pl_val,
                                    },
                                    "geometry": {"type": "Polygon", "coordinates": [circle_coords]},
                                }
                            )

    return point_features, ellipse_features, pl_circle_features, updated_bounds


def collect_ids(args_ids: Optional[Sequence[str]], fallback_path: Path) -> List[str]:
    collected: List[str] = []
    if args_ids:
        for group in args_ids:
            for part in group.split(","):
                part = part.strip()
                if part:
                    collected.append(part)
    if collected:
        return collected

    # Fallback: use first ID from the ground truth CSV.
    with fallback_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            row_id = row.get("id")
            if row_id:
                return [row_id.strip()]
    raise SystemExit("Unable to determine default IDs from ground truth file.")


def parse_observation_highlights(args_highlights: Optional[Sequence[str]]) -> Dict[str, Optional[Set[int]]]:
    highlights: Dict[str, Optional[Set[int]]] = {}
    if not args_highlights:
        return highlights

    for group in args_highlights:
        for token in group.split(","):
            item = token.strip()
            if not item:
                continue
            if item.lower() == "all":
                highlights["__all__"] = None
                continue
            if ":" in item:
                traj_id, seq_text = item.split(":", 1)
                traj_id = traj_id.strip()
                seq_text = seq_text.strip()
                if not traj_id or not seq_text:
                    continue
                try:
                    seq_value = int(seq_text)
                except ValueError:
                    continue
                seq_set = highlights.setdefault(traj_id, set())
                if seq_set is not None:
                    seq_set.add(seq_value)
            else:
                traj_id = item
                highlights[traj_id] = None

    # If special "__all__" key present, treat as highlight all observations.
    if "__all__" in highlights:
        return {"__all__": None}

    return highlights


def collect_ground_truth_features(
    path: Path,
    selected_ids: Sequence[str],
    bounds: Bounds,
    coord_transformer: Optional[CoordinateTransformer],
) -> Tuple[List[Dict], Dict[str, List[str]], Bounds]:
    point_features: List[Dict] = []
    edge_sequences: Dict[str, List[str]] = {}
    updated_bounds = bounds
    id_set = {str(_id) for _id in selected_ids}

    if not path.exists():
        raise SystemExit(f"Ground truth file not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id")
            if traj_id is None:
                continue
            traj_id_str = traj_id.strip()
            if traj_id_str not in id_set:
                continue
            coords = parse_linestring(row.get("geom", ""))
            if coord_transformer:
                coords = coord_transformer.convert_coords(coords)
            if len(coords) < 2:
                continue
            updated_bounds = update_bounds(updated_bounds, coords)
            timestamp_values = parse_value_list(row.get("timestamp", ""))
            raw_edge_ids = parse_value_list(row.get("edge_ids", ""))
            edge_sequences[traj_id_str] = raw_edge_ids
            for idx, (lon, lat) in enumerate(coords):
                timestamp_raw = timestamp_values[idx] if idx < len(timestamp_values) else None
                point_features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "id": traj_id.strip(),
                            "source": "ground_truth",
                            "kind": "ground_truth_point",
                            "seq": idx,
                            "lon": lon,
                            "lat": lat,
                            "timestamp": to_float(timestamp_raw),
                            "timestamp_raw": timestamp_raw,
                            "ep": None,
                            "ep_raw": None,
                            "tp": None,
                            "tp_raw": None,
                            "trustworthiness": None,
                            "trustworthiness_raw": None,
                            "error": None,
                            "error_raw": None,
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat],
                        },
                    }
                )
    return point_features, edge_sequences, updated_bounds


def collect_point_features(
    path: Path,
    selected_ids: Sequence[str],
    dataset_name: str,
    bounds: Bounds,
    coord_transformer: Optional[CoordinateTransformer],
    include_candidates: bool = False,
) -> Tuple[List[Dict], List[Dict], Bounds]:
    features: List[Dict] = []
    candidate_features: List[Dict] = []
    updated_bounds = bounds
    id_set = {str(_id) for _id in selected_ids}

    if not path.exists():
        return features, candidate_features, updated_bounds

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id")
            if traj_id is None or traj_id.strip() not in id_set:
                continue

            coords = parse_linestring(row.get("pgeom", ""))
            if coord_transformer:
                coords = coord_transformer.convert_coords(coords)
            if not coords:
                continue

            candidate_groups: List[CandidateGroup] = []
            if include_candidates:
                raw_groups = parse_candidate_groups(row.get("candidates", ""))
                candidate_groups = convert_candidate_groups(raw_groups, coord_transformer)

            trust_raw = row.get("trustworthiness")
            if not trust_raw:
                trust_raw = row.get("trustworthiness_prob", "")

            values_map: Dict[str, List[str]] = {
                "timestamp": parse_value_list(row.get("timestamp", "")),
                "ep": parse_value_list(row.get("ep", "")),
                "tp": parse_value_list(row.get("tp", "")),
                "trustworthiness": parse_value_list(trust_raw),
                "error": parse_value_list(row.get("error", "")),
            }

            for idx, (lon, lat) in enumerate(coords):
                updated_bounds = update_bounds(updated_bounds, [(lon, lat)])

                def get_value(key: str) -> Optional[str]:
                    series = values_map.get(key, [])
                    return series[idx] if idx < len(series) else None

                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": traj_id.strip(),
                        "source": dataset_name,
                        "kind": f"{dataset_name}_point",
                        "seq": idx,
                        "lon": lon,
                        "lat": lat,
                        "timestamp": to_float(get_value("timestamp")),
                        "timestamp_raw": get_value("timestamp"),
                        "ep": to_float(get_value("ep")),
                        "ep_raw": get_value("ep"),
                        "tp": to_float(get_value("tp")),
                        "tp_raw": get_value("tp"),
                        "trustworthiness": to_float(get_value("trustworthiness")),
                        "trustworthiness_raw": get_value("trustworthiness"),
                        "error": to_float(get_value("error")),
                        "error_raw": get_value("error"),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                }
                features.append(feature)
                if include_candidates and idx < len(candidate_groups):
                    group = candidate_groups[idx]
                    total = len(group)
                    for rank, (candidate_coord, probability) in enumerate(group):
                        if not candidate_coord:
                            continue
                        updated_bounds = update_bounds(updated_bounds, [candidate_coord])
                        candidate_features.append(
                            {
                                "type": "Feature",
                                "properties": {
                                    "id": traj_id.strip(),
                                    "source": "cmm_candidates",
                                    "kind": "cmm_candidate",
                                    "seq": idx,
                                    "rank": f"{rank + 1}/{total}" if total else None,
                                    "probability": probability,
                                },
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [candidate_coord[0], candidate_coord[1]],
                                },
                            }
                        )
    return features, candidate_features, updated_bounds


def requires_token(token: Optional[str]) -> str:
    if token:
        return token
    env_token = os.environ.get("MAPBOX_ACCESS_TOKEN")
    if env_token:
        return env_token
    return MAPBOX_DEFAULT_TOKEN


def bounds_to_json(bounds: Bounds) -> str:
    if bounds_valid(bounds):
        min_lon, min_lat, max_lon, max_lat = bounds
        return json.dumps([[min_lon, min_lat], [max_lon, max_lat]])
    return "null"


def render_html(
    token: str,
    bounds: Bounds,
    selected_ids: Sequence[str],
    ground_truth_points_geojson: Dict,
    ground_truth_edge_geojson: Dict,
    observation_points_geojson: Dict,
    observation_ellipses_geojson: Dict,
    pl_circles_geojson: Dict,
    cmm_geojson: Dict,
    cmm_candidate_geojson: Dict,
    fmm_geojson: Dict,
    edge_sequences: Dict[str, List[str]],
) -> str:
    bounds_js = bounds_to_json(bounds)
    ids_label = ", ".join(str(_id) for _id in selected_ids)
    edge_lines: List[str] = []
    for _id in selected_ids:
        edge_list = edge_sequences.get(str(_id), [])
        if edge_list:
            edge_lines.append(f"{_id}: " + ", ".join(edge_list))
        else:
            edge_lines.append(f"{_id}: (none)")
    edge_info_html = "<br/>".join(edge_lines) if edge_lines else "无可用边ID"

    template = Template("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>CMM vs FMM Mapbox Viewer</title>
<meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
<script src="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.css" rel="stylesheet" />
<style>
  body { margin: 0; padding: 0; }
  #map { position: absolute; top: 0; bottom: 0; width: 100%; }
  #info {
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(0, 0, 0, 0.55);
    color: #fff;
    padding: 10px 14px;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    border-radius: 6px;
    max-width: 320px;
    line-height: 1.5;
  }
  #legend {
    margin-top: 8px;
  }
  #legend span {
    display: inline-flex;
    align-items: center;
    margin-right: 12px;
  }
  #legend i {
    width: 12px;
    height: 12px;
    display: inline-block;
    margin-right: 6px;
    border-radius: 50%;
  }
</style>
</head>
<body>
<div id="map"></div>
<div id="info">
  <strong>Trajectory IDs:</strong> $IDS_LABEL<br/>
  Hover layers to inspect details.
  <div id="search-box" style="margin-top:10px;">
    <label style="display:flex;align-items:center;gap:6px;">
      <span>Search seq:</span>
      <input type="number" id="seq-search-input" placeholder="e.g. 15" style="flex:1;padding:2px 6px;" />
      <button id="seq-search-btn" style="padding:2px 8px;">Go</button>
    </label>
    <div id="seq-search-results" style="margin-top:6px;font-size:12px;color:#ddd;"></div>
  </div>
  <div id="legend">
    <div style="display:flex;align-items:center;margin-bottom:6px;">
      <i style="background:#6a3d9a;border-radius:0;width:18px;height:3px;margin-right:6px;"></i>
      Ground truth edge (always on)
    </div>
    <label style="display:block;margin-bottom:6px;">
      <input type="checkbox" id="toggle-ground-truth-points" style="margin-right:6px;"/>
      Ground truth points
    </label>
    <label style="display:block;margin-bottom:6px;">
      <input type="checkbox" id="toggle-observation" style="margin-right:6px;"/>
      Observation points (hover to reveal covariance/PL)
    </label>
    <label style="display:block;margin-bottom:6px;">
      <input type="checkbox" id="toggle-cmm" style="margin-right:6px;"/>
      CMM points
    </label>
    <label style="display:block;">
      <input type="checkbox" id="toggle-fmm" style="margin-right:6px;"/>
      FMM points
    </label>
  </div>
  <div style="margin-top:6px;font-size:11px;color:#ddd;">
    Click a CMM point to reveal its candidates.
  </div>
  <div id="edge-info" style="margin-top:8px;">
    <strong>Edge IDs:</strong><br/>
    <span id="edge-list">$EDGE_INFO_HTML</span>
  </div>
</div>
<script>
  mapboxgl.accessToken = $TOKEN;
  const selectedIds = $SELECTED_IDS;
  const bounds = $BOUNDS;
  const groundTruthPointsGeojson = $GROUND_POINTS_GEOJSON;
  const groundTruthEdgeGeojson = $GROUND_EDGES_GEOJSON;
  const observationPointGeojson = $OBS_POINT_GEOJSON;
  const observationEllipseGeojson = $OBS_ELLIPSE_GEOJSON;
  const plCircleGeojson = $PL_CIRCLE_GEOJSON;
  const cmmGeojson = $CMM_GEOJSON;
  const cmmCandidatesGeojson = $CMM_CANDIDATES_GEOJSON;
  const fmmGeojson = $FMM_GEOJSON;

  const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/navigation-day-v1',
    center: [0, 0],
    zoom: 2,
    pitch: 60,
    bearing: -20
  });

  map.on('load', () => {
    map.addControl(new mapboxgl.NavigationControl(), 'top-right');

    if (bounds) {
      map.fitBounds(bounds, { padding: 70, maxZoom: 16 });
    }

    map.addSource('mapbox-dem', {
      type: 'raster-dem',
      url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
      tileSize: 512,
      maxzoom: 14
    });
    map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.2 });

    const layers = map.getStyle().layers || [];
    let labelLayerId = null;
    for (const layer of layers) {
      if (layer.type === 'symbol' && layer.layout && layer.layout['text-field']) {
        labelLayerId = layer.id;
        break;
      }
    }

    const buildingLayer = {
      id: 'custom-3d-buildings',
      source: 'composite',
      'source-layer': 'building',
      filter: ['==', ['get', 'extrude'], 'true'],
      type: 'fill-extrusion',
      minzoom: 15,
      paint: {
        'fill-extrusion-color': '#aaa',
        'fill-extrusion-height': ['get', 'height'],
        'fill-extrusion-base': ['get', 'min_height'],
        'fill-extrusion-opacity': 0.6
      }
    };
    if (labelLayerId) {
      map.addLayer(buildingLayer, labelLayerId);
    } else {
      map.addLayer(buildingLayer);
    }

    const hasGroundTruthPoints = groundTruthPointsGeojson.features.length > 0;
    const hasGroundTruthEdges = groundTruthEdgeGeojson.features.length > 0;
    const hasObservationPoints = observationPointGeojson.features.length > 0;
    const hasObservationEllipses = observationEllipseGeojson.features.length > 0;
    const hasPlCircles = plCircleGeojson.features.length > 0;
    const hasCmm = cmmGeojson.features.length > 0;
    const hasCmmCandidates = cmmCandidatesGeojson.features.length > 0;
    const hasFmm = fmmGeojson.features.length > 0;

    if (hasGroundTruthEdges) {
      map.addSource('ground-truth-edges', {
        type: 'geojson',
        data: groundTruthEdgeGeojson
      });
      map.addLayer({
        id: 'ground-truth-edge-layer',
        type: 'line',
        source: 'ground-truth-edges',
        paint: {
          'line-color': '#6a3d9a',
          'line-width': [
            'interpolate', ['linear'], ['zoom'],
            10, 2,
            16, 6
          ],
          'line-dasharray': [2, 2],
          'line-opacity': 0.85
        }
      });
    }

    if (hasGroundTruthPoints) {
      map.addSource('ground-truth-points', {
        type: 'geojson',
        data: groundTruthPointsGeojson
      });
      map.addLayer({
        id: 'ground-truth-points-layer',
        type: 'circle',
        source: 'ground-truth-points',
        layout: {
          'visibility': 'none'
        },
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            10, 4,
            16, 9
          ],
          'circle-color': '#1f78b4',
          'circle-opacity': 0.9,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1
        }
      });
    }

    if (hasObservationPoints) {
      map.addSource('observation-points', {
        type: 'geojson',
        data: observationPointGeojson
      });
    }

    if (hasObservationEllipses) {
      map.addSource('observation-ellipses', {
        type: 'geojson',
        data: observationEllipseGeojson
      });
    }

    if (hasPlCircles) {
      map.addSource('pl-circles', {
        type: 'geojson',
        data: plCircleGeojson
      });
    }

    if (hasObservationPoints) {
      map.addLayer({
        id: 'observation-point-layer',
        type: 'circle',
        source: 'observation-points',
        layout: {
          'visibility': 'none'
        },
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            10, 3,
            16, 8
          ],
          'circle-color': '#17becf',
          'circle-opacity': 0.9,
          'circle-stroke-color': '#103f5c',
          'circle-stroke-width': 1
        }
      });
    }

    if (hasObservationEllipses) {
      map.addLayer({
        id: 'observation-cov-layer',
        type: 'fill',
        source: 'observation-ellipses',
        layout: {
          'visibility': 'none'
        },
        paint: {
          'fill-color': '#17becf',
          'fill-opacity': 0.25,
          'fill-outline-color': '#0f4c5c'
        }
      });
      map.setFilter('observation-cov-layer', ['==', ['get', 'id'], '__none__']);
    }

    if (hasPlCircles) {
      map.addLayer({
        id: 'pl-circle-layer',
        type: 'fill',
        source: 'pl-circles',
        layout: {
          'visibility': 'none'
        },
        paint: {
          'fill-color': '#9467bd',
          'fill-opacity': 0.2,
          'fill-outline-color': '#4a2352'
        }
      });
      map.setFilter('pl-circle-layer', ['==', ['get', 'id'], '__none__']);
    }

    if (hasCmm) {
      map.addSource('cmm-points', {
        type: 'geojson',
        data: cmmGeojson
      });
      map.addLayer({
        id: 'cmm-points-layer',
        type: 'circle',
        source: 'cmm-points',
        layout: {
          'visibility': 'none'
        },
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            10, 4,
            16, 10
          ],
          'circle-color': '#ff7f0e',
          'circle-opacity': 0.85,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.2
        }
      });
    }

    if (hasCmmCandidates) {
      map.addSource('cmm-candidates', {
        type: 'geojson',
        data: cmmCandidatesGeojson
      });
      map.addLayer({
        id: 'cmm-candidates-layer',
        type: 'circle',
        source: 'cmm-candidates',
        layout: {
          'visibility': 'none'
        },
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            10, 3,
            16, 6
          ],
          'circle-color': '#ffe066',
          'circle-opacity': 0.95,
          'circle-stroke-color': '#d9480f',
          'circle-stroke-width': 1.5
        }
      });
      map.setFilter('cmm-candidates-layer', ['==', ['get', 'id'], '__none__']);
    }

    if (hasFmm) {
      map.addSource('fmm-points', {
        type: 'geojson',
        data: fmmGeojson
      });
      map.addLayer({
        id: 'fmm-points-layer',
        type: 'circle',
        source: 'fmm-points',
        layout: {
          'visibility': 'none'
        },
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            10, 4,
            16, 10
          ],
          'circle-color': '#2ca02c',
          'circle-opacity': 0.85,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1.2
        }
      });
    }

    map.addSource('search-highlight', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });
    map.addLayer({
      id: 'search-highlight-layer',
      type: 'circle',
      source: 'search-highlight',
      layout: {
        'visibility': 'none'
      },
      paint: {
        'circle-radius': [
          'interpolate', ['linear'], ['zoom'],
          10, 5,
          16, 12
        ],
        'circle-color': '#ffff00',
        'circle-opacity': 0.9,
        'circle-stroke-color': '#000000',
        'circle-stroke-width': 2
      }
    });

    const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false });

    function formatNumber(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return 'N/A';
      }
      return Number(value).toFixed(digits);
    }

    function formatTimestamp(rawValue) {
      if (!rawValue) {
        return 'N/A';
      }
      const numeric = Number(rawValue);
      if (!Number.isFinite(numeric)) {
        return rawValue;
      }
      return numeric.toFixed(3);
    }

    function buildPopupHTML(props) {
      const kind = props.kind || '';
      if (kind === 'cmm_point' || kind === 'fmm_point') {
        const label = kind === 'cmm_point' ? 'CMM' : 'FMM';
        return '<strong>' + label + ' ID: ' + props.id + '</strong><br/>' +
          'Seq: ' + props.seq + '<br/>' +
          'Lon: ' + formatNumber(props.lon, 6) + '<br/>' +
          'Lat: ' + formatNumber(props.lat, 6) + '<br/>' +
          'Timestamp: ' + formatTimestamp(props.timestamp_raw) + '<br/>' +
          'Error: ' + formatNumber(props.error, 3) + '<br/>' +
          'EP: ' + formatNumber(props.ep, 6) + '<br/>' +
          'TP: ' + formatNumber(props.tp, 6) + '<br/>' +
          'TRUSTWORTHINESS: ' + formatNumber(props.trustworthiness, 6);
      }
      if (kind === 'cmm_candidate') {
        return '<strong>CMM Candidate</strong><br/>' +
          'ID: ' + props.id + '<br/>' +
          'Seq: ' + props.seq + '<br/>' +
          'Rank: ' + (props.rank ?? 'N/A') + '<br/>' +
          'Probability: ' + formatNumber(props.probability, 6);
      }
      if (kind === 'ground_truth_point') {
        return '<strong>Ground Truth ID: ' + props.id + '</strong><br/>' +
          'Seq: ' + props.seq + '<br/>' +
          'Lon: ' + formatNumber(props.lon, 6) + '<br/>' +
          'Lat: ' + formatNumber(props.lat, 6) + '<br/>' +
          'Timestamp: ' + formatTimestamp(props.timestamp_raw);
      }
      if (kind === 'ground_edge') {
        return '<strong>Ground Truth Edge</strong><br/>' +
          'Trajectory: ' + props.id + '<br/>' +
          'Edge ID: ' + props.edge_id + '<br/>' +
          'Order: ' + props.seq;
      }
      if (kind === 'observation_point' || kind === 'observation_cov') {
        const label = kind === 'observation_point' ? 'Observation' : 'Observation Ellipse';
        const coordHtml = kind === 'observation_point'
          ? 'Lon: ' + formatNumber(props.lon, 6) + '<br/>' +
            'Lat: ' + formatNumber(props.lat, 6) + '<br/>'
          : '';
        return '<strong>' + label + ' ID: ' + props.id + '</strong><br/>' +
          'Seq: ' + props.seq + '<br/>' +
          'Timestamp: ' + formatTimestamp(props.timestamp_raw) + '<br/>' +
          coordHtml +
          'PL: ' + formatNumber(props.pl, 3) + '<br/>' +
          'sdN: ' + formatNumber(props.sdn, 6) + '<br/>' +
          'sdE: ' + formatNumber(props.sde, 6) + '<br/>' +
          'sdNE: ' + formatNumber(props.sdne, 6);
      }
      if (kind === 'pl_circle') {
        return '<strong>Protection Level</strong><br/>' +
          'ID: ' + props.id + '<br/>' +
          'Seq: ' + props.seq + '<br/>' +
          'Timestamp: ' + formatTimestamp(props.timestamp_raw) + '<br/>' +
          'PL: ' + formatNumber(props.pl, 3);
      }
      return '<strong>Feature</strong><br/>ID: ' + props.id;
    }

    function attachHover(layerId) {
      if (!map.getLayer(layerId)) {
        return;
      }
      map.on('mouseenter', layerId, (event) => {
        map.getCanvas().style.cursor = 'pointer';
        const features = event.features || [];
        if (!features.length) {
          return;
        }
        const feature = features[0];
        popup
          .setLngLat(feature.geometry.coordinates.slice())
          .setHTML(buildPopupHTML(feature.properties))
          .addTo(map);
      });

      map.on('mouseleave', layerId, () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });
    }

    attachHover('cmm-points-layer');
    attachHover('cmm-candidates-layer');
    attachHover('fmm-points-layer');
    attachHover('ground-truth-points-layer');
    attachHover('ground-truth-edge-layer');

    const toggleCmm = document.getElementById('toggle-cmm');
    const toggleFmm = document.getElementById('toggle-fmm');
    const toggleObservation = document.getElementById('toggle-observation');
    const toggleGroundTruthPoints = document.getElementById('toggle-ground-truth-points');
    const seqSearchInput = document.getElementById('seq-search-input');
    const seqSearchButton = document.getElementById('seq-search-btn');
    const seqSearchResults = document.getElementById('seq-search-results');
    const searchHighlightSource = map.getSource('search-highlight');
    const emptySearchGeojson = { type: 'FeatureCollection', features: [] };

    function setLayerVisibility(layerId, visible) {
      if (!map.getLayer(layerId)) return;
      map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
    }

    let highlightedKey = null;
    let activeCandidateKey = null;

    function setSearchResultsHtml(html) {
      if (seqSearchResults) {
        seqSearchResults.innerHTML = html;
      }
    }

    function focusOnFeatures(features) {
      if (!features.length) {
        return;
      }
      const coords = features
        .map(f => f.geometry && f.geometry.coordinates)
        .filter(Boolean);
      if (!coords.length) {
        return;
      }
      if (coords.length === 1) {
        map.easeTo({ center: coords[0], zoom: Math.max(map.getZoom(), 15) });
      } else {
        const bounds = coords.reduce(
          (acc, coord) => acc.extend(coord),
          new mapboxgl.LngLatBounds(coords[0], coords[0])
        );
        map.fitBounds(bounds, { padding: 80, maxZoom: 17 });
      }
    }

    function updateHighlightLayer(features) {
      if (!searchHighlightSource) return;
      if (!features.length) {
        searchHighlightSource.setData(emptySearchGeojson);
        setLayerVisibility('search-highlight-layer', false);
        return;
      }
      searchHighlightSource.setData({
        type: 'FeatureCollection',
        features: features.map(f => ({
          type: 'Feature',
          properties: {
            dataset: (f.properties && f.properties.source) || f.properties.kind || 'feature',
            seq: f.properties ? f.properties.seq : undefined
          },
          geometry: f.geometry
        }))
      });
      setLayerVisibility('search-highlight-layer', true);
    }

    function ensureLayerVisibility(layerId, toggleEl) {
      if (toggleEl && !toggleEl.checked) {
        toggleEl.checked = true;
      }
      setLayerVisibility(layerId, true);
    }

    function describeFeature(label, feature) {
      const coords = feature.geometry && feature.geometry.coordinates
        ? feature.geometry.coordinates
        : [NaN, NaN];
      const seq = feature.properties ? feature.properties.seq : 'N/A';
      return '<div>' + label + ' seq ' + seq + ' @ (' +
        formatNumber(coords[0], 6) + ', ' + formatNumber(coords[1], 6) + ')</div>';
    }

    function handleSeqSearch() {
      if (!seqSearchInput) return;
      const value = seqSearchInput.value.trim();
      if (!value) {
        setSearchResultsHtml('<em>Enter a seq index.</em>');
        updateHighlightLayer([]);
        return;
      }
      const seqValue = Number(value);
      if (!Number.isFinite(seqValue) || !Number.isInteger(seqValue)) {
        setSearchResultsHtml('<em>Please enter an integer seq.</em>');
        updateHighlightLayer([]);
        return;
      }
      const seq = seqValue;
      const matches = [];
      const summary = [];

      if (hasCmm) {
        const cmmMatches = cmmGeojson.features.filter(f => f.properties && f.properties.seq === seq);
        if (cmmMatches.length) {
          matches.push(...cmmMatches);
          summary.push('<strong>CMM</strong> (' + cmmMatches.length + ')');
          ensureLayerVisibility('cmm-points-layer', toggleCmm);
        }
      }

      if (hasFmm) {
        const fmmMatches = fmmGeojson.features.filter(f => f.properties && f.properties.seq === seq);
        if (fmmMatches.length) {
          matches.push(...fmmMatches);
          summary.push('<strong>FMM</strong> (' + fmmMatches.length + ')');
          ensureLayerVisibility('fmm-points-layer', toggleFmm);
        }
      }

      if (hasObservationPoints) {
        const obsMatches = observationPointGeojson.features.filter(f => f.properties && f.properties.seq === seq);
        if (obsMatches.length) {
          matches.push(...obsMatches);
          summary.push('<strong>Observation</strong> (' + obsMatches.length + ')');
          ensureLayerVisibility('observation-point-layer', toggleObservation);
        }
      }

      if (hasGroundTruthPoints) {
        const gtMatches = groundTruthPointsGeojson.features.filter(f => f.properties && f.properties.seq === seq);
        if (gtMatches.length) {
          matches.push(...gtMatches);
          summary.push('<strong>Ground Truth</strong> (' + gtMatches.length + ')');
          ensureLayerVisibility('ground-truth-points-layer', toggleGroundTruthPoints);
        }
      }

      if (!matches.length) {
        setSearchResultsHtml('<em>No features found for seq ' + seq + '.</em>');
        updateHighlightLayer([]);
        return;
      }

      focusOnFeatures(matches);
      updateHighlightLayer(matches);
      const detailHtml = matches
        .map(f => describeFeature((f.properties && f.properties.source) || f.properties.kind || 'feature', f))
        .join('');
      const summaryHtml = summary.length ? summary.join(' | ') + '<br/>' : '';
      setSearchResultsHtml(summaryHtml + detailHtml);
    }

    function hideObservationEnvelope() {
      highlightedKey = null;
      if (hasObservationEllipses) {
        map.setLayoutProperty('observation-cov-layer', 'visibility', 'none');
        map.setFilter('observation-cov-layer', ['==', ['get', 'id'], '__none__']);
      }
      if (hasPlCircles) {
        map.setLayoutProperty('pl-circle-layer', 'visibility', 'none');
        map.setFilter('pl-circle-layer', ['==', ['get', 'id'], '__none__']);
      }
      if (hasCmm) {
        map.setPaintProperty('cmm-points-layer', 'circle-stroke-width', 1.2);
      }
      if (hasFmm) {
        map.setPaintProperty('fmm-points-layer', 'circle-stroke-width', 1.2);
      }
    }

    function showObservationEnvelope(feature) {
      if (!feature) return;
      const props = feature.properties || {};
      const id = props.id;
      const seq = props.seq;
      if (id === undefined || seq === undefined) return;
      const key = id + ':' + seq;
      if (highlightedKey === key) return;
      highlightedKey = key;
      if (hasObservationEllipses) {
        map.setFilter('observation-cov-layer', ['all',
          ['==', ['get', 'id'], id],
          ['==', ['get', 'seq'], seq]
        ]);
        map.setLayoutProperty('observation-cov-layer', 'visibility', 'visible');
      }
      if (hasPlCircles) {
        map.setFilter('pl-circle-layer', ['all',
          ['==', ['get', 'id'], id],
          ['==', ['get', 'seq'], seq]
        ]);
        map.setLayoutProperty('pl-circle-layer', 'visibility', 'visible');
      }
    }

    function hideCmmCandidates() {
      if (!hasCmmCandidates || !map.getLayer('cmm-candidates-layer')) {
        return;
      }
      activeCandidateKey = null;
      map.setLayoutProperty('cmm-candidates-layer', 'visibility', 'none');
      map.setFilter('cmm-candidates-layer', ['==', ['get', 'id'], '__none__']);
    }

    function showCmmCandidates(feature) {
      if (!hasCmmCandidates || !feature || !map.getLayer('cmm-candidates-layer')) {
        return;
      }
      const props = feature.properties || {};
      const id = props.id;
      const seq = props.seq;
      if (id === undefined || seq === undefined) {
        return;
      }
      const key = id + ':' + seq;
      if (activeCandidateKey === key) {
        return;
      }
      activeCandidateKey = key;
      map.setFilter('cmm-candidates-layer', ['all',
        ['==', ['get', 'id'], id],
        ['==', ['get', 'seq'], seq]
      ]);
      map.setLayoutProperty('cmm-candidates-layer', 'visibility', 'visible');
    }

    if (hasCmm && hasCmmCandidates) {
      map.on('click', 'cmm-points-layer', (event) => {
        const features = event.features || [];
        if (!features.length) {
          return;
        }
        showCmmCandidates(features[0]);
      });
    }

    if (toggleCmm && hasCmm) {
      toggleCmm.addEventListener('change', (event) => {
        const visible = event.target.checked;
        setLayerVisibility('cmm-points-layer', visible);
        if (!visible) {
          hideCmmCandidates();
        }
      });
      setLayerVisibility('cmm-points-layer', toggleCmm.checked);
      if (!toggleCmm.checked) {
        hideCmmCandidates();
      }
    }

    if (toggleFmm && hasFmm) {
      toggleFmm.addEventListener('change', (event) => {
        const visible = event.target.checked;
        setLayerVisibility('fmm-points-layer', visible);
      });
      setLayerVisibility('fmm-points-layer', toggleFmm.checked);
    }

    if (toggleObservation) {
      toggleObservation.addEventListener('change', (event) => {
        const visible = event.target.checked;
        setLayerVisibility('observation-point-layer', visible);
        if (!visible) {
          hideObservationEnvelope();
        }
      });
      setLayerVisibility('observation-point-layer', toggleObservation.checked);
      if (!toggleObservation.checked) {
        hideObservationEnvelope();
      }
    }

    if (seqSearchButton && seqSearchInput) {
      seqSearchButton.addEventListener('click', handleSeqSearch);
      seqSearchInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
          handleSeqSearch();
        }
      });
    }

    if (toggleGroundTruthPoints && hasGroundTruthPoints) {
      toggleGroundTruthPoints.addEventListener('change', (event) => {
        const visible = event.target.checked;
        setLayerVisibility('ground-truth-points-layer', visible);
      });
      setLayerVisibility('ground-truth-points-layer', toggleGroundTruthPoints.checked);
    }

    if (hasObservationPoints) {
      map.on('mouseenter', 'observation-point-layer', (event) => {
        map.getCanvas().style.cursor = 'pointer';
        const features = event.features || [];
        if (!features.length) {
          return;
        }
        const feature = features[0];
        popup
          .setLngLat(feature.geometry.coordinates.slice())
          .setHTML(buildPopupHTML(feature.properties))
          .addTo(map);
        showObservationEnvelope(feature);
        if (hasCmm) {
          map.setPaintProperty('cmm-points-layer', 'circle-stroke-width', [
            'case',
            ['all', ['==', ['get', 'id'], feature.properties.id], ['==', ['get', 'seq'], feature.properties.seq]],
            3,
            1.2
          ]);
        }
        if (hasFmm) {
          map.setPaintProperty('fmm-points-layer', 'circle-stroke-width', [
            'case',
            ['all', ['==', ['get', 'id'], feature.properties.id], ['==', ['get', 'seq'], feature.properties.seq]],
            3,
            1.2
          ]);
        }
      });

      map.on('mouseleave', 'observation-point-layer', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
        hideObservationEnvelope();
      });
    }
  });
</script>
</body>
</html>
""")
    html = template.substitute(
        TOKEN=json.dumps(token),
        SELECTED_IDS=json.dumps(list(selected_ids)),
        BOUNDS=bounds_js,
        GROUND_POINTS_GEOJSON=json.dumps(ground_truth_points_geojson),
        GROUND_EDGES_GEOJSON=json.dumps(ground_truth_edge_geojson),
        OBS_POINT_GEOJSON=json.dumps(observation_points_geojson),
        OBS_ELLIPSE_GEOJSON=json.dumps(observation_ellipses_geojson),
        PL_CIRCLE_GEOJSON=json.dumps(pl_circles_geojson),
        CMM_GEOJSON=json.dumps(cmm_geojson),
        CMM_CANDIDATES_GEOJSON=json.dumps(cmm_candidate_geojson),
        FMM_GEOJSON=json.dumps(fmm_geojson),
        IDS_LABEL=ids_label,
        EDGE_INFO_HTML=edge_info_html,
    )
    return html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an interactive Mapbox viewer for CMM/FMM results alongside ground truth."
    )
    parser.add_argument("--cmm", default="output/cmm_result.csv", help="CMM result CSV path.")
    parser.add_argument("--fmm", default="output/fmm_result.csv", help="FMM result CSV path.")
    parser.add_argument(
        "--ground-truth",
        default="input_cmm/ground_truth.csv",
        help="Ground truth CSV path containing LINESTRING geometries.",
    )
    parser.add_argument(
        "--cmm-trajectory",
        default=DEFAULT_CMM_TRAJECTORY_CSV,
        help="CMM trajectory CSV containing observation geometry, covariance, and protection levels.",
    )
    parser.add_argument(
        "--edges-shapefile",
        default=DEFAULT_EDGES_SHP,
        help="Edge shapefile for reconstructing edge geometries.",
    )
    parser.add_argument(
        "--edge-id-field",
        default=None,
        help="Field name in the edge shapefile matching edge_ids (auto-detected if omitted).",
    )
    parser.add_argument(
        "--input-crs",
        default="auto",
        help=(
            "CRS of the input datasets (e.g., 'EPSG:32649'). "
            "Defaults to 'auto', which attempts to read the CRS from the edge shapefile."
        ),
    )
    parser.add_argument(
        "--show-observation",
        action="append",
        help="Observation identifiers to display covariance/PL (format id or id:seq, can be repeated).",
    )
    parser.add_argument(
        "--ids",
        action="append",
        help="Trajectory IDs to visualize. Accepts multiple --ids arguments or comma separated lists.",
    )
    parser.add_argument("--output", default="output/mapbox_cmm_fmm.html", help="Output HTML file.")
    parser.add_argument(
        "--token",
        default=MAPBOX_DEFAULT_TOKEN,
        help="Mapbox access token (defaults to built-in token; can be overridden).",
    )

    args = parser.parse_args()

    ground_path = Path(args.ground_truth)
    edges_path = Path(args.edges_shapefile)
    cmm_path = Path(args.cmm)
    fmm_path = Path(args.fmm)
    cmm_traj_path = Path(args.cmm_trajectory)

    sample_coord = sample_coordinate_from_inputs(
        [
            (ground_path, "geom"),
            (cmm_path, "pgeom"),
            (fmm_path, "pgeom"),
            (cmm_traj_path, "geom"),
        ]
    )

    source_crs = determine_input_crs(args.input_crs, edges_path, sample_coord)
    coord_transformer = CoordinateTransformer(source_crs) if source_crs is not None else None
    if coord_transformer and coord_transformer.transformer is not None and source_crs is not None:
        try:
            crs_label = source_crs.to_string()
        except Exception:
            crs_label = str(source_crs)
        print(f"Transforming projected coordinates ({crs_label}) to WGS84 for Mapbox rendering.")

    selected_ids = collect_ids(args.ids, ground_path)
    if not selected_ids:
        raise SystemExit("At least one trajectory ID must be provided.")

    bounds = initial_bounds()
    highlight_map = parse_observation_highlights(args.show_observation)
    if "__all__" in highlight_map:
        highlight_map = {str(_id): None for _id in selected_ids}

    (
        ground_point_features,
        edge_sequences,
        bounds,
    ) = collect_ground_truth_features(
        ground_path, selected_ids, bounds, coord_transformer
    )
    edge_geometries = load_edge_geometries(edges_path, args.edge_id_field, coord_transformer)
    ground_edge_features, bounds = build_ground_truth_edge_features(
        edge_sequences, edge_geometries, bounds
    )
    observation_point_features, observation_ellipse_features, pl_circle_features, bounds = collect_observation_features_from_cmm(
        cmm_traj_path, selected_ids, bounds, highlight_map, coord_transformer
    )
    cmm_features, cmm_candidate_features, bounds = collect_point_features(
        cmm_path, selected_ids, "cmm", bounds, coord_transformer, include_candidates=True
    )
    fmm_features, _, bounds = collect_point_features(
        fmm_path, selected_ids, "fmm", bounds, coord_transformer
    )

    if not (
        ground_point_features
        or ground_edge_features
        or observation_point_features
        or observation_ellipse_features
        or pl_circle_features
        or cmm_features
        or cmm_candidate_features
        or fmm_features
    ):
        raise SystemExit("No features found for the requested IDs. Nothing to visualize.")

    ground_points_geojson = {"type": "FeatureCollection", "features": ground_point_features}
    ground_edges_geojson = {"type": "FeatureCollection", "features": ground_edge_features}
    observation_points_geojson = {"type": "FeatureCollection", "features": observation_point_features}
    observation_ellipses_geojson = {"type": "FeatureCollection", "features": observation_ellipse_features}
    pl_circles_geojson = {"type": "FeatureCollection", "features": pl_circle_features}
    cmm_geojson = {"type": "FeatureCollection", "features": cmm_features}
    cmm_candidate_geojson = {"type": "FeatureCollection", "features": cmm_candidate_features}
    fmm_geojson = {"type": "FeatureCollection", "features": fmm_features}

    token = requires_token(args.token)
    html = render_html(
        token,
        bounds,
        selected_ids,
        ground_points_geojson,
        ground_edges_geojson,
        observation_points_geojson,
        observation_ellipses_geojson,
        pl_circles_geojson,
        cmm_geojson,
        cmm_candidate_geojson,
        fmm_geojson,
        edge_sequences,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote Mapbox viewer to {output_path}")


if __name__ == "__main__":
    main()
