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
    from pyproj import Geod
except ImportError:  # pragma: no cover - optional dependency
    Geod = None

try:
    import fiona
except ImportError:  # pragma: no cover - optional dependency
    fiona = None

MAPBOX_DEFAULT_TOKEN = (
    "pk.eyJ1IjoiZG9ua2V5LW5pbmciLCJhIjoiY21kenJ5OTY5MGc5azJqb25hdTVtc2tvNiJ9.CgMc9ZNXaZ1HDrC4Zl2aMQ"
)

DEFAULT_CMM_TRAJECTORY_CSV = "dataset_hainan_06/1.1/trajectory.csv"
ELLIPSE_SCALE = 2.0  # number of standard deviations for covariance ellipses
ELLIPSE_SEGMENTS = 64
CIRCLE_SEGMENTS = 64
GEOD = Geod(ellps="WGS84") if Geod is not None else None

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


def covariance_ellipse(
    center: Coordinate,
    covariance_m: np.ndarray,
    scale: float = ELLIPSE_SCALE,
    segments: int = ELLIPSE_SEGMENTS,
) -> List[Coordinate]:
    """Return polygon coordinates representing a covariance ellipse in metres."""
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


def protection_level_circle(
    center: Coordinate,
    radius_m: float,
    segments: int = CIRCLE_SEGMENTS,
) -> List[Coordinate]:
    """Approximate a circle from a protection level radius in metres."""
    if radius_m is None or radius_m <= 0:
        return []
    lon0, lat0 = center
    coords: List[Coordinate] = []
    if GEOD is None:
        # Fallback: convert metres to degrees with simple approximation.
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


def collect_observation_features_from_cmm(
    path: Path,
    selected_ids: Sequence[str],
    bounds: Bounds,
    highlight_map: Optional[Dict[str, Optional[Set[int]]]] = None,
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

            coords = parse_linestring(row.get("geom", ""))
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
                sdn = to_float(cov_entry[0])
                sde = to_float(cov_entry[1])
                sdne = to_float(cov_entry[3])
                if sdn is None or sde is None or sdne is None:
                    continue

                center = (lon, lat)
                is_lon_lat = (-180.0 <= lon <= 180.0) and (-90.0 <= lat <= 90.0)
                if is_lon_lat:
                    metres_per_lat = 111_320.0
                    metres_per_lon = max(math.cos(math.radians(lat)) * 111_320.0, 1e-6)
                else:
                    metres_per_lat = 1.0
                    metres_per_lon = 1.0
                updated_bounds = update_bounds(updated_bounds, [center])

                point_features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "id": traj_id_str,
                            "kind": "observation_point",
                            "seq": seq,
                            "timestamp": timestamp_val,
                            "timestamp_raw": timestamps[seq] if seq < len(timestamps) else None,
                            "pl": to_float(pl),
                            "sdn": sdn,
                            "sde": sde,
                            "sdne": sdne,
                                },
                                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                            }
                        )

                cov_matrix = np.array([[sde * sde, sdne], [sdne, sdn * sdn]], dtype=float)
                scaling = np.array([[metres_per_lon, 0.0], [0.0, metres_per_lat]], dtype=float)
                cov_matrix_m = scaling @ cov_matrix @ scaling.T

                show_envelope = True
                if highlight_active:
                    seq_set = highlight_entry
                    if not has_highlight_for_id:
                        show_envelope = False
                    elif seq_set is not None:
                        show_envelope = seq in seq_set
                if show_envelope:
                    ellipse_coords = covariance_ellipse(center, cov_matrix_m)
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
                                    "sdn": sdn,
                                    "sde": sde,
                                    "sdne": sdne,
                                },
                                "geometry": {"type": "Polygon", "coordinates": [ellipse_coords]},
                                }
                            )

                    pl_val = to_float(pl)
                    if pl_val is not None and pl_val > 0:
                        radius_m = pl_val * metres_per_lat if is_lon_lat else pl_val
                        circle_coords = protection_level_circle(center, radius_m)
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


def collect_point_features(
    path: Path,
    selected_ids: Sequence[str],
    dataset_name: str,
    bounds: Bounds,
) -> Tuple[List[Dict], Bounds]:
    features: List[Dict] = []
    updated_bounds = bounds
    id_set = {str(_id) for _id in selected_ids}

    if not path.exists():
        return features, updated_bounds

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id")
            if traj_id is None or traj_id.strip() not in id_set:
                continue

            coords = parse_linestring(row.get("pgeom", ""))
            if not coords:
                continue

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
    return features, updated_bounds


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
    observation_points_geojson: Dict,
    observation_ellipses_geojson: Dict,
    pl_circles_geojson: Dict,
    cmm_geojson: Dict,
    fmm_geojson: Dict,
) -> str:
    bounds_js = bounds_to_json(bounds)
    ids_label = ", ".join(str(_id) for _id in selected_ids)
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
</div>
<script>
  mapboxgl.accessToken = $TOKEN;
  const selectedIds = $SELECTED_IDS;
  const bounds = $BOUNDS;
  const observationPointGeojson = $OBS_POINT_GEOJSON;
  const observationEllipseGeojson = $OBS_ELLIPSE_GEOJSON;
  const plCircleGeojson = $PL_CIRCLE_GEOJSON;
  const cmmGeojson = $CMM_GEOJSON;
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

    const hasObservationPoints = observationPointGeojson.features.length > 0;
    const hasObservationEllipses = observationEllipseGeojson.features.length > 0;
    const hasPlCircles = plCircleGeojson.features.length > 0;
    const hasCmm = cmmGeojson.features.length > 0;
    const hasFmm = fmmGeojson.features.length > 0;

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
          'Timestamp: ' + formatTimestamp(props.timestamp_raw) + '<br/>' +
          'Error: ' + formatNumber(props.error, 3) + '<br/>' +
          'EP: ' + formatNumber(props.ep, 6) + '<br/>' +
          'TP: ' + formatNumber(props.tp, 6) + '<br/>' +
          'TRUSTWORTHINESS: ' + formatNumber(props.trustworthiness, 6);
      }
      if (kind === 'observation_point' || kind === 'observation_cov') {
        const label = kind === 'observation_point' ? 'Observation' : 'Observation Ellipse';
        return '<strong>' + label + ' ID: ' + props.id + '</strong><br/>' +
          'Seq: ' + props.seq + '<br/>' +
          'Timestamp: ' + formatTimestamp(props.timestamp_raw) + '<br/>' +
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
    attachHover('fmm-points-layer');

    const toggleCmm = document.getElementById('toggle-cmm');
    const toggleFmm = document.getElementById('toggle-fmm');
    const toggleObservation = document.getElementById('toggle-observation');
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

    if (toggleCmm && hasCmm) {
      toggleCmm.addEventListener('change', (event) => {
        const visible = event.target.checked;
        setLayerVisibility('cmm-points-layer', visible);
      });
      setLayerVisibility('cmm-points-layer', toggleCmm.checked);
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
        OBS_POINT_GEOJSON=json.dumps(observation_points_geojson),
        OBS_ELLIPSE_GEOJSON=json.dumps(observation_ellipses_geojson),
        PL_CIRCLE_GEOJSON=json.dumps(pl_circles_geojson),
        CMM_GEOJSON=json.dumps(cmm_geojson),
        FMM_GEOJSON=json.dumps(fmm_geojson),
        IDS_LABEL=ids_label,
    )
    return html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an interactive Mapbox viewer for CMM/FMM results alongside ground truth."
    )
    parser.add_argument("--cmm", default="dataset_hainan_06/1.1/mr/cmm_results.csv", help="CMM result CSV path.")
    parser.add_argument("--fmm", default="dataset_hainan_06/1.1/mr/fmm_results.csv", help="FMM result CSV path.")
    parser.add_argument(
        "--cmm-trajectory",
        default=DEFAULT_CMM_TRAJECTORY_CSV,
        help="CMM trajectory CSV containing observation geometry, covariance, and protection levels.",
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
    parser.add_argument("--output", default="output/hainan_mapbox_view.html", help="Output HTML file.")
    parser.add_argument(
        "--token",
        default=MAPBOX_DEFAULT_TOKEN,
        help="Mapbox access token (defaults to built-in token; can be overridden).",
    )

    args = parser.parse_args()

    cmm_path = Path(args.cmm)
    selected_ids = collect_ids(args.ids, cmm_path)
    if not selected_ids:
        raise SystemExit("At least one trajectory ID must be provided.")

    bounds = initial_bounds()
    highlight_map = parse_observation_highlights(args.show_observation)
    if "__all__" in highlight_map:
        highlight_map = {str(_id): None for _id in selected_ids}

    observation_point_features, observation_ellipse_features, pl_circle_features, bounds = collect_observation_features_from_cmm(
        Path(args.cmm_trajectory), selected_ids, bounds, highlight_map
    )
    cmm_features, bounds = collect_point_features(cmm_path, selected_ids, "cmm", bounds)
    fmm_features, bounds = collect_point_features(Path(args.fmm), selected_ids, "fmm", bounds)

    if not (
        observation_point_features
        or observation_ellipse_features
        or pl_circle_features
        or cmm_features
        or fmm_features
    ):
        raise SystemExit("No features found for the requested IDs. Nothing to visualize.")

    observation_points_geojson = {"type": "FeatureCollection", "features": observation_point_features}
    observation_ellipses_geojson = {"type": "FeatureCollection", "features": observation_ellipse_features}
    pl_circles_geojson = {"type": "FeatureCollection", "features": pl_circle_features}
    cmm_geojson = {"type": "FeatureCollection", "features": cmm_features}
    fmm_geojson = {"type": "FeatureCollection", "features": fmm_features}

    token = requires_token(args.token)
    html = render_html(
        token,
        bounds,
        selected_ids,
        observation_points_geojson,
        observation_ellipses_geojson,
        pl_circles_geojson,
        cmm_geojson,
        fmm_geojson,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote Mapbox viewer to {output_path}")


if __name__ == "__main__":
    main()