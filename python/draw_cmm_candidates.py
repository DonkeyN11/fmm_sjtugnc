#!/usr/bin/env python3
"""
Interactive Mapbox GL viewer for comparing CMM results (ogeom as obs, max ep candidate as result)
with FMM results. Mimics logic from draw_hainan_map.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from string import Template
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np

try:
    from pyproj import Geod, Transformer
except ImportError:
    Geod = None
    Transformer = None

MAPBOX_DEFAULT_TOKEN = (
    "pk.eyJ1IjoiZG9ua2V5LW5pbmciLCJhIjoiY21kenJ5OTY5MGc5azJqb25hdTVtc2tvNiJ9.CgMc9ZNXaZ1HDrC4Zl2aMQ"
)

COORD_TRANSFORMER = None
USE_PROJECTED_CRS = False
INPUT_CRS = "EPSG:4326"

def parse_linestring(wkt: str) -> List[Tuple[float, float]]:
    """Parse WKT POINT or LINESTRING."""
    text = (wkt or "").strip()
    if not text: return []
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if open_idx == -1 or close_idx == -1: return []
    body = text[open_idx + 1 : close_idx]
    coords = []
    for token in body.split(","):
        parts = token.strip().split()
        if len(parts) >= 2:
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError: continue
    return coords

def transform_coord(x: float, y: float) -> Tuple[float, float]:
    if USE_PROJECTED_CRS and COORD_TRANSFORMER:
        return COORD_TRANSFORMER.transform(x, y)
    return (x, y)

def parse_candidates(candidate_str: str) -> List[Tuple[float, float, float]]:
    """Parse candidate string like '((x1,y1,ep1),(x2,y2,ep2),...)'"""
    if not candidate_str or candidate_str == "()":
        return []
    inner = candidate_str.strip()
    # Match all (x,y,ep) patterns
    import re
    matches = re.findall(r"\(([^)]+)\)", inner)
    results = []
    for m in matches:
        # Some might be nested, e.g. the outer (())
        if "," not in m: continue
        vals = m.split(",")
        if len(vals) >= 3:
            try:
                results.append((float(vals[0]), float(vals[1]), float(vals[2])))
            except ValueError:
                continue
    return results

def collect_features(cmm_csv: Path, fmm_csv: Path, selected_ids: Set[str]):
    obs_points = []
    cmm_points = []
    fmm_points = []
    bounds = [math.inf, math.inf, -math.inf, -math.inf]

    def update_bounds(lon, lat):
        bounds[0] = min(bounds[0], lon)
        bounds[1] = min(bounds[1], lat)
        bounds[2] = max(bounds[2], lon)
        bounds[3] = max(bounds[3], lat)

    # Process CMM CSV: ogeom -> Observation, candidates max ep -> CMM Point
    if cmm_csv.exists():
        with cmm_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                tid = row["id"].strip()
                if tid not in selected_ids: continue
                
                seq = int(row.get("seq", 0))
                # Observation from ogeom
                ogeom = row.get("ogeom", "")
                coords = parse_linestring(ogeom)
                if coords:
                    lon, lat = transform_coord(coords[0][0], coords[0][1])
                    update_bounds(lon, lat)
                    obs_points.append({
                        "type": "Feature",
                        "properties": {
                            "id": tid, "kind": "observation_point", "seq": seq,
                            "timestamp_raw": row.get("timestamp")
                        },
                        "geometry": {"type": "Point", "coordinates": [lon, lat]}
                    })

                # CMM result from candidates
                cand_str = row.get("candidates", "")
                candidates = parse_candidates(cand_str)
                if candidates:
                    best = max(candidates, key=lambda x: x[2])
                    lon, lat = transform_coord(best[0], best[1])
                    update_bounds(lon, lat)
                    cmm_points.append({
                        "type": "Feature",
                        "properties": {
                            "id": tid, "kind": "cmm_point", "seq": seq,
                            "ep": best[2], "source": "cmm_max_ep",
                            "timestamp_raw": row.get("timestamp"),
                            "trustworthiness": float(row.get("trustworthiness", 0)),
                            "tp": float(row.get("tp", 0)),
                            "error": float(row.get("sp_dist", 0)) # Using sp_dist as proxy for error if needed
                        },
                        "geometry": {"type": "Point", "coordinates": [lon, lat]}
                    })

    # Process FMM CSV
    if fmm_csv.exists():
        with fmm_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                tid = row["id"].strip()
                if tid not in selected_ids: continue
                
                pgeom = row.get("pgeom", "")
                coords = parse_linestring(pgeom)
                if coords:
                    lon, lat = transform_coord(coords[0][0], coords[0][1])
                    update_bounds(lon, lat)
                    fmm_points.append({
                        "type": "Feature",
                        "properties": {
                            "id": tid, "kind": "fmm_point", "seq": int(row.get("seq", 0)),
                            "ep": float(row.get("ep", 0)), "source": "fmm",
                            "timestamp_raw": row.get("timestamp"),
                            "tp": float(row.get("tp", 0)),
                            "trustworthiness": float(row.get("trustworthiness", 0))
                        },
                        "geometry": {"type": "Point", "coordinates": [lon, lat]}
                    })

    return obs_points, cmm_points, fmm_points, bounds

def render_html(token, bounds, selected_ids, obs_geo, cmm_geo, fmm_geo):
    ids_label = ", ".join(selected_ids)
    bounds_js = json.dumps([[bounds[0], bounds[1]], [bounds[2], bounds[3]]]) if bounds[0] != math.inf else "null"
    
    # Using the refined Template from draw_hainan_map.py but simplified for this use case
    template = Template("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>CMM Candidates Viewer</title>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.css" rel="stylesheet" />
<style>
  body { margin: 0; padding: 0; }
  #map { position: absolute; top: 0; bottom: 0; width: 100%; }
  #info {
    position: absolute; top: 12px; left: 12px;
    background: rgba(0, 0, 0, 0.6); color: #fff;
    padding: 12px; font-family: "Segoe UI", sans-serif; border-radius: 8px;
    max-width: 300px; line-height: 1.6;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; cursor: pointer; }
  .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
</style>
</head>
<body>
<div id="map"></div>
<div id="info">
  <strong>Trajectory IDs:</strong> $IDS_LABEL<br/>
  <div style="margin-top:8px;">
    <label class="legend-item"><input type="checkbox" id="obs-toggle" checked><span class="dot" style="background:#17becf"></span> Observation (ogeom)</label>
    <label class="legend-item"><input type="checkbox" id="cmm-toggle" checked><span class="dot" style="background:#ff7f0e"></span> CMM (Max EP Candidate)</label>
    <label class="legend-item"><input type="checkbox" id="fmm-toggle" checked><span class="dot" style="background:#2ca02c"></span> FMM (Filtered Results)</label>
  </div>
</div>
<script>
  mapboxgl.accessToken = '$TOKEN';
  const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/satellite-streets-v12',
    center: [0, 0], zoom: 2, pitch: 60, bearing: -20
  });

  map.on('load', () => {
    map.addControl(new mapboxgl.NavigationControl(), 'top-right');
    const bounds = $BOUNDS;
    if (bounds) map.fitBounds(bounds, { padding: 70, maxZoom: 16 });

    map.addSource('mapbox-dem', {
      type: 'raster-dem', url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
      tileSize: 512, maxzoom: 14
    });
    map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.2 });

    map.addSource('obs', { type: 'geojson', data: $OBS_GEO });
    map.addSource('cmm', { type: 'geojson', data: $CMM_GEO });
    map.addSource('fmm', { type: 'geojson', data: $FMM_GEO });

    map.addLayer({
      id: 'obs-layer', type: 'circle', source: 'obs',
      paint: { 'circle-radius': 4, 'circle-color': '#17becf', 'circle-opacity': 0.8 }
    });
    map.addLayer({
      id: 'cmm-layer', type: 'circle', source: 'cmm',
      paint: {
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 4, 16, 10],
        'circle-color': '#ff7f0e', 'circle-opacity': 0.9,
        'circle-stroke-color': '#fff', 'circle-stroke-width': 1.5
      }
    });
    map.addLayer({
      id: 'fmm-layer', type: 'circle', source: 'fmm',
      paint: {
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 4, 16, 10],
        'circle-color': '#2ca02c', 'circle-opacity': 0.9,
        'circle-stroke-color': '#fff', 'circle-stroke-width': 1.5
      }
    });

    const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false });
    const hoverLayers = ['obs-layer', 'cmm-layer', 'fmm-layer'];
    
    hoverLayers.forEach(lyr => {
      map.on('mouseenter', lyr, (e) => {
        map.getCanvas().style.cursor = 'pointer';
        const p = e.features[0].properties;
        let h = '<strong>ID: ' + p.id + '</strong><br>Seq: ' + p.seq;
        if (p.timestamp_raw) h += '<br>Time: ' + p.timestamp_raw;
        if (p.ep) h += '<br>EP: ' + p.ep.toFixed(6);
        if (p.trustworthiness) h += '<br>Trust: ' + p.trustworthiness.toFixed(4);
        popup.setLngLat(e.features[0].geometry.coordinates).setHTML(h).addTo(map);
      });
      map.on('mouseleave', lyr, () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });
    });

    document.getElementById('obs-toggle').onchange = (e) => map.setLayoutProperty('obs-layer', 'visibility', e.target.checked ? 'visible' : 'none');
    document.getElementById('cmm-toggle').onchange = (e) => map.setLayoutProperty('cmm-layer', 'visibility', e.target.checked ? 'visible' : 'none');
    document.getElementById('fmm-toggle').onchange = (e) => map.setLayoutProperty('fmm-layer', 'visibility', e.target.checked ? 'visible' : 'none');
  });
</script>
</body>
</html>
""")
    return template.substitute(
        TOKEN=token, IDS_LABEL=ids_label, BOUNDS=bounds_js,
        OBS_GEO=json.dumps(obs_geo), CMM_GEO=json.dumps(cmm_geo), FMM_GEO=json.dumps(fmm_geo)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmm", default="dataset-hainan-06/mr/cmm_results_bak.csv")
    parser.add_argument("--fmm", default="dataset-hainan-06/mr/fmm_results_filtered.csv")
    parser.add_argument("--ids", default="11")
    parser.add_argument("--output", default="output/cmm_candidates_view.html")
    parser.add_argument("--input-crs", default="EPSG:4326")
    args = parser.parse_args()

    global USE_PROJECTED_CRS, COORD_TRANSFORMER
    if args.input_crs.upper() != "EPSG:4326":
        USE_PROJECTED_CRS = True
        COORD_TRANSFORMER = Transformer.from_crs(args.input_crs, "EPSG:4326", always_xy=True)

    selected_ids = set(args.ids.split(","))
    obs, cmm, fmm, b = collect_features(Path(args.cmm), Path(args.fmm), selected_ids)

    html = render_html(MAPBOX_DEFAULT_TOKEN, b, selected_ids, 
                       {"type": "FeatureCollection", "features": obs},
                       {"type": "FeatureCollection", "features": cmm},
                       {"type": "FeatureCollection", "features": fmm})
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Viewer written to {out_path}")

if __name__ == "__main__":
    main()