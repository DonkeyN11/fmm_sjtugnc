#!/usr/bin/env python3
"""
Interactive Mapbox GL viewer for comparing CMM/FMM results in the Hainan dataset.

Features:
  - CMM/FMM Result Visualization
  - Trajectory ID & Sequence Search
  - Road Network with Hover-to-ID
"""

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from string import Template
from typing import Dict, Iterable, List, Optional, Tuple, Set

import numpy as np

# Increase CSV field size limit for large candidate lists
csv.field_size_limit(10 ** 7)

# Token from environment variable (DO NOT hardcode credentials in source code)
MAPBOX_DEFAULT_TOKEN = os.environ.get("MAPBOX_ACCESS_TOKEN", "")

Coordinate = Tuple[float, float]
Bounds = Tuple[float, float, float, float]


def parse_geometry(wkt: str) -> List[Coordinate]:
    """Parse a WKT POINT or LINESTRING into a list of [lon, lat] coordinate pairs."""
    text = (wkt or "").strip().upper()
    if not text:
        return []
    
    if text.startswith("POINT"):
        open_idx = text.find("(")
        close_idx = text.rfind(")")
        if open_idx == -1 or close_idx == -1: return []
        body = text[open_idx + 1 : close_idx]
        parts = body.strip().split()
        if len(parts) >= 2:
            try:
                return [(float(parts[0]), float(parts[1]))]
            except ValueError:
                return []
    
    if text.startswith("LINESTRING"):
        open_idx = text.find("(")
        close_idx = text.rfind(")")
        if open_idx == -1 or close_idx == -1: return []
        body = text[open_idx + 1 : close_idx]
        coords: List[Coordinate] = []
        for token in body.split(","):
            parts = token.strip().split()
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        return coords
    
    return []


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
        if math.isfinite(val):
            return val
    except (TypeError, ValueError):
        pass
    return None


def initial_bounds() -> Bounds:
    return (math.inf, math.inf, -math.inf, -math.inf)


def update_bounds(bounds: Bounds, coords: Iterable[Coordinate]) -> Bounds:
    min_lon, min_lat, max_lon, max_lat = bounds
    for lon, lat in coords:
        if lon < min_lon: min_lon = lon
        if lat < min_lat: min_lat = lat
        if lon > max_lon: max_lon = lon
        if lat > max_lat: max_lat = lat
    return min_lon, min_lat, max_lon, max_lat


def covariance_ellipse(
    center: Coordinate,
    sde: float,
    sdn: float,
    sdne: float,
    scale: float = 2.0,
    segments: int = 64,
) -> List[Coordinate]:
    """Return polygon coordinates representing a covariance ellipse in degrees."""
    lon0, lat0 = center
    
    # Covariance matrix in degrees (approx)
    cov_matrix_deg = np.array([[sde * sde, sdne], [sdne, sdn * sdn]], dtype=float)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_deg)
    except np.linalg.LinAlgError:
        return []
    
    eigenvalues = np.maximum(eigenvalues, 0.0)
    radii = np.sqrt(eigenvalues) * float(scale)
    transform = eigenvectors @ np.diag(radii)

    coords: List[Coordinate] = []
    for idx in range(segments):
        angle = 2.0 * math.pi * idx / segments
        unit = np.array([math.cos(angle), math.sin(angle)])
        offset = transform @ unit
        coords.append((lon0 + float(offset[0]), lat0 + float(offset[1])))
    
    if coords:
        coords.append(coords[0])
    return coords


def load_observation_points(path: Path, ids: Set[str], bounds: Bounds) -> Tuple[List[Dict], Bounds]:
    features: List[Dict] = []
    updated_bounds = bounds
    if not path.exists():
        return features, updated_bounds

    id_counters: Dict[str, int] = {}

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id", "").strip()
            if ids and traj_id not in ids:
                continue
            
            try:
                lon = float(row["x"])
                lat = float(row["y"])
            except (KeyError, ValueError, TypeError):
                continue
            
            seq = id_counters.get(traj_id, 0)
            id_counters[traj_id] = seq + 1
                
            center = (lon, lat)
            updated_bounds = update_bounds(updated_bounds, [center])
            
            sde = to_float(row.get("sde"))
            sdn = to_float(row.get("sdn"))
            sdne = to_float(row.get("sdne"))
            pl = to_float(row.get("protection_level"))
            
            props = {
                "id": traj_id,
                "kind": "observation",
                "seq": seq,
                "timestamp": row.get("timestamp"),
                "sde": sde,
                "sdn": sdn,
                "sdne": sdne,
                "pl": pl
            }
            
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]}
            })
            
            # Add ellipse if covariance is present
            if sde is not None and sdn is not None and sdne is not None:
                ellipse_coords = covariance_ellipse(center, sde, sdn, sdne)
                if ellipse_coords:
                    features.append({
                        "type": "Feature",
                        "properties": {**props, "kind": "observation_cov"},
                        "geometry": {"type": "Polygon", "coordinates": [ellipse_coords]}
                    })
                    
    return features, updated_bounds


def load_match_results(path: Path, ids: Set[str], kind: str, bounds: Bounds) -> Tuple[List[Dict], Set[str], Bounds]:
    features: List[Dict] = []
    used_edges: Set[str] = set()
    updated_bounds = bounds
    if not path.exists():
        return features, used_edges, updated_bounds

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            traj_id = row.get("id", "").strip()
            if ids and traj_id not in ids:
                continue
            
            pgeom_wkt = row.get("pgeom", "")
            coords = parse_geometry(pgeom_wkt)
            if not coords:
                continue
            
            lon, lat = coords[0]
            updated_bounds = update_bounds(updated_bounds, [coords[0]])
            
            cpath = row.get("cpath", "").strip()
            if cpath:
                for edge in cpath.split(","):
                    if edge.strip(): used_edges.add(edge.strip())
            
            props = {
                "id": traj_id,
                "kind": f"{kind}_point",
                "seq": int(to_float(row.get("seq")) or 0),
                "timestamp": row.get("timestamp"),
                "ep": to_float(row.get("ep")),
                "tp": to_float(row.get("tp")),
                "trustworthiness": to_float(row.get("trustworthiness")),
                "error": to_float(row.get("error")),
                "cpath": cpath
            }
            
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]}
            })
            
    return features, used_edges, updated_bounds


def load_edges_geojson(shapefile: Path, used_edges: Set[str]) -> Dict:
    """Load road network and return as GeoJSON using ogr2ogr/ogrinfo to filter."""
    if not shapefile.exists():
        print(f"Road network not found at {shapefile}")
        return {"type": "FeatureCollection", "features": []}

    print(f"Extracting road network for {len(used_edges)} edges...")
    
    # Construct SQL query for ogr2ogr
    edge_list_str = ",".join(f"'{e}'" for e in used_edges)
    # We allow both 'fid' and 'id' as common fields
    sql = f"SELECT fid as id, * FROM edges"
    if used_edges:
        sql += f" WHERE fid IN ({edge_list_str})"

    temp_json = "/tmp/edges_filtered.json"
    cmd = [
        "ogr2ogr", "-f", "GeoJSON", 
        "-sql", sql,
        temp_json, str(shapefile)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(temp_json, 'r') as f:
            data = json.load(f)
        os.remove(temp_json)
        return data
    except Exception as e:
        print(f"Error loading road network: {e}")
        return {"type": "FeatureCollection", "features": []}


def render_html(
    token: str,
    bounds: Bounds,
    obs_geojson: Dict,
    cmm_geojson: Dict,
    fmm_geojson: Dict,
    road_geojson: Dict,
    ids: List[str]
) -> str:
    bounds_js = json.dumps([[bounds[0], bounds[1]], [bounds[2], bounds[3]]]) if math.isfinite(bounds[0]) else "null"
    
    template = Template("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Hainan Map Match Viewer</title>
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
    background: rgba(0, 0, 0, 0.85);
    color: #fff;
    padding: 14px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    border-radius: 8px;
    width: 300px;
    z-index: 1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  .legend-item { display: flex; align-items: center; margin-bottom: 6px; }
  .legend-color { width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; border: 1px solid #fff; }
  .search-section {
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid #444;
  }
  .search-section h4 { margin: 0 0 10px 0; font-size: 14px; color: #eee; }
  .input-group { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; }
  .input-row { display: flex; align-items: center; justify-content: space-between; }
  .input-row label { flex: 0 0 60px; color: #bbb; }
  .input-row input {
    flex: 1;
    background: #222;
    border: 1px solid #444;
    color: #fff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
  }
  .btn-group { display: flex; gap: 8px; }
  button {
    flex: 1;
    background: #444;
    color: #fff;
    border: none;
    padding: 6px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.2s;
  }
  button:hover { background: #555; }
  button.primary { background: #3887be; }
  button.primary:hover { background: #2b6cb0; }
  .stats { margin-top: 10px; font-size: 11px; color: #888; }
</style>
</head>
<body>
<div id="map"></div>
<div id="info">
  <strong style="font-size: 16px;">Hainan Map Match</strong><br/>
  <div style="color: #888; margin-bottom: 12px;">IDs: $IDS_LABEL</div>
  
  <div class="legend-item"><div class="legend-color" style="background:#17becf;"></div> Observations</div>
  <div class="legend-item"><div class="legend-color" style="background:#ff7f0e;"></div> CMM Result</div>
  <div class="legend-item"><div class="legend-color" style="background:#2ca02c;"></div> FMM Result</div>
  <div class="legend-item"><div style="width: 12px; height: 3px; background:#888; margin-right: 10px;"></div> Road Network</div>
  
  <div style="margin-top:12px; display: flex; flex-direction: column; gap: 4px;">
    <label><input type="checkbox" id="toggle-obs" checked> Observations</label>
    <label><input type="checkbox" id="toggle-cmm" checked> CMM</label>
    <label><input type="checkbox" id="toggle-fmm" checked> FMM</label>
    <label><input type="checkbox" id="toggle-road" checked> Road Network</label>
  </div>

  <div class="search-section">
    <h4>Search & Filter</h4>
    <div class="input-group">
      <div class="input-row">
        <label>Traj ID</label>
        <input type="text" id="search-id" placeholder="e.g. 11">
      </div>
      <div class="input-row">
        <label>Seq</label>
        <input type="number" id="search-seq" placeholder="e.g. 0">
      </div>
    </div>
    <div class="btn-group">
      <button class="primary" id="btn-filter">Filter</button>
      <button id="btn-clear">Clear</button>
    </div>
    <div id="filter-stats" class="stats"></div>
  </div>
</div>

<script>
  mapboxgl.accessToken = '$TOKEN';
  const bounds = $BOUNDS;
  const obsGeojson = $OBS_GEOJSON;
  const cmmGeojson = $CMM_GEOJSON;
  const fmmGeojson = $FMM_GEOJSON;
  const roadGeojson = $ROAD_GEOJSON;

  const map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/light-v11',
    center: [110.4, 20.0],
    zoom: 12,
    pitch: 45
  });

  map.on('load', () => {
    if (bounds) map.fitBounds(bounds, { padding: 50 });

    // Sources
    map.addSource('obs', { type: 'geojson', data: obsGeojson });
    map.addSource('cmm', { type: 'geojson', data: cmmGeojson });
    map.addSource('fmm', { type: 'geojson', data: fmmGeojson });
    map.addSource('road', { type: 'geojson', data: roadGeojson });

    // Road Layers (Bottom)
    map.addLayer({
      id: 'road-layer',
      type: 'line',
      source: 'road',
      paint: {
        'line-color': '#888',
        'line-width': ['interpolate', ['linear'], ['zoom'], 10, 1, 16, 4],
        'line-opacity': 0.6
      }
    });

    map.addLayer({
      id: 'road-hover',
      type: 'line',
      source: 'road',
      filter: ['==', ['get', 'id'], ''],
      paint: {
        'line-color': '#f00',
        'line-width': ['interpolate', ['linear'], ['zoom'], 10, 2, 16, 6],
        'line-opacity': 0.8
      }
    });

    // Point Layers
    map.addLayer({
      id: 'obs-layer',
      type: 'circle',
      source: 'obs',
      filter: ['==', ['get', 'kind'], 'observation'],
      paint: {
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 2, 16, 6],
        'circle-color': '#17becf',
        'circle-opacity': 0.6,
        'circle-stroke-width': 1,
        'circle-stroke-color': '#fff'
      }
    });
    
    map.addLayer({
      id: 'obs-cov-layer',
      type: 'fill',
      source: 'obs',
      filter: ['==', ['get', 'kind'], 'observation_cov'],
      paint: {
        'fill-color': '#17becf',
        'fill-opacity': 0.1
      }
    });

    map.addLayer({
      id: 'cmm-layer',
      type: 'circle',
      source: 'cmm',
      paint: {
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 3, 16, 8],
        'circle-color': '#ff7f0e',
        'circle-stroke-width': 1.5,
        'circle-stroke-color': '#fff'
      }
    });

    map.addLayer({
      id: 'fmm-layer',
      type: 'circle',
      source: 'fmm',
      paint: {
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 3, 16, 8],
        'circle-color': '#2ca02c',
        'circle-stroke-width': 1.5,
        'circle-stroke-color': '#fff'
      }
    });

    // Popups
    const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false });

    // Road Hover
    map.on('mousemove', 'road-layer', (e) => {
      if (e.features.length > 0) {
        const feature = e.features[0];
        map.setFilter('road-hover', ['==', ['get', 'id'], feature.properties.id]);
        map.getCanvas().style.cursor = 'pointer';
        
        popup.setLngLat(e.lngLat)
          .setHTML(`<strong>Road Edge ID:</strong> $${feature.properties.id}`)
          .addTo(map);
      }
    });

    map.on('mouseleave', 'road-layer', () => {
      map.setFilter('road-hover', ['==', ['get', 'id'], '']);
      map.getCanvas().style.cursor = '';
      popup.remove();
    });

    function addPopup(layerId) {
      map.on('mouseenter', layerId, (e) => {
        map.getCanvas().style.cursor = 'pointer';
        const props = e.features[0].properties;
        let content = `<strong>$${props.kind}</strong><br/>`;
        for (const [key, val] of Object.entries(props)) {
          if (val !== null && val !== undefined) {
             let displayVal = (typeof val === 'number' && !Number.isInteger(val)) ? val.toFixed(6) : val;
             content += `<strong>$${key}:</strong> $${displayVal}<br/>`;
          }
        }
        popup.setLngLat(e.lngLat).setHTML(content).addTo(map);
      });
      map.on('mouseleave', layerId, () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });
    }

    addPopup('obs-layer');
    addPopup('cmm-layer');
    addPopup('fmm-layer');

    // Toggles
    document.getElementById('toggle-obs').addEventListener('change', (e) => {
      const visibility = e.target.checked ? 'visible' : 'none';
      map.setLayoutProperty('obs-layer', 'visibility', visibility);
      map.setLayoutProperty('obs-cov-layer', 'visibility', visibility);
    });
    document.getElementById('toggle-cmm').addEventListener('change', (e) => {
      map.setLayoutProperty('cmm-layer', 'visibility', e.target.checked ? 'visible' : 'none');
    });
    document.getElementById('toggle-fmm').addEventListener('change', (e) => {
      map.setLayoutProperty('fmm-layer', 'visibility', e.target.checked ? 'visible' : 'none');
    });
    document.getElementById('toggle-road').addEventListener('change', (e) => {
      map.setLayoutProperty('road-layer', 'visibility', e.target.checked ? 'visible' : 'none');
    });

    // Filter Logic
    const searchIdInput = document.getElementById('search-id');
    const searchSeqInput = document.getElementById('search-seq');
    const filterStats = document.getElementById('filter-stats');

    function applyFilter() {
      const idVal = searchIdInput.value.trim();
      const seqVal = searchSeqInput.value.trim();
      
      let filter = ['all'];
      if (idVal) filter.push(['==', ['get', 'id'], idVal]);
      if (seqVal) filter.push(['==', ['get', 'seq'], parseInt(seqVal)]);

      const obsBaseFilter = ['==', ['get', 'kind'], 'observation'];
      const obsCovBaseFilter = ['==', ['get', 'kind'], 'observation_cov'];
      
      map.setFilter('obs-layer', filter.length > 1 ? [...filter, obsBaseFilter] : obsBaseFilter);
      map.setFilter('obs-cov-layer', filter.length > 1 ? [...filter, obsCovBaseFilter] : obsCovBaseFilter);
      map.setFilter('cmm-layer', filter.length > 1 ? filter : null);
      map.setFilter('fmm-layer', filter.length > 1 ? filter : null);

      if (idVal || seqVal) {
        const features = [
          ...obsGeojson.features,
          ...cmmGeojson.features,
          ...fmmGeojson.features
        ].filter(f => {
          let match = true;
          if (idVal && f.properties.id !== idVal) match = false;
          if (seqVal && f.properties.seq !== parseInt(seqVal)) match = false;
          return match;
        });

        if (features.length > 0) {
          const lons = features.map(f => f.geometry.coordinates[0]);
          const lats = features.map(f => f.geometry.coordinates[1]);
          const newBounds = [
            [Math.min(...lons), Math.min(...lats)],
            [Math.max(...lons), Math.max(...lats)]
          ];
          map.fitBounds(newBounds, { padding: 80, maxZoom: 17 });
          filterStats.innerText = `Found $${features.length} points.`;
        } else {
          filterStats.innerText = "No points found.";
        }
      } else {
        filterStats.innerText = "";
        if (bounds) map.fitBounds(bounds, { padding: 50 });
      }
    }

    document.getElementById('btn-filter').addEventListener('click', applyFilter);
    document.getElementById('btn-clear').addEventListener('click', () => {
      searchIdInput.value = "";
      searchSeqInput.value = "";
      applyFilter();
    });
  });
</script>
</body>
</html>
""")
    return template.substitute(
        TOKEN=token,
        BOUNDS=bounds_js,
        OBS_GEOJSON=json.dumps(obs_geojson),
        CMM_GEOJSON=json.dumps(cmm_geojson),
        FMM_GEOJSON=json.dumps(fmm_geojson),
        ROAD_GEOJSON=json.dumps(road_geojson),
        IDS_LABEL=", ".join(ids) if ids else "All",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmm", default="dataset-hainan-06/mr/cmm_results_0303_only_trust_norm.csv")
    parser.add_argument("--fmm", default="dataset-hainan-06/mr/fmm_results_filtered.csv")
    parser.add_argument("--input", default="dataset-hainan-06/cmm_input_points.csv")
    parser.add_argument("--edges", default="input/map/hainan/edges.shp")
    parser.add_argument("--ids", help="Comma separated trajectory IDs")
    parser.add_argument("--output", default="python_dataset/mapbox_hainan_viewer.html")
    args = parser.parse_args()

    selected_ids = set(args.ids.split(",")) if args.ids else set()
    bounds = initial_bounds()

    print("Loading observation points...")
    obs_features, bounds = load_observation_points(Path(args.input), selected_ids, bounds)

    print("Loading CMM results...")
    cmm_features, cmm_edges, bounds = load_match_results(Path(args.cmm), selected_ids, "cmm", bounds)

    print("Loading FMM results...")
    fmm_features, fmm_edges, bounds = load_match_results(Path(args.fmm), selected_ids, "fmm", bounds)

    all_used_edges = cmm_edges.union(fmm_edges)
    road_geojson = load_edges_geojson(Path(args.edges), all_used_edges)

    obs_geojson = {"type": "FeatureCollection", "features": obs_features}
    cmm_geojson = {"type": "FeatureCollection", "features": cmm_features}
    fmm_geojson = {"type": "FeatureCollection", "features": fmm_features}

    html = render_html(MAPBOX_DEFAULT_TOKEN, bounds, obs_geojson, cmm_geojson, fmm_geojson, road_geojson, list(selected_ids))
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
