#!/usr/bin/env python3
"""Generate a Mapbox GL JS viewer for FMM map-matching output."""
import argparse
import csv
csv.field_size_limit(10 ** 7)
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_linestring(wkt: str) -> List[List[float]]:
    wkt = (wkt or "").strip()
    if not wkt.upper().startswith("LINESTRING"):
        return []
    open_idx = wkt.find("(")
    close_idx = wkt.rfind(")")
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        return []
    body = wkt[open_idx + 1:close_idx]
    coords: List[List[float]] = []
    for token in body.split(","):
        parts = token.strip().split()
        if len(parts) != 2:
            continue
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        coords.append([lon, lat])
    return coords


def update_bounds(bounds: Tuple[float, float, float, float], coords: List[List[float]]):
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


def build_feature(row: Dict[str, str]) -> Dict:
    coords = parse_linestring(row.get("pgeom", ""))
    if len(coords) < 2:
        return {}
    try:
        cumu_prob = float(row.get("cumu_prob", ""))
    except ValueError:
        cumu_prob = None
    feature: Dict = {
        "type": "Feature",
        "properties": {
            "id": row.get("id"),
            "cumu_prob": cumu_prob,
            "duration": row.get("duration"),
            "timestamp": row.get("timestamp"),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
    }
    return feature


def build_raw_point_features(path: Path, ids: Set[str], bounds: Tuple[float, float, float, float]) -> Tuple[List[Dict], Tuple[float, float, float, float]]:
    features: List[Dict] = []
    updated_bounds = bounds
    if not path or not path.exists():
        return features, updated_bounds

    with path.open(newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        for row in reader:
            traj_id = row.get("id")
            if traj_id not in ids:
                continue
            coords = parse_linestring(row.get("geom", ""))
            if not coords:
                continue
            updated_bounds = update_bounds(updated_bounds, coords)
            original_id = row.get("original_id")
            for idx, (lon, lat) in enumerate(coords):
                features.append({
                    "type": "Feature",
                    "properties": {
                        "id": traj_id,
                        "original_id": original_id,
                        "seq": idx,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                })
    return features, updated_bounds


def render_html(token: str, geojson: Dict, bounds: Tuple[float, float, float, float], raw_points: Optional[Dict] = None) -> str:
    min_lon, min_lat, max_lon, max_lat = bounds
    bounds_js = json.dumps([[min_lon, min_lat], [max_lon, max_lat]])
    geojson_js = json.dumps(geojson)
    raw_points_js = json.dumps(raw_points) if raw_points is not None else "null"
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>FMM Mapbox 3D Viewer</title>
<meta name=\"viewport\" content=\"initial-scale=1,maximum-scale=1,user-scalable=no\" />
<script src=\"https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.js\"></script>
<link href=\"https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.css\" rel=\"stylesheet\" />
<style>
  body {{ margin: 0; padding: 0; }}
  #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
  #info {{ position: absolute; top: 12px; left: 12px; background: rgba(0, 0, 0, 0.55); color: #fff; padding: 8px 12px; font-family: sans-serif; font-size: 13px; border-radius: 4px; }}
</style>
</head>
<body>
<div id=\"map\"></div>
<div id=\"info\">Scroll to zoom • Drag to look around <br/>Color encodes \"cumu_prob\"</div>
<script>
  mapboxgl.accessToken = {json.dumps(token)};
  const routeGeojson = {geojson_js};
  const rawPointsGeojson = {raw_points_js};
  const bounds = {bounds_js};
  const map = new mapboxgl.Map({{
    container: 'map',
    style: 'mapbox://styles/mapbox/navigation-day-v1',
    center: [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2],
    zoom: 12,
    pitch: 60,
    bearing: -20
  }});

  map.on('load', () => {{
    map.addControl(new mapboxgl.NavigationControl(), 'top-right');

    map.addSource('mapbox-dem', {{
      type: 'raster-dem',
      url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
      tileSize: 512,
      maxzoom: 14
    }});
    map.setTerrain({{ source: 'mapbox-dem', exaggeration: 1.4 }});

    map.addSource('matched-route', {{
      type: 'geojson',
      data: routeGeojson
    }});

    map.addLayer({{
      id: 'matched-route-line',
      type: 'line',
      source: 'matched-route',
      paint: {{
        'line-color': [
          'interpolate', ['linear'], ['coalesce', ['get', 'cumu_prob'], 0],
          0, '#2b83ba',
          0.5, '#ffffbf',
          1, '#d7191c'
        ],
        'line-width': [
          'interpolate', ['linear'], ['zoom'],
          10, 3,
          16, 12
        ],
        'line-opacity': 0.85
      }}
    }});

    if (rawPointsGeojson) {{
      map.addSource('raw-points', {{
        type: 'geojson',
        data: rawPointsGeojson
      }});
      map.addLayer({{
        id: 'raw-points-layer',
        type: 'circle',
        source: 'raw-points',
        paint: {{
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            10, 3,
            16, 8
          ],
          'circle-color': '#ff8c00',
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 1
        }}
      }});
    }}

    const layers = map.getStyle().layers || [];
    let labelLayerId = null;
    for (const layer of layers) {{
      if (layer.type === 'symbol' && layer.layout && layer.layout['text-field']) {{
        labelLayerId = layer.id;
        break;
      }}
    }}

    const buildingLayer = {{
      id: 'custom-3d-buildings',
      source: 'composite',
      'source-layer': 'building',
      filter: ['==', ['get', 'extrude'], 'true'],
      type: 'fill-extrusion',
      minzoom: 15,
      paint: {{
        'fill-extrusion-color': '#aaa',
        'fill-extrusion-height': ['get', 'height'],
        'fill-extrusion-base': ['get', 'min_height'],
        'fill-extrusion-opacity': 0.6
      }}
    }};
    if (labelLayerId) {{
      map.addLayer(buildingLayer, labelLayerId);
    }} else {{
      map.addLayer(buildingLayer);
    }}

    const hasExtent = bounds[0][0] !== bounds[1][0] || bounds[0][1] !== bounds[1][1];
    if (hasExtent) {{
      map.fitBounds(bounds, {{ padding: 64, maxZoom: 16 }});
    }} else {{
      map.setZoom(15);
    }}
  }});
</script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Create a Mapbox 3D viewer for FMM map-matching results.")
    parser.add_argument("input", help="Path to the FMM output CSV containing a 'pgeom' LINESTRING column")
    parser.add_argument("--output", default="output/fmm_mapbox_view.html", help="HTML file to write")
    parser.add_argument(
        "--raw-trajectories",
        help="CSV with original trajectories (id;original_id;geom) to overlay as point features",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of trajectories to include (0 means all)")
    parser.add_argument("--token", default=os.environ.get("MAPBOX_ACCESS_TOKEN"), help="Mapbox access token (defaults to MAPBOX_ACCESS_TOKEN env var)")

    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Mapbox access token is required. Pass --token or set MAPBOX_ACCESS_TOKEN.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    features: List[Dict] = []
    bounds = (float("inf"), float("inf"), float("-inf"), float("-inf"))

    with input_path.open(newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=';')
        for row in reader:
            feature = build_feature(row)
            if not feature:
                continue
            coords = feature["geometry"]["coordinates"]
            bounds = update_bounds(bounds, coords)
            features.append(feature)
            if args.limit and len(features) >= args.limit:
                break

    if not features:
        raise SystemExit("No valid LINESTRING geometries found in the input file.")

    id_set: Set[str] = set()
    for feature in features:
        props = feature.get("properties", {})
        feature_id = props.get("id")
        if feature_id is not None:
            id_set.add(str(feature_id))

    raw_points_geojson: Optional[Dict] = None
    raw_path: Optional[Path]
    if args.raw_trajectories:
        raw_path = Path(args.raw_trajectories)
    else:
        default_raw_path = Path("input/trajectory/all_2hour_data/all_2hour_data_Jan_parallel_filtered.csv")
        raw_path = default_raw_path if default_raw_path.exists() else None

    if raw_path:
        if not raw_path.exists():
            raise SystemExit(f"Raw trajectory file not found: {raw_path}")
        raw_point_features, bounds = build_raw_point_features(raw_path, id_set, bounds)
        if raw_point_features:
            raw_points_geojson = {"type": "FeatureCollection", "features": raw_point_features}

    geojson = {"type": "FeatureCollection", "features": features}
    html = render_html(args.token, geojson, bounds, raw_points_geojson)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f"Wrote Mapbox viewer to {output_path}")


if __name__ == "__main__":
    main()
