#!/usr/bin/env python3
"""
Mapbox visualization: SPP positions + covariance ellipses vs RTK ground truth.

Layers:
  - Road network (gray lines)
  - SPP points with covariance ellipses (blue)
  - RTK ground truth points (green)
  - Error vectors: SPP → RTK (red lines)

Usage:
  python tests/python/mapbox_spp_rtk.py --traj 11
  python tests/python/mapbox_spp_rtk.py --traj 11 --max-epochs 500 --output experiments/output/spp_error/mapbox_traj11.html
"""

import argparse, csv, json, math, os, sys
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "dataset-hainan-06"


def dm_to_dd(dm_str):
    if not dm_str: return None
    try:
        val = float(dm_str); d = int(val / 100); m = val - d * 100
        return d + m / 60.0
    except: return None


def parse_rtk_nmea(filepath):
    """Return list of (unix_ts, lon, lat)."""
    result = []
    current_date = None
    with open(filepath, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("#"): continue
            idx = line.find("$")
            if idx < 0: continue
            content = line[idx:].split("*")[0]; fields = content.split(",")
            if len(fields) < 2: continue
            mt = fields[0][3:]
            if mt == "RMC" and len(fields) >= 10:
                try: current_date = date(int(fields[9][4:6])+2000, int(fields[9][2:4]), int(fields[9][0:2]))
                except: pass
            if mt == "GGA" and len(fields) >= 10 and current_date:
                ts = fields[1]; lat_dm = fields[2]; ns = fields[3]; lon_dm = fields[4]; ew = fields[5]
                if not ts: continue
                lat = dm_to_dd(lat_dm); lon = dm_to_dd(lon_dm)
                if lat is None or lon is None: continue
                if ns == "S": lat = -lat
                if ew == "W": lon = -lon
                try:
                    h, m = int(ts[0:2]), int(ts[2:4]); s = float(ts[4:])
                    dt = datetime(current_date.year, current_date.month, current_date.day,
                                  h, m, int(s), int((s-int(s))*1e6), tzinfo=timezone.utc)
                    result.append((dt.timestamp(), lon, lat))
                except: pass
    return result


def load_spp(traj_id: int):
    """Load SPP points for trajectory, return list of (ts, lon, lat, sde, sdn, sdne)."""
    spp = []
    spp_file = DATA / "cmm_input_points.csv"
    with open(spp_file, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if int(row["id"]) != traj_id: continue
            ts = float(row["timestamp"])
            lon = float(row["x"]); lat = float(row["y"])
            sde = float(row["sde"]); sdn = float(row["sdn"]); sdne = float(row["sdne"])
            spp.append((ts, lon, lat, sde, sdn, sdne))
    return spp


def compute_ellipse(sde, sdn, sdne, lon, lat, n_pts=32):
    """Compute covariance ellipse polygon in lon/lat, 1-sigma."""
    cov = np.array([[sde*sde, sdne], [sdne, sdn*sdn]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    sigma_major = math.sqrt(max(eigvals[0], eigvals[1], 0))
    sigma_minor = math.sqrt(max(min(eigvals[0], eigvals[1]), 0))
    angle = math.degrees(math.atan2(eigvecs[1, 1], eigvecs[0, 1]))

    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
    m_per_deg_lat = 111320.0

    coords = []
    for i in range(n_pts + 1):
        theta = 2.0 * math.pi * i / n_pts
        dx_m = sigma_major * math.cos(theta)
        dy_m = sigma_minor * math.sin(theta)
        # Rotate
        ang_rad = math.radians(angle)
        rx = dx_m * math.cos(ang_rad) - dy_m * math.sin(ang_rad)
        ry = dx_m * math.sin(ang_rad) + dy_m * math.cos(ang_rad)
        coords.append([lon + rx / m_per_deg_lon, lat + ry / m_per_deg_lat])
    return coords, sigma_major, sigma_minor, angle


def load_road_network(shapefile: str):
    """Load road network as GeoJSON line features."""
    try:
        from osgeo import ogr
    except ImportError:
        print("  GDAL not available, skipping road network")
        return []

    ds = ogr.Open(shapefile)
    if ds is None: return []
    layer = ds.GetLayer()
    features = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None: continue
        coords = []
        for i in range(geom.GetPointCount()):
            pt = geom.GetPoint(i)
            coords.append([pt[0], pt[1]])
        if len(coords) >= 2:
            features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {"id": feat.GetField("key")}})
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=int, default=11, help="Trajectory ID (11-23)")
    parser.add_argument("--max-epochs", type=int, default=2000,
                        help="Max epochs to display (for performance)")
    parser.add_argument("--every", type=int, default=1,
                        help="Show every Nth epoch (1=all, 5=every 5th)")
    parser.add_argument("--ellipse-k", type=float, default=1.0,
                        help="Ellipse scaling factor (1=1-sigma)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shapefile", type=Path,
                        default=PROJECT_ROOT / "input/map/hainan/edges.shp")
    parser.add_argument("--token", type=str, default=None,
                        help="Mapbox access token (or set MAPBOX_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("MAPBOX_TOKEN", "MAPBOX_TOKEN_PLACEHOLDER")

    tid = args.traj
    traj_name = str(tid)
    if tid >= 20:
        traj_dir = f"{tid // 10}.{tid % 10}"
    else:
        traj_dir = f"{tid // 10}.{tid % 10}"

    if args.output is None:
        args.output = PROJECT_ROOT / f"experiments/output/spp_error/mapbox_traj{tid}.html"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading SPP for trajectory {tid}...")
    spp = load_spp(tid)
    print(f"  {len(spp)} SPP epochs")

    print(f"Loading RTK...")
    rtk = parse_rtk_nmea(str(DATA / traj_dir / "实时定位结果" / "rtk_solution_clean.txt"))
    print(f"  {len(rtk)} RTK epochs")

    # Time-align
    rtk_by_ts = {int(t): (lon, lat) for t, lon, lat in rtk}
    print(f"  RTK ts range: {min(rtk_by_ts.keys())} - {max(rtk_by_ts.keys())}")

    # Build GeoJSON for SPP and RTK
    spp_points = []
    rtk_points = []
    error_lines = []
    ellipses = []

    count = 0
    for ts, lon, lat, sde, sdn, sdne in spp:
        if count >= args.max_epochs:
            break
        ts_int = int(ts)
        if ts_int not in rtk_by_ts:
            continue

        if count % args.every != 0:
            count += 1
            continue

        rtk_lon, rtk_lat = rtk_by_ts[ts_int]

        # SPP point
        spp_points.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"epoch": count, "ts": ts_int, "sde_m": sde * 111320,
                           "sdn_m": sdn * 111320}
        })

        # RTK point
        rtk_points.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [rtk_lon, rtk_lat]},
            "properties": {"epoch": count, "ts": ts_int}
        })

        # Error line SPP → RTK
        error_lines.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[lon, lat], [rtk_lon, rtk_lat]]},
            "properties": {"epoch": count}
        })

        # Covariance ellipse
        ell_coords, smaj, smin, ang = compute_ellipse(
            sde * args.ellipse_k, sdn * args.ellipse_k, sdne * args.ellipse_k**2, lon, lat)
        ellipses.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ell_coords]},
            "properties": {"epoch": count, "smaj_m": smaj * 111320, "smin_m": smin * 111320,
                           "angle_deg": ang}
        })

        count += 1

    print(f"  Displaying {len(spp_points)} epochs")

    # Road network (filter to trajectory bounding box)
    roads = []
    if args.shapefile.exists():
        print("Loading road network...")
        all_lons = [p["geometry"]["coordinates"][0] for p in spp_points]
        all_lats = [p["geometry"]["coordinates"][1] for p in spp_points]
        margin = 0.005  # ~500m
        bbox = (min(all_lons) - margin, min(all_lats) - margin,
                max(all_lons) + margin, max(all_lats) + margin)
        all_roads = load_road_network(str(args.shapefile))
        for r in all_roads:
            coords = r["geometry"]["coordinates"]
            if any(bbox[0] <= c[0] <= bbox[2] and bbox[1] <= c[1] <= bbox[3]
                   for c in coords):
                roads.append(r)
        print(f"  {len(roads)} road segments in view (from {len(all_roads)} total)")

    # Build HTML
    center_lon = float(np.mean([p["geometry"]["coordinates"][0] for p in spp_points[:100]])) if spp_points else 110.48
    center_lat = float(np.mean([p["geometry"]["coordinates"][1] for p in spp_points[:100]])) if spp_points else 19.96

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SPP vs RTK — Trajectory {tid}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet">
<style>
  body {{ margin:0; padding:0; }}
  #map {{ position:absolute; top:0; bottom:0; width:100%; }}
  .legend {{ background:white; padding:10px; border-radius:4px; font:12px monospace;
             box-shadow:0 1px 4px rgba(0,0,0,0.3); }}
  .legend div {{ margin:3px 0; }}
  .legend span {{ display:inline-block; width:14px; height:14px; margin-right:6px;
                  border-radius:3px; vertical-align:middle; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="legend" class="legend" style="position:absolute; bottom:30px; right:10px; z-index:1;">
  <div><span style="background:#3388ff;"></span> SPP (n={len(spp_points)})</div>
  <div><span style="background:#33aa33;"></span> RTK Ground Truth</div>
  <div><span style="background:#ff4444; width:14px; height:2px; display:inline-block; margin:6px 6px 4px 0; vertical-align:middle;"></span> Error vector (SPP→RTK)</div>
  <div><span style="background:rgba(51,136,255,0.15); border:1px solid #3388ff;"></span> Cov ellipse ({args.ellipse_k}σ)</div>
  <div><span style="background:#999;"></span> Road network</div>
</div>
<script>
mapboxgl.accessToken = 'MAPBOX_TOKEN_PLACEHOLDER';
var map = new mapboxgl.Map({{
  container: 'map', style: 'mapbox://styles/mapbox/light-v11',
  center: [{center_lon}, {center_lat}], zoom: 16.5, pitch: 45, bearing: 0
}});

map.on('load', function() {{
  // Road network
  map.addSource('roads', {{ type: 'geojson', data: {{ type: 'FeatureCollection',
    features: {json.dumps(roads)} }} }});
  map.addLayer({{ id: 'roads', source: 'roads', type: 'line',
    paint: {{ 'line-color': '#999999', 'line-width': 2, 'line-opacity': 0.5 }} }});

  // Error lines SPP→RTK
  map.addSource('error_lines', {{ type: 'geojson', data: {{ type: 'FeatureCollection',
    features: {json.dumps(error_lines)} }} }});
  map.addLayer({{ id: 'error_lines', source: 'error_lines', type: 'line',
    paint: {{ 'line-color': '#ff4444', 'line-width': 1, 'line-opacity': 0.4 }} }});

  // Covariance ellipses
  map.addSource('ellipses', {{ type: 'geojson', data: {{ type: 'FeatureCollection',
    features: {json.dumps(ellipses)} }} }});
  map.addLayer({{ id: 'ellipses', source: 'ellipses', type: 'fill',
    paint: {{ 'fill-color': '#3388ff', 'fill-opacity': 0.12 }} }});
  map.addLayer({{ id: 'ellipses_outline', source: 'ellipses', type: 'line',
    paint: {{ 'line-color': '#3388ff', 'line-width': 0.8, 'line-opacity': 0.4 }},
    filter: ['==', ['geometry-type'], 'Polygon'] }});

  // SPP points
  map.addSource('spp', {{ type: 'geojson', data: {{ type: 'FeatureCollection',
    features: {json.dumps(spp_points)} }} }});
  map.addLayer({{ id: 'spp', source: 'spp', type: 'circle',
    paint: {{ 'circle-radius': ['interpolate',['linear'],['zoom'], 14,1.5, 18,4],
             'circle-color': '#3388ff', 'circle-opacity': 0.7,
             'circle-stroke-color': '#ffffff', 'circle-stroke-width': 0.5 }} }});

  // RTK points
  map.addSource('rtk', {{ type: 'geojson', data: {{ type: 'FeatureCollection',
    features: {json.dumps(rtk_points)} }} }});
  map.addLayer({{ id: 'rtk', source: 'rtk', type: 'circle',
    paint: {{ 'circle-radius': ['interpolate',['linear'],['zoom'], 14,1.5, 18,4],
             'circle-color': '#33aa33', 'circle-opacity': 0.8,
             'circle-stroke-color': '#ffffff', 'circle-stroke-width': 0.5 }} }});

  // Hover popups
  var popup = new mapboxgl.Popup({{ closeButton: false, closeOnClick: false }});
  ['spp','rtk'].forEach(function(layer) {{
    map.on('mouseenter', layer, function(e) {{
      map.getCanvas().style.cursor = 'pointer';
      var p = e.features[0].properties;
      popup.setLngLat(e.lngLat).setHTML(
        'Epoch: ' + p.epoch + '<br>TS: ' + p.ts +
        (p.sde_m ? '<br>σ_E=' + p.sde_m.toFixed(1) + 'm σ_N=' + p.sdn_m.toFixed(1) + 'm' : '')
      ).addTo(map);
    }});
    map.on('mouseleave', layer, function() {{
      map.getCanvas().style.cursor = ''; popup.remove();
    }});
  }});
}});
</script>
</body>
</html>"""

    html = html.replace("MAPBOX_TOKEN_PLACEHOLDER", token)

    with open(args.output, "w") as f:
        f.write(html)

    file_size = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved: {args.output} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
