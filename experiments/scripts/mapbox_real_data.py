#!/usr/bin/env python3
"""
Mapbox visualization of real-vehicle data using timestamp-aligned CSV.

Layers:
  - Road network (gray, filtered to trajectory bounds)
  - SPP observation points (blue circles, hover shows σ_E/σ_N)
  - RTK ground truth points (green circles)
  - CMM matched path (orange line)
  - FMM matched path (red line)
  - CMM matched points (orange circles)
  - FMM matched points (red circles)
  - Trustworthiness color-coded points (green=high, red=low)

All data aligned by timestamp; no seq mismatch issues.

Usage:
  export MAPBOX_ACCESS_TOKEN="pk.eyJ1..."
  python experiments/scripts/mapbox_real_data.py --output output.html
  python experiments/scripts/mapbox_real_data.py --traj 11 --every 5 --output traj11.html
"""

import argparse, csv, json, math, os, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/real_data"
OUT_DIR = PROJECT_ROOT / "output/spp_error"


def load_roads_near(shapefile, all_lons, all_lats, margin=0.005):
    """Load road segments within bounding box of trajectory points."""
    try:
        from osgeo import ogr
    except ImportError:
        return []
    ds = ogr.Open(str(shapefile))
    if ds is None: return []
    bbox = (min(all_lons) - margin, min(all_lats) - margin,
            max(all_lons) + margin, max(all_lats) + margin)
    layer = ds.GetLayer()
    features = []
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None: continue
        coords = [[geom.GetPoint(i)[0], geom.GetPoint(i)[1]]
                  for i in range(geom.GetPointCount())]
        if len(coords) < 2: continue
        if any(bbox[0] <= c[0] <= bbox[2] and bbox[1] <= c[1] <= bbox[3]
               for c in coords):
            features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": {"id": feat.GetField("key")}})
    return features


def tw_color(tw):
    """Map trustworthiness [0,1] to hex color (red=0, yellow=0.5, green=1)."""
    try: v = float(tw)
    except: return "#888888"
    v = max(0, min(1, v))
    if v < 0.5:
        r = 220; g = int(220 * v * 2); b = 50
    else:
        r = int(220 * (1 - v) * 2); g = 220; b = 50
    return f"#{r:02x}{g:02x}{b:02x}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_DIR / "aligned.csv")
    parser.add_argument("--edges", type=Path,
                        default=PROJECT_ROOT.parent / "input/map/hainan/edges.shp")
    parser.add_argument("--output", type=Path,
                        default=OUT_DIR / "mapbox_real_aligned.html")
    parser.add_argument("--traj", type=str, default=None, help="Trajectory ID (e.g. 11)")
    parser.add_argument("--every", type=int, default=3, help="Show every Nth epoch")
    parser.add_argument("--max-epochs", type=int, default=3000)
    args = parser.parse_args()

    token = os.environ.get("MAPBOX_ACCESS_TOKEN", "MAPBOX_TOKEN_PLACEHOLDER")

    # ── Load aligned data ──
    obs_points, gt_points = [], []
    cmm_points, fmm_points = [], []
    cmm_tw_points, fmm_tw_points = [], []
    cmm_path_by_id, fmm_path_by_id = defaultdict(list), defaultdict(list)
    gt_path_by_id = defaultdict(list)

    all_lons, all_lats = [], []
    count = 0

    with open(args.input, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"]
            if args.traj and tid != args.traj:
                continue
            count += 1
            if count > args.max_epochs:
                break
            if (count - 1) % args.every != 0:
                # Still track path points for all epochs
                pass

            ts = row["timestamp"]

            # SPP observation
            ox, oy = row.get("obs_x", ""), row.get("obs_y", "")
            if ox and oy:
                ox, oy = float(ox), float(oy)
                all_lons.append(ox); all_lats.append(oy)
                if (count - 1) % args.every == 0:
                    obs_points.append({"type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [ox, oy]},
                        "properties": {"kind": "obs", "id": tid, "ts": ts,
                            "sde": row.get("obs_sde",""), "sdn": row.get("obs_sdn","")}})

            # RTK ground truth
            gx, gy = row.get("gt_x", ""), row.get("gt_y", "")
            if gx and gy:
                gx, gy = float(gx), float(gy)
                gt_path_by_id[tid].append([gx, gy])
                if (count - 1) % args.every == 0:
                    gt_points.append({"type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [gx, gy]},
                        "properties": {"kind": "gt", "id": tid, "ts": ts}})

            # CMM
            cx, cy = row.get("cmm_x", ""), row.get("cmm_y", "")
            ctw = row.get("cmm_tw", "")
            if cx and cy:
                cx, cy = float(cx), float(cy)
                cmm_path_by_id[tid].append([cx, cy])
                if (count - 1) % args.every == 0:
                    cmm_points.append({"type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [cx, cy]},
                        "properties": {"kind": "cmm", "id": tid, "ts": ts, "tw": ctw}})
                    cmm_tw_points.append({"type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [cx, cy]},
                        "properties": {"kind": "cmm_tw", "id": tid, "ts": ts, "tw": ctw,
                            "color": tw_color(ctw)}})

            # FMM
            fx, fy = row.get("fmm_x", ""), row.get("fmm_y", "")
            ftw = row.get("fmm_tw", "")
            if fx and fy:
                fx, fy = float(fx), float(fy)
                fmm_path_by_id[tid].append([fx, fy])
                if (count - 1) % args.every == 0:
                    fmm_points.append({"type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [fx, fy]},
                        "properties": {"kind": "fmm", "id": tid, "ts": ts, "tw": ftw}})
                    fmm_tw_points.append({"type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [fx, fy]},
                        "properties": {"kind": "fmm_tw", "id": tid, "ts": ts, "tw": ftw,
                            "color": tw_color(ftw)}})

    print(f"Displaying {len(obs_points)} obs points (every {args.every}th, max {args.max_epochs})")

    # ── Paths as LineStrings ──
    cmm_paths = [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                  "properties": {"kind": "cmm_path", "id": tid}}
                 for tid, coords in cmm_path_by_id.items() if len(coords) >= 2]
    fmm_paths = [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                  "properties": {"kind": "fmm_path", "id": tid}}
                 for tid, coords in fmm_path_by_id.items() if len(coords) >= 2]
    gt_paths = [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                 "properties": {"kind": "gt_path", "id": tid}}
                for tid, coords in gt_path_by_id.items() if len(coords) >= 2]

    # ── Road network ──
    print("Loading roads...")
    roads = load_roads_near(args.edges, all_lons, all_lats) if all_lons else []
    print(f"  {len(roads)} road segments")

    # ── HTML ──
    center_lon = float(np.mean(all_lons)) if all_lons else 110.48
    center_lat = float(np.mean(all_lats)) if all_lats else 19.96

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Real-Vehicle Data: CMM vs FMM vs Ground Truth</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet">
<style>
  body{{margin:0;padding:0}}#map{{position:absolute;top:0;bottom:0;width:100%}}
  .legend{{background:white;padding:10px;border-radius:4px;font:12px monospace;
           box-shadow:0 1px 4px rgba(0,0,0,0.3);line-height:1.6}}
  .legend span{{display:inline-block;width:14px;height:14px;margin-right:6px;
                border-radius:3px;vertical-align:middle}}
</style></head><body>
<div id="map"></div>
<div class="legend" style="position:absolute;bottom:30px;right:10px;z-index:1">
  <div><span style="background:#3388ff"></span> SPP obs (n={len(obs_points)})</div>
  <div><span style="background:#33aa33"></span> RTK Ground Truth</div>
  <div><span style="background:#ff8800"></span> CMM matched path</div>
  <div><span style="background:#ff4444"></span> FMM matched path</div>
  <div><span style="background:linear-gradient(90deg,red,yellow,green);width:100px;display:inline-block;height:10px;margin-right:6px;vertical-align:middle"></span> Trustworthiness</div>
  <div><span style="background:#999"></span> Road network</div>
</div>
<script>
var token = '{token}';
mapboxgl.accessToken = token;
var map = new mapboxgl.Map({{
  container:'map', style:'mapbox://styles/mapbox/light-v11',
  center:[{center_lon},{center_lat}], zoom:16, pitch:45, bearing:0
}});

map.on('load',function(){{
  // Road network
  map.addSource('roads',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(roads)}}}}});
  map.addLayer({{id:'roads',source:'roads',type:'line',
    paint:{{'line-color':'#999','line-width':1.5,'line-opacity':0.4}}}});

  // GT path
  map.addSource('gt_path',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(gt_paths)}}}}});
  map.addLayer({{id:'gt_path',source:'gt_path',type:'line',
    paint:{{'line-color':'#33aa33','line-width':2.5,'line-opacity':0.5}},
    filter:['==',['geometry-type'],'LineString']}});

  // CMM path
  map.addSource('cmm_path',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(cmm_paths)}}}}});
  map.addLayer({{id:'cmm_path',source:'cmm_path',type:'line',
    paint:{{'line-color':'#ff8800','line-width':2,'line-opacity':0.6}},
    filter:['==',['geometry-type'],'LineString']}});

  // FMM path
  map.addSource('fmm_path',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(fmm_paths)}}}}});
  map.addLayer({{id:'fmm_path',source:'fmm_path',type:'line',
    paint:{{'line-color':'#ff4444','line-width':2,'line-opacity':0.6}},
    filter:['==',['geometry-type'],'LineString']}});

  // GT points
  map.addSource('gt',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(gt_points)}}}}});
  map.addLayer({{id:'gt',source:'gt',type:'circle',
    paint:{{'circle-radius':['interpolate',['linear'],['zoom'],14,2,18,5],
            'circle-color':'#33aa33','circle-opacity':0.8,
            'circle-stroke-color':'#fff','circle-stroke-width':0.5}}}});

  // SPP obs points
  map.addSource('obs',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(obs_points)}}}}});
  map.addLayer({{id:'obs',source:'obs',type:'circle',
    paint:{{'circle-radius':['interpolate',['linear'],['zoom'],14,2,18,5],
            'circle-color':'#3388ff','circle-opacity':0.7,
            'circle-stroke-color':'#fff','circle-stroke-width':0.5}}}});

  // CMM matched points
  map.addSource('cmm',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(cmm_points)}}}}});
  map.addLayer({{id:'cmm',source:'cmm',type:'circle',
    paint:{{'circle-radius':['interpolate',['linear'],['zoom'],14,1.5,18,4],
            'circle-color':'#ff8800','circle-opacity':0.7,
            'circle-stroke-color':'#fff','circle-stroke-width':0.3}}}});

  // FMM matched points
  map.addSource('fmm',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(fmm_points)}}}}});
  map.addLayer({{id:'fmm',source:'fmm',type:'circle',
    paint:{{'circle-radius':['interpolate',['linear'],['zoom'],14,1.5,18,4],
            'circle-color':'#ff4444','circle-opacity':0.7,
            'circle-stroke-color':'#fff','circle-stroke-width':0.3}}}});

  // Trustworthiness-colored CMM points (toggle layer)
  map.addSource('cmm_tw',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(cmm_tw_points)}}}}});
  map.addLayer({{id:'cmm_tw',source:'cmm_tw',type:'circle',layout:{{visibility:'none'}},
    paint:{{'circle-radius':['interpolate',['linear'],['zoom'],14,3,18,7],
            'circle-color':['get','color'],'circle-opacity':0.85,
            'circle-stroke-color':'#fff','circle-stroke-width':0.5}}}});

  // FMM trustworthiness points
  map.addSource('fmm_tw',{{type:'geojson',data:{{type:'FeatureCollection',features:{json.dumps(fmm_tw_points)}}}}});
  map.addLayer({{id:'fmm_tw',source:'fmm_tw',type:'circle',layout:{{visibility:'none'}},
    paint:{{'circle-radius':['interpolate',['linear'],['zoom'],14,3,18,7],
            'circle-color':['get','color'],'circle-opacity':0.85,
            'circle-stroke-color':'#fff','circle-stroke-width':0.5}}}});

  // Custom toggle for trustworthiness layer
  map.on('click','cmm_tw',function(e){{new mapboxgl.Popup().setLngLat(e.lngLat)
    .setHTML('TW:'+e.features[0].properties.tw).addTo(map)}});
}});

// Toggle trustworthiness with keyboard 't'
document.addEventListener('keydown',function(e){{
  if(e.key==='t'){{
    var v=map.getLayoutProperty('cmm_tw','visibility');
    var nv=(v==='visible')?'none':'visible';
    map.setLayoutProperty('cmm_tw','visibility',nv);
    map.setLayoutProperty('fmm_tw','visibility',nv);
    map.setLayoutProperty('cmm','visibility',(nv==='visible')?'none':'visible');
    map.setLayoutProperty('fmm','visibility',(nv==='visible')?'none':'visible');
  }}
}});
</script></body></html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    sz = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved: {args.output} ({sz:.1f} MB)")
    print("Press 't' in browser to toggle trustworthiness color mode.")


if __name__ == "__main__":
    main()
