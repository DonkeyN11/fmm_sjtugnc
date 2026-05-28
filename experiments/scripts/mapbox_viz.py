#!/usr/bin/env python3
"""
Interactive Mapbox visualization for synthetic experiment results.

Visualizes:
  - Synthetic observations with covariance ellipses (UTM→WGS84 reprojection)
  - Ground truth path
  - FDE detection status (color-coded)
  - Protection Level circles
  - CMM matching results (if available)
  - Road network with hover-to-ID

Usage:
  python experiments/scripts/mapbox_viz.py \
    --data-dir experiments/data/sigma_30/no_occlusion/with_fault \
    --cmm-result /tmp/cmm_test/cmm_result.csv \
    --output experiments/output/mapbox_sigma30_fault.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

csv.field_size_limit(10 ** 7)
MAPBOX_TOKEN = os.environ.get("MAPBOX_ACCESS_TOKEN", "")

Coordinate = Tuple[float, float]
Bounds = Tuple[float, float, float, float]

# ── Color palette ──────────────────────────────────────────────────────────
COLOR_OBS = "#17becf"        # Observations (cyan)
COLOR_FDE_DETECTED = "#e74c3c"  # FDE triggered (red)
COLOR_FDE_CLEAN = "#2ecc71"     # FDE clean (green)
COLOR_TRUTH = "#f39c12"         # Ground truth path (orange)
COLOR_CMM = "#ff7f0e"           # CMM match (orange)
COLOR_PL = "#9b59b6"            # PL circle (purple)
COLOR_ROAD = "#888888"          # Road network


def parse_geometry(wkt: str) -> List[Coordinate]:
    text = (wkt or "").strip().upper()
    if not text: return []
    if text.startswith("POINT"):
        open_idx = text.find("("); close_idx = text.rfind(")")
        if open_idx == -1 or close_idx == -1: return []
        parts = text[open_idx+1:close_idx].strip().split()
        if len(parts) >= 2:
            try: return [(float(parts[0]), float(parts[1]))]
            except ValueError: pass
    if text.startswith("LINESTRING"):
        open_idx = text.find("("); close_idx = text.rfind(")")
        if open_idx == -1 or close_idx == -1: return []
        coords = []
        for token in text[open_idx+1:close_idx].split(","):
            parts = token.strip().split()
            if len(parts) >= 2:
                try: coords.append((float(parts[0]), float(parts[1])))
                except ValueError: pass
        return coords
    return []


def reproject_utm_to_wgs84_coords(coords: np.ndarray) -> np.ndarray:
    """Reproject UTM (EPSG:32649) → WGS84 lon/lat."""
    import pyproj
    t = pyproj.Transformer.from_crs("EPSG:32649", "EPSG:4326", always_xy=True)
    out = np.zeros_like(coords)
    for i in range(len(coords)):
        out[i, 0], out[i, 1] = t.transform(coords[i, 0], coords[i, 1])
    return out


def initial_bounds() -> Bounds:
    return (math.inf, math.inf, -math.inf, -math.inf)


def update_bounds(bounds: Bounds, coords) -> Bounds:
    min_lon, min_lat, max_lon, max_lat = bounds
    for lon, lat in coords:
        if lon < min_lon: min_lon = lon
        if lat < min_lat: min_lat = lat
        if lon > max_lon: max_lon = lon
        if lat > max_lat: max_lat = lat
    return min_lon, min_lat, max_lon, max_lat


def load_observations(path: Path, bounds: Bounds) -> Tuple[List[Dict], Bounds]:
    """Load synthetic observations (already in WGS84 lon/lat)."""
    features = []
    b = bounds
    if not path.exists(): return features, b

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                lon = float(row["x"])  # Already WGS84
                lat = float(row["y"])
            except (KeyError, ValueError): continue

            b = update_bounds(b, [(lon, lat)])

            sde = float(row.get("sde", 0))
            sdn = float(row.get("sdn", 0))
            sdne = float(row.get("sdne", 0))
            fde_detected = row.get("fde_detected", "0") == "1"

            color = COLOR_FDE_DETECTED if fde_detected else COLOR_FDE_CLEAN
            props = {
                "id": row.get("id", ""),
                "kind": "observation",
                "timestamp": row.get("timestamp", ""),
                "sde": round(sde, 8), "sdn": round(sdn, 8), "sdne": round(sdne, 8),
                "pl": round(float(row.get("protection_level", 0)), 8),
                "fde_detected": fde_detected,
                "fde_excluded": int(row.get("fde_excluded", 0)),
                "color": color,
            }
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]}
            })

            # Covariance ellipse (already in degrees)
            if sde > 0 and sdn > 0:
                cov_deg = np.array([[sde**2, sdne], [sdne, sdn**2]])
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov_deg)
                    eigvals = np.maximum(eigvals, 0)
                    radii = np.sqrt(eigvals) * 2.0
                    T = eigvecs @ np.diag(radii)
                    ellipse = []
                    for a in np.linspace(0, 2*np.pi, 65):
                        off = T @ np.array([math.cos(a), math.sin(a)])
                        ellipse.append([lon+off[0], lat+off[1]])
                    ellipse.append(ellipse[0])
                    features.append({
                        "type": "Feature",
                        "properties": {**props, "kind": "observation_cov"},
                        "geometry": {"type": "Polygon", "coordinates": [ellipse]}
                    })
                except np.linalg.LinAlgError: pass

    return features, b


def load_ground_truth(path: Path, bounds: Bounds) -> Tuple[List[Dict], Bounds]:
    """Load ground truth LINESTRING as path polygons."""
    features = []
    b = bounds
    if not path.exists(): return features, b

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            coords_utm = parse_geometry(row.get("geom", ""))
            if not coords_utm: continue
            coords_wgs = reproject_utm_to_wgs84_coords(np.array(coords_utm))
            points_wgs = [[c[0], c[1]] for c in coords_wgs]
            b = update_bounds(b, points_wgs)
            features.append({
                "type": "Feature",
                "properties": {"id": row["id"], "kind": "ground_truth"},
                "geometry": {"type": "LineString", "coordinates": points_wgs}
            })
    return features, b


def load_cmm_results(path: Path, bounds: Bounds) -> Tuple[List[Dict], Bounds]:
    """Load CMM match result points."""
    features = []
    b = bounds
    if not path.exists(): return features, b
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            coords = parse_geometry(row.get("pgeom", ""))
            if not coords: continue
            lon, lat = coords[0]
            b = update_bounds(b, [(lon, lat)])
            props = {
                "id": row.get("id", ""), "kind": "cmm_point",
                "seq": row.get("seq", ""), "cpath": row.get("cpath", ""),
                "trustworthiness": round(float(row.get("trustworthiness", 0)), 4),
                "ep": round(float(row.get("ep", 0)), 4),
                "error": round(float(row.get("error", 0)), 2) if row.get("error") else None,
            }
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]}
            })
    return features, b


def load_edges_geojson(shapefile: Path) -> Dict:
    if not shapefile.exists(): return {"type": "FeatureCollection", "features": []}
    tmp = "/tmp/edges_exp_viz.json"
    cmd = ["ogr2ogr", "-f", "GeoJSON", "-sql", "SELECT fid as id, * FROM edges", tmp, str(shapefile)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(tmp) as f: data = json.load(f)
        os.remove(tmp)
        return data
    except Exception as e:
        print(f"  Warning: road network load failed: {e}")
        return {"type": "FeatureCollection", "features": []}


def render_html(token: str, bounds: Bounds, obs_geo: Dict, truth_geo: Dict,
                cmm_geo: Dict, road_geo: Dict, title: str) -> str:
    bounds_js = json.dumps([[bounds[0], bounds[1]], [bounds[2], bounds[3]]]) if math.isfinite(bounds[0]) else "null"

    tpl = Template("""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/><title>$TITLE</title>
<meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no"/>
<script src="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.css" rel="stylesheet"/>
<style>
body{margin:0;padding:0}#map{position:absolute;top:0;bottom:0;width:100%}
#info{position:absolute;top:12px;left:12px;background:rgba(0,0,0,0.85);color:#fff;padding:14px;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-size:13px;
  border-radius:8px;width:280px;z-index:1;box-shadow:0 4px 12px rgba(0,0,0,0.3)}
.legend-item{display:flex;align-items:center;margin-bottom:4px}
.legend-color{width:12px;height:12px;border-radius:50%;margin-right:8px;border:1px solid #fff}
.style-btn{flex:1;background:#333;color:#999;border:1px solid #555;padding:4px 6px;
  border-radius:4px;cursor:pointer;font-size:11px;transition:all 0.2s;margin:2px}
.style-btn.active{background:#3887be;color:#fff;border-color:#3887be}
.style-btn:hover{background:#444;color:#fff}
</style></head><body>
<div id="map"></div>
<div id="info">
  <strong style="font-size:16px">$TITLE</strong><br/>
  <div style="color:#888;margin:8px 0">Experiment Results Viewer</div>
  <div class="legend-item"><div class="legend-color" style="background:$OBS_COLOR"></div>Observation (FDE clean)</div>
  <div class="legend-item"><div class="legend-color" style="background:$FDE_COLOR"></div>FDE Triggered</div>
  <div class="legend-item"><div class="legend-color" style="background:$CMM_COLOR"></div>CMM Matched</div>
  <div class="legend-item"><div style="width:12px;height:2px;background:#f39c12;margin-right:8px"></div>Ground Truth</div>
  <div class="legend-item"><div style="width:12px;height:2px;background:#888;margin-right:8px"></div>Road Network</div>
  <div style="margin-top:8px">
    <label><input type="checkbox" id="toggle-obs" checked> Observations</label><br/>
    <label><input type="checkbox" id="toggle-cov" checked> Cov Ellipses</label><br/>
    <label><input type="checkbox" id="toggle-truth" checked> Ground Truth</label><br/>
    <label><input type="checkbox" id="toggle-cmm" checked> CMM</label><br/>
    <label><input type="checkbox" id="toggle-road"> Road Network</label>
  </div>
  <div style="margin-top:10px;display:flex;gap:4px">
    <button class="style-btn active" id="btn-sat">Satellite</button>
    <button class="style-btn" id="btn-light">Light</button>
  </div>
</div>
<script>
mapboxgl.accessToken='$TOKEN';
const map=new mapboxgl.Map({container:'map',style:'mapbox://styles/mapbox/satellite-streets-v12',
  center:[110.4,20],zoom:12,pitch:0});
const bounds=$BOUNDS;
const obsGeo=$OBS_GEO;const truthGeo=$TRUTH_GEO;const cmmGeo=$CMM_GEO;const roadGeo=$ROAD_GEO;
const STYLE_S='mapbox://styles/mapbox/satellite-streets-v12';
const STYLE_L='mapbox://styles/mapbox/light-v11';
let currentStyle=STYLE_S;let init=false;

function addLayers(){
  if(!map.getSource('obs'))map.addSource('obs',{type:'geojson',data:obsGeo});
  if(!map.getSource('truth'))map.addSource('truth',{type:'geojson',data:truthGeo});
  if(!map.getSource('cmm'))map.addSource('cmm',{type:'geojson',data:cmmGeo});
  if(!map.getSource('road'))map.addSource('road',{type:'geojson',data:roadGeo});
  if(!map.getLayer('road-layer'))map.addLayer({id:'road-layer',type:'line',source:'road',
    paint:{'line-color':'#888','line-width':['interpolate',['linear'],['zoom'],10,1,16,4],'line-opacity':0.5}});
  if(!map.getLayer('truth-layer'))map.addLayer({id:'truth-layer',type:'line',source:'truth',
    paint:{'line-color':'#f39c12','line-width':3,'line-opacity':0.8}});
  if(!map.getLayer('obs-point'))map.addLayer({id:'obs-point',type:'circle',source:'obs',
    filter:['==',['get','kind'],'observation'],
    paint:{'circle-radius':['interpolate',['linear'],['zoom'],10,2.5,16,7],
      'circle-color':['get','color'],'circle-opacity':0.7,'circle-stroke-width':1,'circle-stroke-color':'#fff'}});
  if(!map.getLayer('obs-cov'))map.addLayer({id:'obs-cov',type:'fill',source:'obs',
    filter:['==',['get','kind'],'observation_cov'],
    paint:{'fill-color':['get','color'],'fill-opacity':0.08}});
  if(!map.getLayer('cmm-point'))map.addLayer({id:'cmm-point',type:'circle',source:'cmm',
    paint:{'circle-radius':['interpolate',['linear'],['zoom'],10,3,16,8],
      'circle-color':'$CMM_COLOR','circle-opacity':0.7,'circle-stroke-width':1.5,'circle-stroke-color':'#fff'}});
}
map.on('style.load',()=>{
  if(!init&&bounds)map.fitBounds(bounds,{padding:50});
  addLayers();
  if(init)return;init=true;
  const popup=new mapboxgl.Popup({closeButton:false,closeOnClick:false});
  function addPopup(layerId){map.on('mouseenter',layerId,(e)=>{
    map.getCanvas().style.cursor='pointer';const p=e.features[0].properties;
    let h=`<strong>$${p.kind||'point'}</strong><br/>`;
    for(const[k,v]of Object.entries(p)){if(v!=null)h+=`<strong>$${k}:</strong> $${typeof v==='number'&&!Number.isInteger(v)?v.toFixed(4):v}<br/>`;}
    popup.setLngLat(e.lngLat).setHTML(h).addTo(map);
  });map.on('mouseleave',layerId,()=>{map.getCanvas().style.cursor='';popup.remove();});}
  addPopup('obs-point');addPopup('cmm-point');
  document.getElementById('toggle-obs').addEventListener('change',e=>{
    map.setLayoutProperty('obs-point','visibility',e.target.checked?'visible':'none');
    map.setLayoutProperty('obs-cov','visibility',e.target.checked&&document.getElementById('toggle-cov').checked?'visible':'none');
  });
  document.getElementById('toggle-cov').addEventListener('change',e=>{
    map.setLayoutProperty('obs-cov','visibility',e.target.checked&&document.getElementById('toggle-obs').checked?'visible':'none');
  });
  document.getElementById('toggle-truth').addEventListener('change',e=>
    map.setLayoutProperty('truth-layer','visibility',e.target.checked?'visible':'none'));
  document.getElementById('toggle-cmm').addEventListener('change',e=>
    map.setLayoutProperty('cmm-point','visibility',e.target.checked?'visible':'none'));
  document.getElementById('toggle-road').addEventListener('change',e=>
    map.setLayoutProperty('road-layer','visibility',e.target.checked?'visible':'none'));
  document.getElementById('btn-sat').addEventListener('click',()=>{currentStyle=STYLE_S;map.setStyle(STYLE_S);
    document.getElementById('btn-sat').className='style-btn active';document.getElementById('btn-light').className='style-btn';});
  document.getElementById('btn-light').addEventListener('click',()=>{currentStyle=STYLE_L;map.setStyle(STYLE_L);
    document.getElementById('btn-light').className='style-btn active';document.getElementById('btn-sat').className='style-btn';});
});
</script></body></html>""")
    return tpl.substitute(TOKEN=token, BOUNDS=bounds_js, TITLE=title,
                          OBS_GEO=json.dumps(obs_geo), TRUTH_GEO=json.dumps(truth_geo),
                          CMM_GEO=json.dumps(cmm_geo), ROAD_GEO=json.dumps(road_geo),
                          OBS_COLOR=COLOR_FDE_CLEAN, FDE_COLOR=COLOR_FDE_DETECTED,
                          CMM_COLOR=COLOR_CMM)


def main():
    parser = argparse.ArgumentParser(description="Experiment Mapbox visualization")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Dataset directory (e.g., experiments/data/sigma_10/no_occlusion/no_fault)")
    parser.add_argument("--cmm-result", type=Path, default=None,
                        help="CMM match result CSV (optional)")
    parser.add_argument("--edges", type=Path, default=Path("input/map/hainan/edges.shp"))
    parser.add_argument("--output", type=Path, default=Path("experiments/output/mapbox_viewer.html"))
    parser.add_argument("--no-road", action="store_true", help="Skip road network (faster)")
    args = parser.parse_args()

    if not MAPBOX_TOKEN:
        print("WARNING: MAPBOX_ACCESS_TOKEN not set. Set it to render Mapbox tiles.")

    obs_path = args.data_dir / "observations.csv"
    gt_path = args.data_dir / "ground_truth.csv"
    if not obs_path.exists():
        raise SystemExit(f"observations.csv not found in {args.data_dir}")

    bounds = initial_bounds()

    print("Loading observations...")
    obs_features, bounds = load_observations(obs_path, bounds)

    print("Loading ground truth...")
    truth_features, bounds = load_ground_truth(gt_path, bounds)

    cmm_features = []
    if args.cmm_result and args.cmm_result.exists():
        print("Loading CMM results...")
        cmm_features, bounds = load_cmm_results(args.cmm_result, bounds)

    road_geo = {"type": "FeatureCollection", "features": []}
    if not args.no_road:
        print("Loading road network...")
        road_geo = load_edges_geojson(args.edges)

    title = f"Experiment: {args.data_dir.name}"

    html = render_html(MAPBOX_TOKEN, bounds,
                       {"type": "FeatureCollection", "features": obs_features},
                       {"type": "FeatureCollection", "features": truth_features},
                       {"type": "FeatureCollection", "features": cmm_features},
                       road_geo, title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"\n  Saved: {args.output}")
    print(f"  Observations: {len(obs_features)} features")
    print(f"  Open with: firefox {args.output}")


if __name__ == "__main__":
    main()
