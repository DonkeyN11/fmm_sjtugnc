#!/usr/bin/env python3
"""Visualize mismatch epochs with TW > 0.99 on satellite map with road network."""
import csv, json, re, math
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path(__file__).resolve().parents[2]
PP_RE = re.compile(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', re.I)

def pp(wkt):
    m = PP_RE.search(wkt or '')
    return (float(m.group(1)), float(m.group(2))) if m else None

# ── Step 1: Identify mismatch epochs with TW > 0.99 ──
aligned = BASE / 'data/real_vehicle/hainan_06/processed/aligned.csv'
cmm_out = BASE / 'data/real_vehicle/hainan_06/processed/cmm_result_0729.csv'

gt_lookup = {}
with open(aligned, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter=';'):
        ts = row['timestamp'].strip().rstrip('0').rstrip('.')
        gt = row.get('gt_edge','').strip()
        if gt and gt != '0' and gt != '-1':
            gt_lookup[(row['id'], ts)] = gt

# Collect mismatch data
mismatches = []
with open(cmm_out, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter=';'):
        ts = row['timestamp'].strip().rstrip('0').rstrip('.')
        key = (row['id'], ts)
        gt = gt_lookup.get(key)
        if gt is None: continue
        edge = row.get('opath','').strip()
        try: tw = float(row.get('trustworthiness','') or '0')
        except: tw = 0
        if str(edge) != gt and tw > 0.99:
            og = pp(row.get('ogeom',''))
            pg = pp(row.get('pgeom',''))
            if og is None or pg is None: continue
            # Parse candidates
            candidates = []
            cands_raw = row.get('candidates','()')
            try:
                # Format: ((x,y,ep),(x,y,ep),...)
                cands_clean = cands_raw.strip('()')
                for part in re.findall(r'\(([\d.\-]+),([\d.\-]+),([\d.\-e+]+)\)', cands_raw):
                    candidates.append((float(part[0]), float(part[1]), float(part[2])))
            except: pass
            mismatches.append({
                'id': row['id'], 'seq': row['seq'],
                'lon': og[0], 'lat': og[1],
                'p_lon': pg[0], 'p_lat': pg[1],
                'tw': tw, 'matched_edge': edge, 'gt_edge': gt,
                'candidates': candidates,
            })

print(f"Total mismatch epochs with TW>0.99: {len(mismatches)}")

# ── Step 2: Per-trajectory summary ──
traj_counts = defaultdict(int)
traj_edges = defaultdict(lambda: defaultdict(int))
for m in mismatches:
    traj_counts[m['id']] += 1
    traj_edges[m['id']][(m['gt_edge'], m['matched_edge'])] += 1

print("\nPer-trajectory breakdown:")
for tid in sorted(traj_counts.keys()):
    edges = traj_edges[tid]
    top_pairs = sorted(edges.items(), key=lambda x: -x[1])[:3]
    pair_str = '; '.join(f'gt={gt}→match={me} ({c}x)' for (gt, me), c in top_pairs)
    print(f"  Traj {tid}: {traj_counts[tid]} mismatches — {pair_str}")

# ── Step 3: Get center coordinates ──
all_lons = [m['lon'] for m in mismatches]
all_lats = [m['lat'] for m in mismatches]
center_lon = np.mean(all_lons)
center_lat = np.mean(all_lats)
print(f"\nCenter: {center_lat:.6f}, {center_lon:.6f}")

# ── Step 4: Create folium map ──
try:
    import folium
    from folium import plugins
except ImportError:
    print("folium not available, installing...")
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'folium'], capture_output=True)
    import folium
    from folium import plugins

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=15,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery'
)

# Load road network (spatial filter: ±0.02° around center)
import geopandas as gpd
gdf = gpd.read_file(BASE / 'input/map/hainan/edges.shp')
bbox = (center_lon - 0.02, center_lat - 0.02, center_lon + 0.02, center_lat + 0.02)
gdf_sub = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
print(f"Road edges in view: {len(gdf_sub)}")

# Add road network
folium.GeoJson(
    gdf_sub.__geo_interface__,
    style_function=lambda x: {'color': '#888888', 'weight': 2, 'opacity': 0.6},
    name='Road Network'
).add_to(m)

# Add mismatch points by trajectory with colors
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf']
traj_color = {tid: colors[i % len(colors)] for i, tid in enumerate(sorted(traj_counts.keys()))}

# Group by proximity for clustering
from folium.plugins import MarkerCluster
cluster = MarkerCluster(name='Mismatch GPS obs').add_to(m)

for i, mm in enumerate(mismatches):
    tid = mm['id']
    color = traj_color[tid]

    # GPS observation point
    folium.CircleMarker(
        [mm['lat'], mm['lon']],
        radius=4, color=color, fill=True, fill_opacity=0.7,
        popup=f"Traj={tid} seq={mm['seq']}<br>TW={mm['tw']:.4f}<br>GT edge={mm['gt_edge']}<br>Matched={mm['matched_edge']}<br>GPS: {mm['lon']:.6f},{mm['lat']:.6f}",
        name=f'Traj {tid}'
    ).add_to(cluster)

    # Matched point (small arrow from obs to matched)
    folium.CircleMarker(
        [mm['p_lat'], mm['p_lon']],
        radius=2, color='red', fill=True, fill_opacity=0.5,
        popup=f"Matched point<br>Edge={mm['matched_edge']}"
    ).add_to(m)

    # Line from GPS obs to matched point
    folium.PolyLine(
        [[mm['lat'], mm['lon']], [mm['p_lat'], mm['p_lon']]],
        color='black', weight=0.5, opacity=0.3
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

out_path = BASE / 'experiments/output/8_ablation/mismatch_high_tw_map.html'
m.save(str(out_path))
print(f"\nMap saved to {out_path}")
print(f"Open with: firefox {out_path}")
