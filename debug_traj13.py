#!/usr/bin/env python3
"""
Debug Trajectory 13 candidate selection.
"""

import csv
import re

def parse_candidates(candidates_str):
    """Parse the candidates string."""
    if not candidates_str or candidates_str.strip() == '':
        return []
    pattern = r'\(([^,]+),([^,]+),([^)]+)\)'
    matches = re.findall(pattern, candidates_str)
    candidates = []
    for match in matches:
        try:
            x = float(match[0])
            y = float(match[1])
            prob = float(match[2])
            candidates.append((x, y, prob))
        except ValueError:
            continue
    return candidates

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance in meters."""
    import math
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Read GPS data
print("Reading GPS data for Trajectory 13...")
gps_points = []
with open('dataset-hainan-06/cmm_input_points.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        if row['id'] == '13':
            x = float(row['x'])
            y = float(row['y'])
            gps_points.append((x, y))

print(f"Total GPS points: {len(gps_points)}")
print(f"First GPS point: ({gps_points[0][0]:.6f}, {gps_points[0][1]:.6f})")
print(f"Last GPS point: ({gps_points[-1][0]:.6f}, {gps_points[-1][1]:.6f})")

# Read results
print("\nReading CMM results for Trajectory 13...")
result_points = []
with open('dataset-hainan-06/mr/cmm_results_filtered.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        if row['id'] == '13':
            # Parse observed point
            ogeom = row['ogeom']
            if 'POINT(' in ogeom:
                coords = ogeom.replace('POINT(', '').replace(')', '').strip()
                parts = coords.split()
                obs_lon = float(parts[0])
                obs_lat = float(parts[1])

            # Parse candidates
            candidates = parse_candidates(row['candidates'])
            if candidates:
                first_candidate = candidates[0]
                result_points.append({
                    'observed': (obs_lon, obs_lat),
                    'first_candidate': first_candidate,
                    'all_candidates': candidates
                })

print(f"Total result points: {len(result_points)}")

# Analyze first 20 points
print("\n" + "="*80)
print("ANALYZING FIRST 20 POINTS")
print("="*80)

for i in range(min(20, len(result_points))):
    rp = result_points[i]
    obs = rp['observed']
    cand = rp['first_candidate']
    all_cands = rp['all_candidates']

    # Calculate distance from observed to first candidate
    dist = haversine_distance(obs[0], obs[1], cand[0], cand[1])

    print(f"\nPoint {i}:")
    print(f"  Observed:     ({obs[0]:.6f}, {obs[1]:.6f})")
    print(f"  1st Candid:   ({cand[0]:.6f}, {cand[1]:.6f}) prob={cand[2]:.6f}")
    print(f"  Distance:     {dist:.2f} m")

    if len(all_cands) > 1:
        print(f"  Top 5 candidates:")
        for j, c in enumerate(all_cands[:5]):
            d = haversine_distance(obs[0], obs[1], c[0], c[1])
            print(f"    {j}: ({c[0]:.6f}, {c[1]:.6f}) prob={c[2]:.6f} dist={d:.2f}m")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

distances = []
probs = []
for rp in result_points:
    obs = rp['observed']
    cand = rp['first_candidate']
    dist = haversine_distance(obs[0], obs[1], cand[0], cand[1])
    distances.append(dist)
    probs.append(cand[2])

print(f"\nDistance from observed to first candidate:")
print(f"  Mean: {sum(distances)/len(distances):.2f} m")
print(f"  Max:  {max(distances):.2f} m")
print(f"  Min:  {min(distances):.2f} m")

# Find outlier points
print(f"\nPoints with distance > 100m:")
count = 0
for i, dist in enumerate(distances):
    if dist > 100:
        print(f"  Point {i}: {dist:.2f} m")
        count += 1
        if count >= 10:
            print(f"  ... and {sum(1 for d in distances if d > 100) - 10} more")
            break
