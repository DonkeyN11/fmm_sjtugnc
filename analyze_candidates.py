#!/usr/bin/env python3
"""
Analyze first candidate points vs observed GPS points.
Check if they follow the road network.
"""

import csv
import re
import math
from collections import defaultdict

def parse_candidates(candidates_str):
    """Parse the candidates string to extract list of (x, y, prob) tuples."""
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
    """Calculate distance between two points in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def parse_ogeom(geom_str):
    """Parse WKT POINT string."""
    # Format: POINT(110.44366522 19.98353776)
    if not geom_str or 'POINT' not in geom_str:
        return None, None
    coords = geom_str.replace('POINT(', '').replace(')', '').strip()
    parts = coords.split()
    if len(parts) >= 2:
        return float(parts[0]), float(parts[1])
    return None, None

def analyze_trajectory(csv_file, traj_id):
    """Analyze a single trajectory."""
    data = {
        'observed': [],
        'candidates': [],
        'first_candidates': []
    }

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            if row['id'] != str(traj_id):
                continue

            # Parse observed point
            obs_lon, obs_lat = parse_ogeom(row['ogeom'])
            if obs_lon and obs_lat:
                data['observed'].append((obs_lon, obs_lat))

            # Parse candidates
            candidates = parse_candidates(row['candidates'])
            if candidates:
                data['candidates'].append(candidates)
                data['first_candidates'].append(candidates[0])

    return data

def analyze_continuity(points, name="Points"):
    """Analyze spatial continuity of a point sequence."""
    if len(points) < 2:
        print(f"{name}: Not enough points")
        return

    distances = []
    max_gap = 0
    total_distance = 0

    for i in range(1, len(points)):
        lon1, lat1 = points[i-1][0], points[i-1][1]
        lon2, lat2 = points[i][0], points[i][1]
        dist = haversine_distance(lon1, lat1, lon2, lat2)
        distances.append(dist)
        total_distance += dist
        max_gap = max(max_gap, dist)

    print(f"\n{name}:")
    print(f"  Total points: {len(points)}")
    print(f"  Total distance: {total_distance:.2f} m")
    print(f"  Mean step distance: {sum(distances)/len(distances):.2f} m")
    print(f"  Max step distance: {max_gap:.2f} m")
    print(f"  Min step distance: {min(distances):.2f} m")

    # Detect large gaps
    large_gaps = [(i, d) for i, d in enumerate(distances, 1) if d > 100]
    if large_gaps:
        print(f"  Large gaps (>100m): {len(large_gaps)}")
        if len(large_gaps) <= 10:
            for idx, gap in large_gaps[:5]:
                print(f"    Gap at point {idx}: {gap:.2f} m")

def main():
    csv_file = 'dataset-hainan-06/mr/cmm_results_filtered.csv'

    # Get all unique trajectory IDs
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        traj_ids = sorted(set(row['id'] for row in reader))

    print("="*70)
    print("TRAJECTORY ANALYSIS: Observed GPS vs First Candidates")
    print("="*70)

    for traj_id in traj_ids:
        print(f"\n{'='*70}")
        print(f"TRAJECTORY {traj_id}")
        print(f"{'='*70}")

        data = analyze_trajectory(csv_file, traj_id)

        analyze_continuity(data['observed'], "Observed GPS Points")
        analyze_continuity(data['first_candidates'], "First Candidate Points")

        # Calculate distance from observed to first candidate
        if data['observed'] and data['first_candidates']:
            print("\nDistance from Observed to First Candidate:")
            distances = []
            for (obs_lon, obs_lat), (cand_lon, cand_lat, _) in zip(data['observed'], data['first_candidates']):
                dist = haversine_distance(obs_lon, obs_lat, cand_lon, cand_lat)
                distances.append(dist)

            print(f"  Mean: {sum(distances)/len(distances):.2f} m")
            print(f"  Max:  {max(distances):.2f} m")
            print(f"  Min:  {min(distances):.2f} m")

            # Count points far from observed GPS
            far_points = sum(1 for d in distances if d > 50)
            print(f"  Points >50m from observed: {far_points}/{len(distances)} ({100*far_points/len(distances):.1f}%)")

        # Probability statistics for first candidates
        if data['first_candidates']:
            probs = [p[2] for p in data['first_candidates']]
            print(f"\nFirst Candidate Probabilities:")
            print(f"  Mean: {sum(probs)/len(probs):.6f}")
            print(f"  Max:  {max(probs):.6f}")
            print(f"  Min:  {min(probs):.6f}")

            low_prob = sum(1 for p in probs if p < 0.01)
            print(f"  Points with prob < 0.01: {low_prob}/{len(probs)} ({100*low_prob/len(probs):.1f}%)")

if __name__ == '__main__':
    main()
