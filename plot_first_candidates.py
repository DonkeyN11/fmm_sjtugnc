#!/usr/bin/env python3
"""
Extract first candidate points from CMM results and plot on map.
"""

import csv
import re
import folium
from collections import defaultdict

def parse_candidates(candidates_str):
    """Parse the candidates string to extract list of (x, y, prob) tuples."""
    if not candidates_str or candidates_str.strip() == '':
        return []

    # Extract all coordinate tuples using regex
    # Pattern matches: (x,y,prob)
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

def extract_first_candidates(csv_file):
    """Extract first candidate point for each trajectory sequence."""
    traj_points = defaultdict(list)  # traj_id -> list of (x, y, prob)

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            traj_id = row['id']
            candidates_str = row['candidates']

            candidates = parse_candidates(candidates_str)
            if candidates:
                # Get first candidate
                first = candidates[0]
                traj_points[traj_id].append(first)

    return traj_points

def create_map(traj_points, road_network_shp=None):
    """Create folium map with first candidate points."""
    # Calculate center from all points
    all_points = []
    for points in traj_points.values():
        all_points.extend(points)

    if not all_points:
        print("No points to plot!")
        return None

    # Calculate center (median to be robust)
    lons = [p[0] for p in all_points]
    lats = [p[1] for p in all_points]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                   tiles='OpenStreetMap')

    # Add road network if provided
    if road_network_shp:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(road_network_shp)
            folium.GeoJson(
                gdf.__geo_interface__,
                style_function=lambda x: {
                    'color': '#888888',
                    'weight': 2,
                    'opacity': 0.5
                },
                name='Road Network'
            ).add_to(m)
        except ImportError:
            print("geopandas not available, skipping road network visualization")
        except Exception as e:
            print(f"Could not load road network: {e}")

    # Plot each trajectory in a different color
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
              'darkblue', 'darkgreen', 'pink', 'gray']

    for idx, (traj_id, points) in enumerate(traj_points.items()):
        color = colors[idx % len(colors)]

        # Add markers and lines for this trajectory
        for i, (x, y, prob) in enumerate(points):
            # Marker with popup showing probability
            popup_text = f"Traj {traj_id}<br>Point {i}<br>Prob: {prob:.6f}<br>Coords: ({x:.6f}, {y:.6f})"

            folium.CircleMarker(
                location=[y, x],
                radius=4,
                popup=folium.Popup(popup_text, max_width=200),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(m)

        # Connect points with polyline
        if len(points) > 1:
            coords = [[p[1], p[0]] for p in points]  # folium uses [lat, lon]
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=2,
                opacity=0.8,
                popup=f"Trajectory {traj_id}"
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m

def main():
    csv_file = 'dataset-hainan-06/mr/cmm_results_filtered.csv'
    road_network = 'input/map/hainan/edges.shp'

    print(f"Reading {csv_file}...")
    traj_points = extract_first_candidates(csv_file)

    print(f"\nExtracted first candidates for {len(traj_points)} trajectories:")
    for traj_id, points in sorted(traj_points.items()):
        print(f"  Trajectory {traj_id}: {len(points)} points")
        if points:
            print(f"    First point: ({points[0][0]:.6f}, {points[0][1]:.6f}), prob={points[0][2]:.6f}")
            print(f"    Last point:  ({points[-1][0]:.6f}, {points[-1][1]:.6f}), prob={points[-1][2]:.6f}")

    # Create map
    print("\nCreating map...")
    m = create_map(traj_points, road_network)

    if m:
        output_file = '/tmp/first_candidates_map.html'
        m.save(output_file)
        print(f"Map saved to: {output_file}")
        print("\nYou can open the map in a web browser:")
        print(f"  firefox {output_file}")
        print(f"  or")
        print(f"  xdg-open {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    all_probs = []
    for points in traj_points.values():
        all_probs.extend([p[2] for p in points])

    if all_probs:
        print(f"Total points: {len(all_probs)}")
        print(f"Max probability: {max(all_probs):.6f}")
        print(f"Min probability: {min(all_probs):.6f}")
        print(f"Mean probability: {sum(all_probs)/len(all_probs):.6f}")

        # Count low probability points
        low_prob_threshold = 0.5
        low_prob_count = sum(1 for p in all_probs if p < low_prob_threshold)
        print(f"\nPoints with prob < {low_prob_threshold}: {low_prob_count}/{len(all_probs)} ({100*low_prob_count/len(all_probs):.1f}%)")

if __name__ == '__main__':
    main()
