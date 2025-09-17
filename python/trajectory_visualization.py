#!/usr/bin/env python3
"""
Enhanced Trajectory Visualization Script with Parallel Processing
Visualizes trajectory density using color gradients to show traffic distribution
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import multiprocessing as mp
from functools import partial
import time
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def parse_linestring(geom_str):
    """Parse LINESTRING geometry from string format"""
    if geom_str.startswith('LINESTRING('):
        coords_str = geom_str[11:-1]  # Remove 'LINESTRING(' and ')'
        coords = []
        for coord_pair in coords_str.split(','):
            if coord_pair.strip():
                lon, lat = map(float, coord_pair.strip().split())
                coords.append((lon, lat))
        if len(coords) > 1:
            return LineString(coords)
    return None


def process_chunk_parallel(chunk_data, chunk_id):
    """Process a chunk of trajectory data in parallel"""
    chunk_lines, trajectory_column = chunk_data

    trajectories = []
    trajectory_coords = []

    for _, row in chunk_lines.iterrows():
        geom_str = row[trajectory_column]
        if geom_str != 'LINESTRING()':
            geom = parse_linestring(geom_str)
            if geom is not None:
                trajectories.append({
                    'id': row.get('id', len(trajectories)),
                    'geometry': geom
                })
                trajectory_coords.extend(list(geom.coords))

    print(f"Processed chunk {chunk_id}: {len(trajectories)} trajectories")
    return trajectories, trajectory_coords


def load_trajectories_parallel(file_path, trajectory_column='mgeom',
                               batch_size=10000, max_workers=None):
    """Load trajectories using parallel processing"""
    start_time = time.time()

    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    print(f"Loading trajectories from {file_path}...")
    print(f"Using {max_workers} workers, batch size: {batch_size}")

    # Determine file format
    if file_path.endswith('.csv'):
        separator = ';'
    elif file_path.endswith('.txt'):
        separator = ';'
    else:
        # Try to detect separator
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if ';' in first_line:
                separator = ';'
            else:
                separator = ','

    # Read file in chunks
    total_trajectories = 0
    all_trajectories = []
    all_coords = []

    with mp.Pool(processes=max_workers) as pool:
        chunks = []
        chunk_id = 0

        for chunk in pd.read_csv(file_path, sep=separator, chunksize=batch_size):
            chunks.append((chunk, trajectory_column))
            chunk_id += 1

        print(f"Processing {len(chunks)} chunks in parallel...")

        # Process chunks in parallel
        results = pool.starmap(process_chunk_parallel, [(chunk, i) for i, chunk in enumerate(chunks)])

        # Combine results
        for trajectories, coords in results:
            all_trajectories.extend(trajectories)
            all_coords.extend(coords)
            total_trajectories += len(trajectories)

    processing_time = time.time() - start_time
    print(f"Loaded {total_trajectories} trajectories in {processing_time:.2f} seconds")
    print(f"Performance: {total_trajectories/processing_time:.1f} trajectories/second")

    return all_trajectories, np.array(all_coords) if all_coords else np.array([])


def calculate_trajectory_density_grid(trajectories, grid_size=100):
    """Calculate trajectory density on a grid"""
    print("Calculating trajectory density grid...")

    # Collect all coordinates
    all_coords = []
    for traj in trajectories:
        if isinstance(traj['geometry'], LineString):
            all_coords.extend(list(traj['geometry'].coords))

    if not all_coords:
        return None, None, None

    coords_array = np.array(all_coords)

    # Create grid
    min_lon, max_lon = coords_array[:, 0].min(), coords_array[:, 0].max()
    min_lat, max_lat = coords_array[:, 1].min(), coords_array[:, 1].max()

    # Add small buffer
    lon_buffer = (max_lon - min_lon) * 0.05
    lat_buffer = (max_lat - min_lat) * 0.05

    min_lon -= lon_buffer
    max_lon += lon_buffer
    min_lat -= lat_buffer
    max_lat += lat_buffer

    # Create grid cells
    lon_edges = np.linspace(min_lon, max_lon, grid_size + 1)
    lat_edges = np.linspace(min_lat, max_lat, grid_size + 1)

    # Count trajectories passing through each grid cell
    density_grid = np.zeros((grid_size, grid_size))

    for traj in trajectories:
        if isinstance(traj['geometry'], LineString):
            coords = np.array(list(traj['geometry'].coords))

            # Find which grid cells this trajectory passes through
            for i in range(len(coords) - 1):
                # Simple line rasterization
                start, end = coords[i], coords[i + 1]

                # Interpolate points along the line
                num_points = max(int(np.linalg.norm(end - start) * 1000), 1)
                if num_points > 100:  # Limit interpolation points
                    num_points = 100

                for t in np.linspace(0, 1, num_points):
                    point = start + t * (end - start)

                    # Find grid cell
                    lon_idx = np.searchsorted(lon_edges, point[0]) - 1
                    lat_idx = np.searchsorted(lat_edges, point[1]) - 1

                    if 0 <= lon_idx < grid_size and 0 <= lat_idx < grid_size:
                        density_grid[lat_idx, lon_idx] += 1

    return density_grid, (lon_edges, lat_edges), (min_lon, max_lon, min_lat, max_lat)


def calculate_segment_density(trajectories, spatial_threshold=0.001):
    """Calculate density for road segments using spatial clustering"""
    print("Calculating segment density using spatial clustering...")

    # Collect all line segments
    segments = []
    segment_centers = []

    for traj in trajectories:
        if isinstance(traj['geometry'], LineString):
            coords = list(traj['geometry'].coords)
            for i in range(len(coords) - 1):
                segment = [coords[i], coords[i + 1]]
                center = [(coords[i][0] + coords[i + 1][0]) / 2,
                         (coords[i][1] + coords[i + 1][1]) / 2]
                segments.append(segment)
                segment_centers.append(center)

    if not segments:
        return {}

    segment_centers = np.array(segment_centers)

    # Use DBSCAN to cluster nearby segments
    clustering = DBSCAN(eps=spatial_threshold, min_samples=1)
    labels = clustering.fit_predict(segment_centers)

    # Count segments in each cluster
    cluster_counts = defaultdict(int)
    for label in labels:
        cluster_counts[label] += 1

    # Assign density to each segment
    segment_densities = {}
    for i, segment in enumerate(segments):
        label = labels[i]
        segment_densities[tuple(segment[0]), tuple(segment[1])] = cluster_counts[label]

    return segment_densities


def visualize_density_map(trajectories, edges_shp, output_image='trajectory_density_map.png',
                         color_scheme='hot', alpha=0.8, grid_size=150):
    """Create density-based trajectory visualization"""

    print("Creating density-based visualization...")

    # Load road network
    print("Loading road network...")
    edges_gdf = gpd.read_file(edges_shp)
    print(f"Loaded {len(edges_gdf)} road segments")

    # Calculate segment densities
    segment_densities = calculate_segment_density(trajectories)

    if not segment_densities:
        print("No valid segments found for density calculation!")
        return

    # Get density values for color mapping
    density_values = list(segment_densities.values())
    min_density = min(density_values)
    max_density = max(density_values)

    print(f"Density range: {min_density} - {max_density}")

    # Create color map
    if color_scheme == 'hot':
        cmap = plt.cm.hot
    elif color_scheme == 'plasma':
        cmap = plt.cm.plasma
    elif color_scheme == 'viridis':
        cmap = plt.cm.viridis
    else:
        cmap = plt.cm.Reds

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Plot road network as background
    edges_gdf.plot(ax=ax, color='lightgray', linewidth=0.8, alpha=0.4,
                  label='Road Network', zorder=1)

    # Plot trajectories with density-based colors
    line_collections = []
    colors = []

    for traj in trajectories:
        if isinstance(traj['geometry'], LineString):
            coords = list(traj['geometry'].coords)

            # Create line segments
            segments = []
            segment_colors = []

            for i in range(len(coords) - 1):
                segment = [coords[i], coords[i + 1]]
                segment_key = tuple(coords[i]), tuple(coords[i + 1])

                # Get density for this segment
                density = segment_densities.get(segment_key, min_density)

                # Normalize density to [0, 1]
                normalized_density = (density - min_density) / (max_density - min_density) if max_density > min_density else 0

                segments.append(segment)
                segment_colors.append(cmap(normalized_density))

            if segments:
                lc = LineCollection(segments, colors=segment_colors,
                                 linewidths=2.5, alpha=alpha, zorder=2)
                line_collections.append(lc)
                ax.add_collection(lc)

    # Set plot properties
    ax.set_title(f'Trajectory Density Distribution\n{len(trajectories)} trajectories visualized',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_aspect('equal')

    # Create custom legend for density
    create_density_legend(ax, cmap, min_density, max_density, density_values)

    # Add statistics
    add_statistics_text(ax, density_values, len(trajectories))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Density visualization saved to {output_image}")

    return fig, ax


def create_density_legend(ax, cmap, min_density, max_density, density_values):
    """Create a legend showing density distribution"""

    # Create density bins for legend
    num_bins = 6
    density_bins = np.linspace(min_density, max_density, num_bins + 1)

    # Count trajectories in each density bin
    bin_counts = []
    for i in range(num_bins):
        lower_bound = density_bins[i]
        upper_bound = density_bins[i + 1]

        if i == num_bins - 1:
            count = sum(1 for d in density_values if lower_bound <= d <= upper_bound)
        else:
            count = sum(1 for d in density_values if lower_bound <= d < upper_bound)

        bin_counts.append(count)

    # Create legend elements
    legend_elements = []
    for i in range(num_bins):
        color = cmap(i / (num_bins - 1))
        density_range = f"{density_bins[i]:.0f}-{density_bins[i+1]:.0f}"
        count = bin_counts[i]

        legend_elements.append(plt.Line2D([0], [0], color=color, lw=4,
                                         label=f'{density_range} segments\n({count} segments)'))

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right',
             bbox_to_anchor=(1.15, 1), title='Segment Density\n(trajectories per cluster)',
             fontsize=10, title_fontsize=11)


def add_statistics_text(ax, density_values, num_trajectories):
    """Add statistics text to the plot"""

    if density_values:
        stats_text = f"""Trajectory Statistics:
Total trajectories: {num_trajectories:,}
Mean density: {np.mean(density_values):.1f}
Max density: {np.max(density_values):.0f}
Min density: {np.min(density_values):.0f}
Std density: {np.std(density_values):.1f}"""
    else:
        stats_text = f"Total trajectories: {num_trajectories:,}"

    # Add text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_additional_plots(trajectories, output_prefix):
    """Create additional analysis plots"""

    print("Creating additional analysis plots...")

    # Calculate trajectory lengths
    lengths = []
    for traj in trajectories:
        if isinstance(traj['geometry'], LineString):
            lengths.append(traj['geometry'].length)

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Trajectory length distribution
    if lengths:
        ax1.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Trajectory Length Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Length (degrees)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        ax1.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.4f}')
        ax1.axvline(median_len, color='green', linestyle='--', label=f'Median: {median_len:.4f}')
        ax1.legend()

    # Plot 2: Coordinate density heatmap
    all_coords = []
    for traj in trajectories:
        if isinstance(traj['geometry'], LineString):
            all_coords.extend(list(traj['geometry'].coords))

    if all_coords:
        coords_array = np.array(all_coords)
        ax2.hexbin(coords_array[:, 0], coords_array[:, 1], gridsize=80,
                  cmap='YlOrRd', alpha=0.7, mincnt=1)
        ax2.set_title('GPS Point Density Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_aspect('equal')

        # Add colorbar
        cb = plt.colorbar(ax2.collections[0], ax=ax2, shrink=0.8)
        cb.set_label('Point Count', rotation=270, labelpad=15)

    # Plot 3: Trajectory start/end points
    start_points = []
    end_points = []

    for traj in trajectories:
        if isinstance(traj['geometry'], LineString):
            coords = list(traj['geometry'].coords)
            if coords:
                start_points.append(coords[0])
                end_points.append(coords[-1])

    if start_points and end_points:
        start_array = np.array(start_points)
        end_array = np.array(end_points)

        ax3.scatter(start_array[:, 0], start_array[:, 1],
                   c='green', s=20, alpha=0.6, label='Start Points')
        ax3.scatter(end_array[:, 0], end_array[:, 1],
                   c='red', s=20, alpha=0.6, label='End Points')
        ax3.set_title('Trajectory Start/End Points', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        ax3.legend()
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Processing time statistics
    ax4.text(0.1, 0.9, f'Trajectory Processing Summary',
             transform=ax4.transAxes, fontsize=14, fontweight='bold')

    summary_text = f"""
File processed: {len(trajectories):,} trajectories
Parallel processing: {mp.cpu_count() - 1} workers
Data format: CSV/TXT with semicolon separator
Visualization method: Density-based color mapping
Color scheme: Hot colormap (red = high density)
Alpha transparency: {0.8}

Features:
- Spatial clustering for density calculation
- Automatic color scaling
- Interactive legend with density ranges
- Statistical analysis plots
- High-quality output (300 DPI)
    """

    ax4.text(0.1, 0.7, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Processing Information', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to {output_prefix}_analysis.png")


def main():
    """Main function with command line argument support"""
    import argparse

    parser = argparse.ArgumentParser(description='Trajectory Density Visualization')
    parser.add_argument('--input', '-i',
                       default='/home/dell/Czhang/fmm_sjtugnc/output/mr_fields_all_filtered.csv',
                       help='Input trajectory file (CSV or TXT)')
    parser.add_argument('--edges', '-e',
                       default='/home/dell/Czhang/fmm_sjtugnc/input/map/haikou/edges.shp',
                       help='Road network shapefile')
    parser.add_argument('--output', '-o',
                       default='trajectory_density_map',
                       help='Output file prefix (without extension)')
    parser.add_argument('--column', '-c',
                       default='mgeom',
                       help='Column name containing trajectory geometry')
    parser.add_argument('--color',
                       default='hot',
                       choices=['hot', 'plasma', 'viridis', 'Reds'],
                       help='Color scheme for density visualization')
    parser.add_argument('--alpha', type=float, default=0.8,
                       help='Alpha transparency for trajectories (0-1)')
    parser.add_argument('--grid', type=int, default=150,
                       help='Grid size for density calculation')

    args = parser.parse_args()

    print("Enhanced Trajectory Density Visualization")
    print("=" * 60)

    try:
        # Load trajectories using parallel processing
        trajectories, coords_array = load_trajectories_parallel(
            args.input, args.column, max_workers=mp.cpu_count() - 1
        )

        if not trajectories:
            print("No valid trajectories found!")
            return

        # Create main density visualization
        output_image = f"{args.output}.png"
        visualize_density_map(
            trajectories, args.edges, output_image,
            args.color, args.alpha, args.grid
        )

        # Create additional analysis plots
        create_additional_plots(trajectories, args.output)

        print("\n" + "=" * 60)
        print("Visualization completed successfully!")
        print(f"Main visualization: {output_image}")
        print(f"Analysis plots: {args.output}_analysis.png")

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()