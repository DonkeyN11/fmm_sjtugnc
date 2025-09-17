#!/usr/bin/env python3
"""
Simplified Trajectory Visualization Script
Displays road network in gray and trajectories with transparent red overlay
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings('ignore')

# Check available packages
HAS_MATPLOTLIB = False
HAS_SHAPELY = False
HAS_GEOPANDAS = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
    print("matplotlib imported successfully")
except ImportError:
    print("matplotlib not available")

try:
    from shapely.geometry import LineString
    HAS_SHAPELY = True
    print("shapely imported successfully")
except ImportError:
    print("shapely not available")

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
    print("geopandas imported successfully")
except ImportError:
    print("geopandas not available")


def parse_linestring_basic(geom_str):
    """Parse LINESTRING geometry without shapely dependency"""
    if geom_str.startswith('LINESTRING('):
        coords_str = geom_str[11:-1]  # Remove 'LINESTRING(' and ')'
        coords = []
        for coord_pair in coords_str.split(','):
            if coord_pair.strip():
                try:
                    lon, lat = map(float, coord_pair.strip().split())
                    coords.append((lon, lat))
                except:
                    continue
        if len(coords) > 1:
            return coords
    return None


def process_chunk_for_visualization(chunk_data, chunk_id):
    """Process a chunk of trajectory data for visualization"""
    chunk_lines, trajectory_column = chunk_data

    trajectory_lines = []
    trajectory_count = 0

    for _, row in chunk_lines.iterrows():
        geom_str = row[trajectory_column]
        if geom_str != 'LINESTRING()':
            coords = parse_linestring_basic(geom_str)
            if coords is not None:
                trajectory_lines.append(coords)
                trajectory_count += 1

    print(f"Processed chunk {chunk_id}: {trajectory_count} trajectories")
    return trajectory_lines


def load_trajectories_fast(file_path, trajectory_column='mgeom',
                          batch_size=20000, max_workers=None):
    """Load trajectories using parallel processing - optimized for speed"""
    start_time = time.time()

    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    print(f"Loading trajectories from {file_path}...")
    print(f"Using {max_workers} workers, batch size: {batch_size}")

    # Determine file format
    if file_path.endswith('.csv') or file_path.endswith('.txt'):
        separator = ';'
    else:
        # Try to detect separator
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if ';' in first_line:
                separator = ';'
            else:
                separator = ','

    # Read file in chunks and process in parallel
    all_trajectory_lines = []

    with mp.Pool(processes=max_workers) as pool:
        chunks = []
        chunk_id = 0

        for chunk in pd.read_csv(file_path, sep=separator, chunksize=batch_size):
            chunks.append((chunk, trajectory_column))
            chunk_id += 1

        print(f"Processing {len(chunks)} chunks in parallel...")

        # Process chunks in parallel
        results = pool.starmap(process_chunk_for_visualization,
                             [(chunk, i) for i, chunk in enumerate(chunks)])

        # Combine results
        for trajectory_lines in results:
            all_trajectory_lines.extend(trajectory_lines)

    processing_time = time.time() - start_time
    total_trajectories = len(all_trajectory_lines)
    print(f"Loaded {total_trajectories:,} trajectories in {processing_time:.2f} seconds")
    print(f"Performance: {total_trajectories/processing_time:.1f} trajectories/second")

    return all_trajectory_lines


def create_simple_map_visualization(trajectory_lines, edges_shp, output_file='trajectory_map.png',
                                   trajectory_alpha=0.6, trajectory_width=1.5):
    """Create simple map visualization with gray road network and red trajectories"""

    if not HAS_MATPLOTLIB or not HAS_GEOPANDAS:
        print("matplotlib or geopandas not available - cannot create visualization")
        return False

    print("Creating simple map visualization...")
    print(f"Loading road network from {edges_shp}...")

    try:
        # Load road network
        edges_gdf = gpd.read_file(edges_shp)
        print(f"Loaded {len(edges_gdf)} road segments")

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Plot road network in gray
        edges_gdf.plot(ax=ax, color='gray', linewidth=1.0, alpha=0.8, zorder=1)

        # Plot trajectories with transparent red
        trajectory_segments = []
        for line in trajectory_lines:
            if len(line) >= 2:
                # Create line segments for this trajectory
                for i in range(len(line) - 1):
                    segment = [line[i], line[i + 1]]
                    trajectory_segments.append(segment)

        if trajectory_segments:
            # Create line collection for all trajectory segments
            lc = LineCollection(trajectory_segments, colors='red',
                              linewidths=trajectory_width, alpha=trajectory_alpha, zorder=2)
            ax.add_collection(lc)

        # Set plot properties
        ax.set_title(f'Trajectory Map\n{len(trajectory_lines):,} trajectories shown',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add simple legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, alpha=0.8, label='Road Network'),
            Line2D([0], [0], color='red', lw=2, alpha=trajectory_alpha, label='Trajectories')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Map visualization saved to {output_file}")

        return True

    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False


def analyze_trajectory_statistics(trajectory_lines):
    """Analyze and display trajectory statistics"""
    if not trajectory_lines:
        return

    print("\n" + "="*50)
    print("TRAJECTORY STATISTICS")
    print("="*50)

    total_trajectories = len(trajectory_lines)
    print(f"Total trajectories: {total_trajectories:,}")

    # Calculate lengths
    lengths = []
    total_points = 0
    all_coords = []

    for line in trajectory_lines:
        if len(line) >= 2:
            # Calculate approximate length using Euclidean distance
            length = 0
            for i in range(len(line) - 1):
                x1, y1 = line[i]
                x2, y2 = line[i + 1]
                length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            lengths.append(length)
            total_points += len(line)
            all_coords.extend(line)

    if lengths:
        print(f"Mean trajectory length: {np.mean(lengths):.6f} degrees")
        print(f"Median trajectory length: {np.median(lengths):.6f} degrees")
        print(f"Min trajectory length: {np.min(lengths):.6f} degrees")
        print(f"Max trajectory length: {np.max(lengths):.6f} degrees")
        print(f"Total GPS points: {total_points:,}")

    if all_coords:
        coords_array = np.array(all_coords)
        print(f"Coordinate bounds:")
        print(f"  Longitude: {coords_array[:, 0].min():.6f} to {coords_array[:, 0].max():.6f}")
        print(f"  Latitude: {coords_array[:, 1].min():.6f} to {coords_array[:, 1].max():.6f}")

    print("="*50)


def main():
    """Main function with command line support"""
    import argparse

    parser = argparse.ArgumentParser(description='Simple Trajectory Map Visualization')
    parser.add_argument('--input', '-i',
                       default='/home/dell/Czhang/fmm_sjtugnc/output/mr_fields_all_filtered.csv',
                       help='Input trajectory file (CSV or TXT)')
    parser.add_argument('--edges', '-e',
                       default='/home/dell/Czhang/fmm_sjtugnc/input/map/haikou/edges.shp',
                       help='Road network shapefile')
    parser.add_argument('--output', '-o',
                       default='/home/dell/Czhang/fmm_sjtugnc/output/trajectory_density_map',
                       help='Output file prefix (without extension)')
    parser.add_argument('--column', '-c',
                       default='mgeom',
                       help='Column name containing trajectory geometry')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Trajectory transparency (0-1, default: 0.6)')
    parser.add_argument('--width', type=float, default=1.5,
                       help='Trajectory line width (default: 1.5)')
    parser.add_argument('--batch', type=int, default=20000,
                       help='Batch size for processing (default: 20000)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics, no visualization')

    args = parser.parse_args()

    print("Simple Trajectory Map Visualization")
    print("=" * 50)
    print(f"Input file: {args.input}")
    print(f"Road network: {args.edges}")
    print(f"Output: {args.output}.png")
    print(f"Trajectory alpha: {args.alpha}")
    print("=" * 50)

    try:
        # Load trajectories using fast parallel processing
        trajectory_lines = load_trajectories_fast(
            args.input, args.column,
            batch_size=args.batch,
            max_workers=mp.cpu_count() - 1
        )

        if not trajectory_lines:
            print("No valid trajectories found!")
            return

        # Analyze statistics
        analyze_trajectory_statistics(trajectory_lines)

        if not args.stats_only:
            # Create simple map visualization
            output_file = f"{args.output}.png"
            success = create_simple_map_visualization(
                trajectory_lines, args.edges, output_file,
                trajectory_alpha=args.alpha,
                trajectory_width=args.width
            )

            if success:
                print(f"\nVisualization completed successfully!")
                print(f"Output file: {output_file}")
            else:
                print("Visualization failed!")
        else:
            print("\nStatistics only mode - no visualization created")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()