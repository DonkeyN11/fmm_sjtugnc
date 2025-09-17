#!/usr/bin/env python3
"""
High-Performance Trajectory Density Visualization Script
Uses alpha blending to create density maps through trajectory overlap
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# Check available packages
HAS_MATPLOTLIB = False
HAS_SHAPELY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
    print("matplotlib imported successfully")
except ImportError:
    print("matplotlib not available - using basic text output only")

try:
    from shapely.geometry import LineString
    HAS_SHAPELY = True
    print("shapely imported successfully")
except ImportError:
    print("shapely not available - using basic coordinate parsing")


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


def create_density_map_basic(trajectory_lines, bounds=None, resolution=2000):
    """Create basic density map using numpy array for alpha blending"""
    print("Creating density map using alpha blending...")

    if not trajectory_lines:
        return None, None

    # Calculate bounds if not provided
    if bounds is None:
        all_coords = []
        for line in trajectory_lines:
            all_coords.extend(line)

        if not all_coords:
            return None, None

        coords_array = np.array(all_coords)
        min_x, max_x = coords_array[:, 0].min(), coords_array[:, 0].max()
        min_y, max_y = coords_array[:, 1].min(), coords_array[:, 1].max()

        # Add buffer
        x_buffer = (max_x - min_x) * 0.02
        y_buffer = (max_y - min_y) * 0.02
        bounds = (min_x - x_buffer, max_x + x_buffer, min_y - y_buffer, max_y + y_buffer)

    min_x, max_x, min_y, max_y = bounds

    # Create density array
    density_map = np.zeros((resolution, resolution, 4), dtype=np.float32)  # RGBA

    # Convert coordinates to pixel coordinates
    x_scale = resolution / (max_x - min_x)
    y_scale = resolution / (max_y - min_y)

    def coord_to_pixel(x, y):
        px = int((x - min_x) * x_scale)
        py = int((max_y - y) * y_scale)  # Flip y-axis
        return max(0, min(resolution - 1, px)), max(0, min(resolution - 1, py))

    # Process each trajectory with low alpha
    alpha_value = 0.02  # Very low alpha for each trajectory
    red_intensity = 0.8  # Red color intensity

    processed_count = 0
    for line in trajectory_lines:
        if len(line) < 2:
            continue

        # Convert trajectory to pixel coordinates
        pixel_coords = []
        for x, y in line:
            px, py = coord_to_pixel(x, y)
            pixel_coords.append((px, py))

        # Draw trajectory on density map
        for i in range(len(pixel_coords) - 1):
            x1, y1 = pixel_coords[i]
            x2, y2 = pixel_coords[i + 1]

            # Simple line drawing using Bresenham's algorithm
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            x, y = x1, y1
            while True:
                if 0 <= x < resolution and 0 <= y < resolution:
                    # Add red color with alpha
                    density_map[y, x, 0] = min(1.0, density_map[y, x, 0] + red_intensity * alpha_value)
                    density_map[y, x, 3] = min(1.0, density_map[y, x, 3] + alpha_value)

                if x == x2 and y == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count:,} trajectories...")

    print(f"Completed density map with {processed_count:,} trajectories")
    return density_map, bounds


def create_simple_visualization(trajectory_lines, bounds=None, output_file='trajectory_density.png'):
    """Create simple visualization using matplotlib"""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available - cannot create visualization")
        return False

    print("Creating density visualization...")

    # Create density map
    density_map, bounds = create_density_map_basic(trajectory_lines, bounds)

    if density_map is None:
        print("Failed to create density map")
        return False

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Display density map
    ax.imshow(density_map[:, :, :3], extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
             origin='lower', aspect='equal')

    # Set plot properties
    ax.set_title(f'Trajectory Density Map (Alpha Blending)\n{len(trajectory_lines):,} trajectories',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Create custom legend
    create_density_legend_simple(ax, len(trajectory_lines))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_file}")

    return True


def create_density_legend_simple(ax, num_trajectories):
    """Create simple legend explaining the density visualization"""

    legend_text = f"""Trajectory Density Visualization

Method: Alpha Blending
Total Trajectories: {num_trajectories:,}
Color: Red gradient
Density: Darker = Higher traffic
Alpha: 0.02 per trajectory

Interpretation:
• Light areas: Low traffic
• Medium areas: Moderate traffic
• Dark areas: High traffic
• Very dark: Traffic hotspots"""

    # Add text box
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            family='monospace')


def analyze_trajectory_statistics(trajectory_lines):
    """Analyze and display trajectory statistics"""
    if not trajectory_lines:
        return

    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)

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

    print("="*60)


def main():
    """Main function with command line support"""
    import argparse

    parser = argparse.ArgumentParser(description='High-Performance Trajectory Density Visualization')
    parser.add_argument('--input', '-i',
                       default='/home/dell/Czhang/fmm_sjtugnc/output/mr_fields_all_filtered.csv',
                       help='Input trajectory file (CSV or TXT)')
    parser.add_argument('--output', '-o',
                       default='trajectory_density_shade',
                       help='Output file prefix (without extension)')
    parser.add_argument('--column', '-c',
                       default='mgeom',
                       help='Column name containing trajectory geometry')
    parser.add_argument('--resolution', type=int, default=2000,
                       help='Resolution for density map (default: 2000)')
    parser.add_argument('--batch', type=int, default=20000,
                       help='Batch size for processing (default: 20000)')
    parser.add_argument('--alpha', type=float, default=0.02,
                       help='Alpha value for each trajectory (default: 0.02)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics, no visualization')

    args = parser.parse_args()

    print("High-Performance Trajectory Density Visualization")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output prefix: {args.output}")
    print(f"Method: Alpha blending density mapping")
    print("=" * 60)

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
            if HAS_MATPLOTLIB:
                # Create visualization
                output_file = f"{args.output}.png"
                success = create_simple_visualization(trajectory_lines, output_file=output_file)

                if success:
                    print(f"\nVisualization completed successfully!")
                    print(f"Output file: {output_file}")
                else:
                    print("Visualization failed!")
            else:
                print("\nmatplotlib not available - statistics only mode")
                print("To enable visualization, install: pip install matplotlib")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()