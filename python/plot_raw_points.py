#!/usr/bin/env python3
"""
High-performance trajectory point density visualization.
Reads trajectory data, performs density-based clustering, and creates heatmaps on map background.
Optimized for large datasets with parallel processing and memory efficiency.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count, Manager
import psutil
from collections import defaultdict
import argparse
from pathlib import Path
import warnings
import re
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from scipy.ndimage import gaussian_filter
    from sklearn.cluster import DBSCAN
    HAS_REQUIRED_LIBS = True
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Please install: pip install geopandas matplotlib scipy scikit-learn pandas numpy")
    HAS_REQUIRED_LIBS = False

class TrajectoryPointDensity:
    def __init__(self, input_file, output_file=None, shapefile_path=None,
                 pixel_size=0.001, sigma=2.0, min_samples=10, memory_limit_gb=16):
        """
        Initialize the trajectory point density analyzer.

        Args:
            input_file: Path to trajectory data file
            output_file: Path to output image (auto-generated if None)
            shapefile_path: Path to shapefile for map background
            pixel_size: Size of each pixel in degrees (default: 0.001)
            sigma: Gaussian smoothing parameter (default: 2.0)
            min_samples: Minimum samples for DBSCAN clustering (default: 10)
            memory_limit_gb: Memory limit in GB (default: 16)
        """
        if not HAS_REQUIRED_LIBS:
            raise ImportError("Required libraries not available")

        self.input_file = input_file
        self.pixel_size = pixel_size
        self.sigma = sigma
        self.min_samples = min_samples
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024

        # Set output file path
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            self.output_file = f"{os.path.dirname(input_file)}/{base_name}_density_heatmap.png"
        else:
            self.output_file = output_file

        # Set shapefile path
        if shapefile_path is None:
            self.shapefile_path = "/home/dell/Czhang/fmm_sjtugnc/input/map/haikou/edges.shp"
        else:
            self.shapefile_path = shapefile_path

        # Detect file format and trajectory column
        self.file_format, self.trajectory_col, self.delimiter = self.detect_file_format()

        # Load shapefile and get spatial filter
        self.spatial_filter = self.load_spatial_filter()

        # Statistics
        self.stats = {
            'total_lines': 0,
            'points_processed': 0,
            'points_filtered': 0,
            'memory_peak': 0,
            'time_start': time.time(),
            'chunks_processed': 0
        }

        # Bounding box (will be determined from data)
        self.bounds = None
        self.grid_shape = None

    def detect_file_format(self, sample_size=100):
        """
        Automatically detect file format and trajectory column.

        Args:
            sample_size: Number of lines to sample for format detection

        Returns:
            Tuple of (file_format, trajectory_column, delimiter)
        """
        print("Detecting file format...")

        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline().strip() for _ in range(min(sample_size, 10))]

        # Check if it's CSV format (has headers)
        first_line = lines[0] if lines else ""

        # Detect delimiter
        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = ','  # default

        # Try to detect format based on headers
        if delimiter in first_line:
            headers = [h.strip().lower() for h in first_line.split(delimiter)]

            # Look for geometry/trajectory columns
            geom_columns = ['geom', 'geometry', 'pgeom', 'linestring', 'wkt', 'trajectory']
            trajectory_col = None

            for col in geom_columns:
                if col in headers:
                    trajectory_col = headers.index(col)
                    break

            if trajectory_col is not None:
                print(f"Detected CSV format with geometry column '{headers[trajectory_col]}' at index {trajectory_col}")
                return 'csv', trajectory_col, delimiter

        # Check if it's raw trajectory format (no headers, contains coordinates)
        sample_coords = 0
        for line in lines[1:] if delimiter in first_line else lines:  # Skip header if CSV
            parts = line.split(delimiter)
            if len(parts) >= 5:  # ID, timestamp, speed, lon, lat
                try:
                    float(parts[3])  # longitude
                    float(parts[4])  # latitude
                    sample_coords += 1
                except (ValueError, IndexError):
                    continue

        if sample_coords > 0:
            print(f"Detected raw trajectory format (coordinates in columns 3 and 4)")
            return 'raw', None, delimiter

        # Default to raw format
        print("Could not detect format, defaulting to raw trajectory format")
        return 'raw', None, delimiter

    def extract_points_from_linestring(self, linestring_str):
        """
        Extract coordinate points from a LINESTRING WKT string.

        Args:
            linestring_str: LINESTRING WKT string

        Returns:
            List of (longitude, latitude) tuples
        """
        points = []

        # Remove LINESTRING prefix and parentheses
        if linestring_str.startswith('LINESTRING'):
            coords_str = linestring_str.replace('LINESTRING', '').strip('() ')
        elif linestring_str.startswith('POINT'):
            coords_str = linestring_str.replace('POINT', '').strip('() ')
        else:
            return points

        # Split coordinate pairs
        coord_pairs = coords_str.split(',')

        for pair in coord_pairs:
            pair = pair.strip()
            if not pair:
                continue

            try:
                coords = pair.split()
                if len(coords) >= 2:
                    lon = float(coords[0])
                    lat = float(coords[1])
                    points.append([lon, lat])
            except (ValueError, IndexError):
                continue

        return points

    def load_spatial_filter(self):
        """
        Load shapefile and create spatial filter for point filtering.

        Returns:
            Dictionary containing spatial filter information
        """
        print("Loading spatial filter from shapefile...")

        try:
            # Load shapefile
            gdf = gpd.read_file(self.shapefile_path)

            # Create spatial index for faster queries
            if hasattr(gdf, 'sindex'):
                spatial_index = gdf.sindex
                print("Using spatial index for optimized queries")
            else:
                spatial_index = None
                print("Spatial index not available, using bounds-only filtering")

            # Get the combined geometry of all features
            combined_geometry = gdf.union_all()

            # Get bounding box for quick filtering
            bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

            # Create spatial filter
            spatial_filter = {
                'geometry': combined_geometry,
                'bounds': bounds,
                'gdf': gdf,
                'spatial_index': spatial_index
            }

            print(f"Spatial filter loaded: {len(gdf)} features")
            print(f"Bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f}]")

            return spatial_filter

        except Exception as e:
            print(f"Warning: Could not load spatial filter: {e}")
            return None

    def is_point_within_bounds(self, lon, lat):
        """
        Check if point is within the spatial filter bounds.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Boolean indicating if point is within bounds
        """
        if self.spatial_filter is None:
            return True  # No filtering if spatial filter not available

        bounds = self.spatial_filter['bounds']
        return (bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3])

    def is_point_within_geometry(self, lon, lat):
        """
        Check if point is within the shapefile geometry using spatial filter.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Boolean indicating if point is within geometry
        """
        if self.spatial_filter is None:
            return True  # No filtering if spatial filter not available

        try:
            # First check bounds (quick filter)
            if not self.is_point_within_bounds(lon, lat):
                return False

            # Create point geometry
            from shapely.geometry import Point
            point = Point(lon, lat)

            # Use spatial index if available for faster queries
            spatial_index = self.spatial_filter.get('spatial_index')
            if spatial_index is not None:
                # Find potential matches using spatial index
                possible_matches_index = list(spatial_index.intersection(point.bounds))
                if not possible_matches_index:
                    return False

                # Check only the potential matches
                possible_matches = self.spatial_filter['gdf'].iloc[possible_matches_index]
                return any(possible_matches.geometry.contains(point).any())
            else:
                # Fallback to combined geometry check
                return self.spatial_filter['geometry'].contains(point)

        except Exception:
            # Fallback to bounds check if detailed check fails
            return self.is_point_within_bounds(lon, lat)

    def get_file_size(self):
        """Get file size in bytes."""
        return os.path.getsize(self.input_file)

    def count_lines(self):
        """Count total lines in file efficiently."""
        with open(self.input_file, 'rb') as f:
            return sum(1 for _ in f)

    def determine_bounds(self, sample_size=100000):
        """
        Determine geographical bounds from data sample.

        Args:
            sample_size: Number of lines to sample for bounds determination
        """
        print("Determining geographical bounds...")

        min_lon, max_lon = float('inf'), float('-inf')
        min_lat, max_lat = float('inf'), float('-inf')
        valid_points = 0

        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip header if CSV format
            if self.file_format == 'csv':
                next(f)

            for i, line in enumerate(f):
                if i >= sample_size:
                    break

                try:
                    if self.file_format == 'raw':
                        parts = line.strip().split(self.delimiter)
                        if len(parts) >= 5:
                            lon = float(parts[3])
                            lat = float(parts[4])
                    elif self.file_format == 'csv':
                        parts = line.strip().split(self.delimiter)
                        if len(parts) > self.trajectory_col:
                            linestring_str = parts[self.trajectory_col]
                            points = self.extract_points_from_linestring(linestring_str)
                            if points:
                                lon = points[0][0]  # First point
                                lat = points[0][1]  # First point
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue

                    # Validate coordinates (reasonable bounds for China)
                    if (70 <= lon <= 140 and 15 <= lat <= 55):  # China coordinate bounds
                        min_lon = min(min_lon, lon)
                        max_lon = max(max_lon, lon)
                        min_lat = min(min_lat, lat)
                        max_lat = max(max_lat, lat)
                        valid_points += 1

                except (ValueError, IndexError):
                    continue

        if valid_points == 0:
            raise ValueError("No valid coordinate points found in the data")

        # Add small padding
        padding = 0.01
        self.bounds = {
            'min_lon': min_lon - padding,
            'max_lon': max_lon + padding,
            'min_lat': min_lat - padding,
            'max_lat': max_lat + padding
        }

        # Calculate grid dimensions
        width = int((self.bounds['max_lon'] - self.bounds['min_lon']) / self.pixel_size)
        height = int((self.bounds['max_lat'] - self.bounds['min_lat']) / self.pixel_size)
        self.grid_shape = (height, width)

        print(f"Valid points found: {valid_points}")
        print(f"Bounds: {self.bounds}")
        print(f"Grid size: {width} x {height} pixels")

    def process_chunk(self, chunk_lines, chunk_id):
        """
        Process a chunk of lines and return point coordinates.

        Args:
            chunk_lines: List of lines to process
            chunk_id: Chunk identifier

        Returns:
            numpy array of point coordinates
        """
        points = []

        # Skip header line if CSV format
        if self.file_format == 'csv' and chunk_id == 0:
            chunk_lines = chunk_lines[1:]

        for line in chunk_lines:
            line = line.strip()
            if not line:
                continue

            if self.file_format == 'raw':
                # Raw trajectory format: id,timestamp,speed,lon,lat
                try:
                    parts = line.split(self.delimiter)
                    if len(parts) >= 5:
                        lon = float(parts[3])
                        lat = float(parts[4])
                        # Validate coordinates and spatial filter
                        if (70 <= lon <= 140 and 15 <= lat <= 55):
                            if self.is_point_within_geometry(lon, lat):
                                points.append([lon, lat])
                            else:
                                self.stats['points_filtered'] += 1
                except (ValueError, IndexError):
                    continue

            elif self.file_format == 'csv':
                # CSV format with geometry column
                try:
                    parts = line.split(self.delimiter)
                    if len(parts) > self.trajectory_col:
                        linestring_str = parts[self.trajectory_col]
                        line_points = self.extract_points_from_linestring(linestring_str)

                        # Optimize: check bounds first before detailed geometry check
                        if self.spatial_filter is not None:
                            bounds = self.spatial_filter['bounds']
                            filtered_points = []
                            for point in line_points:
                                lon, lat = point
                                # Quick bounds check first
                                if (bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]):
                                    if self.is_point_within_geometry(lon, lat):
                                        filtered_points.append(point)
                                    else:
                                        self.stats['points_filtered'] += 1
                                else:
                                    self.stats['points_filtered'] += 1
                            points.extend(filtered_points)
                        else:
                            points.extend(line_points)
                except (ValueError, IndexError):
                    continue

        return np.array(points, dtype=np.float32) if points else np.array([], dtype=np.float32).reshape(0, 2)

    def create_density_grid(self, all_points):
        """
        Create density grid from point coordinates.

        Args:
            all_points: numpy array of all point coordinates

        Returns:
            density grid as numpy array
        """
        print("Creating density grid...")

        # Convert to pixel coordinates
        lon_pixels = ((all_points[:, 0] - self.bounds['min_lon']) / self.pixel_size).astype(int)
        lat_pixels = ((all_points[:, 1] - self.bounds['min_lat']) / self.pixel_size).astype(int)

        # Ensure coordinates are within bounds
        lon_pixels = np.clip(lon_pixels, 0, self.grid_shape[1] - 1)
        lat_pixels = np.clip(lat_pixels, 0, self.grid_shape[0] - 1)

        # Create density grid
        density_grid = np.zeros(self.grid_shape, dtype=np.float32)

        # Count points per pixel
        for i in range(len(lon_pixels)):
            density_grid[lat_pixels[i], lon_pixels[i]] += 1

        # Apply Gaussian smoothing
        density_grid = gaussian_filter(density_grid, sigma=self.sigma)

        return density_grid

    def process_file_parallel(self, chunk_size=100000):
        """
        Process the entire file in parallel chunks.

        Args:
            chunk_size: Number of lines per chunk

        Returns:
            numpy array of all point coordinates
        """
        print(f"Processing file: {self.input_file}")
        print(f"File size: {self.get_file_size() / (1024*1024):.2f} MB")

        total_lines = self.count_lines()
        self.stats['total_lines'] = total_lines
        print(f"Total lines: {total_lines:,}")

        # Determine optimal chunk size based on memory
        available_memory = psutil.virtual_memory().available
        safe_memory = min(available_memory * 0.6, self.memory_limit)
        avg_line_size = self.get_file_size() / total_lines if total_lines > 0 else 100
        max_lines_in_memory = int(safe_memory / (avg_line_size * 10))  # 10x for overhead
        chunk_size = min(chunk_size, max_lines_in_memory)

        print(f"Using chunk size: {chunk_size:,}")
        print(f"Available memory: {available_memory / (1024*1024):.2f} MB")

        # Process chunks in parallel
        all_points = []
        chunk_lines = []
        chunk_number = 0

        with ThreadPoolExecutor(max_workers=min(cpu_count(), 8)) as executor:
            futures = []

            with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip header if CSV format
                if self.file_format == 'csv':
                    header = f.readline()
                    total_lines -= 1  # Adjust total lines count

                for line_num, line in enumerate(f):
                    chunk_lines.append(line)

                    if len(chunk_lines) >= chunk_size:
                        # Submit chunk for processing
                        future = executor.submit(self.process_chunk, chunk_lines.copy(), chunk_number)
                        futures.append(future)
                        chunk_lines = []
                        chunk_number += 1

                        # Monitor progress
                        if chunk_number % 5 == 0:
                            progress = (line_num / total_lines) * 100
                            print(f"Progress: {progress:.1f}% ({line_num:,}/{total_lines:,} lines)")

                            # Memory monitoring
                            current_memory = psutil.Process().memory_info().rss
                            self.stats['memory_peak'] = max(self.stats['memory_peak'], current_memory)

            # Process remaining lines
            if chunk_lines:
                future = executor.submit(self.process_chunk, chunk_lines, chunk_number)
                futures.append(future)

        # Collect results
        print("Collecting results...")
        for future in futures:
            chunk_points = future.result()
            if len(chunk_points) > 0:
                all_points.append(chunk_points)

        # Combine all points
        if all_points:
            all_points_array = np.vstack(all_points)
            self.stats['points_processed'] = len(all_points_array)
            print(f"Total points processed: {len(all_points_array):,}")
        else:
            all_points_array = np.array([])
            print("No valid points found in data")

        return all_points_array

    def create_visualization(self, density_grid):
        """
        Create visualization with map background and density heatmap.

        Args:
            density_grid: 2D numpy array of density values
        """
        print("Creating visualization...")

        # Load shapefile for map background
        try:
            gdf = gpd.read_file(self.shapefile_path)
            print(f"Loaded shapefile with {len(gdf)} features")
        except Exception as e:
            print(f"Warning: Could not load shapefile: {e}")
            gdf = None

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))

        # Plot map background if available
        if gdf is not None:
            gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.7)

        # Create coordinate arrays for plotting
        lon_coords = np.linspace(self.bounds['min_lon'], self.bounds['max_lon'], self.grid_shape[1])
        lat_coords = np.linspace(self.bounds['min_lat'], self.bounds['max_lat'], self.grid_shape[0])

        # Plot density heatmap
        im = ax.imshow(density_grid, extent=[self.bounds['min_lon'], self.bounds['max_lon'],
                                           self.bounds['min_lat'], self.bounds['max_lat']],
                      origin='lower', cmap='hot', alpha=0.8,
                      norm=colors.PowerNorm(gamma=0.5))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Point Density', rotation=270, labelpad=20)

        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Trajectory Point Density Heatmap\nTotal Points: {self.stats["points_processed"]:,}')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {self.output_file}")

        plt.close()

    def print_stats(self):
        """Print processing statistics."""
        total_time = time.time() - self.stats['time_start']

        print("\n=== Processing Statistics ===")
        print(f"Total lines processed: {self.stats['total_lines']:,}")
        print(f"Points extracted: {self.stats['points_processed']:,}")
        if self.stats['points_filtered'] > 0:
            print(f"Points filtered out: {self.stats['points_filtered']:,}")
            filter_rate = (self.stats['points_filtered'] / (self.stats['points_processed'] + self.stats['points_filtered'])) * 100
            print(f"Filter rate: {filter_rate:.1f}%")
        print(f"Chunks processed: {self.stats['chunks_processed']}")

        if self.stats['memory_peak'] > 0:
            memory_peak_mb = self.stats['memory_peak'] / (1024 * 1024)
            print(f"Peak memory usage: {memory_peak_mb:.2f} MB")

        print(f"Total time: {total_time:.2f} seconds")
        print(f"Lines per second: {self.stats['total_lines'] / total_time:,.0f}")
        print(f"Output file: {self.output_file}")

    def process(self):
        """
        Main processing method.
        """
        try:
            print("Starting trajectory point density analysis...")
            print(f"Input: {self.input_file}")
            print(f"Output: {self.output_file}")
            print(f"Shapefile: {self.shapefile_path}")
            print(f"CPU cores: {cpu_count()}")
            print(f"Detected format: {self.file_format}, delimiter: '{self.delimiter}'")
            if self.file_format == 'csv':
                print(f"Trajectory column: {self.trajectory_col}")

            # Step 1: Determine geographical bounds
            self.determine_bounds()

            # Step 2: Process file and extract points
            all_points = self.process_file_parallel()

            if len(all_points) == 0:
                print("No valid points found in the data")
                return

            # Step 3: Create density grid
            density_grid = self.create_density_grid(all_points)

            # Step 4: Create visualization
            self.create_visualization(density_grid)

            # Print statistics
            self.print_stats()

            return self.output_file

        except Exception as e:
            print(f"Error during processing: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create trajectory point density heatmap')
    parser.add_argument('input_file', help='Path to trajectory data file')
    parser.add_argument('--output', '-o', help='Output image file path')
    parser.add_argument('--shapefile', '-s', help='Path to shapefile for map background')
    parser.add_argument('--pixel-size', '-p', type=float, default=0.001,
                       help='Pixel size in degrees (default: 0.001)')
    parser.add_argument('--sigma', type=float, default=2.0,
                       help='Gaussian smoothing parameter (default: 2.0)')
    parser.add_argument('--memory', '-m', type=float, default=16,
                       help='Memory limit in GB (default: 16)')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    # Create and run processor
    processor = TrajectoryPointDensity(
        input_file=args.input_file,
        output_file=args.output,
        shapefile_path=args.shapefile,
        pixel_size=args.pixel_size,
        memory_limit_gb=args.memory
    )

    try:
        output_path = processor.process()
        print(f"\nProcessing completed successfully!")
        print(f"Heatmap saved to: {output_path}")
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()