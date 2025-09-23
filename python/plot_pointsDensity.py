#!/usr/bin/env python3
"""
Simple fast trajectory point density processing without complex threading.
Based on the optimized version but with simplified multiprocessing.
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
import re
import argparse
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from scipy.ndimage import gaussian_filter
    HAS_GIS_LIBS = True
except ImportError:
    HAS_GIS_LIBS = False
    print("Warning: GIS/plotting libraries not available")

# Pre-compile regex patterns
LINESTRING_PATTERN = re.compile(r'LINESTRING\s*\(([^)]+)\)')
POINT_PATTERN = re.compile(r'POINT\s*\(([^)]+)\)')

class FastTrajectoryProcessor:
    def __init__(self, input_file, output_file=None, shapefile_path=None,
                 pixel_size=0.001, sigma=2.0, workers=None, batch_size=50000):
        self.input_file = input_file
        self.pixel_size = pixel_size
        self.sigma = sigma
        self.batch_size = batch_size

        # Use all available cores by default
        self.num_workers = workers or min(cpu_count(), 64)

        # Set output file path
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            self.output_file = f"{os.path.dirname(input_file)}/{base_name}_density_heatmap_fast.png"
        else:
            self.output_file = output_file

        # Set shapefile path
        self.shapefile_path = shapefile_path or "/home/dell/Czhang/fmm_sjtugnc/input/map/haikou/edges.shp"

        # Detect file format
        self.file_format, self.trajectory_col, self.delimiter = self.detect_file_format()

        # Load shapefile if available
        self.spatial_filter = self.load_spatial_filter() if HAS_GIS_LIBS else None

        # Initialize statistics
        self.stats = {
            'total_lines': 0,
            'points_processed': 0,
            'start_time': time.time()
        }

        print(f"🚀 Fast Trajectory Processor Initialized")
        print(f"   Input: {self.input_file}")
        print(f"   Format: {self.file_format}, delimiter: '{self.delimiter}'")
        print(f"   Workers: {self.num_workers}")
        print(f"   Output: {self.output_file}")

    def detect_file_format(self, sample_size=100):
        """Detect file format and delimiter"""
        print("🔍 Detecting file format...")

        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline().strip() for _ in range(min(sample_size, 10))]

        first_line = lines[0] if lines else ""

        # Detect delimiter
        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = ','

        # Check if CSV with geometry column
        if delimiter in first_line:
            headers = [h.strip().lower() for h in first_line.split(delimiter)]
            geom_columns = ['geom', 'geometry', 'pgeom', 'linestring', 'wkt', 'trajectory']

            for i, header in enumerate(headers):
                if header in geom_columns:
                    print(f"✓ Detected CSV format with geometry column '{header}' at index {i}")
                    return 'csv', i, delimiter

        print("✓ Detected raw trajectory format")
        return 'raw', None, delimiter

    def load_spatial_filter(self):
        """Load spatial filter from shapefile"""
        if not HAS_GIS_LIBS:
            return None

        try:
            print("🗺️ Loading spatial filter...")
            gdf = gpd.read_file(self.shapefile_path)
            bounds = gdf.total_bounds
            combined_geometry = gdf.union_all() if len(gdf) > 1 else gdf.geometry.iloc[0]

            spatial_filter = {
                'geometry': combined_geometry,
                'bounds': bounds,
                'gdf': gdf
            }

            print(f"✓ Spatial filter loaded: {len(gdf)} features")
            return spatial_filter
        except Exception as e:
            print(f"⚠ Could not load spatial filter: {e}")
            return None

    def parse_linestring(self, linestring_str):
        """Parse LINESTRING coordinates"""
        # Fast extraction without regex when possible
        if linestring_str.startswith('LINESTRING'):
            start_idx = linestring_str.find('(')
            end_idx = linestring_str.rfind(')')
            if start_idx != -1 and end_idx != -1:
                coords_str = linestring_str[start_idx + 1:end_idx]
            else:
                coords_str = linestring_str.replace('LINESTRING', '').strip('() ')
        else:
            coords_str = linestring_str.strip('() ')

        if not coords_str:
            return []

        # Parse coordinates
        points = []
        coord_pairs = coords_str.split(',')
        for pair in coord_pairs:
            pair = pair.strip()
            if not pair:
                continue
            coords = pair.split()
            if len(coords) >= 2:
                try:
                    points.append([float(coords[0]), float(coords[1])])
                except (ValueError, IndexError):
                    continue

        return points

    def process_chunk(self, chunk_lines):
        """Process a chunk of lines"""
        points = []

        for line in chunk_lines:
            line = line.strip()
            if not line:
                continue

            if self.file_format == 'raw':
                try:
                    parts = line.split(self.delimiter)
                    if len(parts) >= 5:
                        lon = float(parts[3])
                        lat = float(parts[4])
                        if 70 <= lon <= 140 and 15 <= lat <= 55:  # China bounds
                            points.append([lon, lat])
                except (ValueError, IndexError):
                    continue
            elif self.file_format == 'csv':
                try:
                    parts = line.split(self.delimiter)
                    if len(parts) > self.trajectory_col:
                        line_points = self.parse_linestring(parts[self.trajectory_col])
                        points.extend(line_points)
                except (ValueError, IndexError):
                    continue

        return np.array(points, dtype=np.float32) if points else np.array([], dtype=np.float32).reshape(0, 2)

    def determine_bounds(self, sample_size=200000):
        """Determine geographical bounds from sample data"""
        print("🗺️ Determining geographical bounds...")

        sample_data = []
        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip header if CSV
            if self.file_format == 'csv':
                next(f)

            for i, line in enumerate(f):
                if i >= sample_size:
                    break

                if self.file_format == 'raw':
                    parts = line.strip().split(self.delimiter)
                    if len(parts) >= 5:
                        try:
                            lon = float(parts[3])
                            lat = float(parts[4])
                            if 70 <= lon <= 140 and 15 <= lat <= 55:
                                sample_data.append([lon, lat])
                        except (ValueError, IndexError):
                            continue
                elif self.file_format == 'csv':
                    parts = line.strip().split(self.delimiter)
                    if len(parts) > self.trajectory_col:
                        points = self.parse_linestring(parts[self.trajectory_col])
                        if points:
                            sample_data.append(points[0])

        if not sample_data:
            raise ValueError("No valid coordinate points found")

        sample_array = np.array(sample_data, dtype=np.float32)
        min_lon, max_lon = np.min(sample_array[:, 0]), np.max(sample_array[:, 0])
        min_lat, max_lat = np.min(sample_array[:, 1]), np.max(sample_array[:, 1])

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

        print(f"✓ Bounds determined: {self.bounds}")
        print(f"✓ Grid size: {width} x {height} pixels")

    def read_file_batches(self, batch_size=50000):
        """Stream file in batches without loading everything into memory"""
        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            if self.file_format == 'csv':
                # Skip header but keep it available for downstream use if needed
                next(f, None)

            batch_lines = []
            for line in f:
                stripped = line.rstrip('\n')
                if not stripped:
                    continue

                batch_lines.append(stripped)
                if len(batch_lines) >= batch_size:
                    yield batch_lines
                    batch_lines = []

            if batch_lines:
                yield batch_lines

    def update_density_grid(self, density_grid, points):
        """Update density grid with new points"""
        if len(points) == 0:
            return

        # Convert to pixel coordinates
        lon_pixels = ((points[:, 0] - self.bounds['min_lon']) / self.pixel_size).astype(int)
        lat_pixels = ((points[:, 1] - self.bounds['min_lat']) / self.pixel_size).astype(int)

        # Clip to grid bounds
        lon_pixels = np.clip(lon_pixels, 0, self.grid_shape[1] - 1)
        lat_pixels = np.clip(lat_pixels, 0, self.grid_shape[0] - 1)

        # Fast frequency counting scoped to touched cells only
        flat_indices = lat_pixels * self.grid_shape[1] + lon_pixels
        unique_indices, counts = np.unique(flat_indices, return_counts=True)
        density_grid.ravel()[unique_indices] += counts

    def process(self):
        """Main processing method"""
        try:
            start_time = time.time()

            # Step 1: Determine bounds
            self.determine_bounds()

            print("⚡ Starting parallel processing...")
            density_grid = np.zeros(self.grid_shape, dtype=np.float32)

            total_lines = 0

            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                pending = set()
                max_pending = max(1, self.num_workers * 2)

                for batch_lines in self.read_file_batches(self.batch_size):
                    total_lines += len(batch_lines)

                    # Split batch into chunks for parallel processing
                    chunk_size = max(1, len(batch_lines) // self.num_workers)
                    chunks = [batch_lines[i:i + chunk_size] for i in range(0, len(batch_lines), chunk_size)]

                    for chunk in chunks:
                        future = executor.submit(self.process_chunk, chunk)
                        pending.add(future)

                        if len(pending) >= max_pending:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for completed in done:
                                self._integrate_future_result(completed, density_grid)

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for completed in done:
                        self._integrate_future_result(completed, density_grid)

            self.stats['total_lines'] = total_lines
            print(f"📊 Total lines: {total_lines:,}")

            # Step 4: Apply smoothing
            if HAS_GIS_LIBS:
                density_grid = gaussian_filter(density_grid, sigma=self.sigma)

            # Step 5: Create visualization
            if HAS_GIS_LIBS:
                self.create_visualization(density_grid)

            # Print statistics
            elapsed_time = time.time() - start_time
            print(f"\n🎉 Processing completed!")
            print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
            print(f"📊 Points processed: {self.stats['points_processed']:,}")
            print(f"⚡ Processing speed: {self.stats['points_processed']/elapsed_time:,.0f} points/sec")
            print(f"💾 Output saved to: {self.output_file}")

            return self.output_file

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise

    def _integrate_future_result(self, future, density_grid):
        """Safely merge results from a completed worker"""
        try:
            chunk_points = future.result()
            if len(chunk_points) > 0:
                self.update_density_grid(density_grid, chunk_points)
                self.stats['points_processed'] += len(chunk_points)
        except Exception as e:
            print(f"⚠ Error processing chunk: {e}")

    def create_visualization(self, density_grid):
        """Create visualization"""
        print("🎨 Creating visualization...")

        # Load shapefile for map background
        try:
            gdf = gpd.read_file(self.shapefile_path)
            print(f"✓ Loaded shapefile with {len(gdf)} features")
        except Exception as e:
            print(f"⚠ Could not load shapefile: {e}")
            gdf = None

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))

        # Plot map background if available
        if gdf is not None:
            gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.7)

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

        # Save figure
        plt.savefig(self.output_file, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Fast trajectory point density processing')
    parser.add_argument('input_file', help='Path to trajectory data file')
    parser.add_argument('--output', '-o', help='Output image file path')
    parser.add_argument('--shapefile', '-s', help='Path to shapefile for map background')
    parser.add_argument('--pixel-size', '-p', type=float, default=0.001, help='Pixel size in degrees')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian smoothing parameter')
    parser.add_argument('--workers', '-w', type=int, help='Number of worker processes')
    parser.add_argument('--batch-size', type=int, default=50000, help='Batch size for processing')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    try:
        processor = FastTrajectoryProcessor(
            input_file=args.input_file,
            output_file=args.output,
            shapefile_path=args.shapefile,
            pixel_size=args.pixel_size,
            sigma=args.sigma,
            workers=args.workers,
            batch_size=args.batch_size
        )

        output_path = processor.process()
        print(f"✅ Success! Heatmap saved to: {output_path}")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
