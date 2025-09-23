#!/usr/bin/env python3
"""
High-performance trajectory point density visualization - OPTIMIZED VERSION.
Implements all 6 optimization strategies:
1. Vectorized computation using numpy
2. Streaming memory processing
3. ProcessPoolExecutor parallel processing
4. LRU caching for repeated computations
5. Batch I/O optimization
6. Fast density grid calculation with histogram2d
7. REAL-TIME PERFORMANCE VISUALIZATION with progress bars and speed metrics
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager, shared_memory
import psutil
from collections import defaultdict, deque
import argparse
from pathlib import Path
import warnings
import re
from functools import lru_cache
from typing import Tuple, List, Optional
import threading
from datetime import datetime
warnings.filterwarnings('ignore')

# Pre-compile regex patterns for performance
COORD_PATTERN = re.compile(r'[-+]?\d*\.\d+|\d+')
LINESTRING_PATTERN = re.compile(r'LINESTRING\s*\(([^)]+)\)')
POINT_PATTERN = re.compile(r'POINT\s*\(([^)]+)\)')

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    END = '\033[0m'

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


class RealTimeMonitor:
    """Real-time performance monitoring and visualization"""

    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.stats_queue = deque(maxlen=1000)
        self.start_time = time.time()
        self.last_update = time.time()

        # Performance metrics
        self.total_lines = 0
        self.processed_lines = 0
        self.processed_points = 0
        self.memory_usage = 0
        self.cache_hits = 0
        self.io_operations = 0

        # Speed tracking
        self.speed_history = deque(maxlen=50)
        self.point_speed_history = deque(maxlen=50)

    def update_stats(self, **kwargs):
        """Update statistics"""
        timestamp = time.time()
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Calculate current speeds
        elapsed = timestamp - self.last_update
        if elapsed >= self.update_interval:
            lines_per_sec = (kwargs.get('processed_lines', self.processed_lines) -
                           getattr(self, '_last_processed_lines', 0)) / elapsed
            points_per_sec = (kwargs.get('processed_points', self.processed_points) -
                             getattr(self, '_last_processed_points', 0)) / elapsed

            self.speed_history.append(lines_per_sec)
            self.point_speed_history.append(points_per_sec)

            self._last_processed_lines = self.processed_lines
            self._last_processed_points = self.processed_points
            self.last_update = timestamp

            # Update memory usage
            try:
                self.memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            except:
                pass

            self.stats_queue.append({
                'timestamp': timestamp,
                'processed_lines': self.processed_lines,
                'processed_points': self.processed_points,
                'memory_usage': self.memory_usage,
                'lines_per_sec': lines_per_sec,
                'points_per_sec': points_per_sec,
                'cache_hits': self.cache_hits,
                'io_operations': self.io_operations
            })

    def start(self):
        """Start monitoring thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def _monitor_loop(self):
        """Monitoring loop - displays real-time stats"""
        import shutil

        while self.running:
            time.sleep(self.update_interval)

            if not self.stats_queue:
                continue

            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')

            # Get latest stats
            latest = self.stats_queue[-1]

            # Calculate progress
            progress = (latest['processed_lines'] / self.total_lines * 100) if self.total_lines > 0 else 0
            elapsed = latest['timestamp'] - self.start_time

            # Calculate average speeds
            avg_lines_per_sec = latest['processed_lines'] / elapsed if elapsed > 0 else 0
            avg_points_per_sec = latest['processed_points'] / elapsed if elapsed > 0 else 0

            # Calculate ETA
            if progress > 0 and progress < 100:
                remaining_lines = self.total_lines - latest['processed_lines']
                eta_seconds = remaining_lines / avg_lines_per_sec if avg_lines_per_sec > 0 else 0
                eta_time = datetime.fromtimestamp(time.time() + eta_seconds).strftime('%H:%M:%S')
            else:
                eta_time = "N/A"

            # Get terminal width
            try:
                terminal_width = shutil.get_terminal_size().columns
            except:
                terminal_width = 80

            # Display header
            print(f"{Colors.BOLD}{Colors.CYAN}{'=' * terminal_width}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.CYAN}  TRAJECTORY POINT DENSITY ANALYSIS - REAL-TIME MONITOR{Colors.END}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'=' * terminal_width}{Colors.END}")
            print()

            # Progress bar
            bar_width = min(60, terminal_width - 40)
            filled_width = int(bar_width * progress / 100)
            bar = '█' * filled_width + '░' * (bar_width - filled_width)
            print(f"{Colors.BOLD}Progress:{Colors.END} [{bar}] {progress:.1f}% ({latest['processed_lines']:,}/{self.total_lines:,})")
            print(f"{Colors.BOLD}ETA:{Colors.END} {eta_time} | {Colors.BOLD}Elapsed:{Colors.END} {elapsed:.1f}s")
            print()

            # Speed metrics
            print(f"{Colors.BOLD}{Colors.YELLOW}🚀 PERFORMANCE METRICS{Colors.END}")
            print(f"  {Colors.GREEN}Current Speed:{Colors.END} {latest['lines_per_sec']:,.0f} lines/sec | {latest['points_per_sec']:,.0f} points/sec")
            print(f"  {Colors.BLUE}Average Speed:{Colors.END} {avg_lines_per_sec:,.0f} lines/sec | {avg_points_per_sec:,.0f} points/sec")

            # Speed trend
            if len(self.speed_history) >= 5:
                recent_speeds = list(self.speed_history)[-5:]
                if recent_speeds[-1] > recent_speeds[0] * 1.1:
                    trend = f"{Colors.GREEN}⬆ Accelerating{Colors.END}"
                elif recent_speeds[-1] < recent_speeds[0] * 0.9:
                    trend = f"{Colors.RED}⬇ Decelerating{Colors.END}"
                else:
                    trend = f"{Colors.YELLOW}➡ Stable{Colors.END}"
                print(f"  {Colors.MAGENTA}Trend:{Colors.END} {trend}")

            print()

            # Memory usage
            memory_percent = (self.memory_usage / (self.memory_usage + 100)) * 100 if self.memory_usage > 0 else 0
            print(f"{Colors.BOLD}{Colors.YELLOW}💾 MEMORY USAGE{Colors.END}")
            print(f"  {Colors.CYAN}Current:{Colors.END} {self.memory_usage:.1f} MB")

            # Memory bar
            memory_bar_width = min(30, terminal_width - 50)
            memory_filled = int(memory_bar_width * min(memory_percent / 100, 1.0))
            memory_bar = '█' * memory_filled + '░' * (memory_bar_width - memory_filled)
            print(f"  {Colors.CYAN}Usage:{Colors.END} [{memory_bar}] {memory_percent:.1f}%")
            print()

            # System stats
            print(f"{Colors.BOLD}{Colors.YELLOW}📊 SYSTEM STATISTICS{Colors.END}")
            print(f"  {Colors.WHITE}Cache Hits:{Colors.END} {self.cache_hits:,}")
            print(f"  {Colors.WHITE}I/O Operations:{Colors.END} {self.io_operations:,}")
            print(f"  {Colors.WHITE}CPU Cores:{Colors.END} {cpu_count()}")

            # CPU usage
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                print(f"  {Colors.WHITE}CPU Usage:{Colors.END} {cpu_percent:.1f}%")
            except:
                pass

            print()

            # Performance indicators
            print(f"{Colors.BOLD}{Colors.YELLOW}⚡ PERFORMANCE INDICATORS{Colors.END}")

            # Lines per second rating
            if avg_lines_per_sec > 100000:
                line_rating = f"{Colors.GREEN}EXCELLENT (>100K/s){Colors.END}"
            elif avg_lines_per_sec > 50000:
                line_rating = f"{Colors.CYAN}VERY GOOD (>50K/s){Colors.END}"
            elif avg_lines_per_sec > 20000:
                line_rating = f"{Colors.YELLOW}GOOD (>20K/s){Colors.END}"
            elif avg_lines_per_sec > 10000:
                line_rating = f"{Colors.MAGENTA}FAIR (>10K/s){Colors.END}"
            else:
                line_rating = f"{Colors.RED}SLOW (<10K/s){Colors.END}"

            print(f"  {Colors.WHITE}Processing Speed:{Colors.END} {line_rating}")

            # Memory efficiency
            if self.memory_usage < 1000:
                memory_rating = f"{Colors.GREEN}EXCELLENT (<1GB){Colors.END}"
            elif self.memory_usage < 2000:
                memory_rating = f"{Colors.CYAN}GOOD (<2GB){Colors.END}"
            elif self.memory_usage < 4000:
                memory_rating = f"{Colors.YELLOW}FAIR (<4GB){Colors.END}"
            else:
                memory_rating = f"{Colors.RED}HIGH (>4GB){Colors.END}"

            print(f"  {Colors.WHITE}Memory Efficiency:{Colors.END} {memory_rating}")

            print()
            print(f"{Colors.BOLD}{Colors.CYAN}{'=' * terminal_width}{Colors.END}")
            print(f"{Colors.DIM}Press Ctrl+C to interrupt processing...{Colors.END}")

    def get_final_stats(self):
        """Get final performance summary"""
        if not self.stats_queue:
            return {}

        elapsed = time.time() - self.start_time
        latest = self.stats_queue[-1]

        return {
            'total_time': elapsed,
            'avg_lines_per_sec': latest['processed_lines'] / elapsed if elapsed > 0 else 0,
            'avg_points_per_sec': latest['processed_points'] / elapsed if elapsed > 0 else 0,
            'peak_memory': max([s['memory_usage'] for s in self.stats_queue]) if self.stats_queue else 0,
            'total_cache_hits': self.cache_hits,
            'total_io_operations': self.io_operations
        }


class OptimizedTrajectoryPointDensity:
    def __init__(self, input_file, output_file=None, shapefile_path=None,
                 pixel_size=0.001, sigma=2.0, min_samples=10, memory_limit_gb=16):
        """
        Initialize optimized trajectory point density analyzer.
        """
        if not HAS_REQUIRED_LIBS:
            raise ImportError("Required libraries not available")

        self.input_file = input_file
        self.pixel_size = pixel_size
        self.sigma = sigma
        self.min_samples = min_samples
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024

        # Performance optimization settings
        self.num_workers = min(cpu_count(), 32)  # Limit to prevent memory overload
        self.batch_size = 50000  # Optimal batch size for I/O
        self.chunk_size = 200000  # Larger chunks for better parallelization

        # Cache for repeated computations
        self.parse_cache = {}
        self.bounds_cache = None

        # Real-time monitoring
        self.monitor = RealTimeMonitor()
        self.monitor.total_lines = 0

        # Set output file path
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            self.output_file = f"{os.path.dirname(input_file)}/{base_name}_density_heatmap_optimized.png"
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
            'chunks_processed': 0,
            'cache_hits': 0,
            'batch_io_operations': 0
        }

        # Bounding box (will be determined from data)
        self.bounds = None
        self.grid_shape = None

    def detect_file_format(self, sample_size=100):
        """Optimized file format detection."""
        print(f"{Colors.BOLD}{Colors.BLUE}🔍 Detecting file format...{Colors.END}")

        # Use optimized reading
        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline().strip() for _ in range(min(sample_size, 10))]

        first_line = lines[0] if lines else ""

        # Optimized delimiter detection
        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            delimiter = ','

        # Vectorized header analysis
        if delimiter in first_line:
            headers = np.array([h.strip().lower() for h in first_line.split(delimiter)])

            # Vectorized column search
            geom_columns = ['geom', 'geometry', 'pgeom', 'linestring', 'wkt', 'trajectory']
            mask = np.isin(headers, geom_columns)

            if np.any(mask):
                trajectory_col = np.where(mask)[0][0]
                print(f"{Colors.GREEN}✓{Colors.END} Detected CSV format with geometry column '{headers[trajectory_col]}' at index {trajectory_col}")
                return 'csv', int(trajectory_col), delimiter

        # Fast coordinate detection using vectorized operations
        sample_coords = 0
        test_lines = lines[1:] if delimiter in first_line else lines

        for line in test_lines:
            parts = line.split(delimiter)
            if len(parts) >= 5:
                try:
                    # Fast float conversion check
                    if parts[3].replace('.', '', 1).replace('-', '', 1).isdigit() and \
                       parts[4].replace('.', '', 1).replace('-', '', 1).isdigit():
                        sample_coords += 1
                except (ValueError, IndexError):
                    continue

        if sample_coords > 0:
            print(f"{Colors.GREEN}✓{Colors.END} Detected raw trajectory format (coordinates in columns 3 and 4)")
            return 'raw', None, delimiter

        print(f"{Colors.YELLOW}!{Colors.END} Could not detect format, defaulting to raw trajectory format")
        return 'raw', None, delimiter

    @lru_cache(maxsize=5000)
    def parse_linestring_cached(self, linestring_str):
        """
        Cached optimized linestring parsing.
        """
        return self._parse_linestring_fast(linestring_str)

    def _parse_linestring_fast(self, linestring_str):
        """
        Fast linestring parsing using pre-compiled regex and vectorized operations.
        """
        # Use cached result if available
        cache_key = hash(linestring_str)
        if cache_key in self.parse_cache:
            self.stats['cache_hits'] += 1
            self.monitor.update_stats(cache_hits=self.stats['cache_hits'])
            return self.parse_cache[cache_key]

        points = []

        # Fast extraction using regex
        if linestring_str.startswith('LINESTRING'):
            coords_match = LINESTRING_PATTERN.search(linestring_str)
            if coords_match:
                coords_str = coords_match.group(1)
            else:
                coords_str = linestring_str.replace('LINESTRING', '').strip('() ')
        elif linestring_str.startswith('POINT'):
            coords_match = POINT_PATTERN.search(linestring_str)
            if coords_match:
                coords_str = coords_match.group(1)
            else:
                coords_str = linestring_str.replace('POINT', '').strip('() ')
        else:
            return points

        # Vectorized coordinate extraction
        coord_pairs = coords_str.split(',')
        for pair in coord_pairs:
            pair = pair.strip()
            if not pair:
                continue

            coords = pair.split()
            if len(coords) >= 2:
                try:
                    lon = float(coords[0])
                    lat = float(coords[1])
                    points.append([lon, lat])
                except (ValueError, IndexError):
                    continue

        # Cache result
        if len(self.parse_cache) < 10000:  # Limit cache size
            self.parse_cache[cache_key] = points

        return points

    def load_spatial_filter(self):
        """Optimized spatial filter loading with caching."""
        print(f"{Colors.BOLD}{Colors.BLUE}🗺️  Loading spatial filter from shapefile...{Colors.END}")

        try:
            # Load shapefile
            gdf = gpd.read_file(self.shapefile_path)

            # Create spatial index for faster queries
            if hasattr(gdf, 'sindex'):
                spatial_index = gdf.sindex
                print(f"{Colors.GREEN}✓{Colors.END} Using spatial index for optimized queries")
            else:
                spatial_index = None
                print(f"{Colors.YELLOW}!{Colors.END} Spatial index not available, using bounds-only filtering")

            # Vectorized bounds calculation
            bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

            # Optimized geometry union
            if len(gdf) > 1:
                combined_geometry = gdf.unary_union
            else:
                combined_geometry = gdf.geometry.iloc[0]

            spatial_filter = {
                'geometry': combined_geometry,
                'bounds': bounds,
                'gdf': gdf,
                'spatial_index': spatial_index
            }

            print(f"{Colors.GREEN}✓{Colors.END} Spatial filter loaded: {len(gdf)} features")
            print(f"{Colors.CYAN}  Bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f}]{Colors.END}")

            return spatial_filter

        except Exception as e:
            print(f"{Colors.RED}✗{Colors.END} Warning: Could not load spatial filter: {e}")
            return None

    def is_point_within_bounds_vectorized(self, points_array):
        """
        Vectorized bounds checking for multiple points.
        """
        if self.spatial_filter is None:
            return np.ones(len(points_array), dtype=bool)

        bounds = self.spatial_filter['bounds']
        mask = (
            (points_array[:, 0] >= bounds[0]) &
            (points_array[:, 0] <= bounds[2]) &
            (points_array[:, 1] >= bounds[1]) &
            (points_array[:, 1] <= bounds[3])
        )
        return mask

    def get_file_size(self):
        """Get file size in bytes."""
        return os.path.getsize(self.input_file)

    def count_lines_optimized(self):
        """Optimized line counting."""
        print(f"{Colors.BOLD}{Colors.BLUE}📊 Counting lines...{Colors.END}")

        with open(self.input_file, 'rb') as f:
            buf_size = 1024 * 1024  # 1MB buffer
            count = sum(buf.count(b'\n') for buf in iter(lambda: f.read(buf_size), b''))

        print(f"{Colors.GREEN}✓{Colors.END} Total lines: {count:,}")
        return count

    def determine_bounds_optimized(self, sample_size=200000):
        """
        Optimized bounds determination using vectorized operations.
        """
        print(f"{Colors.BOLD}{Colors.BLUE}🗺️  Determining geographical bounds...{Colors.END}")

        if self.bounds_cache is not None:
            self.bounds = self.bounds_cache
            print(f"{Colors.GREEN}✓{Colors.END} Using cached bounds")
            return

        # Read sample data
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
                            if (70 <= lon <= 140 and 15 <= lat <= 55):
                                sample_data.append([lon, lat])
                        except (ValueError, IndexError):
                            continue
                elif self.file_format == 'csv':
                    parts = line.strip().split(self.delimiter)
                    if len(parts) > self.trajectory_col:
                        points = self.parse_linestring_cached(parts[self.trajectory_col])
                        if points:
                            sample_data.append(points[0])

        if not sample_data:
            raise ValueError("No valid coordinate points found in the data")

        # Vectorized bounds calculation
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

        # Cache bounds
        self.bounds_cache = self.bounds

        # Calculate grid dimensions
        width = int((self.bounds['max_lon'] - self.bounds['min_lon']) / self.pixel_size)
        height = int((self.bounds['max_lat'] - self.bounds['min_lat']) / self.pixel_size)
        self.grid_shape = (height, width)

        print(f"{Colors.GREEN}✓{Colors.END} Valid points found: {len(sample_data)}")
        print(f"{Colors.CYAN}  Bounds: {self.bounds}{Colors.END}")
        print(f"{Colors.CYAN}  Grid size: {width} x {height} pixels{Colors.END}")

    def process_chunk_isolated(self, chunk_lines, chunk_id):
        """
        Isolated chunk processing for parallel execution.
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
                try:
                    parts = line.split(self.delimiter)
                    if len(parts) >= 5:
                        lon = float(parts[3])
                        lat = float(parts[4])
                        if (70 <= lon <= 140 and 15 <= lat <= 55):
                            points.append([lon, lat])
                except (ValueError, IndexError):
                    continue

            elif self.file_format == 'csv':
                try:
                    parts = line.split(self.delimiter)
                    if len(parts) > self.trajectory_col:
                        line_points = self.parse_linestring_cached(parts[self.trajectory_col])
                        points.extend(line_points)
                except (ValueError, IndexError):
                    continue

        return np.array(points, dtype=np.float32) if points else np.array([], dtype=np.float32).reshape(0, 2)

    def read_file_batches(self, batch_size=None):
        """
        Optimized batch file reading.
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch_lines = []
        batch_number = 0

        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Skip header if CSV format
            if self.file_format == 'csv':
                header = f.readline()
                yield [header], 0  # Yield header separately
                batch_number += 1

            for line in f:
                batch_lines.append(line)

                if len(batch_lines) >= batch_size:
                    yield batch_lines, batch_number
                    batch_lines = []
                    batch_number += 1
                    self.stats['batch_io_operations'] += 1
                    self.monitor.update_stats(io_operations=self.stats['batch_io_operations'])

            # Yield remaining lines
            if batch_lines:
                yield batch_lines, batch_number
                self.stats['batch_io_operations'] += 1
                self.monitor.update_stats(io_operations=self.stats['batch_io_operations'])

    def process_streaming_parallel(self):
        """
        Streaming parallel processing with optimized memory usage.
        """
        print(f"{Colors.BOLD}{Colors.BLUE}⚡ Processing file: {self.input_file}{Colors.END}")
        print(f"{Colors.CYAN}  File size: {self.get_file_size() / (1024*1024):.2f} MB{Colors.END}")

        total_lines = self.count_lines_optimized()
        self.stats['total_lines'] = total_lines
        self.monitor.total_lines = total_lines

        print(f"{Colors.GREEN}✓{Colors.END} Total lines: {total_lines:,}")
        print(f"{Colors.CYAN}  Using {self.num_workers} workers{Colors.END}")
        print(f"{Colors.CYAN}  Batch size: {self.batch_size:,}{Colors.END}")

        print(f"\n{Colors.BOLD}{Colors.YELLOW}🚀 Starting parallel processing...{Colors.END}")
        print(f"{Colors.DIM}Real-time monitoring will begin shortly...{Colors.END}")

        # Start monitoring
        self.monitor.start()

        # Pre-allocate shared density grid
        density_grid = np.zeros(self.grid_shape, dtype=np.float32)

        # Process in batches
        processed_chunks = 0
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for batch_lines, batch_id in self.read_file_batches():
                # Split batch into chunks for parallel processing
                chunks = [batch_lines[i:i + self.chunk_size]
                         for i in range(0, len(batch_lines), self.chunk_size)]

                for chunk_id, chunk in enumerate(chunks):
                    future = executor.submit(self.process_chunk_isolated, chunk, batch_id * 100 + chunk_id)
                    futures.append((future, batch_id, chunk_id))

                # Process completed futures
                completed_futures = []
                for future, batch_id, chunk_id in futures:
                    if future.done():
                        try:
                            chunk_points = future.result()
                            if len(chunk_points) > 0:
                                self.update_density_grid_vectorized(density_grid, chunk_points)
                                self.stats['points_processed'] += len(chunk_points)

                            processed_chunks += 1
                            self.stats['chunks_processed'] = processed_chunks

                            # Update monitoring
                            lines_processed = processed_chunks * self.chunk_size
                            self.monitor.update_stats(
                                processed_lines=lines_processed,
                                processed_points=self.stats['points_processed'],
                                cache_hits=self.stats['cache_hits'],
                                io_operations=self.stats['batch_io_operations']
                            )

                        except Exception as e:
                            print(f"{Colors.RED}✗{Colors.END} Error processing chunk: {e}")

                        completed_futures.append((future, batch_id, chunk_id))

                # Remove completed futures
                futures = [(f, b, c) for f, b, c in futures if (f, b, c) not in completed_futures]

            # Wait for remaining futures
            for future, batch_id, chunk_id in futures:
                try:
                    chunk_points = future.result()
                    if len(chunk_points) > 0:
                        self.update_density_grid_vectorized(density_grid, chunk_points)
                        self.stats['points_processed'] += len(chunk_points)

                        # Final monitoring update
                        self.monitor.update_stats(
                            processed_lines=self.stats['total_lines'],
                            processed_points=self.stats['points_processed'],
                            cache_hits=self.stats['cache_hits'],
                            io_operations=self.stats['batch_io_operations']
                        )
                except Exception as e:
                    print(f"{Colors.RED}✗{Colors.END} Error processing final chunk: {e}")

        # Stop monitoring
        self.monitor.stop()

        print(f"\n{Colors.BOLD}{Colors.GREEN}✓ Processing completed!{Colors.END}")

        return density_grid

    def update_density_grid_vectorized(self, density_grid, points):
        """
        Vectorized density grid update.
        """
        if len(points) == 0:
            return

        # Convert to pixel coordinates
        lon_pixels = ((points[:, 0] - self.bounds['min_lon']) / self.pixel_size).astype(int)
        lat_pixels = ((points[:, 1] - self.bounds['min_lat']) / self.pixel_size).astype(int)

        # Clip to grid bounds
        lon_pixels = np.clip(lon_pixels, 0, self.grid_shape[1] - 1)
        lat_pixels = np.clip(lat_pixels, 0, self.grid_shape[0] - 1)

        # Vectorized counting using numpy bincount (much faster than manual counting)
        # Flatten coordinates for 2D histogram
        flat_indices = lat_pixels * self.grid_shape[1] + lon_pixels

        # Use bincount for fast frequency counting
        counts = np.bincount(flat_indices, minlength=self.grid_shape[0] * self.grid_shape[1])

        # Reshape back to 2D and add to density grid
        density_grid += counts.reshape(self.grid_shape)

    def create_density_grid_optimized(self, points):
        """
        Optimized density grid creation using histogram2d.
        """
        print(f"{Colors.BOLD}{Colors.BLUE}📊 Creating optimized density grid...{Colors.END}")

        if len(points) == 0:
            return np.zeros(self.grid_shape, dtype=np.float32)

        # Vectorized bounds filtering
        bounds_mask = self.is_point_within_bounds_vectorized(points)
        filtered_points = points[bounds_mask]

        if len(filtered_points) == 0:
            return np.zeros(self.grid_shape, dtype=np.float32)

        # Use optimized histogram2d for fast density calculation
        density_grid, x_edges, y_edges = np.histogram2d(
            filtered_points[:, 1], filtered_points[:, 0],  # lat, lon order for histogram2d
            bins=[self.grid_shape[0], self.grid_shape[1]],
            range=[[self.bounds['min_lat'], self.bounds['max_lat']],
                   [self.bounds['min_lon'], self.bounds['max_lon']]],
            density=False
        )

        # Apply Gaussian smoothing
        density_grid = gaussian_filter(density_grid.astype(np.float32), sigma=self.sigma)

        return density_grid

    def create_visualization(self, density_grid):
        """
        Create visualization (same as original, already optimized).
        """
        print(f"{Colors.BOLD}{Colors.BLUE}🎨 Creating visualization...{Colors.END}")

        # Load shapefile for map background
        try:
            gdf = gpd.read_file(self.shapefile_path)
            print(f"{Colors.GREEN}✓{Colors.END} Loaded shapefile with {len(gdf)} features")
        except Exception as e:
            print(f"{Colors.RED}✗{Colors.END} Warning: Could not load shapefile: {e}")
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
        ax.set_title(f'Trajectory Point Density Heatmap (OPTIMIZED)\nTotal Points: {self.stats["points_processed"]:,}')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_file, dpi=300, bbox_inches='tight')
        print(f"{Colors.GREEN}✓{Colors.END} Visualization saved to: {self.output_file}")

        plt.close()

    def print_stats(self):
        """Print enhanced processing statistics."""
        final_stats = self.monitor.get_final_stats()
        total_time = final_stats.get('total_time', time.time() - self.stats['time_start'])

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}  FINAL PERFORMANCE SUMMARY{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")
        print()

        print(f"{Colors.BOLD}{Colors.YELLOW}📊 PROCESSING STATISTICS{Colors.END}")
        print(f"  {Colors.WHITE}Total lines processed:{Colors.END} {self.stats['total_lines']:,}")
        print(f"  {Colors.WHITE}Points extracted:{Colors.END} {self.stats['points_processed']:,}")
        if self.stats['points_filtered'] > 0:
            print(f"  {Colors.WHITE}Points filtered out:{Colors.END} {self.stats['points_filtered']:,}")
            filter_rate = (self.stats['points_filtered'] / (self.stats['points_processed'] + self.stats['points_filtered'])) * 100
            print(f"  {Colors.WHITE}Filter rate:{Colors.END} {filter_rate:.1f}%")
        print(f"  {Colors.WHITE}Chunks processed:{Colors.END} {self.stats['chunks_processed']}")
        print(f"  {Colors.WHITE}Batch I/O operations:{Colors.END} {self.stats['batch_io_operations']}")
        print(f"  {Colors.WHITE}Cache hits:{Colors.END} {self.stats['cache_hits']:,}")
        print()

        print(f"{Colors.BOLD}{Colors.YELLOW}⚡ PERFORMANCE METRICS{Colors.END}")
        print(f"  {Colors.GREEN}Total time:{Colors.END} {total_time:.2f} seconds")
        print(f"  {Colors.GREEN}Lines per second:{Colors.END} {final_stats.get('avg_lines_per_sec', 0):,.0f}")
        print(f"  {Colors.GREEN}Points per second:{Colors.END} {final_stats.get('avg_points_per_sec', 0):,.0f}")
        print(f"  {Colors.BLUE}Peak memory usage:{Colors.END} {final_stats.get('peak_memory', 0):.1f} MB")
        print()

        print(f"{Colors.BOLD}{Colors.YELLOW}🏆 PERFORMANCE RATINGS{Colors.END}")

        # Performance rating
        avg_lines_per_sec = final_stats.get('avg_lines_per_sec', 0)
        if avg_lines_per_sec > 100000:
            perf_rating = f"{Colors.GREEN}⭐⭐⭐⭐⭐ EXCELLENT (>100K/s){Colors.END}"
        elif avg_lines_per_sec > 50000:
            perf_rating = f"{Colors.CYAN}⭐⭐⭐⭐ VERY GOOD (>50K/s){Colors.END}"
        elif avg_lines_per_sec > 20000:
            perf_rating = f"{Colors.YELLOW}⭐⭐⭐ GOOD (>20K/s){Colors.END}"
        elif avg_lines_per_sec > 10000:
            perf_rating = f"{Colors.MAGENTA}⭐⭐ FAIR (>10K/s){Colors.END}"
        else:
            perf_rating = f"{Colors.RED}⭐ SLOW (<10K/s){Colors.END}"

        print(f"  {Colors.WHITE}Processing Speed:{Colors.END} {perf_rating}")

        # Memory efficiency rating
        peak_memory = final_stats.get('peak_memory', 0)
        if peak_memory < 1000:
            mem_rating = f"{Colors.GREEN}⭐⭐⭐⭐⭐ EXCELLENT (<1GB){Colors.END}"
        elif peak_memory < 2000:
            mem_rating = f"{Colors.CYAN}⭐⭐⭐⭐ VERY GOOD (<2GB){Colors.END}"
        elif peak_memory < 4000:
            mem_rating = f"{Colors.YELLOW}⭐⭐⭐ GOOD (<4GB){Colors.END}"
        else:
            mem_rating = f"{Colors.RED}⭐⭐ HIGH (>4GB){Colors.END}"

        print(f"  {Colors.WHITE}Memory Efficiency:{Colors.END} {mem_rating}")

        print()
        print(f"{Colors.BOLD}{Colors.CYAN}Output file:{Colors.END} {self.output_file}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")

    def process(self):
        """
        Main processing method with all optimizations.
        """
        try:
            print(f"\n{Colors.BOLD}{Colors.GREEN}🚀 STARTING OPTIMIZED TRAJECTORY POINT DENSITY ANALYSIS{Colors.END}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")
            print(f"{Colors.CYAN}Input:{Colors.END} {self.input_file}")
            print(f"{Colors.CYAN}Output:{Colors.END} {self.output_file}")
            print(f"{Colors.CYAN}Shapefile:{Colors.END} {self.shapefile_path}")
            print(f"{Colors.CYAN}CPU cores:{Colors.END} {cpu_count()}")
            print(f"{Colors.CYAN}Workers:{Colors.END} {self.num_workers}")
            print(f"{Colors.CYAN}Format:{Colors.END} {self.file_format}, delimiter: '{self.delimiter}'")
            if self.file_format == 'csv':
                print(f"{Colors.CYAN}Trajectory column:{Colors.END} {self.trajectory_col}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")

            # Step 1: Determine geographical bounds
            self.determine_bounds_optimized()

            # Step 2: Process file using streaming parallel approach
            density_grid = self.process_streaming_parallel()

            if np.sum(density_grid) == 0:
                print(f"{Colors.RED}✗{Colors.END} No valid points found in the data")
                return

            # Step 3: Apply final smoothing if needed
            density_grid = gaussian_filter(density_grid, sigma=self.sigma)

            # Step 4: Create visualization
            self.create_visualization(density_grid)

            # Print statistics
            self.print_stats()

            return self.output_file

        except Exception as e:
            print(f"{Colors.RED}✗{Colors.END} Error during processing: {e}")
            raise
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠{Colors.END} Processing interrupted by user")
            self.monitor.stop()
            sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create OPTIMIZED trajectory point density heatmap with REAL-TIME monitoring')
    parser.add_argument('input_file', help='Path to trajectory data file')
    parser.add_argument('--output', '-o', help='Output image file path')
    parser.add_argument('--shapefile', '-s', help='Path to shapefile for map background')
    parser.add_argument('--pixel-size', '-p', type=float, default=0.001,
                       help='Pixel size in degrees (default: 0.001)')
    parser.add_argument('--sigma', type=float, default=2.0,
                       help='Gaussian smoothing parameter (default: 2.0)')
    parser.add_argument('--memory', '-m', type=float, default=16,
                       help='Memory limit in GB (default: 16)')
    parser.add_argument('--workers', '-w', type=int,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Disable real-time monitoring')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"{Colors.RED}✗{Colors.END} Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    # Create and run processor
    processor = OptimizedTrajectoryPointDensity(
        input_file=args.input_file,
        output_file=args.output,
        shapefile_path=args.shapefile,
        pixel_size=args.pixel_size,
        memory_limit_gb=args.memory
    )

    # Override worker count if specified
    if args.workers:
        processor.num_workers = min(args.workers, 128)  # Allow up to 128 workers

    # Disable monitoring if requested
    if args.no_monitor:
        processor.monitor.update_interval = float('inf')

    try:
        output_path = processor.process()
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Processing completed successfully!{Colors.END}")
        print(f"{Colors.CYAN}Heatmap saved to: {output_path}{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.END} Processing failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠{Colors.END} Processing interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()