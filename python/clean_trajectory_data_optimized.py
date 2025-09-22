#!/usr/bin/env python3
"""
Optimized Trajectory Data Cleaning and Filtering Script

High-performance version with:
- Memory-efficient chunked reading
- Multi-threaded parallel processing using all CPU cores
- Optimized data structures and algorithms
- Maintains original functionality and output format

Input format: ID,timestamp,speed,longitude,latitude
Output format: id;original_id;geom;timestamps
"""

import json
import sys
import os
from datetime import datetime
from collections import defaultdict
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Manager, cpu_count
import queue
import time
import gc


class OptimizedTrajectoryCleaner:
    def __init__(self, input_file, bounds_file, output_file, chunk_size=100000, num_workers=None):
        """
        Initialize the optimized trajectory cleaner.

        Args:
            input_file: Path to input trajectory file
            bounds_file: Path to JSON bounds file
            output_file: Path to output cleaned file
            chunk_size: Number of lines per processing chunk
            num_workers: Number of worker threads/processes
        """
        self.input_file = input_file
        self.bounds_file = bounds_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.num_workers = num_workers or min(cpu_count(), 64)  # Use up to 64 cores

        # Load bounds
        self.bounds = self.load_bounds()

        # Thread-safe statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            'total_points': 0,
            'filtered_points': 0,
            'vehicles_processed': 0,
            'output_lines': 0,
            'processing_time': 0
        }

        # Shared data structures for parallel processing
        self.result_queue = queue.Queue()
        self.all_vehicle_ids = set()
        self.processed_chunks = []

        # Track order of vehicle appearance to match original script
        self.vehicle_first_appearance = {}

    def load_bounds(self):
        """Load map bounds from JSON file."""
        try:
            with open(self.bounds_file, 'r') as f:
                bounds = json.load(f)
            print(f"Loaded bounds: {bounds}")
            return bounds
        except Exception as e:
            print(f"Error loading bounds file: {e}")
            sys.exit(1)

    def is_point_within_bounds(self, lon, lat):
        """Check if point is within the map bounds."""
        return (self.bounds['min_lon'] <= lon <= self.bounds['max_lon'] and
                self.bounds['min_lat'] <= lat <= self.bounds['max_lat'])

    def convert_timestamp_to_utc(self, timestamp_str):
        """Convert yyyymmddhhmmss timestamp to UTC format."""
        try:
            dt = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            return int(dt.timestamp())
        except Exception as e:
            return None

    def read_file_chunks(self):
        """
        Generator that yields chunks of lines from the input file.
        Memory-efficient reading for large files.
        """
        print(f"Reading file in chunks of {self.chunk_size} lines using {self.num_workers} workers...")

        with open(self.input_file, 'r', encoding='utf-8') as f:
            chunk = []
            for line_num, line in enumerate(f, 1):
                chunk.append((line_num, line.strip()))

                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []

            # Yield remaining lines
            if chunk:
                yield chunk

    def process_chunk(self, chunk_data):
        """
        Process a chunk of lines in parallel.
        Returns vehicle points for this chunk and their first appearance line numbers.
        """
        chunk_vehicle_points = defaultdict(list)
        chunk_stats = {
            'total_points': 0,
            'filtered_points': 0
        }
        chunk_vehicle_min_lines = {}  # Track minimum line number for each vehicle in this chunk

        for line_num, line in chunk_data:
            if not line:
                continue

            chunk_stats['total_points'] += 1

            try:
                parts = line.split(',')
                if len(parts) != 5:
                    continue

                vehicle_id = parts[0].strip()
                timestamp_str = parts[1].strip()
                longitude = float(parts[3])
                latitude = float(parts[4])

                # Filter by bounds
                if not self.is_point_within_bounds(longitude, latitude):
                    chunk_stats['filtered_points'] += 1
                    continue

                # # Convert timestamp
                # utc_timestamp = self.convert_timestamp_to_utc(timestamp_str)
                # if utc_timestamp is None:
                #     continue

                # Store point
                point_data = {
                    'lon': longitude,
                    'lat': latitude,
                    # 'timestamp': utc_timestamp,
                    'timestamp': timestamp_str,  # Keep original for ordering
                    # 'original_timestamp': timestamp_str
                }
                chunk_vehicle_points[vehicle_id].append(point_data)

                # Track minimum line number for this vehicle in the chunk
                if vehicle_id not in chunk_vehicle_min_lines or line_num < chunk_vehicle_min_lines[vehicle_id]:
                    chunk_vehicle_min_lines[vehicle_id] = line_num

            except Exception as e:
                continue

        return chunk_vehicle_points, chunk_stats, chunk_vehicle_min_lines

    def parallel_process_chunks(self):
        """
        Process all chunks in parallel using thread pool.
        Returns aggregated vehicle points and statistics with original order preserved.
        """
        # Use OrderedDict to preserve insertion order (first appearance in file)
        all_vehicle_points = {}
        total_stats = {
            'total_points': 0,
            'filtered_points': 0
        }

        print(f"Starting parallel processing with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks for processing
            futures = []
            for chunk_data in self.read_file_chunks():
                future = executor.submit(self.process_chunk, chunk_data)
                futures.append(future)

            print(f"Submitted {len(futures)} chunks for processing")

            # Collect results as they complete, but store in order of completion
            completed = 0
            chunk_results = []
            for future in futures:
                try:
                    result = future.result()
                    chunk_results.append(result)
                    total_stats['total_points'] += result[1]['total_points']
                    total_stats['filtered_points'] += result[1]['filtered_points']

                    completed += 1
                    if completed % 100 == 0:
                        print(f"Completed {completed}/{len(futures)} chunks...")

                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue

        # Now merge results in order of chunk completion to preserve file order
        for chunk_vehicle_points, chunk_stats, chunk_vehicle_min_lines in chunk_results:
            for vehicle_id, points in chunk_vehicle_points.items():
                if vehicle_id not in all_vehicle_points:
                    # First time seeing this vehicle - add to ordered dict
                    all_vehicle_points[vehicle_id] = []
                all_vehicle_points[vehicle_id].extend(points)

        return all_vehicle_points, total_stats

    def create_linestring(self, points):
        """Create LineString WKT from points."""
        if not points:
            return None

        # Sort points by timestamp to ensure correct order
        points_sorted = sorted(points, key=lambda x: x['timestamp'])

        # Create coordinate pairs
        coords = []
        for point in points_sorted:
            coords.append(f"{point['lon']} {point['lat']}")

        # Create LineString WKT
        linestring = f"LINESTRING ({', '.join(coords)})"
        return linestring

    def create_timestamp_sequence(self, points):
        """Create timestamp sequence string."""
        if not points:
            return ""

        # Sort points by timestamp
        points_sorted = sorted(points, key=lambda x: x['timestamp'])

        # Create comma-separated timestamp sequence
        timestamps = [str(point['timestamp']) for point in points_sorted]
        return ','.join(timestamps)

    def write_output_optimized(self, vehicle_points):
        """
        Optimized output writing with buffering and original order preserved.
        """
        print(f"Writing output to: {self.output_file}")

        # Use the natural order of the dictionary (preserves insertion order in Python 3.7+)
        # This matches the original script's behavior where vehicles appear in first-encounter order
        vehicle_ids = list(vehicle_points.keys())

        # Use buffered writing for better performance
        buffer_size = 1000
        output_buffer = []

        with open(self.output_file, 'w', encoding='utf-8', buffering=8192*8) as f:
            # Write header
            f.write("id;original_id;geom;timestamp\n")

            # Process each vehicle in the order they first appeared in the file
            new_id = 1
            for i, vehicle_id in enumerate(vehicle_ids):
                points = vehicle_points[vehicle_id]
                if not points:
                    continue

                # Create LineString
                linestring = self.create_linestring(points)
                if not linestring:
                    continue

                # Create timestamp sequence
                timestamp_sequence = self.create_timestamp_sequence(points)

                # Add to buffer
                output_line = f"{new_id};{vehicle_id};{linestring};{timestamp_sequence}\n"
                output_buffer.append(output_line)

                # Write buffer when full or at the end
                if len(output_buffer) >= buffer_size or i == len(vehicle_ids) - 1:
                    f.writelines(output_buffer)
                    output_buffer.clear()

                new_id += 1

        print(f"Written {new_id - 1} vehicle trajectories")

    def print_statistics(self, processing_time):
        """Print processing statistics."""
        print("\n" + "="*60)
        print("OPTIMIZED TRAJECTORY CLEANING STATISTICS")
        print("="*60)
        print(f"Total points processed: {self.stats['total_points']:,}")
        print(f"Points filtered (out of bounds): {self.stats['filtered_points']:,}")
        print(f"Vehicles with valid data: {self.stats['vehicles_processed']:,}")
        print(f"Output trajectories: {self.stats['output_lines']:,}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Workers used: {self.num_workers}")
        print(f"Memory usage optimized: Chunked processing + parallel execution")
        print("="*60)

    def process(self):
        """Main processing method with optimizations."""
        try:
            print("Starting optimized trajectory data cleaning...")
            start_time = time.time()

            # Process chunks in parallel
            vehicle_points, chunk_stats = self.parallel_process_chunks()

            # Update final statistics
            with self.stats_lock:
                self.stats['total_points'] = chunk_stats['total_points']
                self.stats['filtered_points'] = chunk_stats['filtered_points']
                self.stats['vehicles_processed'] = len(vehicle_points)

            print(f"Processed {self.stats['total_points']:,} total points")
            print(f"Filtered {self.stats['filtered_points']:,} out-of-bounds points")
            print(f"Found {self.stats['vehicles_processed']:,} vehicles with valid data")

            # Write optimized output
            self.write_output_optimized(vehicle_points)
            self.stats['output_lines'] = len([v for v in vehicle_points.values() if v])

            # Calculate processing time
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time

            # Print statistics
            self.print_statistics(processing_time)

            return True

        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Clean and filter trajectory data (OPTIMIZED)')
    parser.add_argument('input_file', help='Path to input trajectory file')
    parser.add_argument('bounds_file', help='Path to JSON bounds file')
    parser.add_argument('output_file', help='Path to output cleaned file')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Number of lines per processing chunk (default: 100000)')
    parser.add_argument('--workers', type=int,
                       help='Number of worker threads (default: auto-detect)')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.bounds_file):
        print(f"Error: Bounds file '{args.bounds_file}' does not exist")
        sys.exit(1)

    # Create and run optimized cleaner
    cleaner = OptimizedTrajectoryCleaner(
        input_file=args.input_file,
        bounds_file=args.bounds_file,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        num_workers=args.workers
    )

    success = cleaner.process()

    if success:
        print("\nProcessing completed successfully!")
        sys.exit(0)
    else:
        print("\nProcessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()