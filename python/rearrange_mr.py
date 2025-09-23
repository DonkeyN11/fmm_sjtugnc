#!/usr/bin/env python3
"""
High-performance CSV/TXT file sorter by ID field.
Uses efficient data structures, parallel processing, and memory optimization.
"""

import os
import sys
import csv
import mmap
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager, Queue, Process
from threading import Lock
from functools import partial
import heapq
import tempfile
import shutil

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class HighPerformanceSorter:
    def __init__(self, input_file, output_file=None, delimiter=';', id_col=0,
                 chunk_size=100000, memory_limit_gb=128):
        """
        Initialize the high-performance sorter.

        Args:
            input_file: Path to input CSV/TXT file
            output_file: Path to output file (auto-generated if None)
            delimiter: Field delimiter (default: ';')
            id_col: ID column index (default: 0)
            chunk_size: Number of lines per chunk (default: 100000)
            memory_limit_gb: Memory limit in GB (default: 4)
        """
        self.input_file = input_file
        self.delimiter = delimiter
        self.id_col = id_col
        self.chunk_size = chunk_size
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # Convert to bytes

        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            self.output_file = f"output/{base_name}_rearranged.csv"
        else:
            self.output_file = output_file

        self.temp_dir = tempfile.mkdtemp(prefix="sorter_temp_")
        self.chunk_files = []
        self.stats = {
            'total_lines': 0,
            'chunks_processed': 0,
            'memory_peak': 0,
            'time_start': time.time()
        }

    def get_file_size(self):
        """Get file size in bytes."""
        return os.path.getsize(self.input_file)

    def count_lines(self):
        """Count total lines in file efficiently."""
        try:
            # Try using memory mapping first (most efficient)
            with open(self.input_file, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return sum(1 for _ in iter(mm.readline, b''))
        except:
            # Fallback to simple line counting
            with open(self.input_file, 'rb') as f:
                return sum(1 for _ in f)

    def has_header(self):
        """Check if file has a header row."""
        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            if first_line:
                # Check if first line contains non-numeric ID field
                parts = first_line.split(self.delimiter)
                if len(parts) > self.id_col:
                    try:
                        int(parts[self.id_col])
                        return False  # First line has numeric ID, no header
                    except ValueError:
                        return True  # First line has non-numeric ID, likely header
            return False

    def parse_line(self, line):
        """Parse a single line and extract ID and line data."""
        try:
            parts = line.strip().split(self.delimiter)
            if len(parts) > self.id_col:
                try:
                    line_id = int(parts[self.id_col])
                    return line_id, line.strip()
                except ValueError:
                    return float('inf'), line.strip()  # Put invalid IDs at end
            return float('inf'), line.strip()
        except:
            return float('inf'), line.strip()

    def process_chunk_parallel(self, chunk_lines):
        """Process a chunk of lines in parallel using multiple processes."""
        if not chunk_lines:
            return []

        # Use ThreadPoolExecutor for I/O-bound parsing
        with ThreadPoolExecutor(max_workers=min(cpu_count(), 8)) as executor:
            parsed_data = list(executor.map(self.parse_line, chunk_lines))

        # Sort the parsed data
        parsed_data.sort(key=lambda x: x[0])
        return parsed_data

    def create_sorted_chunks(self):
        """Read file in chunks, sort each chunk, and save to temporary files."""
        print(f"Processing file: {self.input_file}")
        print(f"File size: {self.get_file_size() / (1024*1024):.2f} MB")

        # Check for header
        has_header = self.has_header()
        print(f"Has header: {has_header}")

        total_lines = self.count_lines()
        self.stats['total_lines'] = total_lines
        print(f"Total lines: {total_lines}")

        # Memory-based chunking strategy
        if HAS_PSUTIL:
            available_memory = psutil.virtual_memory().available
            safe_memory = min(available_memory * 0.7, self.memory_limit)

            # Adjust chunk size based on available memory
            avg_line_size = self.get_file_size() / total_lines if total_lines > 0 else 100
            max_lines_in_memory = int(safe_memory / (avg_line_size * 3))  # 3x for overhead
            self.chunk_size = min(self.chunk_size, max_lines_in_memory)

            print(f"Using chunk size: {self.chunk_size}")
            print(f"Available memory: {available_memory / (1024*1024):.2f} MB")
        else:
            print(f"Using fixed chunk size: {self.chunk_size}")

        # Read and process chunks
        chunk_lines = []
        chunk_number = 0
        header_line = None

        with open(self.input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if line_num == 0 and has_header:
                    header_line = line.strip()
                    continue

                chunk_lines.append(line)

                if len(chunk_lines) >= self.chunk_size:
                    self._process_chunk(chunk_lines, chunk_number)
                    chunk_lines = []
                    chunk_number += 1

                    # Memory monitoring
                    if HAS_PSUTIL:
                        current_memory = psutil.Process().memory_info().rss
                        self.stats['memory_peak'] = max(self.stats['memory_peak'], current_memory)

                    if chunk_number % 10 == 0:
                        progress = (line_num / total_lines) * 100
                        print(f"Progress: {progress:.1f}% ({line_num}/{total_lines} lines)")

        # Process remaining lines
        if chunk_lines:
            self._process_chunk(chunk_lines, chunk_number)

        print(f"Created {len(self.chunk_files)} sorted chunks")
        return self.chunk_files, header_line

    def _process_chunk(self, chunk_lines, chunk_number):
        """Process a single chunk and save to temporary file."""
        try:
            # Sort the chunk using parallel processing
            sorted_data = self.process_chunk_parallel(chunk_lines)

            # Save to temporary file
            chunk_file = os.path.join(self.temp_dir, f"chunk_{chunk_number:04d}.tmp")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                for _, line in sorted_data:
                    f.write(line + '\n')

            self.chunk_files.append(chunk_file)
            self.stats['chunks_processed'] += 1

        except Exception as e:
            print(f"Error processing chunk {chunk_number}: {e}")
            raise

    def merge_sorted_chunks(self, header_line=None):
        """Merge sorted chunks using a min-heap for efficient merging."""
        if len(self.chunk_files) == 1:
            # If only one chunk, copy it and add header if exists
            with open(self.chunk_files[0], 'r', encoding='utf-8') as src_file:
                with open(self.output_file, 'w', encoding='utf-8') as dst_file:
                    # Write header first if it exists
                    if header_line:
                        dst_file.write(header_line + '\n')
                    # Copy the rest of the file
                    shutil.copyfileobj(src_file, dst_file)
            return

        print(f"Merging {len(self.chunk_files)} sorted chunks...")

        # Open all chunk files
        file_handles = []
        for chunk_file in self.chunk_files:
            f = open(chunk_file, 'r', encoding='utf-8')
            file_handles.append(f)

        # Create heap for merging
        heap = []
        for i, f in enumerate(file_handles):
            line = f.readline().strip()
            if line:
                line_id, _ = self.parse_line(line)
                heapq.heappush(heap, (line_id, i, line))

        # Merge sorted chunks
        with open(self.output_file, 'w', encoding='utf-8') as out_file:
            lines_written = 0

            # Write header first if it exists
            if header_line:
                out_file.write(header_line + '\n')

            while heap:
                line_id, file_idx, line = heapq.heappop(heap)
                out_file.write(line + '\n')
                lines_written += 1

                # Read next line from the same file
                next_line = file_handles[file_idx].readline().strip()
                if next_line:
                    next_id, _ = self.parse_line(next_line)
                    heapq.heappush(heap, (next_id, file_idx, next_line))

                # Progress reporting
                if lines_written % 10000 == 0:
                    progress = (lines_written / self.stats['total_lines']) * 100
                    print(f"Merge progress: {progress:.1f}% ({lines_written}/{self.stats['total_lines']})")

        # Close file handles
        for f in file_handles:
            f.close()

    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def sort(self):
        """Main sorting method."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            print(f"Starting sort process...")
            print(f"Input: {self.input_file}")
            print(f"Output: {self.output_file}")
            print(f"CPU cores: {cpu_count()}")

            # Create sorted chunks
            self.chunk_files, header_line = self.create_sorted_chunks()

            # Merge sorted chunks
            self.merge_sorted_chunks(header_line)

            # Print statistics
            self.stats['time_end'] = time.time()
            self.print_stats()

            return self.output_file

        except Exception as e:
            print(f"Error during sorting: {e}")
            raise
        finally:
            self.cleanup()

    def print_stats(self):
        """Print sorting statistics."""
        total_time = self.stats['time_end'] - self.stats['time_start']

        print("\n=== Sorting Statistics ===")
        print(f"Total lines processed: {self.stats['total_lines']:,}")
        print(f"Chunks created: {self.stats['chunks_processed']}")

        if HAS_PSUTIL and self.stats['memory_peak'] > 0:
            memory_peak_mb = self.stats['memory_peak'] / (1024 * 1024)
            print(f"Peak memory usage: {memory_peak_mb:.2f} MB")

        print(f"Total time: {total_time:.2f} seconds")
        print(f"Lines per second: {self.stats['total_lines'] / total_time:,.0f}")
        print(f"Output file: {self.output_file}")
        print(f"Output size: {os.path.getsize(self.output_file) / (1024*1024):.2f} MB")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python rearrange_mr.py <input_file> [output_file]")
        print("Example: python rearrange_mr.py output/mr_cumu_ts_rearranged.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)

    # Create and run sorter
    sorter = HighPerformanceSorter(
        input_file=input_file,
        output_file=output_file,
        delimiter=';',      # Based on observed file format
        id_col=0,          # ID is first column
        chunk_size=50000,  # Optimized for performance
        memory_limit_gb=128 # Use up to 128GB of memory
    )

    try:
        output_path = sorter.sort()
        print(f"\nSorting completed successfully!")
        print(f"Sorted file saved to: {output_path}")
    except Exception as e:
        print(f"Sorting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()