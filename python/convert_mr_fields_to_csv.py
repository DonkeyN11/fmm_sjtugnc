#!/usr/bin/env python3
import pandas as pd
import time
import os
import multiprocessing as mp
from functools import partial
import csv

def is_empty_row(line):
    """Check if a line represents an empty row (failed map matching)"""
    return line.strip() == "0;;;;;LINESTRING();;;"

def process_chunk(chunk_lines, chunk_id):
    """Process a chunk of lines and return filtered non-empty rows"""
    filtered_lines = []

    for line in chunk_lines:
        line = line.strip()
        # Skip empty rows and header
        if line and not is_empty_row(line) and not line.startswith("id;"):
            filtered_lines.append(line)

    print(f"Processed chunk {chunk_id}: {len(chunk_lines)} input lines, {len(filtered_lines)} valid rows")
    return filtered_lines

def convert_mr_fields_parallel(input_file, output_file, batch_size=100000, max_workers=None):
    """
    Convert mr_fields.txt to CSV format using parallel processing
    Filters out empty rows (failed map matching results)
    """
    start_time = time.time()

    # Determine optimal number of workers
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    print(f"Parallel mr_fields conversion started...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Batch size: {batch_size}")
    print(f"Using {max_workers} workers")

    # Count total lines first
    print("Counting total lines...")
    total_lines = sum(1 for _ in open(input_file))
    print(f"Total lines: {total_lines}")

    # Process in parallel
    processed_lines = 0
    batch_num = 0
    all_results = []

    print("Processing in parallel...")

    # Create process pool
    with mp.Pool(processes=max_workers) as pool:
        # Read file in chunks and process in parallel
        with open(input_file, 'r') as f:
            batch_lines = []

            for line in f:
                batch_lines.append(line)

                if len(batch_lines) >= batch_size:
                    # Submit chunk to process pool
                    result = pool.apply_async(process_chunk, (batch_lines.copy(), batch_num))
                    all_results.append(result)

                    processed_lines += len(batch_lines)
                    batch_num += 1

                    # Progress update
                    progress = (processed_lines / total_lines) * 100
                    print(f"Batch {batch_num}: {processed_lines:,}/{total_lines:,} lines ({progress:.1f}%)")

                    # Clear batch for next iteration
                    batch_lines = []

            # Process remaining lines
            if batch_lines:
                result = pool.apply_async(process_chunk, (batch_lines, batch_num))
                all_results.append(result)
                processed_lines += len(batch_lines)
                progress = (processed_lines / total_lines) * 100
                print(f"Final batch: {processed_lines:,}/{total_lines:,} lines ({progress:.1f}%)")

        # Collect all results
        print("Collecting results...")
        all_filtered_lines = []
        for result in all_results:
            filtered_lines = result.get()
            all_filtered_lines.extend(filtered_lines)

    print(f"Filtered {processed_lines} lines into {len(all_filtered_lines)} valid rows")

    # Write output
    print("Writing output CSV...")

    # Parse all filtered lines and write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')

        # Write header
        writer.writerow(['id', 'opath', 'spdist', 'cpath', 'tpath', 'mgeom', 'ep', 'tp', 'cumu_prob'])

        # Write data rows
        for line in all_filtered_lines:
            # Split by semicolon
            fields = line.split(';')
            writer.writerow(fields)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Parallel conversion completed!")
    print(f"Total valid rows: {len(all_filtered_lines)}")
    print(f"Total lines processed: {processed_lines}")
    print(f"Empty rows filtered: {processed_lines - len(all_filtered_lines)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {processed_lines/processing_time:.2f} lines/second")
    print(f"Workers used: {max_workers}")

def convert_simple_fast(input_file, output_file):
    """
    Simple and fast conversion for moderate sized files
    """
    start_time = time.time()

    print("Simple fast conversion...")

    # Read file and filter lines
    filtered_lines = []
    total_lines = 0

    with open(input_file, 'r') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            # Skip empty rows and header
            if line and not is_empty_row(line) and not line.startswith("id;"):
                filtered_lines.append(line)

    print(f"Read {total_lines} lines, filtered to {len(filtered_lines)} valid rows")

    # Write output
    print("Writing output CSV...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')

        # Write header
        writer.writerow(['id', 'opath', 'spdist', 'cpath', 'tpath', 'mgeom', 'ep', 'tp', 'cumu_prob'])

        # Write data rows
        for line in filtered_lines:
            fields = line.split(';')
            writer.writerow(fields)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Conversion completed!")
    print(f"Total valid rows: {len(filtered_lines)}")
    print(f"Total lines processed: {total_lines}")
    print(f"Empty rows filtered: {total_lines - len(filtered_lines)}")
    print(f"Processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    input_file = "/home/dell/Czhang/fmm_sjtugnc/output/mr_filtered_fields_all.txt"
    output_file = "/home/dell/Czhang/fmm_sjtugnc/output/mr_filtered_fields_all.csv"

    # Check file size
    file_size = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Input file size: {file_size:.2f} MB")

    # Choose method based on file size
    if file_size > 100:  # Use parallel processing for large files
        # Adjust batch size based on file size
        if file_size > 500:
            batch_size = 500000
        elif file_size > 200:
            batch_size = 300000
        else:
            batch_size = 200000

        print(f"Using parallel processing with batch size: {batch_size}")
        convert_mr_fields_parallel(input_file, output_file, batch_size=batch_size)
    else:
        print("Using simple fast conversion")
        convert_simple_fast(input_file, output_file)