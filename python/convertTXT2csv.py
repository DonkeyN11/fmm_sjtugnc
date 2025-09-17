#!/usr/bin/env python3
import pandas as pd
import time
import os
from collections import defaultdict
import multiprocessing as mp
from functools import partial

def process_chunk(chunk, chunk_id):
    """Process a single chunk of data and return trajectory groups"""
    trajectory_groups = defaultdict(list)

    for _, row in chunk.iterrows():
        trajectory_groups[row['id']].append((row['longitude'], row['latitude']))

    print(f"Processed chunk {chunk_id}: {len(chunk)} rows")
    return trajectory_groups

def merge_trajectory_groups(groups_list):
    """Merge multiple trajectory groups dictionaries"""
    merged = defaultdict(list)

    for groups in groups_list:
        for trajectory_id, coordinates in groups.items():
            merged[trajectory_id].extend(coordinates)

    return merged

def convert_trajectory_to_linestring(args):
    """Convert a single trajectory to LINESTRING format with sequential ID"""
    item, new_id = args
    trajectory_id, coordinates = item
    coord_strs = [f"{lon} {lat}" for lon, lat in coordinates]
    return {
        'id': new_id,
        'original_id': trajectory_id,
        'geom': f"LINESTRING({','.join(coord_strs)})"
    }

def convert_parallel(input_file, output_file, batch_size=1000000, max_workers=None):
    """
    Convert using parallel processing to maximize CPU utilization
    """
    start_time = time.time()

    # Determine optimal number of workers
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    print(f"Parallel conversion started...")
    print(f"Input file: {input_file}")
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
        chunk_iterator = pd.read_csv(input_file, header=None, chunksize=batch_size)

        for chunk in chunk_iterator:
            chunk.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

            # Submit chunk to process pool
            result = pool.apply_async(process_chunk, (chunk, batch_num))
            all_results.append(result)

            processed_lines += len(chunk)
            batch_num += 1

            # Progress update
            progress = (processed_lines / total_lines) * 100
            print(f"Batch {batch_num}: {processed_lines:,}/{total_lines:,} lines ({progress:.1f}%)")

            # Process results as they complete to manage memory
            if len(all_results) >= max_workers * 2:
                completed_results = []
                for result in all_results:
                    if result.ready():
                        completed_results.append(result.get())
                all_results = [r for r in all_results if not r.ready()]

                if completed_results:
                    # Save intermediate results
                    intermediate_groups = merge_trajectory_groups(completed_results)
                    save_intermediate_results(intermediate_groups, f"{output_file}.batch_{batch_num}")

        # Get remaining results
        if all_results:
            completed_results = [result.get() for result in all_results]
            final_groups = merge_trajectory_groups(completed_results)
        else:
            final_groups = defaultdict(list)

    print(f"Grouped {processed_lines} points into {len(final_groups)} trajectories")

    # Final conversion using parallel processing
    print("Converting to LINESTRING format...")

    # Convert trajectory groups to list for parallel processing
    trajectory_items = list(final_groups.items())
    total_trajectories = len(trajectory_items)

    # Process trajectories in parallel
    print(f"Converting {total_trajectories:,} trajectories using {max_workers} workers...")

    # Create a list of (trajectory_item, new_id) pairs
    trajectory_with_new_ids = [(item, i+1) for i, item in enumerate(trajectory_items)]

    # Process trajectories in parallel
    with mp.Pool(processes=max_workers) as pool:
        # Process individual trajectories in parallel with new IDs
        results = pool.imap_unordered(convert_trajectory_to_linestring, trajectory_with_new_ids)

        output_data = []
        processed_trajectories = 0

        for result in results:
            output_data.append(result)
            processed_trajectories += 1

            if processed_trajectories % 1000 == 0:
                progress = (processed_trajectories / total_trajectories) * 100
                print(f"Processed {processed_trajectories:,}/{total_trajectories:,} trajectories ({progress:.1f}%)")

    # Save final result
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Parallel conversion completed!")
    print(f"Total trajectories: {len(output_df)}")
    print(f"Total points: {processed_lines}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {processed_lines/processing_time:.2f} points/second")
    print(f"Workers used: {max_workers}")

def save_intermediate_results(trajectory_groups, filename):
    """Save intermediate results to avoid data loss with sequential IDs"""
    output_data = []

    for new_id, (trajectory_id, coordinates) in enumerate(trajectory_groups.items(), 1):
        coord_strs = [f"{lon} {lat}" for lon, lat in coordinates]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        output_data.append({
            'id': new_id,
            'original_id': trajectory_id,
            'geom': linestring
        })

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(filename, index=False, sep=';')
    print(f"Saved intermediate results to: {filename}")

def convert_simple_fast(input_file, output_file):
    """
    Simple and fast conversion for moderate sized files
    """
    start_time = time.time()

    print("Simple fast conversion...")

    # Read file
    df = pd.read_csv(input_file, header=None)
    df.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

    print(f"Read {len(df)} coordinate points")

    # Group and convert
    print("Grouping and converting...")
    result = []

    # Group by ID and assign sequential IDs
    for new_id, (trajectory_id, group) in enumerate(df.groupby('id'), 1):
        coords = list(zip(group['longitude'], group['latitude']))
        coord_strs = [f"{lon} {lat}" for lon, lat in coords]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        result.append({
            'id': new_id,
            'original_id': trajectory_id,
            'geom': linestring
        })

    # Save
    output_df = pd.DataFrame(result)
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Conversion completed!")
    print(f"Total trajectories: {len(output_df)}")
    print(f"Total points: {len(df)}")
    print(f"Processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    input_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/all_2hour_data/all_2hour_data_Jan.txt"
    output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/all_2hour_data/all_2hour_data_Jan_parallel.csv"

    file_size = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Input file size: {file_size:.2f} MB")

    # Always use parallel processing for maximum performance
    # Adjust batch size based on file size
    if file_size > 500:
        batch_size = 500000
    elif file_size > 100:
        batch_size = 200000
    else:
        batch_size = 100000

    print(f"Using batch size: {batch_size}")
    convert_parallel(input_file, output_file, batch_size=batch_size)