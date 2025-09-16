#!/usr/bin/env python3
import pandas as pd
import time
import os
from collections import defaultdict

def convert_batch(input_file, output_file, batch_size=1000000):
    """
    Convert in batches to balance speed and memory
    """
    start_time = time.time()

    print(f"Batch conversion started...")
    print(f"Input file: {input_file}")
    print(f"Batch size: {batch_size}")

    # Count total lines first
    print("Counting total lines...")
    total_lines = sum(1 for _ in open(input_file))
    print(f"Total lines: {total_lines}")

    # Process in batches
    trajectory_groups = defaultdict(list)
    processed_lines = 0
    batch_num = 0

    print("Processing in batches...")

    # Read file in chunks
    for chunk in pd.read_csv(input_file, header=None, chunksize=batch_size):
        chunk.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

        # Process batch
        for _, row in chunk.iterrows():
            trajectory_groups[row['id']].append((row['longitude'], row['latitude']))

        processed_lines += len(chunk)
        batch_num += 1

        # Progress update
        progress = (processed_lines / total_lines) * 100
        print(f"Batch {batch_num}: {processed_lines:,}/{total_lines:,} lines ({progress:.1f}%)")

        # Convert and save intermediate results every few batches
        if batch_num % 5 == 0:
            save_intermediate_results(trajectory_groups, f"{output_file}.batch_{batch_num}")

    print(f"Grouped {processed_lines} points into {len(trajectory_groups)} trajectories")

    # Final conversion
    print("Converting to LINESTRING format...")
    output_data = []

    for i, (trajectory_id, coordinates) in enumerate(trajectory_groups.items()):
        coord_strs = [f"{lon} {lat}" for lon, lat in coordinates]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        output_data.append({
            'id': trajectory_id,
            'geom': linestring
        })

        if i % 10000 == 0:
            print(f"Processed {i:,}/{len(trajectory_groups):,} trajectories")

    # Save final result
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Batch conversion completed!")
    print(f"Total trajectories: {len(output_df)}")
    print(f"Total points: {processed_lines}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {processed_lines/processing_time:.2f} points/second")

def save_intermediate_results(trajectory_groups, filename):
    """Save intermediate results to avoid data loss"""
    output_data = []

    for trajectory_id, coordinates in trajectory_groups.items():
        coord_strs = [f"{lon} {lat}" for lon, lat in coordinates]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        output_data.append({
            'id': trajectory_id,
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

    # Group by ID
    for trajectory_id, group in df.groupby('id'):
        coords = list(zip(group['longitude'], group['latitude']))
        coord_strs = [f"{lon} {lat}" for lon, lat in coords]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        result.append({
            'id': trajectory_id,
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
    input_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan.txt"
    output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan_batch.csv"

    file_size = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Input file size: {file_size:.2f} MB")

    if file_size > 500:
        convert_batch(input_file, output_file, batch_size=500000)
    else:
        convert_simple_fast(input_file, output_file)