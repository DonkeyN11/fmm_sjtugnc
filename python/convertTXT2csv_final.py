#!/usr/bin/env python3
"""
Final optimized script to convert trajectory data from TXT to CSV format with integer IDs.
This script integrates ID conversion functionality directly into the conversion process.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import time
import os

def convert_txt_to_csv_with_integer_ids(input_file, output_file, batch_size=1000000):
    """
    Convert trajectory data from TXT format to CSV format with integer IDs.

    Input format: ID, timestamp, speed, longitude, latitude
    Output format: id (integer), geom (LINESTRING with coordinates)
    """
    start_time = time.time()

    print(f"=== Trajectory Data Conversion with Integer IDs ===")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Batch size: {batch_size}")

    # Count total lines first for progress tracking
    print("Counting total lines...")
    total_lines = sum(1 for _ in open(input_file))
    print(f"Total lines: {total_lines:,}")

    # Process in batches to manage memory
    trajectory_groups = defaultdict(list)
    processed_lines = 0
    batch_num = 0

    print("Processing trajectory data...")

    # Read file in chunks
    for chunk in pd.read_csv(input_file, header=None, chunksize=batch_size):
        chunk.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

        # Process batch - group coordinates by trajectory ID
        for _, row in chunk.iterrows():
            trajectory_groups[row['id']].append((row['longitude'], row['latitude']))

        processed_lines += len(chunk)
        batch_num += 1

        # Progress update
        progress = (processed_lines / total_lines) * 100
        print(f"Batch {batch_num}: {processed_lines:,}/{total_lines:,} lines ({progress:.1f}%)")

    print(f"Grouped {processed_lines:,} coordinate points into {len(trajectory_groups):,} trajectories")

    # Create ID mapping from string to integer (1-based)
    print("Creating integer ID mapping...")
    unique_ids = sorted(trajectory_groups.keys())
    id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_ids)}

    print(f"ID mapping created with {len(id_mapping):,} unique trajectories")
    print(f"Sample mapping (first 5):")
    for old_id, new_id in list(id_mapping.items())[:5]:
        print(f"  {old_id} -> {new_id}")

    # Convert to final format with integer IDs
    print("Converting to LINESTRING format with integer IDs...")
    output_data = []

    for i, (trajectory_id, coordinates) in enumerate(trajectory_groups.items()):
        # Format coordinates for LINESTRING
        coord_strs = [f"{lon} {lat}" for lon, lat in coordinates]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        # Use integer ID
        integer_id = id_mapping[trajectory_id]

        output_data.append({
            'id': integer_id,
            'geom': linestring
        })

        if i % 10000 == 0:
            print(f"Processed {i:,}/{len(trajectory_groups):,} trajectories")

    # Create and save final result
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False, sep=';')

    # Save ID mapping for reference
    mapping_file = output_file.replace('.csv', '_id_mapping.csv')
    mapping_df = pd.DataFrame(list(id_mapping.items()), columns=['original_id', 'integer_id'])
    mapping_df.to_csv(mapping_file, index=False)
    print(f"ID mapping saved to: {mapping_file}")

    # Performance statistics
    end_time = time.time()
    processing_time = end_time - start_time

    print("\n=== Conversion Completed ===")
    print(f"✓ Total trajectories: {len(output_df):,}")
    print(f"✓ Total coordinate points: {processed_lines:,}")
    print(f"✓ Processing time: {processing_time:.2f} seconds")
    print(f"✓ Performance: {processed_lines/processing_time:,.0f} points/second")
    print(f"✓ Output file: {output_file}")
    print(f"✓ ID mapping file: {mapping_file}")

    # Validate output
    print("\n=== Output Validation ===")
    print(f"✓ Output file exists: {os.path.exists(output_file)}")
    print(f"✓ ID mapping file exists: {os.path.exists(mapping_file)}")
    print(f"✓ Output file size: {os.path.getsize(output_file)/(1024*1024):.1f} MB")

    # Check a few sample IDs
    sample_ids = output_df['id'].head(10).tolist()
    print(f"✓ Sample integer IDs: {sample_ids}")
    print(f"✓ All IDs are integers: {all(isinstance(id, int) or id.isdigit() for id in sample_ids)}")

    return id_mapping

def convert_simple_with_integer_ids(input_file, output_file):
    """
    Simple conversion for smaller files with integer IDs
    """
    start_time = time.time()

    print("Simple conversion with integer IDs...")

    # Read file
    df = pd.read_csv(input_file, header=None)
    df.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

    print(f"Read {len(df):,} coordinate points")

    # Create ID mapping
    unique_ids = df['id'].unique()
    id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_ids)}

    print(f"Found {len(unique_ids):,} unique trajectories")
    print(f"Sample ID mapping (first 3):")
    for old_id, new_id in list(id_mapping.items())[:3]:
        print(f"  {old_id} -> {new_id}")

    # Group and convert with integer IDs
    print("Grouping and converting...")
    result = []

    for trajectory_id, group in df.groupby('id'):
        coords = list(zip(group['longitude'], group['latitude']))
        coord_strs = [f"{lon} {lat}" for lon, lat in coords]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        # Use integer ID
        integer_id = id_mapping[trajectory_id]

        result.append({
            'id': integer_id,
            'geom': linestring
        })

    # Save
    output_df = pd.DataFrame(result)
    output_df.to_csv(output_file, index=False, sep=';')

    # Save ID mapping
    mapping_file = output_file.replace('.csv', '_id_mapping.csv')
    mapping_df = pd.DataFrame(list(id_mapping.items()), columns=['original_id', 'integer_id'])
    mapping_df.to_csv(mapping_file, index=False)
    print(f"ID mapping saved to: {mapping_file}")

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Simple conversion completed!")
    print(f"Total trajectories: {len(output_df):,}")
    print(f"Total points: {len(df):,}")
    print(f"Processing time: {processing_time:.2f} seconds")

    return id_mapping

if __name__ == "__main__":
    # Input and output files
    input_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan.txt"
    output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan_final.csv"

    # Check file size to determine best approach
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"Input file size: {file_size:.1f} MB")

    # Choose appropriate method based on file size
    if file_size > 500:
        print("Large file detected, using batch processing...")
        convert_txt_to_csv_with_integer_ids(input_file, output_file, batch_size=1000000)
    else:
        print("Medium file detected, using simple processing...")
        convert_simple_with_integer_ids(input_file, output_file)