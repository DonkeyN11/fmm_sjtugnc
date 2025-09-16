#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import os

def convert_txt_to_csv_fast(input_file, output_file):
    """
    Fast conversion using optimized pandas operations
    """
    start_time = time.time()

    print(f"Fast conversion started...")
    print(f"Input file: {input_file}")

    # Read file with optimized parameters
    print("Reading file...")
    df = pd.read_csv(
        input_file,
        header=None,
        dtype={
            0: 'object',  # ID as string
            1: 'int64',   # timestamp
            2: 'float32', # speed
            3: 'float64', # longitude
            4: 'float64'  # latitude
        }
    )

    df.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

    print(f"Read {len(df)} coordinate points")

    # Group by ID and aggregate coordinates
    print("Grouping coordinates...")
    grouped = df.groupby('id', as_index=False).apply(
        lambda x: pd.Series({
            'coordinates': list(zip(x['longitude'], x['latitude']))
        })
    )

    print(f"Found {len(grouped)} unique trajectories")

    # Convert to LINESTRING format
    print("Converting to LINESTRING format...")

    def coords_to_linestring(coords):
        coord_strs = [f"{lon} {lat}" for lon, lat in coords]
        return f"LINESTRING({','.join(coord_strs)})"

    grouped['geom'] = grouped['coordinates'].apply(coords_to_linestring)

    # Prepare final output
    output_df = grouped[['id', 'geom']]

    # Write to CSV
    print(f"Writing output to: {output_file}")
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Fast conversion completed!")
    print(f"Total trajectories: {len(output_df)}")
    print(f"Total points: {len(df)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {len(df)/processing_time:.2f} points/second")

def convert_txt_to_csv_chunked_fast(input_file, output_file, chunk_size=100000):
    """
    Fast chunked processing for very large files
    """
    start_time = time.time()

    print(f"Fast chunked conversion (chunk_size={chunk_size})")
    print(f"Input file: {input_file}")

    # Use defaultdict for efficient grouping
    trajectory_groups = defaultdict(list)
    total_lines = 0

    print("Reading and grouping in chunks...")
    chunk_count = 0

    for chunk in pd.read_csv(
        input_file,
        header=None,
        chunksize=chunk_size,
        dtype={
            0: 'object',
            1: 'int64',
            2: 'float32',
            3: 'float64',
            4: 'float64'
        }
    ):
        chunk.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

        # Process chunk efficiently
        for _, row in chunk.iterrows():
            trajectory_groups[row['id']].append((row['longitude'], row['latitude']))

        total_lines += len(chunk)
        chunk_count += 1

        if chunk_count % 10 == 0:
            print(f"Processed {chunk_count} chunks ({total_lines} lines)")

    print(f"Grouped {total_lines} points into {len(trajectory_groups)} trajectories")

    # Convert to output format
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
            print(f"Processed {i}/{len(trajectory_groups)} trajectories")

    # Create DataFrame and save
    output_df = pd.DataFrame(output_data)

    print(f"Writing output to: {output_file}")
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Fast chunked conversion completed!")
    print(f"Total trajectories: {len(output_df)}")
    print(f"Total points: {total_lines}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {total_lines/processing_time:.2f} points/second")

if __name__ == "__main__":
    input_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan.txt"
    output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan_fast.csv"

    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"Input file size: {file_size:.2f} MB")

    if file_size > 1000:  # Use chunked for very large files
        convert_txt_to_csv_chunked_fast(input_file, output_file, chunk_size=200000)
    else:
        convert_txt_to_csv_fast(input_file, output_file)