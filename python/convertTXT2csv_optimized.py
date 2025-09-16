#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time
import gc
from tqdm import tqdm
import os

def process_trajectory_group(group_data):
    """
    Process a group of trajectories in parallel
    """
    trajectory_id, coordinates = group_data
    coord_strs = [f"{lon} {lat}" for lon, lat in coordinates]
    linestring = f"LINESTRING({','.join(coord_strs)})"
    return {
        'id': trajectory_id,
        'geom': linestring
    }

def read_in_chunks(file_path, chunk_size=100000):
    """
    Read large file in chunks to save memory
    """
    chunks = []
    for chunk in pd.read_csv(file_path, header=None, chunksize=chunk_size):
        chunk.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']
        chunks.append(chunk)
    return chunks

def convert_txt_to_csv_parallel(input_file, output_file, num_processes=None, chunk_size=100000):
    """
    Convert trajectory data with parallel processing and memory optimization
    """
    start_time = time.time()

    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free

    print(f"Starting conversion with {num_processes} processes...")
    print(f"Chunk size: {chunk_size}")

    # Method 1: Memory-efficient chunk processing for very large files
    trajectory_groups = defaultdict(list)
    total_lines = 0

    print("Reading file in chunks...")
    for i, chunk in enumerate(read_in_chunks(input_file, chunk_size)):
        print(f"Processing chunk {i+1}...")

        # Group coordinates by trajectory ID
        for _, row in chunk.iterrows():
            trajectory_id = row['id']
            lon = row['longitude']
            lat = row['latitude']
            trajectory_groups[trajectory_id].append((lon, lat))

        total_lines += len(chunk)

        # Free memory
        del chunk
        gc.collect()

    print(f"Grouped {total_lines} coordinate points into {len(trajectory_groups)} trajectories")

    # Prepare data for parallel processing
    group_data = list(trajectory_groups.items())

    # Method 2: Parallel processing with progress bar
    print("Converting trajectories in parallel...")
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_trajectory_group, group_data),
            total=len(group_data),
            desc="Processing trajectories"
        ))

    # Create output DataFrame
    output_df = pd.DataFrame(results)

    # Write to CSV
    print("Writing output file...")
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Conversion completed!")
    print(f"Total trajectories processed: {len(output_df)}")
    print(f"Total coordinate points processed: {total_lines}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {total_lines/processing_time:.2f} points/second")

def convert_txt_to_csv_vectorized(input_file, output_file):
    """
    Vectorized approach using pandas operations for better performance
    """
    start_time = time.time()

    print(f"Reading input file: {input_file}")

    # Read file with optimized dtype
    dtypes = {
        0: 'object',  # ID as string
        1: 'int64',   # timestamp
        2: 'float32', # speed
        3: 'float64', # longitude
        4: 'float64'  # latitude
    }

    df = pd.read_csv(input_file, header=None, dtype=dtypes)
    df.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

    # Vectorized grouping using pandas groupby
    print("Grouping coordinates by trajectory ID...")
    grouped = df.groupby('id')[['longitude', 'latitude']].apply(
        lambda x: list(zip(x['longitude'], x['latitude']))
    ).to_dict()

    # Vectorized coordinate formatting
    print("Converting to LINESTRING format...")
    output_data = []

    for trajectory_id in tqdm(grouped.keys(), desc="Formatting trajectories"):
        coordinates = grouped[trajectory_id]
        coord_array = np.array(coordinates)
        coord_strs = [f"{lon} {lat}" for lon, lat in coord_array]
        linestring = f"LINESTRING({','.join(coord_strs)})"

        output_data.append({
            'id': trajectory_id,
            'geom': linestring
        })

    # Create output DataFrame
    output_df = pd.DataFrame(output_data)

    # Write to CSV
    print(f"Writing output file: {output_file}")
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Vectorized conversion completed!")
    print(f"Total trajectories processed: {len(output_df)}")
    print(f"Total coordinate points processed: {len(df)}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {len(df)/processing_time:.2f} points/second")

def convert_txt_to_csv_memory_efficient(input_file, output_file, chunk_size=50000):
    """
    Memory-efficient approach for extremely large files
    """
    start_time = time.time()

    print(f"Memory-efficient conversion (chunk_size={chunk_size})")

    # First pass: collect all unique trajectory IDs
    print("First pass: collecting trajectory IDs...")
    unique_ids = set()
    total_lines = 0

    for chunk in pd.read_csv(input_file, header=None, chunksize=chunk_size):
        unique_ids.update(chunk[0].unique())
        total_lines += len(chunk)

    print(f"Found {len(unique_ids)} unique trajectories in {total_lines} lines")

    # Second pass: process each trajectory separately
    print("Second pass: processing trajectories...")
    output_data = []

    for trajectory_id in tqdm(unique_ids, desc="Processing trajectories"):
        coordinates = []

        # Read file again to extract coordinates for this trajectory
        for chunk in pd.read_csv(input_file, header=None, chunksize=chunk_size):
            chunk.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']
            trajectory_data = chunk[chunk['id'] == trajectory_id]

            if not trajectory_data.empty:
                coords = list(zip(trajectory_data['longitude'], trajectory_data['latitude']))
                coordinates.extend(coords)

        if coordinates:
            coord_strs = [f"{lon} {lat}" for lon, lat in coordinates]
            linestring = f"LINESTRING({','.join(coord_strs)})"

            output_data.append({
                'id': trajectory_id,
                'geom': linestring
            })

    # Create output DataFrame
    output_df = pd.DataFrame(output_data)

    # Write to CSV
    print(f"Writing output file: {output_file}")
    output_df.to_csv(output_file, index=False, sep=';')

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Memory-efficient conversion completed!")
    print(f"Total trajectories processed: {len(output_df)}")
    print(f"Total coordinate points processed: {total_lines}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Performance: {total_lines/processing_time:.2f} points/second")

if __name__ == "__main__":
    input_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan.txt"

    # Check file size to determine best approach
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB

    print(f"Input file size: {file_size:.2f} MB")
    print(f"Available CPU cores: {cpu_count()}")

    # Choose the best method based on file size
    if file_size > 500:  # Very large files
        print("Very large file detected, using memory-efficient approach...")
        output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan_memory_efficient.csv"
        convert_txt_to_csv_memory_efficient(input_file, output_file, chunk_size=100000)
    elif file_size > 100:  # Large files
        print("Large file detected, using parallel processing...")
        output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan_parallel.csv"
        convert_txt_to_csv_parallel(input_file, output_file, num_processes=4)
    else:  # Small to medium files
        print("Medium file detected, using vectorized approach...")
        output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan_vectorized.csv"
        convert_txt_to_csv_vectorized(input_file, output_file)