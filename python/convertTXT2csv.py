#!/usr/bin/env python3
import pandas as pd
import sys
from collections import defaultdict

def convert_txt_to_csv(input_file, output_file):
    """
    Convert trajectory data from TXT format to CSV format with LINESTRING geometry.

    Input format: ID, timestamp, speed, longitude, latitude
    Output format: id, geom (LINESTRING with coordinates)
    """

    # Read the input file
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file, header=None)
    df.columns = ['id', 'timestamp', 'speed', 'longitude', 'latitude']

    # Group by trajectory ID
    trajectory_groups = defaultdict(list)

    print("Processing trajectories...")
    for _, row in df.iterrows():
        trajectory_id = row['id']
        lon = row['longitude']
        lat = row['latitude']
        trajectory_groups[trajectory_id].append((lon, lat))

    # Convert to target format
    output_data = []
    for trajectory_id in sorted(trajectory_groups.keys()):
        coordinates = trajectory_groups[trajectory_id]

        # Format coordinates for LINESTRING
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

    print(f"Conversion completed!")
    print(f"Total trajectories processed: {len(output_data)}")
    print(f"Total coordinate points processed: {len(df)}")

if __name__ == "__main__":
    input_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan.txt"
    output_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/all_2hour_data_Jan_converted.csv"

    convert_txt_to_csv(input_file, output_file)