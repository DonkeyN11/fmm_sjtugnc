#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Chenzhang Ning

import pandas as pd
import re

def parse_pos_file(file_path):
    """
    Parse RTKLIB .pos file and extract coordinate data
    """
    coordinates = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip header lines (lines starting with %)
    data_lines = [line.strip() for line in lines if not line.strip().startswith('%')]

    for line in data_lines:
        if line.strip():
            # Split by whitespace and extract coordinates
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 3:
                # Format: time, latitude, longitude, height, Q, ns, ...
                try:
                    timestamp = parts[0] + ' ' + parts[1]
                    latitude = float(parts[2])
                    longitude = float(parts[3])
                    height = float(parts[4])

                    coordinates.append({
                        'timestamp': timestamp,
                        'latitude': latitude,
                        'longitude': longitude,
                        'height': height
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    continue

    return coordinates

def coordinates_to_linestring(coordinates):
    """
    Convert list of coordinate dictionaries to LINESTRING format
    """
    if not coordinates:
        return ""

    # Format: LINESTRING(longitude latitude, longitude latitude, ...)
    coord_pairs = []
    for coord in coordinates:
        # Note: LINESTRING format is (longitude latitude), not (latitude longitude)
        coord_str = f"{coord['longitude']} {coord['latitude']}"
        coord_pairs.append(coord_str)

    return f"LINESTRING({','.join(coord_pairs)})"

def convert_pos_to_csv(input_file, output_file):
    """
    Convert RTKLIB .pos file to CSV with LINESTRING format
    """
    # Parse the input file
    coordinates = parse_pos_file(input_file)

    if not coordinates:
        print("No valid coordinates found in the input file")
        return

    # Convert to LINESTRING format
    linestring = coordinates_to_linestring(coordinates)

    # Create DataFrame with trips.csv format
    df = pd.DataFrame({
        'id': [1],
        'geom': [linestring]
    })

    # Save to CSV
    df.to_csv(output_file, sep=';', index=False)
    print(f"Converted {len(coordinates)} coordinate points to {output_file}")
    print(f"Generated LINESTRING with {len(coordinates)} coordinate pairs")

def main():
    """
    Main function to convert GGA/POS data to trips.csv format
    """
    input_file = "/home/dell/Czhang/fmm_sjtugnc/input/trajectory/zhai/rtk_data/20240228/hpm100_SPP.pos"
    output_file = "/home/dell/Czhang/fmm_sjtugnc/output/trips_converted.csv"

    convert_pos_to_csv(input_file, output_file)

if __name__ == "__main__":
    main()