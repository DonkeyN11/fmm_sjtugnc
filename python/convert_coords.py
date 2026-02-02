#!/usr/bin/env python3
"""
This script converts coordinates in a CSV file's 'geom' column from a source
Coordinate Reference System (CRS) to a target CRS. It is designed to work with
WKT LINESTRING geometries.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

try:
    from pyproj import CRS, Transformer
    from pyproj.exceptions import CRSError
except ImportError:
    print("pyproj is not installed. Please install it using: pip install pyproj")
    exit(1)

# Set a high field size limit for CSV reader to handle large WKT strings
csv.field_size_limit(10**7)

Coordinate = Tuple[float, float]

def parse_linestring(wkt: str) -> List[Coordinate]:
    """Parse a WKT LINESTRING into a list of [lon, lat] coordinate pairs."""
    text = (wkt or "").strip()
    if not text or not text.upper().startswith("LINESTRING"):
        return []
    open_idx = text.find("(")
    close_idx = text.rfind(")")
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        return []
    body = text[open_idx + 1 : close_idx]
    coords: List[Coordinate] = []
    for token in body.split(","):
        parts = token.strip().split()
        if len(parts) != 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        coords.append((lon, lat))
    return coords

def format_linestring(coords: List[Coordinate]) -> str:
    """Format a list of coordinate pairs into a WKT LINESTRING string."""
    if not coords:
        return "LINESTRING EMPTY"
    coord_strs = [f"{lon:.8f} {lat:.8f}" for lon, lat in coords]
    return f"LINESTRING ({', '.join(coord_strs)})"

def main():
    """Main function to parse arguments and run the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert LINESTRING geometries in a CSV file to a new CRS."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input CSV file (e.g., 'cmm_trajectory.csv')."
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path for the output CSV file with converted coordinates."
    )
    parser.add_argument(
        "--source-crs",
        default="EPSG:4326",
        help="Source CRS (default: 'EPSG:4326' for WGS84 lon/lat)."
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:32649",
        help="Target CRS (default: 'EPSG:32649' for UTM Zone 49N/Hainan)."
    )
    parser.add_argument(
        "--geom-column",
        default="geom",
        help="Name of the geometry column containing WKT LINESTRINGs."
    )
    args = parser.parse_args()

    try:
        source_crs = CRS.from_user_input(args.source_crs)
        target_crs = CRS.from_user_input(args.target_crs)
    except CRSError as e:
        print(f"Error creating CRS: {e}")
        return

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    print(f"Reading from: {args.input_file}")
    print(f"Writing to: {args.output_file}")
    print(f"Converting from '{source_crs.name}' to '{target_crs.name}'")

    try:
        with args.input_file.open("r", newline="", encoding="utf-8") as infile, \
             args.output_file.open("w", newline="", encoding="utf-8") as outfile:

            reader = csv.reader(infile, delimiter=";")
            writer = csv.writer(outfile, delimiter=";")

            header = next(reader)
            writer.writerow(header)

            try:
                geom_col_idx = header.index(args.geom_column)
            except ValueError:
                print(f"Error: Geometry column '{args.geom_column}' not found in header.")
                return

            processed_rows = 0
            for row in reader:
                if len(row) <= geom_col_idx:
                    writer.writerow(row)
                    continue

                original_wkt = row[geom_col_idx]
                coords = parse_linestring(original_wkt)

                if coords:
                    lons = [pt[0] for pt in coords]
                    lats = [pt[1] for pt in coords]

                    # Perform the coordinate transformation
                    proj_xs, proj_ys = transformer.transform(lons, lats)

                    projected_coords = list(zip(proj_xs, proj_ys))
                    new_wkt = format_linestring(projected_coords)
                    row[geom_col_idx] = new_wkt
                
                writer.writerow(row)
                processed_rows += 1
            
            print(f"Successfully processed {processed_rows} rows.")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
