#!/usr/bin/env python3
"""
Merge multiple CMM trajectory CSV files into a single file.
"""

import argparse
import csv
import glob
import sys
from pathlib import Path
from typing import List

# Increase CSV field size limit to handle large trajectories
max_int = sys.maxsize
while True:
    # Decrease the value until it fits in a C long
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)


def merge_csv_files(
    input_pattern: str,
    output_file: str,
    delimiter: str = ';',
    skip_duplicates: bool = False,
) -> None:
    """
    Merge multiple CSV files with the same structure.

    Args:
        input_pattern: Glob pattern for input files (e.g., "dataset/*/mr/cmm_trajectory.csv")
        output_file: Path to output merged CSV
        delimiter: CSV delimiter (default: ';')
        skip_duplicates: Whether to skip duplicate trajectory IDs
    """
    files = sorted(glob.glob(input_pattern))

    if not files:
        print(f"Error: No files found matching pattern: {input_pattern}")
        return

    print(f"Found {len(files)} files to merge")

    # Read header from first file
    with open(files[0], 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)

    print(f"Header: {header}")

    # Track seen IDs if skip_duplicates is enabled
    seen_ids = set()
    rows_written = 0
    skipped_count = 0

    # Write merged file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=delimiter)
        writer.writerow(header)

        for file_path in files:
            print(f"  Processing: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f_in:
                reader = csv.DictReader(f_in, delimiter=delimiter)

                for row in reader:
                    # Skip duplicate trajectory IDs if requested
                    if skip_duplicates and 'id' in row:
                        traj_id = row['id']
                        if traj_id in seen_ids:
                            skipped_count += 1
                            continue
                        seen_ids.add(traj_id)

                    # Write row
                    writer.writerow(row.values())
                    rows_written += 1

    print(f"\nMerge complete!")
    print(f"  Total rows written: {rows_written}")
    if skip_duplicates:
        print(f"  Duplicates skipped: {skipped_count}")
    print(f"  Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV files with the same structure"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Glob pattern for input files (e.g., 'dataset/*/mr/cmm_trajectory.csv')"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="merged.csv",
        help="Path to output merged CSV file (default: merged.csv)"
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default=';',
        help="CSV delimiter (default: ';')"
    )

    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip rows with duplicate 'id' field values"
    )

    args = parser.parse_args()

    merge_csv_files(
        input_pattern=args.input,
        output_file=args.output,
        delimiter=args.delimiter,
        skip_duplicates=args.skip_duplicates,
    )


if __name__ == "__main__":
    main()
