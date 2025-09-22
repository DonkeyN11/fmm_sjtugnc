#!/usr/bin/env python3
"""
CSV ID Correspondence Checker

Check if IDs in two CSV files correspond to each other.
Writes non-matching IDs to contradict.csv file.
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
from pathlib import Path
import time

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)

class IDChecker:
    def __init__(self, file1_path, file2_path, output_file="contradict.csv"):
        """
        Initialize the ID checker.

        Args:
            file1_path: Path to first CSV file
            file2_path: Path to second CSV file
            output_file: Path to output file for non-matching IDs
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.output_file = output_file

        # Statistics
        self.stats = {
            'file1_ids': set(),
            'file2_ids': set(),
            'matching_ids': set(),
            'file1_only': set(),
            'file2_only': set(),
            'total_processed': 0,
            'start_time': time.time()
        }

    def detect_delimiter(self, file_path, sample_lines=5):
        """
        Detect the delimiter used in a CSV file.

        Args:
            file_path: Path to CSV file
            sample_lines: Number of lines to sample for detection

        Returns:
            Detected delimiter (',' or ';')
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                if ';' in line:
                    return ';'
                elif ',' in line:
                    return ','
        return ','  # Default to comma

    def read_csv_ids(self, file_path, id_column_name='id'):
        """
        Read IDs from a CSV file.

        Args:
            file_path: Path to CSV file
            id_column_name: Name of ID column

        Returns:
            Set of IDs found in the file
        """
        print(f"Reading IDs from {file_path}...")

        # Detect delimiter
        delimiter = self.detect_delimiter(file_path)
        print(f"Detected delimiter: '{delimiter}'")

        ids = set()
        total_rows = 0

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read header to find ID column
            header = f.readline().strip()
            if not header:
                print(f"Warning: Empty file {file_path}")
                return ids

            # Parse header
            columns = [col.strip().lower() for col in header.split(delimiter)]
            print(f"Columns found: {columns}")

            # Find ID column
            id_column_index = None
            for i, col in enumerate(columns):
                if col == id_column_name.lower():
                    id_column_index = i
                    break

            if id_column_index is None:
                print(f"Error: ID column '{id_column_name}' not found in {file_path}")
                print(f"Available columns: {columns}")
                return ids

            print(f"Using column '{columns[id_column_index]}' at index {id_column_index} for IDs")

            # Read IDs
            reader = csv.DictReader(f, delimiter=delimiter, fieldnames=columns)
            for row in reader:
                if id_column_index < len(row):
                    id_value = row[columns[id_column_index]]
                    if id_value:
                        ids.add(str(id_value).strip())
                total_rows += 1

        print(f"Read {len(ids)} unique IDs from {total_rows} total rows")
        return ids

    def check_correspondence(self):
        """
        Check ID correspondence between the two files.

        Returns:
            Dictionary containing correspondence results
        """
        print("Checking ID correspondence...")

        # Read IDs from both files
        self.stats['file1_ids'] = self.read_csv_ids(self.file1_path, 'id')
        self.stats['file2_ids'] = self.read_csv_ids(self.file2_path, 'id')

        # Find matching and non-matching IDs
        self.stats['matching_ids'] = self.stats['file1_ids'] & self.stats['file2_ids']
        self.stats['file1_only'] = self.stats['file1_ids'] - self.stats['file2_ids']
        self.stats['file2_only'] = self.stats['file2_ids'] - self.stats['file1_ids']
        self.stats['total_processed'] = len(self.stats['file1_ids']) + len(self.stats['file2_ids'])

        file1_total = len(self.stats['file1_ids'])
        file2_total = len(self.stats['file2_ids'])

        return {
            'matching_count': len(self.stats['matching_ids']),
            'file1_only_count': len(self.stats['file1_only']),
            'file2_only_count': len(self.stats['file2_only']),
            'file1_total': file1_total,
            'file2_total': file2_total
        }

    def write_contradictions(self):
        """
        Write non-matching IDs to contradict.csv file.
        """
        print(f"Writing contradictions to {self.output_file}...")

        with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'source_file', 'status'])

            # Write IDs only in file1
            for id_value in sorted(self.stats['file1_only']):
                writer.writerow([id_value, os.path.basename(self.file1_path), 'only_in_file1'])

            # Write IDs only in file2
            for id_value in sorted(self.stats['file2_only']):
                writer.writerow([id_value, os.path.basename(self.file2_path), 'only_in_file2'])

        print(f"Written {len(self.stats['file1_only']) + len(self.stats['file2_only'])} contradictions")

    def print_statistics(self):
        """Print detailed statistics."""
        end_time = time.time()
        processing_time = end_time - self.stats['start_time']

        print("\n" + "="*60)
        print("ID CORRESPONDENCE CHECK RESULTS")
        print("="*60)
        print(f"File 1: {os.path.basename(self.file1_path)}")
        print(f"  - Total IDs: {len(self.stats['file1_ids']):,}")
        print(f"  - IDs only in this file: {len(self.stats['file1_only']):,}")
        print()
        print(f"File 2: {os.path.basename(self.file2_path)}")
        print(f"  - Total IDs: {len(self.stats['file2_ids']):,}")
        print(f"  - IDs only in this file: {len(self.stats['file2_only']):,}")
        print()
        print(f"Matching IDs: {len(self.stats['matching_ids']):,}")
        print(f"Total contradictions: {len(self.stats['file1_only']) + len(self.stats['file2_only']):,}")
        print()
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Contradictions saved to: {self.output_file}")
        print("="*60)

    def process(self):
        """
        Main processing method.
        """
        try:
            print("Starting ID correspondence check...")
            print(f"File 1: {self.file1_path}")
            print(f"File 2: {self.file2_path}")
            print(f"Output: {self.output_file}")

            # Check correspondence
            results = self.check_correspondence()

            # Write contradictions
            if results['file1_only_count'] > 0 or results['file2_only_count'] > 0:
                self.write_contradictions()
            else:
                print("No contradictions found - all IDs match perfectly!")

            # Print statistics
            self.print_statistics()

            return results

        except Exception as e:
            print(f"Error during processing: {e}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Check ID correspondence between two CSV files')
    parser.add_argument('file1', help='Path to first CSV file')
    parser.add_argument('file2', help='Path to second CSV file')
    parser.add_argument('--output', '-o', default='contradict.csv',
                       help='Output file for contradictions (default: contradict.csv)')

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.file1):
        print(f"Error: File '{args.file1}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.file2):
        print(f"Error: File '{args.file2}' does not exist")
        sys.exit(1)

    # Create and run checker
    checker = IDChecker(
        file1_path=args.file1,
        file2_path=args.file2,
        output_file=args.output
    )

    try:
        results = checker.process()
        print("\nProcessing completed successfully!")

        # Exit with error code if contradictions found
        if results['file1_only_count'] > 0 or results['file2_only_count'] > 0:
            print(f"Found {results['file1_only_count'] + results['file2_only_count']} contradictions")
            sys.exit(1)
        else:
            print("All IDs match perfectly!")
            sys.exit(0)

    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()