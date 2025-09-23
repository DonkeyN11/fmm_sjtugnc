#!/bin/bash

# FMM Processing Pipeline Script
# This script runs the FMM algorithm, processes the output, and generates visualizations

set -e  # Exit on any error

echo "=== FMM Processing Pipeline Started ==="

# Step 1: Run FMM with the specified configuration
echo "Step 1: Running FMM algorithm..."
fmm input/config/fmm_config_omp.xml

# Check if FMM output was generated
if [ ! -f "output/mr_cumu_ts.txt" ]; then
    echo "Error: FMM output file output/mr_cumu_ts.txt not found!"
    exit 1
fi

echo "FMM completed successfully. Output file: output/mr_cumu_ts.txt"

# Step 2: Process the output with rearrange_mr.py
echo "Step 2: Processing FMM output..."
python python/rearrange_mr.py output/mr_cumu_ts.txt

# Check if rearranged file was generated
if [ ! -f "output/mr_cumu_ts_rearranged.txt" ]; then
    echo "Error: Rearranged output file output/mr_cumu_ts_rearranged.txt not found!"
    exit 1
fi

echo "Processing completed. Rearranged file: output/mr_cumu_ts_rearranged.txt"

# Step 3: Generate visualization
echo "Step 3: Generating visualization..."
python python/plot_raw_points.py output/mr_cumu_ts_rearranged.csv -s input/map/haikou/edges.shp

echo "=== FMM Processing Pipeline Completed Successfully ==="
echo "All steps completed. Check the output directory for results."