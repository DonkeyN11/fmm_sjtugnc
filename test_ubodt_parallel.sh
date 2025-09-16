#!/bin/bash
# Test script for optimized ubodt_gen parallel processing

echo "=== Testing Optimized UBODT Generation ==="
echo "System Information:"
echo "Number of CPU cores: $(nproc)"
echo "Available memory: $(free -h)"
echo ""

# Test with different configurations
CONFIGS=(
    "input/config/ubodt_config_omp.xml"
)

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        echo "Testing with config: $config"
        echo "Start time: $(date)"

        # Run ubodt_gen with optimized parallel processing
        time ./bin/ubodt_gen --config $config

        echo "End time: $(date)"
        echo "----------------------------------------"
    else
        echo "Config file not found: $config"
    fi
done

echo "=== Test Completed ==="