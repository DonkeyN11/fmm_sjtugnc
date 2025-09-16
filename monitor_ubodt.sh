#!/bin/bash
# Monitor ubodt_gen process

PID=4040885
OUTPUT_FILE="/home/dell/Czhang/input/map/shanghai_ubodt.txt"

if [ -d "/proc/$PID" ]; then
    echo "Process $PID is still running"
    echo "Running time: $(ps -p $PID -o etime= | xargs)"
    echo "CPU time: $(cat /proc/$PID/stat | cut -d' ' -f14-15 | awk '{print $1 + $2}') seconds"
    echo "Memory usage: $(ps -p $PID -o %mem= | xargs)%"
    echo "Output file size: $(ls -lh $OUTPUT_FILE | awk '{print $5}')"
    echo "System load: $(cat /proc/loadavg | cut -d' ' -f1-3)"
else
    echo "Process $PID has completed"
    echo "Final output file size: $(ls -lh $OUTPUT_FILE | awk '{print $5}')"
    echo "File exists: $([ -f "$OUTPUT_FILE" ] && echo 'Yes' || echo 'No')"
fi