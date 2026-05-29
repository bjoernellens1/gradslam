#!/bin/bash
set -e

TARGET_DIR="/mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity_extracted"
CSV_FILE="${TARGET_DIR}/frames.csv"
IMAGES_DIR="${TARGET_DIR}/images"

echo "=========================================================="
echo "Checking status of extraction and starting SLAM pipeline..."
echo "=========================================================="

if [ -f "$CSV_FILE" ]; then
    echo "✓ Extraction is already complete! frames.csv is present."
else
    echo "Waiting for extraction to finish (checking for frames.csv)..."
    while [ ! -f "$CSV_FILE" ]; do
        if [ -d "$IMAGES_DIR" ]; then
            # Using find to count files efficiently
            COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -type f | wc -l)
            echo "Still extracting... currently $COUNT frames extracted."
        else
            echo "Extraction directory not found yet..."
        fi
        sleep 10
    done
    echo "✓ Extraction complete! frames.csv has been written."
fi

# Print metadata info
echo ""
echo "Dataset summary:"
if [ -f "${TARGET_DIR}/camera_info.json" ]; then
    cat "${TARGET_DIR}/camera_info.json"
fi
echo "Total frames in CSV: $(wc -l < "$CSV_FILE")"
echo ""

echo "Running SLAM on the extracted dataset..."
python3 scripts/run_slam.py normalized --capture-dir "$TARGET_DIR"
