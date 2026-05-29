#!/bin/bash
set -e

echo "=========================================="
echo "Step 1: Finalizing partially extracted dataset..."
echo "=========================================="
python3 scripts/finalize_extracted_dataset.py

echo ""
echo "=========================================="
echo "Step 2: Running SLAM on extracted frames..."
echo "=========================================="
python3 scripts/run_slam.py normalized \
    --capture-dir /mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity_extracted \
    --gt-file /mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity/groundtruth_tum.txt
