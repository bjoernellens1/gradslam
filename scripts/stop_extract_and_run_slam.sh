#!/bin/bash
set -e

echo "=========================================="
echo "Step 1: Stopping active extraction process..."
echo "=========================================="
pkill -f extract_realsense_bag.py || echo "No active extraction process found."
sleep 1

echo ""
echo "=========================================="
echo "Step 2: Instantly finalising dataset..."
echo "=========================================="
python3 scripts/fast_finalize.py

echo ""
echo "=========================================="
echo "Step 3: Running SLAM inside the ROCm Docker container..."
echo "=========================================="
docker compose run --rm gradslam python3 scripts/run_slam.py normalized \
    --capture-dir /workspace/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity_extracted \
    --gt-file /workspace/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity/groundtruth_tum.txt \
    --output /workspace/outputs/normalized_slam_results
