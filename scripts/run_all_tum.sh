#!/bin/bash
# Run SLAM on all TUM sequences in ROCm container

SEQUENCES=(
    "freiburg1_desk"
    "freiburg1_xyz"
    "freiburg2_desk"
    "freiburg3_long_office_household"
)

echo "Running benchmarks on all TUM sequences..."
echo "=========================================="

for seq in "${SEQUENCES[@]}"; do
    echo ""
    echo "Running: $seq"
    python scripts/run_slam.py tum \
        --dataset-root /workspace/datasets/public/TUM/tum_rgbd/ \
        --sequence "$seq" \
        --output "/workspace/outputs/$seq" \
        2>&1 | tail -20
done

echo ""
echo "=========================================="
echo "Benchmark complete! Check /workspace/outputs/"
