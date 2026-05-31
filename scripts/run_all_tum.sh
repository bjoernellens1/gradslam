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
    # Run the system AS DESIGNED for accuracy: hybrid frame-to-model TSDF
    # tracking + mapping + periodic keyframe anchoring + feature PnP. The bare
    # `run_slam.py tum` defaults (fast_rgbd, mapping off, kf-tracking 0,
    # feature 0) disable every drift-mitigation path and are a speed preset,
    # not the recommended accuracy config — passing no flags badly
    # underrepresents the system. See BENCHMARK_RESULTS.md.
    python scripts/run_slam.py tum \
        --dataset-root /workspace/datasets/public/TUM/tum_rgbd/ \
        --sequence "$seq" \
        --output "/workspace/outputs/$seq" \
        --slam-backend rgbdtsdf \
        --enable-mapping \
        --keyframe-tracking-interval 10 \
        --feature-interval 5 \
        2>&1 | tail -20
done

echo ""
echo "=========================================="
echo "Benchmark complete! Check /workspace/outputs/"
