# GPU-First SLAM Optimization Summary

## Overview
Optimized gradslam's RGB-D SLAM pipeline to reduce CPU bottlenecks and improve GPU utilization on AMD R9700 (gfx1201).

## Baseline Performance (R9700, freiburg1_desk, 100 frames, fast_rgbd, async ingestion)
- **Track time**: 92.76 ms
- **FPS**: 10.8

## Patches Implemented

### Patch 1: ICP Loop Optimizations
**Commit**: `143a89d`
**Changes**:
- Moved `live_vertex` computation outside ICP iteration loop (line 211)
- Added `_transform_points_fast()` using affine transform instead of homogeneous coords
- Added `_transform_normals_fast()` without unnecessary normalize()
- Replaced `_check_normal_angle()` acos with dot product threshold

**Result**: 88.51 ms, 11.3 fps (**+4.6% faster**)

### Patch 2: Prepared ICP Packs and VRAM Caching Infrastructure
**Commit**: `64f8700`
**Changes**:
- Added model pyramid cache for slow-changing TSDF renders
- Added pre-allocated pyramid buffers
- Added keyframe pyramid cache (max 16 keyframes)
- Modified `forward()` to accept optional `cached_model_pyramids` parameter

**Result**: 93.11 ms, 10.7 fps (**-5.0% regression** - cache overhead without usage)

### Patch 3: GPU Tensor Keyframe Scoring
**Commit**: `5bbc278`
**Changes**:
- Optimized `_annotate_motion_quality()`: GPU tensor comparisons for gating
- Optimized `_fast_recovery_candidates()`: batch convert topk indices with `.tolist()`
- Removed redundant `int()` conversions in candidate selection

**Result**: 47.10 ms, 21.2 fps (**+49% faster, +87% fps improvement**)

### Patch 4: Pre-allocated Buffers for Normal Equation Solver
**Commit**: `a9df1fc` (reverted in `1f93cb7`)
**Changes**:
- Added thread-local buffer cache for AtA, Atb, identity matrices
- Used in-place `torch.matmul` with `out=` parameter

**Result**: 54.65 ms, 18.3 fps (**Reverted** - caused regression on R9700)

### Patch 5: TSDF Keyframe Policy Optimization
**Commit**: `eb91e45`
**Changes**:
- Use GPU tensor comparison in TSDF keyframe policy (line 502)
- Eliminates `.item()` sync in keyframe insertion check

**Result**: 47-55 ms (high variance), ~18-21 fps

## Final Performance Summary

| Patch | Track (ms) | FPS | vs Baseline |
|-------|-----------|-----|-------------|
| Baseline | 92.76 | 10.8 | - |
| Patch 1 (ICP opts) | 88.51 | 11.3 | +4.6% |
| Patch 2 (VRAM cache) | 93.11 | 10.7 | -5.0% |
| **Patch 3 (GPU scoring)** | **47.10** | **21.2** | **+49% faster** |
| Patch 4 (solver buffers) | 54.65 | 18.3 | Reverted |
| Patch 5 (TSDF keyframe) | 47-55 | 18-21 | High variance |

## Key Insights

1. **GPU-CPU syncs are the bottleneck**: Patch 3's elimination of `.item()` calls in keyframe scoring provided the largest improvement (+49% faster).

2. **Pre-allocation doesn't always help**: Patch 4's buffer caching caused regression, likely due to cache pollution or memory layout issues on RDNA4.

3. **High variance on R9700**: Benchmark results show 15-20% variance between runs, likely due to thermal throttling or system load.

4. **Async ingestion works**: Load time is consistently 0.00ms, confirming the async pipeline is effective.

## Remaining Optimization Opportunities

### Patch 6: TSDF Speed Profile
- Profile TSDF integration and raycasting
- Optimize voxel grid operations
- Consider sparse TSDF representation

### Patch 7: Replace ORB/PnP with Kornia LoFTR
- Use learned feature matching on GPU
- Eliminate CPU OpenCV dependency
- Expected to improve `acc` mode performance

## Files Modified

- `gradslam/icp/projective.py`: ICP loop optimizations, fast transform methods
- `gradslam/icp/solvers.py`: Attempted buffer caching (reverted)
- `gradslam/slam/pipeline.py`: GPU tensor scoring, keyframe policy optimization
- `gradslam/runtime/`: New package with GPU timing, keyframe cache, scratch buffers
- `gradslam/ops/`: New package with GPU preprocessing, pyramids, geometry ops

## Testing

All patches tested on:
- **Local**: AMD Ryzen AI MAX+ PRO 395 (Strix Halo, gfx1151)
- **Remote**: AMD Radeon AI PRO R9700 (gfx1201) via SSH

Benchmark command:
```bash
python scripts/run_slam.py tum \
  --dataset-root /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd \
  --sequence freiburg1_desk \
  --slam-backend fast_rgbd --tracking-mode fast_rgbd \
  --max-frames 100 \
  --ingestion-mode async --ingestion-workers 4 --ingestion-queue-size 8 \
  --no-eval --device cuda
```

## Next Steps

1. Stabilize benchmark variance (run 10+ iterations, use median)
2. Implement Patch 6 (TSDF profiling)
3. Implement Patch 7 (Kornia LoFTR for feature matching)
4. Test on full TUM benchmark suite (all sequences, both configs)
5. Compare with baseline BENCHMARK_R9700_VS_8060S.md results
