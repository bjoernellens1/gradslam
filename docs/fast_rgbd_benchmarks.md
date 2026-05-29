# Fast RGB-D Benchmark Notes

This report captures the current ROCm fast-tracking path after the TUM desk
benchmark was pushed to full 300-frame runs.

## Current Defaults

These are the runner defaults now used by `scripts/run_slam.py` for the fast
path:

| Flag | Default | Why |
|---|---:|---|
| `--tracking-mode` | `fast_rgbd` | Tracks against cached RGB-D references instead of raycasting every frame. |
| `--process-scale` | `0.5` | Half-resolution tracking is the best speed/accuracy tradeoff on TUM. |
| `--feature-interval` | `0` | ORB+PnP was not helping the benchmarked sequences. |
| `--keyframe-tracking-interval` | `0` | Keeps the hot path simple and fastest for the common case. |
| `--robust-loss` | `none` | Robust loss was slightly worse than the plain solver on the best TUM run. |
| `--max-depth-diff` | `0.14` | Best correspondence gate found on the TUM desk sweep. |
| `--max-normal-angle-deg` | `75` | Best normal gate found on the TUM desk sweep. |
| `--min-track-inliers` | `100` | Keeps clearly broken frames marked as lost. |
| `--lost-inlier-ratio` | `0.02` | Conservative fail-fast threshold. |
| `--borderline-inlier-ratio` | `0.08` | Allows TSDF recovery only when the local tracker is weak. |

## Full 300-Frame Results

| Dataset | Frames | Process Scale | Tracking FPS | End-to-End FPS | Lost Frames | ATE / Stability |
|---|---:|---:|---:|---:|---:|---|
| TUM `freiburg1_desk` | 300 | `0.5` | `42.1` | `25.9` | `1` | `ATE RMSE 0.0999 m`, `RPE delta-1 0.0072 m`, `RPE delta-10 0.0382 m` |
| RealSense hallway capture | 300 | `0.5` | `50.6` | `22.4` | `298` | No GT in the normalized capture; same TUM params do not hold up |
| RealSense hallway capture, relaxed gates | 300 | `0.5` | `65.3` | `25.7` | `224` | Still unstable; the issue is not just one gate |
| Orbbec kitchen ICL export | 300 | `0.5` | `18.4` | `11.1` | `0` | Stable, but too slow at half scale |
| Orbbec kitchen ICL export | 300 | `0.25` | `53.7` | `31.7` | `0` | Stable and above 30 FPS at quarter scale |

## Interpretation

| Finding | Practical result |
|---|---|
| Half-scale fast RGB-D tracking is enough for TUM `freiburg1_desk` | It clears the `>30 fps` tracking gate and stays under `10 cm` ATE on the full 300-frame run. |
| The same gate set does not generalize to every handheld RealSense capture | The RealSense hallway segment needs dataset-specific preprocessing or a different tracker regime. |
| Orbbec can stay stable, but the resolution matters | The kitchen export only crosses `30 fps` when processing is reduced to `0.25` scale. |

## Reproduction

Use the container and full 300-frame runs:

```bash
podman compose run --rm gradslam python scripts/run_slam.py tum \
  --dataset-root /workspace/datasets/public/TUM/tum_rgbd \
  --sequence freiburg1_desk \
  --max-frames 300 \
  --device cuda:0
```

For normalized captures:

```bash
podman compose run --rm gradslam python scripts/run_slam.py normalized \
  --capture-dir /workspace/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity_extracted \
  --max-frames 300 \
  --device cuda:0
```
