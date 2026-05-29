# Fast RGB-D Tracking Findings

This report captures the current ROCm tracking state after fixing two evaluation
bugs that made earlier runs misleading:

- TUM tracking now uses every RGB-D pair and only associates ground truth during
  evaluation. This avoids sparse GT pose association creating multi-second frame
  jumps on long Freiburg2 scenes.
- RealSense `.bag` extraction now preserves raw aligned depth units and writes a
  matching `depth_factor`; the previous helper overflowed depth PNGs and produced
  median depths around 30 m.

## Current Defaults

| Dataset type | Default | Why |
|---|---:|---|
| TUM `--process-scale` | `0.5` | Keeps 640x480 TUM tracking above 30 FPS. |
| Normalized/RealSense `--process-scale` | `0.25` | Keeps 1280x720 handheld tracking VRAM-safe and above 30 FPS. |
| Normalized/RealSense `--max-processed-pixels` | `100000` | Refuses accidental half-scale 1280x720 runs unless explicitly overridden. |
| `--slam-backend` | `fast_rgbd` | Hot path for runtime gates. |
| `--tracking-mode` | `fast_rgbd` | Tracks against cached RGB-D references instead of TSDF raycasting. |
| `--keyframe-tracking-interval` | `0` | Periodic keyframe matching was too slow in current kernels. |
| `--feature-interval` | `0` | ORB/PnP did not improve checked runs. |
| `--max-depth-diff` | `0.14` | Best current TUM correspondence gate. |
| `--max-normal-angle-deg` | `75` | Best current TUM normal gate. |

`ICPSLAM` and `PointFusion` are available through `--slam-backend`, but they are
bounded to 30 frames unless `--allow-slow-upstream` is passed. PointFusion still
uses a growing global point map and is not a safe full-sequence ROCm runtime path.

## 300-Frame Results

| Dataset / mode | Frames | Scale | Tracking FPS | End-to-end FPS | Lost | ATE RMSE | RPE1 trans | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TUM `freiburg1_desk`, corrected RGB-D stream | 300 | `0.5` | `41.9` | `26.6` | `1` | `0.1080 m` | `0.0093 m` | Just above the 10 cm gate. |
| TUM `freiburg2_large_no_loop`, corrected RGB-D stream | 300 | `0.5` | `44.6` | `29.2` | not accepted | `6.6418 m` | `0.1672 m` | No sparse-GT jump, but geometry/tracker still fails. |
| TUM `freiburg2_large_no_loop`, official intrinsics | 300 | `0.5` | `44.3` | `28.9` | not accepted | `6.7556 m` | `0.1687 m` | Official intrinsics do not fix Freiburg2. |
| RealSense hallway, corrected extraction | 300 | `0.25` | `61.8` | `25.9` | `0` | `4.6883 m` | `0.1491 m` | Depth is now sane; tracking accuracy is still unusable. |

## VRAM Safety Findings

| Finding | Action |
|---|---|
| PointFusion on 300 frames left a stale ROCm KFD process and reported impossible VRAM accounting. | `--slam-backend pointfusion` and `icpslam` now refuse more than 30 frames by default. |
| RealSense half-scale 1280x720 is too heavy for the current hot path. | Normalized captures now default to quarter-scale and reject workloads above `100000` processed pixels/frame. |
| Stopped containers can leave stale KFD accounting briefly. | Verify with `rocm-smi --showpids`; after cleanup, expected state is `No KFD PIDs currently running`. |

## Reproduction

TUM desk gate:

```bash
podman compose run --rm gradslam python scripts/run_slam.py tum \
  --dataset-root /workspace/datasets/public/TUM/tum_rgbd \
  --sequence freiburg1_desk \
  --max-frames 300 \
  --device cuda:0
```

Corrected RealSense extraction and run:

```bash
podman compose run --rm gradslam bash -lc \
  'python -m pip install "pyrealsense2>=2.57" >/tmp/pyrealsense-install.log &&
   python scripts/extract_realsense_bag.py \
     /workspace/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity/20260522_115050.bag \
     --output /workspace/data/realsense_eval/cps_1stfloor_hallway_mediumdensity_300 \
     --max-frames 300'

podman compose run --rm gradslam python scripts/run_slam.py normalized \
  --capture-dir /workspace/data/realsense_eval/cps_1stfloor_hallway_mediumdensity_300 \
  --gt-file /workspace/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity/groundtruth_tum.txt \
  --max-frames 300 \
  --device cuda:0
```

## Next Fix Direction

The current fast tracker is now safe enough to benchmark, but not accurate
enough beyond easy TUM desk. The next implementation step is a real local-map
tracker: previous-frame tracking for normal frames, keyframe/local-map correction
for weak frames, and a sliding keyframe pose graph for long sequences. Further
gate tuning alone is unlikely to close the Freiburg2 or RealSense error gap.
