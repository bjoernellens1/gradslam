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
| `--local-map-candidates` | `1` | Lightweight recovery mode tests one nearby keyframe only after a quality or motion gate trips. |
| `--max-frame-translation` | `0.12` | Suppresses the worst over-scaled relative-motion failures while keeping fast/default TUM desk stable. |
| `--max-frame-rotation-deg` | `30` | Rejects implausible single-frame rotations. |
| `--max-depth-diff` | `0.14` | Best current TUM correspondence gate. |
| `--max-normal-angle-deg` | `75` | Best current TUM normal gate. |

`ICPSLAM` and `PointFusion` are available through `--slam-backend`, but they are
bounded to 30 frames unless `--allow-slow-upstream` is passed. PointFusion still
uses a growing global point map and is not a safe full-sequence ROCm runtime path.

## 300-Frame Results

| Dataset / mode | Frames | Scale | Tracking FPS | End-to-end FPS | Lost | ATE RMSE | RPE1 trans | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TUM `freiburg1_desk`, corrected RGB-D stream | 300 | `0.5` | `41.9` | `26.6` | `1` | `0.1080 m` | `0.0093 m` | Just above the 10 cm gate. |
| TUM `freiburg1_desk`, default motion gate `0.12` | 300 | `0.5` | `41.5` | `25.8` | `1` | `0.1080 m` | `0.0093 m` | No regression from the new default gate. |
| TUM `freiburg2_large_no_loop`, corrected RGB-D stream | 300 | `0.5` | `44.6` | `29.2` | not accepted | `6.6418 m` | `0.1672 m` | No sparse-GT jump, but geometry/tracker still fails. |
| TUM `freiburg2_large_no_loop`, official intrinsics | 300 | `0.5` | `44.3` | `28.9` | not accepted | `6.7556 m` | `0.1687 m` | Official intrinsics do not fix Freiburg2. |
| TUM `freiburg2_large_no_loop`, heavy local-map gate `0.08`, 3 candidates | 300 | `0.5` | `10.3` | `9.0` | not accepted | `2.3014 m` | `0.0529 m` | Accuracy improves, but it is far below the FPS gate. |
| TUM `freiburg2_large_no_loop`, adaptive local-map gate `0.12`, 1 candidate | 300 | `0.5` | `35.4` | `25.0` | `90` | `3.4597 m` | `0.0753 m` | Clears tracking FPS and halves RPE1, but ATE remains unusable. |
| RealSense hallway, corrected extraction | 300 | `0.25` | `61.8` | `25.9` | `0` | `4.6883 m` | `0.1491 m` | Depth is now sane; tracking accuracy is still unusable. |
| RealSense hallway, adaptive local-map gate `0.12`, 1 candidate | 300 | `0.25` | `57.1` | `25.8` | `46` | `1.7564 m` | `0.0750 m` | Large improvement, still not robust enough. |
| Orbbec kitchen ICL export, adaptive local-map gate `0.12`, 1 candidate | 300 | `0.25` | `52.5` | `20.9` | `0` | n/a | n/a | Stable no-GT speed/stability check. |

## Motion Diagnostics

`scripts/diagnose_tracking_motion.py` compares estimated and GT relative motion
from TUM-format trajectories and writes `motion_diagnostics.json` plus
`relative_motion.csv`. On the broken Freiburg2 fast run it showed:

| Diagnostic | Value |
|---|---:|
| Associated GT pairs | `230 / 300` |
| Direct ATE RMSE | `6.6418 m` |
| Direct RPE1 translation RMSE | `0.1672 m` |
| Median relative translation scale ratio | `15.28x` |
| First relative translation error above 10 cm | frame-pair `2 -> 3` |
| Inverted trajectory ATE RMSE | `7.3179 m` |

The failure is therefore not just a pose convention inversion. The current dense
projective tracker can report good inlier ratios while producing badly
over-scaled relative translation on some long/handheld scenes.

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

Adaptive local-map recovery:

```bash
podman compose run --rm gradslam python scripts/run_slam.py tum \
  --dataset-root /workspace/data/tum_freiburg2/freiburg2_large_no_loop \
  --sequence rgbd_dataset_freiburg2_large_no_loop \
  --slam-backend local_map \
  --max-frames 300 \
  --device cuda:0
```

Relative-motion diagnostics:

```bash
podman compose run --rm gradslam python scripts/diagnose_tracking_motion.py \
  --estimated /workspace/outputs/tum_fr2_large_no_loop_fast_rgbd_300_trackingpairs_container/estimated_poses.txt \
  --gt /workspace/data/tum_freiburg2/freiburg2_large_no_loop/rgbd_dataset_freiburg2_large_no_loop/groundtruth.txt \
  --tracking-metrics /workspace/outputs/tum_fr2_large_no_loop_fast_rgbd_300_trackingpairs_container/tracking_metrics.txt \
  --output-dir /workspace/outputs/tum_fr2_large_no_loop_fast_rgbd_300_trackingpairs_container/diagnostics
```

## Next Fix Direction

The current fast tracker is now safe enough to benchmark, but not accurate
enough beyond easy TUM desk. Adaptive local-map recovery is a useful guardrail,
not a complete fix: it lowers Freiburg2/RealSense RPE while preserving tracking
FPS, but it does not solve long-horizon ATE. The next implementation step should
be an actual local correction layer: candidate triggering from relative-motion
diagnostics, per-keyframe pose correction, and a small sliding pose graph that can
feed corrected keyframe poses back into tracking without blocking the frame loop.
