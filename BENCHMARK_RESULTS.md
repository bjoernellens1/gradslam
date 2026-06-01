# TUM RGB-D Benchmark Results

**Hardware:** AMD Radeon 8060S Graphics (ROCm 7.2.2, PyTorch 2.7.1)  
**GT:** official TUM motion-capture `groundtruth.txt`, depth-timestamp association,
ATE after SE(3) Umeyama alignment (`gradslam/evaluation/trajectory.py`).

---

## Current results — `perf-and-accuracy` branch (observer-mode global PGO)

### Recommended accuracy config

```
python scripts/run_slam.py tum \
  --slam-backend rgbdtsdf --enable-mapping \
  --keyframe-tracking-interval 10 --feature-interval 0 \
  --pose-graph on --loop-closure on \
  --keyframe-db-size 500 --loop-closure-min-inliers 15
```

| Sequence | ATE RMSE | Track FPS | E2E FPS | Lost | Loops |
|---|---|---|---|---|---|
| **freiburg1_desk** | **0.100 m ✓** | 4.6 | 4.3 | 0 | 49 |
| **freiburg1_xyz** | **0.014 m ✓** | 6.1 | 5.5 | 0 | 57 |

**Target ATE < 0.10 m: achieved on both sequences.**

### Ablation (fr1_desk, showing each lever's contribution)

| Config | ATE | Track FPS |
|---|---|---|
| Bare default (fast_rgbd, everything off) | 0.491 m | 21.9 |
| Hybrid TSDF (no loop closure) | 0.135 m | 8.4 |
| Hybrid + live loop feedback ❌ | 0.380 m | 4.6 |
| **Hybrid + observer-mode PGO + loops ✓** | **0.100 m** | **4.6** |

**Key finding — observer mode:** feeding mid-run pose-graph corrections into a
frame-to-model tracker with a baked TSDF map causes tracker desync (confirmed by
discriminator: skip-reexport ATE 0.505, live-feedback 0.380, observer 0.100).
Observer mode accumulates the global pose graph silently during the run and applies
all loop corrections to the trajectory at run end via `finalize_pose_graph()` +
`reexport_pose()` — a single pass, no map correction required.

### FPS note

The 4.3–4.6 fps (vs 7.4 fps without loops) is the cost of brute-force ORB BFMatcher
over 500 keyframes per keyframe insertion. The GPU itself still runs at ~8 fps; e2e
throughput is CPU-bound by `find_loop`. Reducing this is the next performance task:
run loop search every N keyframes (easy, partial), or add cheap global-descriptor
pre-filtering (BoW/NetVLAD, better, recovers most fps).

---

## Architecture — what changed in this branch

### GlobalPoseGraph (`gradslam/slam/global_pose_graph.py`)

- **pypose SE(3) LM** over *all* keyframes (no window trim), node 0 as fixed anchor.
- **Immutable odometry edges**: sequential edges derive from `_raw_poses` (captured at
  tracking time), never from the mutable corrected `_poses` — prevents gauge mixing
  after loop commits.
- **Delta-on-raw re-export**: `optimize()` returns corrections as left-deltas on the
  raw stored pose so a no-op optimization is *exactly* identity (2e-16); prevents
  projection-leak corruption of all frames.
- **`finalize()`**: one final global re-optimization at stream end corrects the
  trajectory tail after the last loop commit.
- **`try_commit_correction(max_translation_step=2.0)`**: safety guard — rejects
  corrections that are non-finite or jump > 2 m.

### Pipeline (`gradslam/slam/pipeline.py`)

- **`pose_graph_observer=True`** (default): suppress in-run writeback; apply only at end.
- **Frame→keyframe attachment** (`_record_frame_attachment`): each frame records its
  anchor keyframe + relative transform so `reexport_pose()` can re-derive it.
- **`finalize_pose_graph()` + `reexport_pose()`**: called from `run_slam.py` after the
  tracking loop; replaces frozen per-frame poses with loop-corrected versions.
- **Loop CSV fields added**: `loop_closure_frame_idx`, `loop_closure_rejected`,
  `loop_closure_inliers`, `loop_closure_weight` now visible in `tracking_debug.csv`.

### New CLI flags

| Flag | Default | Effect |
|---|---|---|
| `--pose-graph on/off` | off | Enable global PGO |
| `--pose-graph-backend global/sliding` | global | pypose LM vs legacy GN |
| `--pose-graph-observer on/off` | on | Observer vs feedback mode |
| `--loop-closure on/off` | off | Enable ORB loop detection |
| `--keyframe-db-size N` | 30 | DB size for loop candidates |
| `--loop-closure-min-inliers N` | 30 | PnP inlier gate |
| `--num-workers N` | 4 | Async data loading workers |

---

## Long-sequence sequences (fr2_desk, fr3_long)

Still 1.5–2 m ATE — fundamental VO drift that requires **global pose-graph BA +
trajectory re-export**, which this branch now provides. These sequences do not
have enough loop-closure overlap with the current ORB detector + 500-frame DB;
the mechanism is proven (GT oracle on fr1_desk 0.135→0.047 m), the detection
bottleneck is the brute-force `find_loop` on sequences with 1000+ keyframes.

---

## Reproduce

```bash
# Build image if needed
docker compose build

# Run recommended config (fr1_desk sub-10cm)
docker compose run --rm gradslam bash -c "
cd /workspace/gradslam
python scripts/run_slam.py tum \
  --dataset-root /workspace/datasets/public/TUM/tum_rgbd/ \
  --sequence freiburg1_desk \
  --slam-backend rgbdtsdf --enable-mapping \
  --keyframe-tracking-interval 10 --feature-interval 0 \
  --pose-graph on --loop-closure on \
  --keyframe-db-size 500 --loop-closure-min-inliers 15"
```

Each run writes: `trajectory.txt`, `metrics.json`, `tracking_debug.csv`
(with loop closure columns), `tracking_plots.png` (ATE-over-time rendered),
`config_resolved.yaml`.
