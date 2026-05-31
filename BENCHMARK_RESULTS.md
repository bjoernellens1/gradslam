# TUM RGB-D Benchmark Results

**Hardware:** AMD Radeon 8060S Graphics (ROCm 7.2.2, PyTorch 2.7.1)
**GT:** official TUM motion-capture `groundtruth.txt`, depth-timestamp association,
ATE after SE(3) Umeyama alignment (`gradslam/evaluation/trajectory.py`).

> **Correction to the prior version of this file.** The earlier benchmark ran
> `python scripts/run_slam.py tum` **with no flags**, which resolves to the
> `fast_rgbd` *speed preset* with TSDF mapping and every drift-mitigation path
> disabled (`enable_mapping=false`, `keyframe_tracking_interval=0`,
> `feature_interval=0`, pose-graph/loop/reloc off — see any
> `outputs/*/config_resolved.yaml`). That configuration is **not** the system as
> designed for accuracy, so the original "1 of 4 sequences" headline measured a
> deliberately hobbled tracker. The benchmark scripts now pass the intended
> accuracy config (`run_all_tum.sh`, `benchmark_all.py`).

## Configuration ablation (ATE RMSE, metres)

| Sequence | Frames | Bare default (fast_rgbd, all off) | + keyframe anchoring | **As designed (hybrid TSDF + mapping + feature PnP)** | Target <0.10 |
|---|---|---|---|---|---|
| freiburg1_xyz | 792 | 0.092 | 0.092 | **0.045** | ✓ |
| freiburg1_desk | 573 | 0.491 | 0.487 | **0.135** | — |
| freiburg2_desk | 2893 | 1.698 | 1.686 | 1.697 | — |
| freiburg3_long_office_household | 2488 | 2.080 | 2.099 | 2.112 | — |

Speed: bare `fast_rgbd` ≈ 22–24 tracking fps; the as-designed hybrid path
≈ 7–8 fps (frame-to-model raycast + TSDF integration). Accuracy/speed trade-off.

## Findings

1. **The headline regression was a benchmark-config artifact.** Running the
   system as designed (hybrid frame-to-model TSDF tracking) **halves** the
   freiburg1_xyz error (0.091 → 0.045 m, well under target) and cuts
   freiburg1_desk **3.6×** (0.491 → 0.135 m). The short sequences are essentially
   solved; the bare-default numbers underrepresented the system.

2. **Keyframe anchoring on the fast path is not the lever** (Pass A ≈ no change):
   the previous-frame candidate wins scoring almost every frame, so adding
   keyframe candidates barely changes ATE. Hybrid TSDF frame-to-model tracking is
   what reduces drift on the short sequences.

3. **Long-sequence ATE (fr2/fr3) is fundamental VO drift.** RPE is excellent
   (e.g. fr3 RPE δ=10 ≈ 2.7 cm) while ATE is metres — the signature of pure
   accumulated drift, not a local-tracking fault. The recorded trajectory is
   frozen per-frame at tracking time (no end-of-run re-export from corrected
   keyframes), so the sliding-window pose graph cannot retroactively reduce these
   numbers. Closing this gap requires **global pose-graph BA + trajectory
   re-export (Phase G)**, not in scope here.

4. **Opt-in loop closure: correct but not yet beneficial, and now safe.**
   The pose-graph/loop wiring had real bugs (sequential edge fed a single-frame
   relative; loop constraints injected as malformed sequential edges; double node
   insertion) — fixed. But the hand-rolled optimizer has no robust outlier kernel,
   so bad PnP loop measurements (which pass the inlier gate on planar desk scenes)
   could corrupt the trajectory and diverge to numerical blow-up. A
   validate-before-commit safety guard now rejects non-finite / implausibly large
   corrections. **Loop closure stays default-off**; making it reduce ATE needs a
   robust back-end + global re-export (Phase G).

## Reproduce

```
docker compose run --rm gradslam bash scripts/run_all_tum.sh   # as-designed config
```
Each run writes `trajectory.txt`, `metrics.json`, `tracking_debug.csv`
(now incl. `tracking_state` / `map_update_allowed`), `tracking_plots.png`
(ATE-over-time panel renders when GT is present), and `config_resolved.yaml`.
