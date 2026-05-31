# Performance Tuning Ablation — FPS vs ATE (hard constraint ATE < 10 cm)

**Hardware:** AMD Radeon 8060S APU (ROCm 7.2.2, PyTorch 2.7.1), single GPU.
**Harness:** `scripts/ablation/` — `run_one.sh` (flock-serialized, single APU → GPU runs
must serialize for valid FPS; resumable via `results.csv`), shard job lists run by parallel
executor agents, `analyze.py` (Pareto + cross-scene feasibility ranking).
**Targets:** TUM `freiburg1_xyz` (792f), `freiburg1_desk` (573f), `freiburg1_room` (1352f).
**Objective:** among configs with ATE < 0.10 m, prefer best ATE margin, tie-break by tracking FPS.
**Runs:** 48 rows (34 ok, 14 failed pre-bf16-fix); 2 coarse waves + 1 finalist wave.

## Headline verdict

**ATE < 10 cm is achievable on `freiburg1_xyz` but NOT on `freiburg1_desk` / `freiburg1_room`**,
and no front-end tuning closes that gap. fr1_desk has a hard ATE floor at **~0.135 m** and
fr1_room sits at **0.94 m** — these are accumulated visual-odometry drift over loopy
trajectories, removable only by global pose-graph BA + trajectory re-export (Phase G), which is
out of scope for a parameter sweep. The ablation exhaustively confirms the floor (below).

## Recommended configs (feasible operating points on `freiburg1_xyz`)

| Operating point | Config | ATE | tracking FPS | Flags |
|---|---|---|---|---|
| **Accuracy (recommended)** | `H_map10` | **0.045 m** | 8.1 | `--slam-backend rgbdtsdf --enable-mapping --keyframe-tracking-interval 10 --feature-interval 5 --voxel-size 0.02 --mapping-interval 10` |
| **Speed (feasible)** | `F_nofeat` | 0.092 m | **22.2** | `--slam-backend fast_rgbd --keyframe-tracking-interval 10 --feature-interval 0` |

`H_map10` gives the largest ATE margin (0.045, comfortably under 0.10) at the per-run-best 8.1 fps
among hybrids. `F_nofeat` is the only sub-10 cm config above 8 fps — 2.7× faster but at the very
edge of the constraint (0.092). No config is feasible across *all three* scenes (desk/room infeasible).

## fr1_xyz Pareto front (tracking FPS vs ATE)

```
F_it42   29.4 fps / 0.177 m   (infeasible)
F_nofeat 22.2 fps / 0.092 m   ✓ feasible — speed pick
H_map10   8.1 fps / 0.045 m   ✓ feasible — accuracy pick
```
Everything between 8.1 and 22 fps is dominated; the front is a sharp accuracy/speed knee at the
fast_rgbd→hybrid backend switch.

## Lever effects (what moved the needle)

| Lever | Effect on ATE | Effect on FPS | Keep? |
|---|---|---|---|
| **Backend: hybrid (rgbdtsdf) vs fast_rgbd** | **3–5× better ATE** (0.045 vs 0.092–0.17 on xyz; 0.135 vs 0.49–0.69 on desk) | ~3× slower (8 vs 22 fps) | hybrid for accuracy |
| **Feature-PnP (`--feature-interval`)** | **HURTS** (0.167 vs 0.092 on xyz fast; lost frames ↑) | slower | **drop it** (interval 0) |
| `--mapping-interval 10` | neutral (0.045) | slightly faster (8.1 vs 7.9) | yes (free win) |
| `--raycast-normal-mode image` | neutral-to-noisier (0.045–0.342 on desk) | slightly faster | marginal |
| voxel 0.015 / 0.02 / 0.04 | flat (~0.045 xyz; 0.135–0.165 desk) | flat | 0.02 default fine |
| `--process-scale 1.0` (full-res) | **worse** (0.183 xyz, 0.168 desk) | **3× slower** (2.9 fps) | no — half-res is better |
| `--photometric-weight 0.1` | slightly worse (0.056) | neutral | no |
| ICP iters 5 3 / 4 2 | neutral-to-worse | faster at 4 2 (but ATE ↑) | default |
| **`--autocast bf16`** | neutral (after fix) | ~neutral on this APU | optional |

## Code finding fixed during the study

**bf16/fp16 autocast crashed the ICP solver** (`gradslam/icp/solvers.py`): under mixed-precision
autocast the 6×6 normal-equation solve received mismatched dtypes →
`torch.linalg.solve/lstsq: Expected ... Float and BFloat16`, silently aborting **every**
`--autocast` run (14 failed rows). Fixed by doing the tiny 6×6 solve in float32 and casting back
(also more numerically stable). bf16 then runs but is ~speed-neutral on this APU. 264 tests green.

## Why desk/room can't meet <10 cm (the floor is structural)

Every accuracy lever was tried on fr1_desk and none crossed 0.10 m (best 0.135). RPE is tiny
(~7 mm/frame) while ATE is decimetres — the signature of pure accumulated drift on a loopy
trajectory, not a local-tracking fault. The recorded trajectory is frozen per-frame at tracking
time (no global optimization / re-export), so a front-end sweep cannot reduce it. fr1_room (large
loop, 471 lost frames) is harder still at 0.94 m. Closing this is the Phase-G global-BA work, not
a tuning problem.

## Reproduce / extend

```
bash scripts/ablation/run_sweep.sh scripts/ablation/shardA.txt   # any shard; GPU-locked, resumable
python3 scripts/ablation/analyze.py --ate-max 0.10 --targets freiburg1_xyz
```
Note: the held-out fr1_room validation of the geometry-only finalists was only partially run
(session limit); the one completed row (Hb_imgn 0.94 m / 471 lost) already shows room is
infeasible, consistent with the desk floor. Re-run shardE_validate.txt to complete it.
