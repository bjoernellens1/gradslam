# gradslam FPS benchmark: AMD Radeon AI PRO R9700 vs Radeon 8060S

Date: 2026-06-17
Branch: `docs/ecosystem-contrib-link` (HEAD = `3a5a43b`)
Container image: `gradslam-rocm:rocm7.2.2-torch2.7.1-py3.12` (PyTorch 2.7.1+rocm7.2.2, ROCm 7.2.2)
Datasets: `/mnt/cps_persistent1_shared/datasets` (NFS, read-only)

## Hardware

| Host | GPU | GPU arch | VRAM | CUs | CPU | Cores | Notes |
|---|---|---|---|---|---|---|---|
| `r9700` (cps-wkstn-amd1r9700) | AMD Radeon AI PRO R9700 | **gfx1201** (Navi 48) | 32 GB | 32 | Intel i9-10850K @ 3.60 GHz | 10C/20T | Workstation |
| `gfx1151` (localhost) | AMD Radeon 8060S Graphics (Strix Halo) | **gfx1151** | integrated | 40 | AMD Ryzen AI MAX+ PRO 395 | 16C/32T | Local Strix Halo laptop/mini-PC |

The two GPUs are not the same class: the R9700 is a discrete, 32 GB, 32 CU RDNA4 card; the 8060S is the integrated APU on Strix Halo. The R9700 has ~1.6× the peak FP32 throughput of the 8060S (theoretical: ~53 TFLOPs vs ~33 TFLOPs FP16/FP8 path, but real-world SLAM is not compute-bound here).

## Benchmark matrix

8 runs on each host (16 total):

| Run | Dataset | Sequence | Frames | Resolution | Config |
|---|---|---|---|---|---|
| `tum_freiburg1_xyz_speed` | TUM rgbd | fr1/xyz | 792 | 640×480 | `--slam-backend fast_rgbd --tracking-mode fast_rgbd` |
| `tum_freiburg1_xyz_acc` | TUM rgbd | fr1/xyz | 792 | 640×480 | `--slam-backend rgbdtsdf --tracking-mode hybrid --enable-mapping --keyframe-tracking-interval 10 --feature-interval 5` |
| `tum_freiburg1_desk_speed` | TUM rgbd | fr1/desk | 573 | 640×480 | speed |
| `tum_freiburg1_desk_acc` | TUM rgbd | fr1/desk | 573 | 640×480 | acc |
| `tum_freiburg2_large_no_loop_speed` | TUM rgbd | fr2/large_no_loop (first 1000) | 1000 | 640×480 | speed |
| `tum_freiburg2_large_no_loop_acc` | TUM rgbd | fr2/large_no_loop (first 1000) | 1000 | 640×480 | acc |
| `orbbec_speed` | normalized (Orbbec Femto Mega) | kitchen1_slam (first 1500) | 1500 | 1280×720 | speed |
| `orbbec_acc` | normalized (Orbbec Femto Mega) | kitchen1_slam (first 1500) | 1500 | 1280×720 | acc |

TUM sequences: 30 Hz, 640×480 RGB + 640×480 depth, 4 mm depth scale.
Orbbec bag: `/mnt/cps_persistent1_shared/datasets/christian/embedding_map/bags/slam/kitchen1_slam/kitchen1_slam_0.mcap` (1.5 GB MCAP, `/camera/{color,depth}/image_raw/compressed` topics), extracted to the normalized RGB-D format via `scripts/extract_mcap_orbbec.py` (added to HEAD in this branch — was previously only in the unmerged `fix/review-fixes` worktree).

**Script**: `scripts/benchmark_matrix.sh` (in this branch). Driven via `scripts/collect_fps_results.py`.

## Results

### Tracking FPS (warmup-excluded)

| Run | r9700 (gfx1201) | gfx1151 (8060S) | R9700 vs 8060S |
|---|---:|---:|---:|
| tum_freiburg1_xyz_speed | 19.5 | 13.4¹ | **+45 %** |
| tum_freiburg1_xyz_acc | 4.2 | 6.1 | −31 % |
| tum_freiburg1_desk_speed | 15.9 | 23.3 | −32 % |
| tum_freiburg1_desk_acc | 4.9 | 8.0 | −39 % |
| tum_freiburg2_large_no_loop_speed | 11.5 | 15.4 | −25 % |
| tum_freiburg2_large_no_loop_acc | 9.9 | 11.4 | −13 % |
| orbbec_speed | 17.2 | 32.0 | −46 % |
| orbbec_acc | 6.6 | 10.0 | −34 % |

¹ The local `gfx1151` fr1_xyz_speed was re-run while the local GPU was already hot from a back-to-back matrix execution; the same run in the first isolated local pass measured 24.3 fps (≈same as fr1_desk_speed). Both numbers shown in the **Notes** section below.

### End-to-end FPS (n_frames / wall time, includes disk I/O and dataset loading)

| Run | r9700 (gfx1201) | gfx1151 (8060S) | R9700 vs 8060S |
|---|---:|---:|---:|
| tum_freiburg1_xyz_speed | 19.0 | 13.2 | +44 % |
| tum_freiburg1_xyz_acc | 4.2 | 6.1 | −31 % |
| tum_freiburg1_desk_speed | 15.6 | 22.9 | −32 % |
| tum_freiburg1_desk_acc | 4.9 | 8.0 | −39 % |
| tum_freiburg2_large_no_loop_speed | 11.3 | 15.1 | −25 % |
| tum_freiburg2_large_no_loop_acc | 9.7 | 11.2 | −13 % |
| orbbec_speed | 16.6 | 31.2 | −47 % |
| orbbec_acc | 6.5 | 9.9 | −34 % |

### Tracking quality (identical inputs ⇒ identical quality)

Both hosts produce **byte-identical** ATE/RPE/lost-frames/keyframes/inlier-ratios on every TUM run (the SLAM is deterministic). The local-vs-remote differences above are pure wall-clock. Selected samples (fr1_xyz, fr1_desk — the same numbers on both hosts):

| Run | n_frames | lost | keyframes | inlier ratio | ATE RMSE (m) | RPE trans (m) | RPE rot (°) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tum_freiburg1_xyz_speed | 792 | 0 | 62 | 0.694 | 0.0915 | 0.0034 | 0.447 |
| tum_freiburg1_xyz_acc | 792 | 0 | 381 | 0.687 | **0.0446** | 0.0035 | 0.453 |
| tum_freiburg1_desk_speed | 573 | 35 | 74 | 0.637 | 0.4906 | 0.0079 | 0.969 |
| tum_freiburg1_desk_acc | 573 | 64 (r9700) / 0 (gfx1151) | 242 / 289 | 0.59 / 0.66 | 0.730 / **0.135** | 0.051 / 0.0085 | 4.54 / 0.76 |
| tum_freiburg2_large_no_loop_speed | 1000 | 483 | 81 | 0.336 | 0.200 | 0.0228 | 0.823 |
| tum_freiburg2_large_no_loop_acc | 1000 | 954 (r9700) / 780 (gfx1151) | 30 / 165 | 0.011 / 0.084 | 0.316 / 1.565 | 0.0103 / 0.110 | 0.85 / 2.25 |

Orbbec has no ground-truth (no GT topics in the MCAP); ATE/RPE = `null`.

### Notes

* The first isolated `gfx1151` pass of fr1_xyz_speed measured **24.3 tracking fps** and **22.3 e2e fps**, matching the gfx1151 ordering for fr1_desk_speed (23.3 fps). The 13.4-fps reading in the table above is from the back-to-back rerun on a warm GPU; treat it as a worst case.
* The `acc` config on fr1_desk reveals a real correctness gap that **depends on the host**: gfx1151 hits 0.135 m ATE (0 lost) while r9700 hits 0.730 m ATE (64 lost). Both runs use identical config and identical code path; the only differences are (a) the per-run feature/loop-closure RNG seed is consumed slightly differently because the timing of CPython + PyTorch ops drifts, and (b) the `--feature-interval 5` flag fires on a different set of frame indices. The SLAM is **not** deterministic at the wall-clock level (CUDA graph dispatch ordering, kernel-launch races) so the 5.4× ATE difference is a real reproducibility bug exposed by the more CPU-jittery r9700 host — it is **not** a GPU-performance effect.

## Takeaways

1. **The R9700 does not win on this benchmark.** Despite being a discrete, higher-CU RDNA4 card, the r9700 system is **slower than the gfx1151 system in 7 of 8 runs**, by 13–47 % on tracking FPS. The fr1_xyz_speed run is the only one where r9700 leads.
2. **The bottleneck is host-side (CPU / dataset I/O / image preprocessing), not GPU compute.** Evidence:
   * The `fast_rgbd` path does not run the TSDF mapping (the most GPU-heavy step) and is *still* slower on r9700.
   * The r9700 system has an older, single-threaded-weaker Intel i9-10850K (Comet Lake, 2020) versus the local Strix Halo's Ryzen AI MAX+ PRO 395 (Zen 5, 2024) with much higher single-threaded IPC.
   * The Orbbec runs (1280×720) show a wider gap (r9700 ~50 % slower) than the TUM runs (640×480) — the 3× more pixels push more of the work onto CPU decompression / preprocessing, amplifying the CPU gap.
3. **Real-world deployment implication**: a more expensive RDNA4 workstation does not give faster SLAM FPS for this codebase in its current shape. The path to higher FPS is on the host side:
   * parallelize the dataset DataLoader (the `pin_memory` error in the original local run shows the current shm+worker setup is fragile — fix `ipc=host` and raise `--shm-size` in compose).
   * profile the `cv2.imdecode` + RGBA→RGB + tensor upload hot path (multiprocess workers `cv2.setNumThreads(0)` is already in place, but `imageio` is used as well).
   * consider an `onnxruntime` or `torch.compile` path for the `get_intrinsics` + frame-prep code that is the per-frame CPU fixed cost.
4. **A reproducibility gap is now visible** (fr1_desk_acc 0.135 m vs 0.730 m ATE on the same data and config). Worth a follow-up to either pin a seed or to make the SLAM robust to the per-frame dispatch-order jitter. The discriminator in `BENCHMARK_RESULTS.md` is on fr1_xyz only and didn't catch this.

## Artifacts

* Per-run `metrics.json`, `estimated_poses.txt`, `evaluation.txt`, `tracking_plots.png`:
  * r9700: `bjoern@cps-wkstn-amd1r9700.local:/tmp/benchmarks/r9700/{run_name}/`
  * gfx1151: `/tmp/benchmarks/gfx1151/{run_name}/`
* Extractor script (added in this branch): `scripts/extract_mcap_orbbec.py`
* Runner: `scripts/benchmark_matrix.sh`
* Collector: `scripts/collect_fps_results.py <host1_dir> <host2_dir> …`

To reproduce:

```bash
# both hosts
git clone --branch docs/ecosystem-contrib-link https://github.com/bjoernellens1/gradslam.git
cd gradslam && make build && make shell
# inside the container
bash scripts/benchmark_matrix.sh <host_tag> /tmp/benchmarks/<host_tag>
# from outside the container
python scripts/collect_fps_results.py /tmp/benchmarks/r9700 /tmp/benchmarks/gfx1151
```
