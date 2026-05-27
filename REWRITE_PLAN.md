# gradslam ROCm-first modernization

## Context

`bjoernellens1/gradslam` is a fork of the differentiable dense-SLAM library, still on an
ancient stack: `torch>=1.6.0`, `chamferdist==1.0.0`, `open3d==0.10.0.0`, `setup.py` with
`python_requires=">3.6"`, `sphinx==2.2.0`, and a CircleCI pipeline. The goal is a fully
PyTorch, cross-platform library whose **first-priority runtime is AMD ROCm** (container
`rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1`), with good auto-built
API docs.

**Strategy (decided): modernize in place, do not rewrite greenfield.** gradslam already
ships ~6,200 lines of working *differentiable* SLAM — `geometry/se3utils.py`,
`geometry/projutils.py`, `geometry/geometryutils.py`, differentiable `odometry/icp.py` +
`gradicp.py`, `slam/pointfusion.py` + `icpslam.py`, and `structures/{pointclouds,rgbdimages}.py`.
These are gradslam's identity and stay. The actual ROCm blockers are narrow:
`chamferdist` (a CUDA-native KNN), `open3d` (native, segfault-ordering hack), and stale
pins. We remove those, modernize packaging/container/CI/docs, and **add** a fast,
non-differentiable KinectFusion-style projective-ICP + TSDF tracker as a *new optional
backend* that reuses the existing geometry math.

**Decisions captured:** Docs = modernize Sphinx + ReadTheDocs (keep the 4 tutorial
notebooks). CI = migrate to GitHub Actions (delete `.circleci/`). Differentiable path
stays differentiable; the new fast tracker is `@torch.no_grad()`.

**Cross-platform scope:** Linux ROCm (priority) + CUDA + CPU; Apple MPS opportunistic;
Windows out of scope. ROCm PyTorch reports as `torch.cuda.*` with device string `"cuda"`,
so device code keys off `torch.cuda.is_available()` and detects ROCm via `torch.version.hip`.

## Hard rules (unchanged from brief)

1. Primary runtime container: `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1`. **Never** `pip install torch` — it comes from the base image.
2. No CUDA-only core dependencies; no custom CUDA kernels in core.
3. No `open3d`/`chamferdist`/`pytorch3d`/`faiss`/`torch-cluster` imported by core modules. `open3d` allowed only as lazy optional viz/export.
4. Every new hot path: CPU test (CI) + ROCm test (local, in-container — CI has no AMD GPU). Eager mode + opt-in `torch.compile` mode.
5. Preserve license/copyright on any borrowed code (KinectFusion is MIT).

## Reuse map (do NOT reimplement)

| Need | Reuse existing |
|---|---|
| SE(3) exp/hat/log | `gradslam/geometry/se3utils.py` (`se3_exp`, `so3_hat`, `se3_hat`) |
| project / unproject | `gradslam/geometry/projutils.py` |
| transform points, normals | `gradslam/geometry/geometryutils.py` (`transform_pointcloud`, normal helpers) |
| differentiable ICP / gradICP | `gradslam/odometry/{icp,gradicp,icputils}.py` (kept; only KNN dep swapped) |
| data structures | `gradslam/structures/{pointclouds,rgbdimages}.py` |
| test fixtures | `tests/data/msrd_b2s3/*.npy` (colors, depths, intrinsics, poses, vertex_map, normal_map) — reuse for new ICP/TSDF/raycast tests instead of synthesizing |

`pytorch3d` is **already not a dependency** — `structutils.py` only *copied* two pure-torch
helpers (`list_to_padded`/`padded_to_list`) with attribution. Nothing to remove; just keep
it out of new code.

---

## Milestone 1 — ROCm unblock (top priority: existing library runs on ROCm)

Goal: `import gradslam` works with no native deps, existing tests pass on CPU and in the
ROCm container.

**Packaging / container**
- Add `pyproject.toml` (setuptools backend). `requires-python = ">=3.12"` *(breaking change for non-container users — call out in README)*. Core deps: `numpy>=1.26, kornia>=0.8.3, imageio>=2.34, opencv-python-headless>=4.10, pyyaml>=6.0, natsort>=8.4, tqdm>=4.66`. **No `torch`.** Extras: `optim` (pypose), `mesh` (scikit-image, trimesh), `vis` (matplotlib, plotly, open3d), `docs` (sphinx + theme + nbsphinx), `dev` (pytest, pytest-cov, ruff). Configure `[tool.ruff]` (line-length 100, py312) and `[tool.pytest.ini_options]`.
- Delete `setup.py` + `requirements.txt` (superseded). `gradslam/version.py` stays as version source (read in `pyproject` or hardcode `0.2.0`).
- Add `Dockerfile` (FROM the ROCm base; apt: git/build-essential/cmake/ninja/ffmpeg/libgl1/libglib2.0-0/libsm6/libxext6/libxrender1; `pip install -e ".[optim,mesh,vis,docs,dev]"`). **Verify the base tag with `docker manifest inspect` before building** — do not assume it resolves.
- Add `compose.yaml` (devices `/dev/kfd` + `/dev/dri`, groups `video`/`render`, `ipc: host`, seccomp unconfined, mount repo). Add `Makefile` targets: `build`, `shell`, `check-rocm`, `test`, `docs`, `lint`.
- Add `scripts/check_rocm_stack.py` — prints backend report, asserts `torch.cuda.is_available()` and `torch.version.hip is not None`, runs a GPU matmul.

**Backend layer (new)**
- `gradslam/backend/device.py`: `accelerator_backend()` (cpu/rocm/cuda via `torch.version.hip`/`.cuda`), `default_device()`, `backend_report()`.
- `gradslam/backend/compile.py`: `compile_if_requested(fn, fullgraph=False)` gated on env `GRADSLAM_COMPILE`, using `mode="reduce-overhead"`. *(Caveat: `fullgraph=True` only for fixed-shape unit tests; varying H/W/N paths may need `dynamic=True` or selective compile.)*

**Dependency removal**
- **chamferdist → torch KNN.** Add `gradslam/geometry/knn.py` with a pure-torch `knn_points` matching the signature used in `odometry/icputils.py:3` (1-NN, returns dists + indices), chunked over query points to bound memory. Swap the import in `icputils.py`. This is the single change that unblocks differentiable ICP on ROCm.
- **open3d → lazy.** Remove the top-of-file `import open3d` in `gradslam/__init__.py` (the "avoid segfault" hack) and `gradslam/structures/pointclouds.py:3`; move the import *inside* `Pointclouds.open3d()` (and any other o3d call site). **Before removing the top-level import, verify in the ROCm container that bare `import gradslam` (then `import torch`) does not segfault**; add a regression test `tests/test_import.py` that imports gradslam with open3d absent.

**CI**
- Delete `.circleci/`. Add `.github/workflows/ci.yml`: setup-python 3.12 → install CPU torch from the PyTorch CPU index → `pip install -e ".[dev,optim,mesh,vis]"` → `ruff check` → `pytest -q --cov=gradslam`.

**Verify M1**
```
docker manifest inspect rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.7.1
make build && make check-rocm
docker compose run --rm gradslam python -c "import gradslam; print('ok')"
docker compose run --rm gradslam pytest -q          # existing suite, in container
pytest -q                                           # existing suite, CPU host
```

---

## Milestone 2 — fast KinectFusion-style backend (new, optional, reuses geometry)

Goal: a real-time projective-ICP + TSDF tracker, non-differentiable, ROCm/CPU tested.
Borrow architecture from JingwenWang95/KinectFusion (MIT — preserve notice in a header
comment); reuse gradslam's `se3utils`/`projutils`/`geometryutils` rather than re-deriving.

- `gradslam/geometry/normals.py`, `gradslam/geometry/image_pyramid.py`: depth/vertex/normal maps + multi-scale pyramids + scaled-K. Use `torch.meshgrid(..., indexing="ij")`, cache pixel grids per (shape, device, dtype), `torch.linalg.solve`/`inv` (not `torch.inverse`).
- `gradslam/icp/{__init__,solvers,residuals,projective}.py`:
  - `solvers.solve_lm_6x6` (damped Levenberg–Marquardt 6×6 normal equations via `torch.linalg.solve`).
  - `residuals.point_to_plane_projective`.
  - `projective.ProjectiveICPTracker(torch.nn.Module)` with `ProjectiveICPConfig` (pyramid levels, per-level iters/damping, depth/normal rejection, Huber). `forward(live_depth, model_depth, intrinsics, init_T)` → `(T_model_live, quality_dict)`. Coarse→fine over the pyramid; left-multiply SE(3) updates. `@torch.no_grad()`. Quality dict: `num_valid, inlier_ratio, rmse, mean/median_abs_residual, condition_number, update_norm, converged`.
- `gradslam/mapping/tsdf.py`: `TSDFVolume(torch.nn.Module)` + `TSDFConfig`. `@torch.no_grad() integrate(depth, intrinsics, T_world_cam, color=None)` projecting voxel centers, frustum check, TSDF/weight/color update. Marching-cubes export lives in `gradslam/io/mesh.py` (uses scikit-image/trimesh — NOT in `mapping`).
- `gradslam/rendering/tsdf_raycast.py`: `raycast(...) -> RenderedModelFrame(depth, color, vertex, normal, mask)` with torch trilinear sampling.
- `gradslam/slam/pipeline.py`: `RGBDTSDFSLAM` + `RGBDFrame`/`TrackingResult` dataclasses. `process_frame`: first frame init+integrate; later frames raycast model from prev pose → projective ICP → quality gate → on success update pose + integrate; keyframe policy (inlier-ratio / motion / interval thresholds). Lives **alongside** existing `pointfusion.py`/`icpslam.py`, does not replace them.
- Opt-in `torch.compile` (via `backend/compile.py`) on: `geometry.normals.*`, pyramid scaling, `icp.residuals.point_to_plane_projective`, `icp.solvers.solve_lm_6x6`, `mapping.tsdf.integrate`, `rendering.tsdf_raycast` trilinear sampler. Add compile tests with `fullgraph=True` on fixed shapes.

**Tests (reuse `tests/data/msrd_b2s3/`)**
- `tests/geometry/test_normals.py`, `test_image_pyramid.py`
- `tests/icp/test_projective_icp_identity.py` (zero motion → identity), `test_projective_icp_known_transform.py` (recover a small synthetic SE(3)), `test_solvers.py`, `test_compile_icp.py`
- `tests/mapping/test_tsdf_integrate.py`, `tests/rendering/test_tsdf_raycast.py` (non-empty depth after integrate), compile tests
- `tests/slam/test_pipeline_two_frames.py`

**Verify M2**
```
pytest tests/geometry tests/icp tests/mapping tests/rendering tests/slam -q
docker compose run --rm gradslam pytest tests/icp tests/mapping -q   # ROCm
GRADSLAM_COMPILE=1 docker compose run --rm gradslam pytest tests/icp/test_compile_icp.py -q
```

---

## Milestone 3 — docs + polish

- Modernize `docs/conf.py` to current Sphinx + a maintained theme (e.g. furo), `autodoc` + `napoleon` (Google-style docstrings), `nbsphinx` for the 4 notebooks in `docs/tutorials/`. Add `docs/modules/{backend,icp,mapping,rendering}.rst` for the new packages; keep existing module pages.
- Update `.readthedocs.yml` to v2 (build.os ubuntu-24.04, python 3.12, install `.[docs]`). *(Note: `gradslam.readthedocs.io` is the upstream's site; the fork must connect its own RTD project or rely on the Pages deploy below.)*
- Add `.github/workflows/docs.yml`: build Sphinx on PRs (catch breakage) and deploy to GitHub Pages on `main`.
- Google-style docstrings (Args / Returns / Shapes / Device-dtype) on all new public classes/functions.
- README: update install (container-first + `pip install -e .`), document `requires-python>=3.12` breaking change, ROCm quickstart, and the new fast-tracker vs. differentiable-blocks distinction.

**Verify M3**
```
docker compose run --rm gradslam make docs        # sphinx-build, 0 errors
```

---

## Later / optional (interfaces only — keep lean, do not block M1–M3)

- `gradslam/icp/pointcloud.py`: DiffICP-*inspired* pure-torch point-cloud ICP (point-to-point SVD/Procrustes, point-to-plane, symmetric, robust rejection, chunked `cdist` correspondences). **Do not copy DiffICP code** (license unverified) or depend on PyTorch3D. Reuse the M1 torch `knn_points`.
- `gradslam/optimization/{pgo_torch,pgo_pypose}.py`: pose-graph optimization, PyPose preferred with a dense-torch fallback + import guard.
- `gradslam/slam/loop_closure.py`: `LoopCandidateProvider`/`LoopVerifier` protocols + an ICP-based verifier. Descriptor retrieval (AnyLoc/DINOv2) optional, separate from tracking core.
- `scripts/benchmark_*.py`: projective-ICP / TSDF / TUM-pipeline benchmarks emitting CSV.

## Risks / watch-items

- ROCm container tag must resolve (`docker manifest inspect`) before the Dockerfile is trusted.
- `open3d` top-level import removal must be container-verified for the segfault-ordering issue before deletion (regression test guards it).
- `torch.compile` may hit recompilation guards on dynamic H/W/N — `fullgraph` only in fixed-shape tests.
- CI runs CPU only; ROCm coverage is local-in-container by necessity.
- `requires-python>=3.12` drops Python <3.12 users (acceptable: container-first).
