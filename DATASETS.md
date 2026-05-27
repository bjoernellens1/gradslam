# gradslam Datasets

Shared dataset storage is at `/mnt/cps_persistent1_shared/datasets/` on the CPS cluster.
Inside the container (via compose.yaml) it is mounted read-only at `/workspace/datasets/`.

---

## TUM RGB-D Dataset (recommended for SLAM testing)

**18 extracted sequences** ready to use at:
```
Host:      /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/
Container: /workspace/datasets/public/TUM/tum_rgbd/
```

Each sequence lives at `<root>/<seq_name>/rgbd_dataset_<seq_name>/` and contains:
```
rgb/          PNG color images
depth/        PNG depth images (16-bit, scale factor 5000)
rgb.txt       frame list: timestamp filename
depth.txt     frame list: timestamp filename
groundtruth.txt  poses: timestamp tx ty tz qx qy qz qw
```

### Available sequences

| Sequence | Category | Difficulty |
|---|---|---|
| `freiburg1_desk` | desktop/tabletop | easy |
| `freiburg1_desk2` | desktop/tabletop | easy |
| `freiburg1_room` | indoor room | medium |
| `freiburg1_rpy` | rotation-only | easy |
| `freiburg1_xyz` | translation-only | easy |
| `freiburg2_desk` | desktop/tabletop | easy |
| `freiburg2_desk_with_person` | desktop + motion | medium |
| `freiburg2_360_kidnap` | kidnap/loop | hard |
| `freiburg2_large_no_loop` | large scene | hard |
| `freiburg2_large_with_loop` | large scene + loop | hard |
| `freiburg2_rpy` | rotation-only | easy |
| `freiburg3_long_office_household` | office | medium |
| `freiburg3_nostructure_notexture_near_withloop` | degenerate | hard |
| `freiburg3_nostructure_texture_near_withloop` | degenerate | hard |
| `freiburg3_sitting_xyz` | dynamic (person) | hard |
| `freiburg3_structure_notexture_near` | featureless | medium |
| `freiburg3_structure_texture_near` | standard | medium |
| `freiburg3_walking_xyz` | dynamic (walking) | hard |

### Running SLAM on TUM sequences

```bash
# Inside container — default sequence (freiburg1_desk)
podman compose run --rm gradslam python scripts/run_tum_slam.py --device cuda:0

# Specific sequence
podman compose run --rm gradslam python scripts/run_tum_slam.py \
    --dataset-root /workspace/datasets/public/TUM/tum_rgbd \
    --sequence freiburg1_desk \
    --device cuda:0 \
    --output /workspace/gradslam/outputs/tum_freiburg1_desk

# Or on host (CPU only)
python scripts/run_tum_slam.py \
    --dataset-root /mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd \
    --sequence freiburg1_desk \
    --device cpu

# Process every 2nd frame (faster end-to-end test)
podman compose run --rm gradslam python scripts/run_tum_slam.py \
    --sequence freiburg1_room \
    --stride 2 \
    --device cuda:0
```

**Note:** The `--dataset-root` must point to the directory containing the named sequence subfolder.
For the tum_rgbd layout, use `tum_rgbd` as root and e.g. `freiburg1_desk` as sequence name.
The script resolves the path as `<root>/<sequence>/rgbd_dataset_<sequence>/`.

---

## Custom RealSense RGB-D Sequences (Bjoern's captures)

Captured with Intel RealSense D435i on CPS premises.
```
Host:      /mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/
```

| Directory | Scene | Format |
|---|---|---|
| `cps_1stfloor_hallway_mediumdensity/` | 1st floor hallway | .bag + groundtruth_tum.txt |
| `cps_2ndfloor_hallway_hand/` | 2nd floor hallway | .bag + groundtruth_tum.txt |
| `cps_2ndfloor_office1_highdensity/` | office room | .bag + groundtruth_tum.txt |

These have ground truth poses but frames are stored in ROS 2 bags — requires extraction
before use with the TUM loader. Use `ros2 bag convert` or a bag reader to extract PNG
frames and generate `rgb.txt`/`depth.txt` index files.

---

## ScanNet Dataset

Partially downloaded (13 scenes):
```
Host:      /mnt/cps_persistent1_shared/datasets/public/ScanNet/scans/
```

Scenes: `scene0000_00`, `scene0011_*`, `scene0050_*`, `scene0231_*`, `scene0378_*`.
Requires ScanNet account for full access. Format: raw sensor `.sens` files;
needs the ScanNet reader to extract RGB-D frames.

---

## Replica Dataset

19 synthetic indoor scenes (apartment, office, hotel, room variants):
```
Host:      /mnt/cps_persistent1_shared/datasets/public/Replica-Dataset/
           /mnt/cps_persistent1_shared/datasets/public/Replica-NICE-SLAM/
```

The `Replica-NICE-SLAM` variant (1.8 GB) includes pre-rendered RGB-D trajectories
usable directly for SLAM evaluation.

---

## Performance Reference (ROCm, Radeon 8060S)

| Dataset | Frames | Resolution | Speed |
|---|---|---|---|
| TUM freiburg1_desk | 572 | 640×480 | **44.5 fps** |

---

## Adding a New Custom Dataset

Implement a dataset loader that returns `(rgb, depth, intrinsics, pose)` tuples
following the TUM interface in `gradslam/datasets/tum.py`. Alternatively, convert
your sequence to TUM format:
1. Extract frames to `rgb/` and `depth/` directories
2. Write `rgb.txt` and `depth.txt` index files (`timestamp filename` per line)
3. Write `groundtruth.txt` in TUM pose format (`timestamp tx ty tz qx qy qz qw`)
4. Set depth scale factor to 5000 (standard TUM — adjust if needed)
