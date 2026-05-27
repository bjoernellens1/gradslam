# gradslam Datasets

Available datasets are stored in `/mnt/cps_persistent1_shared/datasets/` on the CPS cluster.

## TUM RGB-D Dataset

The TUM RGB-D benchmark is available at:
```
/mnt/cps_persistent1_shared/datasets/public/TUM/
```

### Available TUM Sequences

Extracted sequences with RGB-D frames, depth maps, and ground truth:
- `groundtruth/rgbd_dataset_freiburg1_desk/` — Freiburg 1 desk sequence (598 RGB + depth pairs)

Archived sequences (as ROS bag files):
- Freiburg 1: 360°, desk, desk2, floor, room
- Freiburg 2: 360° hemisphere, 360° kidnap, desk, large with/without loop, pioneer variants
- Freiburg 3: long office household

### TUM Dataset Format

Each extracted sequence directory contains:
```
rgbd_dataset_freiburg1_desk/
├── rgb/                       # RGB images (PNG)
├── depth/                     # Depth maps (PNG, 16-bit)
├── rgb.txt                     # RGB frame list: timestamp filename
├── depth.txt                   # Depth frame list: timestamp filename
├── groundtruth.txt             # Ground truth poses: timestamp tx ty tz qx qy qz qw
├── accelerometer.txt           # IMU accelerometer data
└── *.ply                       # Pre-computed meshes (if available)
```

### Loading TUM Data in gradslam

```python
from gradslam.datasets import TUM

# Load a TUM sequence
dataset = TUM(
    root_dir="/mnt/cps_persistent1_shared/datasets/public/TUM/groundtruth",
    sequence_name="rgbd_dataset_freiburg1_desk",
    stride=1  # Use every frame; increase for faster processing
)

# Iterate over RGB-D frames
for rgb, depth, intrinsics, gt_pose in dataset:
    # rgb: (3, H, W) uint8
    # depth: (1, H, W) float32 (meters)
    # intrinsics: (3, 3) camera intrinsics
    # gt_pose: (4, 4) ground truth pose
    pass
```

## Other Available Public Datasets

- **KITTI**: `/mnt/cps_persistent1_shared/datasets/public/KITTI/`
- **ScanNet**: `/mnt/cps_persistent1_shared/datasets/public/ScanNet/`
- **Replica-Dataset**: `/mnt/cps_persistent1_shared/datasets/public/Replica-Dataset/`
- **COCO**: `/mnt/cps_persistent1_shared/datasets/public/coco/`

## User Datasets

Individual user datasets are available under:
- `bjoern/` — Personal datasets (RealSense handheld, cps-seggy)
- `puneeth/` — Puneeth's datasets
- `christian/` — Christian's datasets
- `Vedant/` — Vedant's datasets

## Running SLAM with Datasets

### On CPU

```bash
# Run SLAM pipeline on TUM sequence
python scripts/run_tum_slam.py \
    --dataset-root /mnt/cps_persistent1_shared/datasets/public/TUM/groundtruth \
    --sequence rgbd_dataset_freiburg1_desk \
    --stride 2 \
    --output ./outputs/tum_slam_results
```

### In ROCm Container

```bash
# Build and enter container
make build
make shell

# Inside container:
python scripts/run_tum_slam.py \
    --dataset-root /mnt/cps_persistent1_shared/datasets/public/TUM/groundtruth \
    --sequence rgbd_dataset_freiburg1_desk \
    --device cuda:0 \
    --output ./outputs/tum_slam_results
```

Or via podman compose:
```bash
podman compose run --rm gradslam python scripts/run_tum_slam.py \
    --dataset-root /mnt/cps_persistent1_shared/datasets/public/TUM/groundtruth \
    --sequence rgbd_dataset_freiburg1_desk
```

## Dataset Mounting with podman-compose

The datasets directory is automatically accessible in the container via the host network. To mount datasets explicitly:

```yaml
# compose.yaml
services:
  gradslam:
    volumes:
      - /mnt/cps_persistent1_shared/datasets:/workspace/datasets:Z,ro
```

Then use:
```bash
python scripts/run_tum_slam.py \
    --dataset-root /workspace/datasets/public/TUM/groundtruth
```

## Performance Notes

- TUM sequences: 300–3,000 frames at 640×480 resolution
- Estimated processing on ROCm: 20–50 fps for full SLAM pipeline (ICP + TSDF integration)
- CPU (multi-threaded): 1–5 fps
- Use `--stride` to skip frames for faster end-to-end testing

## Adding Custom RGB-D Datasets

To add a new dataset, implement a dataset class in `gradslam/datasets/` following the TUM interface:

```python
from gradslam.datasets.base import RGBDDataset

class MyDataset(RGBDDataset):
    def __init__(self, root_dir, **kwargs):
        self.root_dir = root_dir
        self._load_frames()
    
    def _load_frames(self):
        # Load RGB, depth, intrinsics, poses from disk
        pass
    
    def __getitem__(self, idx):
        # Return (rgb, depth, intrinsics, pose) tuple
        pass
```
