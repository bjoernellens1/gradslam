"""Dataset loader for the normalized RGB-D format from ros2-jazzy-realsense-fedora.

Expected directory layout::

    <capture_dir>/
      frames.csv          # header: index,timestamp,rgb_file,depth_file
      images/
        00000001.png      # RGB, uint8
        ...
      depth/
        00000001.png      # depth, uint16, millimeters
        ...
      camera_info.json    # fx,fy,cx,cy,width,height,depth_factor,...
      groundtruth_tum.txt # optional: timestamp tx ty tz qx qy qz qw (space-sep)
      <name>_rtabmap_odom_tum.csv  # optional: same fields with CSV header
      best_pseudo_gt_tum.csv       # optional sibling pseudo-GT from the ROS2 pipeline
      candidate_best_unreliable_tum.csv  # optional sibling fallback pseudo-GT

This is the canonical output of the pseudo_gt_pipeline.py normalizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils import data

from ..geometry.geometryutils import relative_transformation
from . import datautils

__all__ = ["NormalizedRGBD"]


class NormalizedRGBD(data.Dataset):
    """Dataset loader for the normalized RGB-D format.

    Reads from the canonical output of the ros2-jazzy-realsense-fedora
    pseudo_gt_pipeline, which converts any bag/TUM/Hypersim dataset into
    a standard layout with frames.csv, images/, depth/, and camera_info.json.

    Args:
        capture_dir: Path to the normalized capture directory.
        seqlen: Frames per sequence chunk. Default: 1.
        dilation: Frames to skip between consecutive frames. Default: 0.
        stride: Step between consecutive sequence start frames. Default: 1.
        start: First frame index. Default: 0.
        end: Last frame index (exclusive, -1 = all). Default: -1.
        channels_first: Use ``(L, C, H, W)``. Default: False.
        normalize_color: Normalize RGB to ``[0, 1]``. Default: True.
        gt_file: Path to ground-truth TUM file (auto-detected if None).
        return_depth: Include depth. Default: True.
        return_intrinsics: Include intrinsics. Default: True.
        return_pose: Include GT poses (requires gt_file or auto-detected). Default: True.
        return_transform: Include relative transforms. Default: True.
        return_names: Include frame names. Default: True.
        return_timestamps: Include timestamps. Default: True.
    """

    def __init__(
        self,
        capture_dir: str,
        seqlen: int = 1,
        dilation: int = 0,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
        channels_first: bool = False,
        normalize_color: bool = True,
        gt_file: Optional[str] = None,
        *,
        return_depth: bool = True,
        return_intrinsics: bool = True,
        return_pose: bool = True,
        return_transform: bool = True,
        return_names: bool = True,
        return_timestamps: bool = True,
    ):
        super().__init__()
        self.capture_dir = Path(capture_dir)
        self.seqlen = seqlen
        self.dilation = dilation
        self.stride = stride
        self.channels_first = channels_first
        self.normalize_color = normalize_color
        self.return_depth = return_depth
        self.return_intrinsics = return_intrinsics
        self.return_pose = return_pose
        self.return_transform = return_transform
        self.return_names = return_names
        self.return_timestamps = return_timestamps
        self.load_poses = return_pose or return_transform

        frames_csv = self.capture_dir / "frames.csv"
        camera_info_file = self.capture_dir / "camera_info.json"

        if not frames_csv.exists():
            raise FileNotFoundError(f"frames.csv not found: {frames_csv}")
        if not camera_info_file.exists():
            raise FileNotFoundError(f"camera_info.json not found: {camera_info_file}")

        # Load frames index
        self._frames = self._load_frames_csv(frames_csv, start, end)

        if len(self._frames) == 0:
            raise ValueError(f"No frames in {frames_csv} within [{start}, {end})")

        # Load camera intrinsics
        with open(camera_info_file) as f:
            ci = json.load(f)
        self.depth_factor = float(ci.get("depth_factor", 1000.0))
        self.height = int(ci["height"])
        self.width = int(ci["width"])
        fx = float(ci["fx"])
        fy = float(ci["fy"])
        cx = float(ci["cx"])
        cy = float(ci["cy"])

        K = torch.tensor(
            [
                [fx,  0.0, cx, 0.0],
                [0.0,  fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.intrinsics = K.unsqueeze(0)  # [1, 4, 4]

        # Load GT poses if available
        self.gt_file: Optional[Path] = None
        self._pose_by_ts: dict[float, np.ndarray] = {}
        if self.load_poses:
            gt_path = self._resolve_gt_file(gt_file)
            if gt_path is not None:
                self.gt_file = gt_path
                self._pose_by_ts = _load_gt_file(gt_path)
            elif return_pose or return_transform:
                import warnings
                warnings.warn(
                    f"No GT file found in {capture_dir}. Poses will be None.",
                    UserWarning,
                )

        # Build sequence index
        step = dilation + 1
        seq_end = step * (seqlen - 1)
        n = len(self._frames)
        self._starts = list(range(0, n - seq_end, stride))
        self.num_sequences = len(self._starts)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        """Return a sequence chunk.

        Returns:
            color_seq, [depth_seq], [intrinsics], [pose_seq], [transform_seq],
            [names], [timestamps].
        """
        import cv2

        s = self._starts[idx]
        step = self.dilation + 1
        chunk = [self._frames[s + i * step] for i in range(self.seqlen)]

        color_seq, depth_seq, pose_seq, names, timestamps = [], [], [], [], []

        for row in chunk:
            ts = row["timestamp"]
            rgb_path = self.capture_dir / row["rgb_file"]
            dep_path = self.capture_dir / row["depth_file"]
            names.append(rgb_path.stem)
            timestamps.append(ts)

            # Color
            _img = cv2.imread(str(rgb_path))
            if _img is None:
                # fallback to imageio
                import imageio.v2 as imageio
                color = np.array(imageio.imread(str(rgb_path)), dtype=np.float32)
                if color.ndim == 2:
                    color = np.stack([color] * 3, axis=-1)
                elif color.shape[2] == 4:
                    color = color[:, :, :3]
            else:
                color = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            if self.normalize_color:
                color = color / 255.0
            if self.channels_first:
                color = color.transpose(2, 0, 1)
            color_seq.append(torch.from_numpy(color))

            # Depth
            if self.return_depth:
                _dep = cv2.imread(str(dep_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
                if _dep is None:
                    # fallback to imageio
                    import imageio.v2 as imageio
                    _dep_arr = np.array(imageio.imread(str(dep_path)), dtype=np.float32)
                else:
                    _dep_arr = np.ascontiguousarray(_dep).astype(np.float32)
                depth = _dep_arr / self.depth_factor
                depth = depth[:, :, np.newaxis]
                if self.channels_first:
                    depth = depth.transpose(2, 0, 1)
                depth_seq.append(torch.from_numpy(depth))

            # Pose
            if self.load_poses and self._pose_by_ts:
                T = _lookup_pose(self._pose_by_ts, ts, max_dt=0.05)
                pose_seq.append(T)

        output = []
        output.append(torch.stack(color_seq, 0))

        if self.return_depth:
            output.append(torch.stack(depth_seq, 0))

        if self.return_intrinsics:
            output.append(self.intrinsics)

        if self.return_pose and pose_seq:
            output.append(torch.stack([torch.from_numpy(p.astype(np.float32)) for p in pose_seq], 0))

        if self.return_transform and pose_seq:
            transforms = datautils.poses_to_transforms(pose_seq)
            output.append(
                torch.stack([torch.from_numpy(t.astype(np.float32)) for t in transforms], 0)
            )

        if self.return_names:
            output.append(names)

        if self.return_timestamps:
            output.append(timestamps)

        return tuple(output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_frames_csv(self, path: Path, start: int, end: int) -> list[dict]:
        import csv
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "index": int(row.get("index", row.get("frame_index", len(rows)))),
                    "timestamp": float(row["timestamp"]),
                    "rgb_file": row.get("rgb_file", row.get("color_file", "")),
                    "depth_file": row.get("depth_file", ""),
                })
        end_idx = len(rows) if end < 0 else min(end, len(rows))
        return rows[start:end_idx]

    def _resolve_gt_file(self, gt_file: Optional[str]) -> Optional[Path]:
        if gt_file is not None:
            p = Path(gt_file)
            return p if p.exists() else None

        # Auto-detect canonical pseudo-GT sidecars first, then local GT files.
        # The ROS2 handheld pipeline often persists the normalized dataset in
        # <output>/dataset/ while writing best_pseudo_gt_tum.csv next to it.
        candidates = [
            self.capture_dir / "best_pseudo_gt_tum.csv",
            self.capture_dir / "candidate_best_unreliable_tum.csv",
            self.capture_dir.parent / "best_pseudo_gt_tum.csv",
            self.capture_dir.parent / "candidate_best_unreliable_tum.csv",
            self.capture_dir.parent / "groundtruth_tum.txt",
            self.capture_dir.parent / "groundtruth.txt",
            self.capture_dir / "groundtruth_tum.txt",
            self.capture_dir / "groundtruth.txt",
        ]
        for p in candidates:
            if p.exists():
                return p
        # Try CSV variants
        for p in sorted(self.capture_dir.glob("*_tum.csv")):
            return p
        for p in sorted(self.capture_dir.parent.glob("*_tum.csv")):
            return p
        return None


def _load_gt_file(path: Path) -> dict[float, np.ndarray]:
    """Load GT poses from TUM format (space or comma separated, optional header)."""
    from scipy.spatial.transform import Rotation

    poses: dict[float, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip CSV header
            if line.lower().startswith("timestamp") or line.lower().startswith("frame"):
                continue
            # Handle both comma and space separated
            vals = line.replace(",", " ").split()
            if len(vals) < 8:
                continue
            ts = float(vals[0])
            t = np.array([float(v) for v in vals[1:4]])
            q = np.array([float(v) for v in vals[4:8]])  # qx qy qz qw
            if not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
                continue
            if np.linalg.norm(q) < 1e-8:
                continue
            R = Rotation.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses[ts] = T
    return poses


def _lookup_pose(
    poses: dict[float, np.ndarray],
    ts: float,
    max_dt: float = 0.05,
) -> np.ndarray:
    """Return the closest GT pose to timestamp ts."""
    ts_arr = np.array(list(poses.keys()))
    nearest_idx = np.argmin(np.abs(ts_arr - ts))
    if abs(ts_arr[nearest_idx] - ts) <= max_dt:
        return poses[ts_arr[nearest_idx]]
    # Return identity if no match within max_dt
    return np.eye(4, dtype=np.float64)
