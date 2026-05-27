"""Dataset loader for the Replica dataset (NICE-SLAM pre-rendered format)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils import data

from ..geometry.geometryutils import relative_transformation
from . import datautils

__all__ = ["ReplicaNICESLAM"]


# Known intrinsics for Replica NICE-SLAM renders (1200×680)
# Source: NICE-SLAM paper / iMAP supplementary
_REPLICA_INTRINSICS = torch.tensor(
    [
        [600.0,   0.0, 599.5, 0.0],
        [  0.0, 600.0, 339.5, 0.0],
        [  0.0,   0.0,   1.0, 0.0],
        [  0.0,   0.0,   0.0, 1.0],
    ],
    dtype=torch.float64,
)

# Depth scale: uint16 value → meters
_REPLICA_DEPTH_SCALE = 6553.5


class ReplicaNICESLAM(data.Dataset):
    """Loader for the Replica dataset in NICE-SLAM pre-rendered format.

    Expected directory layout::

        <scene_dir>/
          results/
            frame000000.jpg   # RGB 1200×680
            frame000001.jpg
            ...
            depth000000.png   # uint16 depth at scale 6553.5 → meters
            depth000001.png
            ...
          traj.txt            # GT poses: one 4×4 matrix per line (16 floats, row-major)

    Each line of ``traj.txt`` is a flattened row-major ``T_world_camera`` (4×4).

    Args:
        basedir: Path to the scene directory (e.g. ``.../Replica/room0``).
        seqlen: Frames per sequence chunk. Default: 1.
        dilation: Frames to skip between consecutive frames in a chunk. Default: 0.
        stride: Step between consecutive sequence start frames. Default: 1.
        start: First frame index to use. Default: 0.
        end: Last frame index (exclusive, -1 = all). Default: -1.
        height: Resize height. Default: 680 (native).
        width: Resize width. Default: 1200 (native).
        channels_first: Use ``(L, C, H, W)`` instead of ``(L, H, W, C)``. Default: False.
        normalize_color: Normalize RGB to ``[0, 1]``. Default: True.
        return_depth: Include depth in output. Default: True.
        return_intrinsics: Include intrinsics in output. Default: True.
        return_pose: Include GT poses in output. Default: True.
        return_transform: Include relative transforms in output. Default: True.
        return_names: Include frame names in output. Default: True.
    """

    def __init__(
        self,
        basedir: str,
        seqlen: int = 1,
        dilation: int = 0,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
        height: int = 680,
        width: int = 1200,
        channels_first: bool = False,
        normalize_color: bool = True,
        *,
        return_depth: bool = True,
        return_intrinsics: bool = True,
        return_pose: bool = True,
        return_transform: bool = True,
        return_names: bool = True,
    ):
        super().__init__()
        self.basedir = Path(basedir)
        self.seqlen = seqlen
        self.dilation = dilation
        self.stride = stride
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.normalize_color = normalize_color
        self.return_depth = return_depth
        self.return_intrinsics = return_intrinsics
        self.return_pose = return_pose
        self.return_transform = return_transform
        self.return_names = return_names
        self.load_poses = return_pose or return_transform

        results_dir = self.basedir / "results"
        traj_file = self.basedir / "traj.txt"

        if not results_dir.exists():
            raise FileNotFoundError(f"results/ dir not found: {results_dir}")
        if not traj_file.exists():
            raise FileNotFoundError(f"traj.txt not found: {traj_file}")

        # Discover all frames (sorted)
        color_files = sorted(results_dir.glob("frame*.jpg"))
        depth_files = sorted(results_dir.glob("depth*.png"))

        if len(color_files) == 0:
            raise FileNotFoundError(f"No frame*.jpg files in {results_dir}")
        if len(depth_files) != len(color_files):
            raise ValueError(
                f"Mismatch: {len(color_files)} color vs {len(depth_files)} depth files"
            )

        # Trim to [start, end)
        total = len(color_files)
        end_idx = total if end < 0 else min(end, total)
        color_files = color_files[start:end_idx]
        depth_files = depth_files[start:end_idx]

        # Load GT poses
        poses_raw = np.loadtxt(str(traj_file))  # [N, 16] or shape depends on file
        if poses_raw.ndim == 1:
            poses_raw = poses_raw.reshape(1, -1)
        poses_raw = poses_raw[start:end_idx]          # [N, 16]
        self._poses_4x4 = poses_raw.reshape(-1, 4, 4) # [N, 4, 4]

        # Build sequence indices
        step = dilation + 1
        seq_end = step * (seqlen - 1)   # last frame offset within one sequence
        n_frames = len(color_files)
        starts = list(range(0, n_frames - seq_end, stride))

        self.colorfiles = [
            [str(color_files[s + i * step]) for i in range(seqlen)] for s in starts
        ]
        self.depthfiles = [
            [str(depth_files[s + i * step]) for i in range(seqlen)] for s in starts
        ]
        self.pose_indices = [
            [s + i * step for i in range(seqlen)] for s in starts
        ]
        self.framenames = [
            [color_files[s + i * step].stem for i in range(seqlen)] for s in starts
        ]
        self.num_sequences = len(starts)

        # Scale intrinsics if resizing
        native_h, native_w = 680, 1200
        K = _REPLICA_INTRINSICS.clone()
        K[0, :] *= width / native_w
        K[1, :] *= height / native_h
        self.intrinsics = K.float().unsqueeze(0)  # [1, 4, 4]

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        """Return a sequence chunk.

        Returns:
            color_seq: ``(L, H, W, 3)`` float32 in ``[0,1]`` (or ``[0,255]``).
            depth_seq: ``(L, H, W, 1)`` float32 in meters.
            intrinsics: ``(1, 4, 4)`` float32.
            pose_seq: ``(L, 4, 4)`` float32 T_world_camera.
            transform_seq: ``(L, 4, 4)`` float32 relative transforms.
            framenames: list of frame stem strings.
        """
        import cv2
        import imageio.v2 as imageio

        color_paths = self.colorfiles[idx]
        depth_paths = self.depthfiles[idx]
        pose_ids = self.pose_indices[idx]

        color_seq, depth_seq, pose_seq = [], [], []

        for c_path, d_path, p_idx in zip(color_paths, depth_paths, pose_ids):
            # Color
            color = np.array(imageio.imread(c_path), dtype=np.float32)
            if color.shape[0] != self.height or color.shape[1] != self.width:
                color = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.normalize_color:
                color = color / 255.0
            if self.channels_first:
                color = color.transpose(2, 0, 1)
            color_seq.append(torch.from_numpy(color))

            # Depth
            if self.return_depth:
                depth = np.array(imageio.imread(d_path), dtype=np.float32)
                depth = depth / _REPLICA_DEPTH_SCALE  # → meters
                if depth.shape[0] != self.height or depth.shape[1] != self.width:
                    depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                depth = depth[:, :, np.newaxis]  # [H, W, 1]
                if self.channels_first:
                    depth = depth.transpose(2, 0, 1)
                depth_seq.append(torch.from_numpy(depth))

            # Pose
            if self.load_poses:
                T = self._poses_4x4[p_idx].astype(np.float32)
                pose_seq.append(torch.from_numpy(T))

        output = []

        color_out = torch.stack(color_seq, 0)  # [L, H, W, 3]
        output.append(color_out)

        if self.return_depth:
            depth_out = torch.stack(depth_seq, 0)  # [L, H, W, 1]
            output.append(depth_out)

        if self.return_intrinsics:
            output.append(self.intrinsics)

        if self.return_pose:
            pose_out = torch.stack(pose_seq, 0)  # [L, 4, 4]
            output.append(pose_out)

        if self.return_transform:
            poses_np = [p.numpy() for p in pose_seq]
            transforms = datautils.poses_to_transforms(poses_np)
            transform_out = torch.stack(
                [torch.from_numpy(t.astype(np.float32)) for t in transforms], 0
            )
            output.append(transform_out)

        if self.return_names:
            output.append(self.framenames[idx])

        return tuple(output)
