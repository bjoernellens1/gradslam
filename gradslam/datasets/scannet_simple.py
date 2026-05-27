"""Simple ScanNet loader for directly-extracted scenes (no seqmeta required)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from natsort import natsorted
from torch.utils import data

from ..geometry.geometryutils import relative_transformation
from . import datautils

__all__ = ["ScanNetSimple"]


# Depth is stored as uint16 millimeters → divide by 1000 to get meters
_SCANNET_DEPTH_SCALE = 1000.0


class ScanNetSimple(data.Dataset):
    """Loader for directly-extracted ScanNet scenes (no seqmeta file needed).

    Expected directory layout for each scene::

        <scene_dir>/             e.g. scans/scene0011_00/
          color/
            0.jpg
            1.jpg
            ...
          depth/
            0.png                # uint16, millimeters
            1.png
            ...
          pose/
            0.txt                # T_world_camera: 4×4 matrix (4 lines × 4 space-separated floats)
            1.txt
            ...
          intrinsic/
            intrinsic_depth.txt  # 4×4 matrix (used for depth/RGB after resize to depth res)

    Color images (often 968×1296) are resized to match depth resolution (default 480×640).

    Args:
        scene_dir: Path to a single ScanNet scene directory.
        seqlen: Frames per sequence chunk. Default: 1.
        dilation: Frames to skip between consecutive frames. Default: 0.
        stride: Step between consecutive sequence start indices. Default: 1.
        start: First frame index. Default: 0.
        end: Last frame index (exclusive, -1 = all). Default: -1.
        height: Output height after resize. Default: 480.
        width: Output width after resize. Default: 640.
        channels_first: Use ``(L, C, H, W)``. Default: False.
        normalize_color: Normalize RGB to ``[0, 1]``. Default: True.
        return_depth: Include depth. Default: True.
        return_intrinsics: Include intrinsics. Default: True.
        return_pose: Include GT poses. Default: True.
        return_transform: Include relative transforms. Default: True.
        return_names: Include frame names. Default: True.
    """

    def __init__(
        self,
        scene_dir: str,
        seqlen: int = 1,
        dilation: int = 0,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
        height: int = 480,
        width: int = 640,
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
        self.scene_dir = Path(scene_dir)
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

        color_dir = self.scene_dir / "color"
        depth_dir = self.scene_dir / "depth"
        pose_dir = self.scene_dir / "pose"
        intrinsic_file = self.scene_dir / "intrinsic" / "intrinsic_depth.txt"

        for d in (color_dir, depth_dir, pose_dir, intrinsic_file):
            if not d.exists():
                raise FileNotFoundError(f"Required path not found: {d}")

        # Discover frames (natural sort to handle 0, 1, 10, 11, ...)
        color_files = natsorted(color_dir.glob("*.jpg"))
        depth_files = natsorted(depth_dir.glob("*.png"))
        pose_files = natsorted(pose_dir.glob("*.txt"))

        # Align by stem so missing frames are skipped
        depth_stems = {f.stem: f for f in depth_files}
        pose_stems = {f.stem: f for f in pose_files}
        frames = []
        for cf in color_files:
            stem = cf.stem
            if stem in depth_stems and stem in pose_stems:
                frames.append((cf, depth_stems[stem], pose_stems[stem]))

        if len(frames) == 0:
            raise ValueError(f"No valid frames found in {scene_dir}")

        n_total = len(frames)
        end_idx = n_total if end < 0 else min(end, n_total)
        frames = frames[start:end_idx]

        self._frames = frames

        # Read intrinsics (4×4 matrix in depth resolution)
        K_raw = np.loadtxt(str(intrinsic_file))  # [4, 4]
        # Scale to target resolution
        native_h, native_w = 480, 640
        K = K_raw.copy()
        K[0, :] *= width / native_w
        K[1, :] *= height / native_h
        self.intrinsics = torch.from_numpy(K.astype(np.float32)).unsqueeze(0)  # [1, 4, 4]

        # Build sequences
        step = dilation + 1
        seq_end = step * (seqlen - 1)
        n_frames = len(frames)
        starts = list(range(0, n_frames - seq_end, stride))
        self.seq_starts = starts
        self.num_sequences = len(starts)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        """Return a sequence chunk.

        Returns:
            color_seq: ``(L, H, W, 3)`` float32.
            depth_seq: ``(L, H, W, 1)`` float32 in meters.
            intrinsics: ``(1, 4, 4)`` float32.
            pose_seq: ``(L, 4, 4)`` float32 T_world_camera.
            transform_seq: ``(L, 4, 4)`` float32 relative transforms.
            framenames: list of frame stem strings.
        """
        import cv2
        import imageio.v2 as imageio

        s = self.seq_starts[idx]
        step = self.dilation + 1
        frame_indices = [s + i * step for i in range(self.seqlen)]

        color_seq, depth_seq, pose_seq, names = [], [], [], []

        for fi in frame_indices:
            cf, df, pf = self._frames[fi]
            names.append(cf.stem)

            # Color
            color = np.array(imageio.imread(str(cf)), dtype=np.float32)
            if color.ndim == 2:
                color = np.stack([color] * 3, axis=-1)
            if color.shape[0] != self.height or color.shape[1] != self.width:
                color = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.normalize_color:
                color = color / 255.0
            if self.channels_first:
                color = color.transpose(2, 0, 1)
            color_seq.append(torch.from_numpy(color))

            # Depth
            if self.return_depth:
                depth = np.array(imageio.imread(str(df)), dtype=np.float32)
                depth = depth / _SCANNET_DEPTH_SCALE  # mm → m
                if depth.shape[0] != self.height or depth.shape[1] != self.width:
                    depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                depth = depth[:, :, np.newaxis]
                if self.channels_first:
                    depth = depth.transpose(2, 0, 1)
                depth_seq.append(torch.from_numpy(depth))

            # Pose
            if self.load_poses:
                T = np.loadtxt(str(pf)).astype(np.float32)  # [4, 4]
                pose_seq.append(torch.from_numpy(T))

        output = []
        output.append(torch.stack(color_seq, 0))

        if self.return_depth:
            output.append(torch.stack(depth_seq, 0))

        if self.return_intrinsics:
            output.append(self.intrinsics)

        if self.return_pose:
            output.append(torch.stack(pose_seq, 0))

        if self.return_transform:
            poses_np = [p.numpy() for p in pose_seq]
            transforms = datautils.poses_to_transforms(poses_np)
            output.append(
                torch.stack([torch.from_numpy(t.astype(np.float32)) for t in transforms], 0)
            )

        if self.return_names:
            output.append(names)

        return tuple(output)
