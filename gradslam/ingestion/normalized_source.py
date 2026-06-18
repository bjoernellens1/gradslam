from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .source import RGBDSource, RGBDFrame


class NormalizedSource(RGBDSource):

    def __init__(self, capture_dir: str):
        self._dir = Path(capture_dir)

        frames_csv = self._dir / "frames.csv"
        self._frames = []
        with open(frames_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._frames.append({
                    "index": int(row.get("index", row.get("frame_index", len(self._frames)))),
                    "timestamp": float(row["timestamp"]),
                    "rgb_file": row.get("rgb_file", row.get("color_file", "")),
                    "depth_file": row.get("depth_file", ""),
                })

        with open(self._dir / "camera_info.json") as f:
            ci = json.load(f)
        self._depth_factor = float(ci.get("depth_factor", 1000.0))
        fx = float(ci["fx"])
        fy = float(ci["fy"])
        cx = float(ci["cx"])
        cy = float(ci["cy"])

        self._K = torch.tensor([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> RGBDFrame:
        row = self._frames[idx]

        rgb_path = str(self._dir / row["rgb_file"])
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb)

        depth_path = str(self._dir / row["depth_file"])
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            raise RuntimeError(f"Failed to read {depth_path}")
        depth_t = torch.from_numpy(depth_raw.astype(np.uint16))

        return RGBDFrame(
            rgb=rgb_t,
            depth=depth_t,
            intrinsics=self._K.clone(),
            depth_factor=self._depth_factor,
            timestamp=row["timestamp"],
            name=Path(row["rgb_file"]).stem,
        )
