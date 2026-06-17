from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .source import RGBDSource, RGBDFrame


class TumSource(RGBDSource):

    _INTRINSICS = {
        "freiburg1": (525.0, 525.0, 319.5, 239.5),
        "freiburg2": (520.9, 521.0, 325.1, 249.7),
        "freiburg3": (535.4, 539.2, 320.1, 247.6),
    }
    _DEFAULT_INTRINSICS = (525.0, 525.0, 319.5, 239.5)

    def __init__(
        self,
        basedir: str,
        sequence: Optional[str] = None,
        height: int = 480,
        width: int = 640,
    ):
        basedir = Path(basedir)
        if sequence is not None:
            seq_subdir = basedir / sequence / f"rgbd_dataset_{sequence}"
            if seq_subdir.exists():
                self._seq_dir = seq_subdir
            else:
                self._seq_dir = basedir / sequence
        else:
            self._seq_dir = basedir

        self._height = height
        self._width = width

        self._rgb_files, self._depth_files, self._timestamps = self._load_associations()

        seq_name = self._seq_dir.name
        fx, fy, cx, cy = self._DEFAULT_INTRINSICS
        for key, vals in self._INTRINSICS.items():
            if key in seq_name:
                fx, fy, cx, cy = vals
                break

        self._K = torch.tensor([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

    def _load_associations(self):
        rgb_txt = self._seq_dir / "rgb.txt"
        depth_txt = self._seq_dir / "depth.txt"

        rgb_entries = self._parse_tum_txt(rgb_txt)
        depth_entries = self._parse_tum_txt(depth_txt)

        rgb_files, depth_files, timestamps = [], [], []
        di = 0
        for ts_rgb, rgb_file in rgb_entries:
            best_di = None
            best_dt = float("inf")
            while di < len(depth_entries):
                dt = abs(depth_entries[di][0] - ts_rgb)
                if dt < best_dt:
                    best_dt = dt
                    best_di = di
                if depth_entries[di][0] < ts_rgb - 0.02:
                    di += 1
                else:
                    break
            if best_di is not None and best_dt < 0.02:
                rgb_files.append(str(self._seq_dir / rgb_file))
                depth_files.append(str(self._seq_dir / depth_entries[best_di][1]))
                timestamps.append(ts_rgb)

        return rgb_files, depth_files, timestamps

    @staticmethod
    def _parse_tum_txt(path):
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                ts = float(parts[0])
                filename = parts[1]
                entries.append((ts, filename))
        return entries

    def __len__(self) -> int:
        return len(self._rgb_files)

    def __getitem__(self, idx: int) -> RGBDFrame:
        rgb_bgr = cv2.imread(self._rgb_files[idx], cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read {self._rgb_files[idx]}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb)

        depth_raw = cv2.imread(self._depth_files[idx], cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            raise RuntimeError(f"Failed to read {self._depth_files[idx]}")
        depth_t = torch.from_numpy(depth_raw.astype(np.uint16))

        return RGBDFrame(
            rgb=rgb_t,
            depth=depth_t,
            intrinsics=self._K.clone(),
            depth_factor=5000.0,
            timestamp=self._timestamps[idx],
            name=f"frame_{idx:06d}",
        )
