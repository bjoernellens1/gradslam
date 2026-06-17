"""Preallocated GPU scratch buffers for SLAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GPUScratch:
    H: int
    W: int
    device: torch.device
    n_levels: int = 3

    vertex_map: Optional[torch.Tensor] = None
    normal_map: Optional[torch.Tensor] = None

    rgb_pyr: Optional[list[torch.Tensor]] = None
    depth_pyr: Optional[list[torch.Tensor]] = None
    vertex_pyr: Optional[list[torch.Tensor]] = None
    normal_pyr: Optional[list[torch.Tensor]] = None

    icp_residual: Optional[torch.Tensor] = None
    icp_weights: Optional[torch.Tensor] = None

    def __post_init__(self):
        self._allocate()

    def _allocate(self):
        self.vertex_map = torch.empty((3, self.H, self.W), device=self.device, dtype=torch.float32)
        self.normal_map = torch.empty((3, self.H, self.W), device=self.device, dtype=torch.float16)

        sizes = [(self.H, self.W)]
        for _ in range(1, self.n_levels):
            sizes.append((sizes[-1][0] // 2, sizes[-1][1] // 2))

        self.rgb_pyr = [
            torch.empty((3, h, w), device=self.device, dtype=torch.float16)
            for h, w in sizes
        ]
        self.depth_pyr = [
            torch.empty((1, h, w), device=self.device, dtype=torch.float32)
            for h, w in sizes
        ]
        self.vertex_pyr = [
            torch.empty((3, h, w), device=self.device, dtype=torch.float32)
            for h, w in sizes
        ]
        self.normal_pyr = [
            torch.empty((3, h, w), device=self.device, dtype=torch.float16)
            for h, w in sizes
        ]

        max_valid = self.H * self.W
        self.icp_residual = torch.empty((max_valid, 6), device=self.device, dtype=torch.float32)
        self.icp_weights = torch.empty((max_valid,), device=self.device, dtype=torch.float32)

    def clear(self):
        if self.vertex_map is not None:
            self.vertex_map.zero_()
        if self.normal_map is not None:
            self.normal_map.zero_()
