"""GPU-resident keyframe cache for SLAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class GPUKeyframeCache:
    max_capacity: int = 256
    device: torch.device = torch.device("cuda")
    dtype_rgb: torch.dtype = torch.float16
    dtype_depth: torch.dtype = torch.float32
    dtype_pose: torch.dtype = torch.float32
    n_levels: int = 3

    rgb_pyr: Optional[list[torch.Tensor]] = None
    depth_pyr: Optional[list[torch.Tensor]] = None
    vertex_pyr: Optional[list[torch.Tensor]] = None
    normal_pyr: Optional[list[torch.Tensor]] = None
    T_wc: Optional[torch.Tensor] = None
    K_intrinsics: Optional[torch.Tensor] = None
    age: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    count: int = 0

    _H: Optional[int] = None
    _W: Optional[int] = None

    def initialize(self, H: int, W: int):
        self._H = H
        self._W = W
        K = self.max_capacity

        sizes = [(H, W)]
        for _ in range(1, self.n_levels):
            sizes.append((sizes[-1][0] // 2, sizes[-1][1] // 2))

        self.rgb_pyr = [
            torch.empty((K, 3, h, w), device=self.device, dtype=self.dtype_rgb)
            for h, w in sizes
        ]

        self.depth_pyr = [
            torch.empty((K, 1, h, w), device=self.device, dtype=self.dtype_depth)
            for h, w in sizes
        ]

        self.vertex_pyr = [
            torch.empty((K, 3, h, w), device=self.device, dtype=self.dtype_rgb)
            for h, w in sizes
        ]

        self.normal_pyr = [
            torch.empty((K, 3, h, w), device=self.device, dtype=self.dtype_rgb)
            for h, w in sizes
        ]

        self.T_wc = torch.eye(4, device=self.device, dtype=self.dtype_pose).unsqueeze(0).expand(K, -1, -1).clone()
        self.K_intrinsics = torch.eye(3, device=self.device, dtype=self.dtype_pose).unsqueeze(0).expand(K, -1, -1).clone()
        self.age = torch.zeros(K, device=self.device, dtype=torch.int32)
        self.valid = torch.zeros(K, device=self.device, dtype=torch.bool)
        self.count = 0

    def insert(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        normal: torch.Tensor,
        T_wc: torch.Tensor,
        K: torch.Tensor,
        vertex: Optional[torch.Tensor] = None,
    ) -> int:
        if self.rgb_pyr is None:
            self.initialize(rgb.shape[1], rgb.shape[2])

        if self.count < self.max_capacity:
            invalid = (~self.valid).nonzero(as_tuple=True)[0]
            slot = int(invalid[0].item())
            self.count += 1
        else:
            slot = int(torch.argmin(self.age).item())

        self.rgb_pyr[0][slot] = rgb.to(dtype=self.dtype_rgb)
        self.depth_pyr[0][slot] = depth.to(dtype=self.dtype_depth)
        self.normal_pyr[0][slot] = normal.to(dtype=self.dtype_rgb)
        if vertex is not None:
            self.vertex_pyr[0][slot] = vertex.to(dtype=self.dtype_rgb)

        for level in range(1, self.n_levels):
            self.rgb_pyr[level][slot] = F.avg_pool2d(
                self.rgb_pyr[level-1][slot].unsqueeze(0), kernel_size=2, stride=2
            ).squeeze(0)
            self.depth_pyr[level][slot] = F.avg_pool2d(
                self.depth_pyr[level-1][slot].unsqueeze(0), kernel_size=2, stride=2
            ).squeeze(0)
            self.normal_pyr[level][slot] = F.avg_pool2d(
                self.normal_pyr[level-1][slot].unsqueeze(0), kernel_size=2, stride=2
            ).squeeze(0)
            if vertex is not None:
                self.vertex_pyr[level][slot] = F.avg_pool2d(
                    self.vertex_pyr[level-1][slot].unsqueeze(0), kernel_size=2, stride=2
                ).squeeze(0)

        self.T_wc[slot] = T_wc.to(dtype=self.dtype_pose)
        self.K_intrinsics[slot] = K.to(dtype=self.dtype_pose)
        self.age[slot] = 0
        self.valid[slot] = True

        self.age[self.valid] += 1

        return slot

    def get(self, slot: int, level: int = 0) -> dict[str, torch.Tensor]:
        result = {
            "rgb": self.rgb_pyr[level][slot],
            "depth": self.depth_pyr[level][slot],
            "normal": self.normal_pyr[level][slot],
            "vertex": self.vertex_pyr[level][slot] if self.vertex_pyr is not None else None,
            "T_wc": self.T_wc[slot],
            "K": self.K_intrinsics[slot],
        }
        return result

    def get_all_poses(self) -> torch.Tensor:
        return self.T_wc[self.valid]

    def get_all_ages(self) -> torch.Tensor:
        return self.age[self.valid]

    def clear(self):
        if self.valid is not None:
            self.valid.zero_()
        self.count = 0
