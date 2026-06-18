"""GPU frame ring buffer for async H2D staging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GPUFrameRing:
    capacity: int = 4
    H: int = 480
    W: int = 640
    device: torch.device = torch.device("cuda")

    rgb: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    K: Optional[torch.Tensor] = None

    write_idx: int = 0
    read_idx: int = 0
    ready: Optional[torch.Tensor] = None

    def __post_init__(self):
        self._allocate()

    def _allocate(self):
        N = self.capacity

        self.rgb = torch.empty((N, 3, self.H, self.W), device=self.device, dtype=torch.float16)
        self.depth = torch.empty((N, 1, self.H, self.W), device=self.device, dtype=torch.float32)
        self.K = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1).clone()
        self.ready = torch.zeros(N, device=self.device, dtype=torch.bool)

    def next_write_slot(self) -> int:
        slot = self.write_idx
        self.write_idx = (self.write_idx + 1) % self.capacity
        return slot

    def mark_ready(self, slot: int):
        self.ready[slot] = True

    def next_read_slot(self) -> Optional[int]:
        if not self.ready.any():
            return None

        for i in range(self.capacity):
            idx = (self.read_idx + i) % self.capacity
            if self.ready[idx]:
                self.read_idx = (idx + 1) % self.capacity
                return idx

        return None

    def mark_consumed(self, slot: int):
        self.ready[slot] = False

    def get_frame(self, slot: int) -> dict[str, torch.Tensor]:
        return {
            "rgb": self.rgb[slot],
            "depth": self.depth[slot],
            "K": self.K[slot],
        }
