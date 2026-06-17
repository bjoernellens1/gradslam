from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RGBDFrame:
    rgb: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    depth_factor: float = 1000.0
    timestamp: Optional[float] = None
    name: Optional[str] = None


class RGBDSource:

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> RGBDFrame:
        raise NotImplementedError

    def close(self) -> None:
        pass
