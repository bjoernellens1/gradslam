"""GPU pyramid construction for multi-scale processing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.compile(mode="reduce-overhead", fullgraph=False)
def build_pyramid_gpu(
    img: torch.Tensor,
    n_levels: int,
    mode: str = "bilinear",
) -> list[torch.Tensor]:
    pyramid = [img]

    for _ in range(1, n_levels):
        prev = pyramid[-1]

        if prev.ndim == 2:
            down = F.avg_pool2d(prev.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2)
            down = down.squeeze(0).squeeze(0)
        else:
            down = F.avg_pool2d(prev.unsqueeze(0), kernel_size=2, stride=2)
            down = down.squeeze(0)

        pyramid.append(down)

    return pyramid


@torch.compile(mode="reduce-overhead", fullgraph=False)
def build_rgbd_pyramid_gpu(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    n_levels: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    rgb_pyr = build_pyramid_gpu(rgb, n_levels, mode="bilinear")
    depth_pyr = build_pyramid_gpu(depth, n_levels, mode="nearest")

    return rgb_pyr, depth_pyr
