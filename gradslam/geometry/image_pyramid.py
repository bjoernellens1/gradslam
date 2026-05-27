"""Multi-scale image pyramids for coarse-to-fine ICP."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_depth_pyramid(
    depth: torch.Tensor,
    n_levels: int,
) -> list[torch.Tensor]:
    """Build a Gaussian (average-pool) depth pyramid, coarsest first.

    Args:
        depth: Full-resolution depth image [H, W].
        n_levels: Number of pyramid levels (including full resolution).

    Returns:
        List of n_levels tensors, index 0 = most downsampled (coarsest),
        index -1 = original resolution (finest).

    Shapes:
        - depth: [H, W]
        - output[i]: [H / 2^(n-1-i), W / 2^(n-1-i)]
    """
    levels = [depth]
    for _ in range(n_levels - 1):
        downed = F.avg_pool2d(
            levels[-1].unsqueeze(0).unsqueeze(0),
            kernel_size=2,
            stride=2,
        ).squeeze(0).squeeze(0)
        levels.append(downed)
    return list(reversed(levels))


def build_normal_pyramid(
    normal: torch.Tensor,
    n_levels: int,
) -> list[torch.Tensor]:
    """Build a Gaussian (average-pool + renormalize) normal pyramid, coarsest first.

    Args:
        normal: Full-resolution normal map [H, W, 3].
        n_levels: Number of pyramid levels (including full resolution).

    Returns:
        List of n_levels tensors, index 0 = coarsest, index -1 = finest.

    Shapes:
        - normal: [H, W, 3]
        - output[i]: [H / 2^(n-1-i), W / 2^(n-1-i), 3]
    """
    levels = [normal]
    for _ in range(n_levels - 1):
        prev = levels[-1]
        downed = F.avg_pool2d(
            prev.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
            kernel_size=2,
            stride=2,
        ).squeeze(0).permute(1, 2, 0)            # [H/2, W/2, 3]
        downed = F.normalize(downed, dim=-1)
        levels.append(downed)
    return list(reversed(levels))


def scale_intrinsics(intrinsics: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale camera intrinsics for a downsampled image.

    Divides fx, fy, cx, cy by `scale`.

    Args:
        intrinsics: Camera intrinsics [3, 3] or [4, 4].
        scale: Downscale factor (e.g. 2 for half-resolution).

    Returns:
        Scaled intrinsics, same shape as input.

    Shapes:
        - intrinsics: [3, 3] or [4, 4]
        - output: same shape
    """
    K = intrinsics.clone()
    K[0, 0] = intrinsics[0, 0] / scale  # fx
    K[1, 1] = intrinsics[1, 1] / scale  # fy
    K[0, 2] = intrinsics[0, 2] / scale  # cx
    K[1, 2] = intrinsics[1, 2] / scale  # cy
    return K
