"""Depth/vertex/normal map utilities for dense SLAM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_to_vertex(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Back-project a depth image to a vertex map in camera coordinates.

    Args:
        depth: Depth image [H, W] in meters (zero = invalid).
        intrinsics: Camera intrinsics [3, 3] (K matrix).

    Returns:
        Vertex map [H, W, 3] in camera coordinates.

    Shapes:
        - depth: [H, W]
        - intrinsics: [3, 3]
        - output: [H, W, 3]
    """
    H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    fx, fy = intrinsics[0, 0].to(dtype), intrinsics[1, 1].to(dtype)
    cx, cy = intrinsics[0, 2].to(dtype), intrinsics[1, 2].to(dtype)

    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="xy",
    )

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return torch.stack([x, y, z], dim=-1)


def estimate_normals(depth: torch.Tensor) -> torch.Tensor:
    """Estimate surface normals from a depth map via cross-product of finite differences.

    Pixels with zero depth in the neighborhood produce unreliable normals;
    they are still computed but callers should mask on depth > 0.

    Args:
        depth: Depth image [H, W] in meters.

    Returns:
        Normal map [H, W, 3] in camera space, unit-length.

    Shapes:
        - depth: [H, W]
        - output: [H, W, 3]
    """
    d = F.pad(depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    d = d.squeeze(0).squeeze(0)  # [H+2, W+2]

    dz_dx = (d[1:-1, 2:] - d[1:-1, :-2]) * 0.5
    dz_dy = (d[2:, 1:-1] - d[:-2, 1:-1]) * 0.5

    ones = torch.ones_like(dz_dx)
    n = torch.stack([-dz_dx, -dz_dy, ones], dim=-1)
    return F.normalize(n, dim=-1)


def vertex_to_normal(vertex: torch.Tensor) -> torch.Tensor:
    """Estimate surface normals from a vertex map via cross-product of neighbors.

    Args:
        vertex: Vertex map [H, W, 3].

    Returns:
        Normal map [H, W, 3], unit-length.

    Shapes:
        - vertex: [H, W, 3]
        - output: [H, W, 3]
    """
    # Use central differences on the vertex map
    H, W, _ = vertex.shape

    # Pad to handle borders
    v = F.pad(
        vertex.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
        (1, 1, 1, 1),
        mode="replicate",
    ).squeeze(0).permute(1, 2, 0)  # [H+2, W+2, 3]

    dx = v[1:-1, 2:] - v[1:-1, :-2]   # [H, W, 3]
    dy = v[2:, 1:-1] - v[:-2, 1:-1]   # [H, W, 3]

    n = torch.linalg.cross(dx, dy)     # [H, W, 3]
    return F.normalize(n, dim=-1)
