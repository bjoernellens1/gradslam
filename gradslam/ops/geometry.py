"""GPU geometry operations for RGB-D SLAM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.compile(mode="reduce-overhead", fullgraph=False)
def depth_to_vertex_map(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    x = (uu - cx) * depth / fx
    y = (vv - cy) * depth / fy
    z = depth

    return torch.stack([x, y, z], dim=-1)


@torch.compile(mode="reduce-overhead", fullgraph=False)
def vertex_to_normal_map(vertex: torch.Tensor) -> torch.Tensor:
    vp = F.pad(vertex.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    vp = vp.squeeze(0).permute(1, 2, 0)

    dx = vp[1:-1, 2:] - vp[1:-1, :-2]
    dy = vp[2:, 1:-1] - vp[:-2, 1:-1]

    n = torch.linalg.cross(dx, dy, dim=-1)
    n = F.normalize(n, dim=-1)

    valid = (vertex.norm(dim=-1) > 0)
    n = torch.where(valid.unsqueeze(-1), n, torch.zeros_like(n))

    return n


@torch.compile(mode="reduce-overhead", fullgraph=False)
def depth_to_normal_map(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    vertex = depth_to_vertex_map(depth, K)
    return vertex_to_normal_map(vertex)
