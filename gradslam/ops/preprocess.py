"""GPU preprocessing operations for RGB-D frames."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.compile(mode="reduce-overhead", fullgraph=False)
def preprocess_rgbd_gpu(
    rgb_u8: torch.Tensor,
    depth_u16: torch.Tensor,
    K: torch.Tensor,
    target_hw: tuple[int, int],
    depth_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_h, target_w = target_hw
    in_h, in_w = depth_u16.shape

    rgb = rgb_u8.permute(2, 0, 1).unsqueeze(0).float().mul_(1.0 / 255.0)
    depth = depth_u16.unsqueeze(0).unsqueeze(0).float().mul_(1.0 / depth_scale)

    rgb = F.interpolate(rgb, size=(target_h, target_w), mode="bilinear", align_corners=False)
    depth = F.interpolate(depth, size=(target_h, target_w), mode="nearest")

    sx = target_w / in_w
    sy = target_h / in_h
    K_scaled = K[:3, :3].clone()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy

    return rgb.squeeze(0).permute(1, 2, 0), depth.squeeze(0).squeeze(0), K_scaled


@torch.compile(mode="reduce-overhead", fullgraph=False)
def normalize_rgb_gpu(rgb_u8: torch.Tensor) -> torch.Tensor:
    return rgb_u8.float().mul_(1.0 / 255.0)


@torch.compile(mode="reduce-overhead", fullgraph=False)
def convert_depth_gpu(depth_u16: torch.Tensor, depth_scale: float) -> torch.Tensor:
    return depth_u16.float().mul_(1.0 / depth_scale)
