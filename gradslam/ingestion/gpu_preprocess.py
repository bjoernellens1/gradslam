from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .source import RGBDFrame


def gpu_preprocess(
    frame: RGBDFrame,
    device: torch.device,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    normalize_color: bool = True,
    non_blocking: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = frame.rgb.to(device, non_blocking=non_blocking)
    depth = frame.depth.to(device, non_blocking=non_blocking)
    K = frame.intrinsics.to(device, non_blocking=non_blocking)

    depth_m = depth.to(dtype=torch.float32) / frame.depth_factor

    if target_height is not None and target_width is not None:
        H, W = rgb.shape[:2]
        if H != target_height or W != target_width:
            if normalize_color:
                rgb = rgb.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                rgb = F.interpolate(rgb, size=(target_height, target_width), mode="bilinear", align_corners=False)
                rgb = rgb.squeeze(0).permute(1, 2, 0)
            else:
                rgb = rgb.permute(2, 0, 1).unsqueeze(0).float()
                rgb = F.interpolate(rgb, size=(target_height, target_width), mode="nearest")
                rgb = rgb.squeeze(0).permute(1, 2, 0).to(torch.uint8)

            depth_m = depth_m.unsqueeze(0).unsqueeze(0)
            depth_m = F.interpolate(depth_m, size=(target_height, target_width), mode="nearest")
            depth_m = depth_m.squeeze(0).squeeze(0)

            scale_h = target_height / H
            scale_w = target_width / W
            K = K.clone()
            K[0, 0] *= scale_w
            K[1, 1] *= scale_h
            K[0, 2] *= scale_w
            K[1, 2] *= scale_h

    if normalize_color and (target_height is None or target_width is None or rgb.dtype == torch.uint8):
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

    return rgb, depth_m, K
