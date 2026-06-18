"""GPU correspondence finding for RGB-D SLAM."""

from __future__ import annotations

import torch


@torch.no_grad()
def find_rgbd_correspondences_gpu(
    rgb_ref: torch.Tensor,
    rgb_live: torch.Tensor,
    depth_ref: torch.Tensor,
    depth_live: torch.Tensor,
    K: torch.Tensor,
    T_ref_live_init: torch.Tensor | None = None,
    max_corr: int = 500,
    photo_thresh: float = 0.1,
    depth_thresh: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    H, W = rgb_ref.shape[:2]
    device = rgb_ref.device
    dtype = rgb_ref.dtype

    if T_ref_live_init is None:
        T_ref_live_init = torch.eye(4, device=device, dtype=dtype)

    step = max(1, int((H * W / max_corr) ** 0.5))
    u_ref = torch.arange(0, W, step, device=device)
    v_ref = torch.arange(0, H, step, device=device)
    vv, uu = torch.meshgrid(v_ref, u_ref, indexing="ij")

    z_ref = depth_ref[vv, uu]
    valid = z_ref > 0
    uu, vv, z_ref = uu[valid], vv[valid], z_ref[valid]

    if len(z_ref) < 20:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X_ref = torch.stack([
        (uu.float() - cx) * z_ref / fx,
        (vv.float() - cy) * z_ref / fy,
        z_ref
    ], dim=-1)

    rgb_ref_sampled = rgb_ref[vv, uu]

    X_ref_h = torch.cat([X_ref, torch.ones(len(X_ref), 1, device=device, dtype=dtype)], dim=-1)
    X_ref_live = (T_ref_live_init @ X_ref_h.T).T[:, :3]

    u_proj = (X_ref_live[:, 0] * fx / X_ref_live[:, 2] + cx).long()
    v_proj = (X_ref_live[:, 1] * fy / X_ref_live[:, 2] + cy).long()

    valid_proj = (
        (u_proj >= 0) & (u_proj < W) &
        (v_proj >= 0) & (v_proj < H)
    )
    u_proj = u_proj[valid_proj]
    v_proj = v_proj[valid_proj]
    X_ref = X_ref[valid_proj]
    X_ref_live = X_ref_live[valid_proj]
    rgb_ref_sampled = rgb_ref_sampled[valid_proj]

    if len(X_ref) < 20:
        return None

    z_live = depth_live[v_proj, u_proj]
    rgb_live_sampled = rgb_live[v_proj, u_proj]

    valid_live = z_live > 0
    X_ref = X_ref[valid_live]
    X_ref_live = X_ref_live[valid_live]
    z_live = z_live[valid_live]
    rgb_ref_sampled = rgb_ref_sampled[valid_live]
    rgb_live_sampled = rgb_live_sampled[valid_live]
    u_proj = u_proj[valid_live]
    v_proj = v_proj[valid_live]

    if len(X_ref) < 20:
        return None

    X_live = torch.stack([
        (u_proj.float() - cx) * z_live / fx,
        (v_proj.float() - cy) * z_live / fy,
        z_live
    ], dim=-1)

    photo_diff = (rgb_ref_sampled - rgb_live_sampled).norm(dim=-1)
    photo_valid = photo_diff < photo_thresh

    geom_diff = (X_ref_live - X_live).norm(dim=-1)
    geom_valid = geom_diff < depth_thresh

    valid_corr = photo_valid & geom_valid
    X_ref = X_ref[valid_corr]
    X_live = X_live[valid_corr]

    if len(X_ref) < 20:
        return None

    if len(X_ref) > max_corr:
        indices = torch.randperm(len(X_ref), device=device)[:max_corr]
        X_ref = X_ref[indices]
        X_live = X_live[indices]

    photo_conf = 1.0 - (photo_diff[valid_corr] / photo_thresh).clamp(0, 1)
    geom_conf = 1.0 - (geom_diff[valid_corr] / depth_thresh).clamp(0, 1)
    weights = (photo_conf + geom_conf) / 2.0

    return X_ref, X_live, weights
