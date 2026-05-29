"""TSDF raycasting for frame rendering — fully vectorized GPU implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RenderedFrame:
    """Output from TSDF raycasting.

    Attributes:
        depth: Rendered depth image [H, W].
        normal: Rendered surface normals [H, W, 3].
        mask: Valid pixel mask [H, W].
    """

    depth: torch.Tensor
    normal: torch.Tensor
    mask: torch.Tensor


@torch.no_grad()
def raycast_tsdf(
    tsdf_volume: torch.Tensor,
    tsdf_origin: torch.Tensor,
    voxel_size: float,
    T_world_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    near: float = 0.1,
    far: float = 5.0,
    n_samples: int = 128,
    normal_mode: str = "gradient",
    precomputed_ray_cam_unit: torch.Tensor | None = None,
) -> RenderedFrame:
    """Raycast a TSDF volume to render depth and normals.

    Fully vectorized: all H*W rays processed simultaneously on GPU.

    Args:
        tsdf_volume: TSDF tensor [nx, ny, nz].
        tsdf_origin: Origin of TSDF volume [3].
        voxel_size: Voxel size in meters.
        T_world_camera: Camera pose [4, 4] (camera-to-world).
        intrinsics: Camera intrinsics [3, 3] or [4, 4].
        height: Output image height.
        width: Output image width.
        near: Near plane distance in meters.
        far: Far plane distance in meters.
        n_samples: Samples along each ray.
        normal_mode: How to compute surface normals. ``"gradient"`` (default)
            computes normals via 6 TSDF finite-difference ``grid_sample`` calls.
            ``"image"`` skips normal computation entirely and returns a
            zero-filled normal map; the caller is responsible for deriving
            normals from the depth image instead (faster).
        precomputed_ray_cam_unit: Optional pre-cached ray directions in camera
            frame [H*W, 3], already unit-normalised. When provided, the per-
            pixel ray grid is not recomputed from scratch (only depends on
            intrinsics, not on pose), saving the meshgrid and normalise step.
            When ``None`` (default), ray directions are computed from
            ``intrinsics``, ``height``, and ``width`` as usual.

    Returns:
        RenderedFrame with depth, normals, and mask.
    """
    device = tsdf_volume.device
    dtype = tsdf_volume.dtype

    nx, ny, nz = tsdf_volume.shape

    # Camera rotation and position in world frame
    R_cw = T_world_camera[:3, :3]  # camera axes in world [3,3]
    t_w = T_world_camera[:3, 3]    # camera origin in world [3]

    fx, fy = intrinsics[0, 0].to(dtype), intrinsics[1, 1].to(dtype)
    cx, cy = intrinsics[0, 2].to(dtype), intrinsics[1, 2].to(dtype)

    # Ray directions in camera frame → world frame [H*W, 3]
    if precomputed_ray_cam_unit is not None:
        # Use caller-supplied unit ray directions (camera frame), skipping the
        # meshgrid construction.  The tensor is already normalised in camera
        # space; we still re-normalise after rotation for numerical safety.
        ray_cam_flat = precomputed_ray_cam_unit  # [N, 3], already unit-normalized in camera frame
        ray_world = ray_cam_flat @ R_cw.t()     # rotate to world frame [N, 3]
        ray_world = F.normalize(ray_world, dim=-1)  # re-normalize after rotation (numerical safety)
    else:
        u = torch.arange(width, device=device, dtype=dtype)
        v = torch.arange(height, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing="ij")   # [H, W] each
        ray_cam = torch.stack(
            [(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], dim=-1
        )  # [H, W, 3], unnormalized
        ray_cam_flat = ray_cam.reshape(-1, 3)            # [N, 3]
        ray_world = ray_cam_flat @ R_cw.t()             # [N, 3]
        ray_world = F.normalize(ray_world, dim=-1)       # [N, 3]

    # Sample depths along rays [n_samples]
    t_vals = torch.linspace(near, far, n_samples, device=device, dtype=dtype)

    # Sample points in world space [N, n_samples, 3]
    # t_w[None, None, :] + t_vals[None, :, None] * ray_world[:, None, :]
    pts_world = t_w[None, None, :] + t_vals[None, :, None] * ray_world[:, None, :]

    # Convert to voxel coordinates [N, n_samples, 3]
    pts_vox = (pts_world - tsdf_origin[None, None, :]) / voxel_size

    # Sample TSDF via trilinear grid_sample (expects input [1, 1, nz, ny, nx], grid [..., (x,y,z)])
    # grid_sample 3D convention: input [N,C,D,H,W], grid [N,D_out,H_out,W_out,3] as (x,y,z) in [-1,1]
    # Our tsdf is indexed [x, y, z] → we treat tsdf.permute(2,1,0) as [z, y, x] = [D, H, W]
    tsdf_for_gs = tsdf_volume.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)  # [1, 1, nz, ny, nx]
    vol_size = torch.tensor([nx - 1, ny - 1, nz - 1], device=device, dtype=dtype)

    # Normalize voxel coords to [-1, 1] as (x, y, z)
    pts_norm = 2.0 * pts_vox / vol_size[None, None, :] - 1.0  # [N, n_samples, 3] as (x, y, z)

    N = ray_world.shape[0]
    # Reshape for grid_sample: [1, N, n_samples, 1, 3]
    grid = pts_norm.unsqueeze(0).unsqueeze(3)  # [1, N, n_samples, 1, 3]

    # grid_sample 3D: output [1, 1, N, n_samples, 1]
    tsdf_samples = F.grid_sample(
        tsdf_for_gs, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )  # [1, 1, N, n_samples, 1]
    tsdf_samples = tsdf_samples.squeeze(0).squeeze(0).squeeze(-1)  # [N, n_samples]

    # Mark out-of-bounds samples as +1 (no surface)
    oob = (
        (pts_vox[..., 0] < 0) | (pts_vox[..., 0] >= nx)
        | (pts_vox[..., 1] < 0) | (pts_vox[..., 1] >= ny)
        | (pts_vox[..., 2] < 0) | (pts_vox[..., 2] >= nz)
    )
    tsdf_samples = tsdf_samples.masked_fill(oob, 1.0)

    # Find first negative TSDF (surface crossing): sign goes + to -)
    sign_curr = tsdf_samples[:, :-1]   # [N, n_samples-1]
    sign_next = tsdf_samples[:, 1:]    # [N, n_samples-1]
    crossing = (sign_curr > 0) & (sign_next <= 0)  # [N, n_samples-1]

    has_crossing = crossing.any(dim=1)              # [N]
    first_idx = crossing.float().argmax(dim=1)      # [N] — index of step before crossing

    # Linear interpolation to find exact depth
    t0 = t_vals[first_idx]           # [N]
    t1 = t_vals[(first_idx + 1).clamp(max=n_samples - 1)]  # [N]
    s0 = tsdf_samples[torch.arange(N, device=device), first_idx]          # [N]
    s1 = tsdf_samples[torch.arange(N, device=device), (first_idx + 1).clamp(max=n_samples - 1)]  # [N]
    denom = (s0 - s1).abs().clamp(min=1e-6)
    t_surface = t0 + (t1 - t0) * s0.abs() / denom  # linear interp

    # Convert ray-parameter (distance along unit ray) to z-depth (camera-frame z-coordinate).
    # z_depth = t * cos(θ) = t * ray_cam_unit_z, where θ is the angle from the optical axis.
    # This makes rendered depth compatible with the RGB-D sensor convention (z-depth).
    if precomputed_ray_cam_unit is not None:
        ray_cam_unit_z = precomputed_ray_cam_unit[:, 2]
    else:
        # ray_cam_flat is unnormalized [(u-cx)/fx, (v-cy)/fy, 1]; unit-z = 1/|ray_cam_flat|
        ray_cam_unit_z = 1.0 / ray_cam_flat.norm(dim=-1)
    depth_flat = torch.where(has_crossing, t_surface * ray_cam_unit_z, torch.zeros_like(t_surface))  # [N]

    # Normals
    if normal_mode == "gradient":
        # Normals: TSDF gradient at surface position (6 grid_sample calls).
        # Use t_surface (ray parameter) × unit ray direction — not depth_flat — for correct 3D position.
        surf_pts = t_w[None, :] + t_surface[:, None] * ray_world  # [N, 3]
        surf_vox = (surf_pts - tsdf_origin[None, :]) / voxel_size  # [N, 3]
        eps = 1.0  # 1 voxel step for gradient

        def sample_at(offsets):
            """Sample tsdf at surf_vox + offsets (offsets: [3])."""
            pts = surf_vox + offsets.to(device=device, dtype=dtype)[None, :]
            pts_n = (2.0 * pts / vol_size[None, :] - 1.0).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            out = F.grid_sample(tsdf_for_gs, pts_n, mode="bilinear", align_corners=True, padding_mode="border")
            return out.squeeze()  # [N]

        gx = sample_at(torch.tensor([eps, 0, 0])) - sample_at(torch.tensor([-eps, 0, 0]))
        gy = sample_at(torch.tensor([0, eps, 0])) - sample_at(torch.tensor([0, -eps, 0]))
        gz = sample_at(torch.tensor([0, 0, eps])) - sample_at(torch.tensor([0, 0, -eps]))

        grad = torch.stack([gx, gy, gz], dim=-1)  # [N, 3] in world space
        grad_cam = grad @ R_cw  # rotate to camera frame [N, 3]
        normal_flat = F.normalize(grad_cam, dim=-1)  # [N, 3]
        # Flip normals facing away from camera
        flip = (normal_flat[:, 2] > 0).float() * 2 - 1
        normal_flat = normal_flat * flip[:, None]
        normal_flat = torch.where(has_crossing[:, None].expand_as(normal_flat), normal_flat,
                                  torch.zeros_like(normal_flat))
    else:
        # normal_mode == "image": skip normals computation entirely.
        # The pipeline will derive normals from the depth map instead (faster).
        normal_flat = torch.zeros(N, 3, device=device, dtype=dtype)

    depth_map = depth_flat.reshape(height, width)
    normal_map = normal_flat.reshape(height, width, 3)
    mask = has_crossing.reshape(height, width)

    return RenderedFrame(depth=depth_map, normal=normal_map, mask=mask)
