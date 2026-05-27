"""TSDF raycasting for frame rendering."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
    n_samples: int = 256,
) -> RenderedFrame:
    """Raycast a TSDF volume to render depth and normals.

    Args:
        tsdf_volume: TSDF tensor [nx, ny, nz].
        tsdf_origin: Origin of TSDF volume [3].
        voxel_size: Voxel size in meters.
        T_world_camera: Camera pose [4, 4].
        intrinsics: Camera intrinsics [3, 3].
        height: Output image height.
        width: Output image width.
        near: Near plane distance.
        far: Far plane distance.
        n_samples: Number of samples along each ray.

    Returns:
        RenderedFrame with depth, normals, and mask.
    """
    device = tsdf_volume.device
    dtype = tsdf_volume.dtype

    # Inverse camera transform
    T_camera_world = torch.linalg.inv(T_world_camera)
    R = T_camera_world[:3, :3]
    t = T_camera_world[:3, 3]

    # Pixel to ray direction in camera frame
    u = torch.arange(width, device=device, dtype=dtype)
    v = torch.arange(height, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing="xy")

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Ray directions in camera frame
    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy
    ray_z = torch.ones_like(u)

    ray_length = torch.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
    ray_x /= ray_length
    ray_y /= ray_length
    ray_z /= ray_length

    # Initialize output
    depth_map = torch.zeros(height, width, device=device, dtype=dtype)
    normal_map = torch.zeros(height, width, 3, device=device, dtype=dtype)
    mask = torch.zeros(height, width, dtype=torch.bool, device=device)

    # Sample along rays
    for h_idx in range(height):
        for w_idx in range(width):
            # Ray from camera center in world frame
            ray_dir_cam = torch.tensor(
                [ray_x[h_idx, w_idx], ray_y[h_idx, w_idx], ray_z[h_idx, w_idx]],
                device=device,
                dtype=dtype,
            )
            ray_dir_world = torch.matmul(R.t(), ray_dir_cam)
            camera_pos = -torch.matmul(R.t(), t)

            # Find zero crossing along ray
            best_depth = None
            for t_val in torch.linspace(near, far, n_samples, device=device):
                pos_world = camera_pos + t_val * ray_dir_world
                tsdf_val = _sample_tsdf(
                    tsdf_volume, pos_world, tsdf_origin, voxel_size
                )
                if tsdf_val is not None and tsdf_val < 0:
                    best_depth = t_val
                    break

            if best_depth is not None:
                depth_map[h_idx, w_idx] = best_depth
                mask[h_idx, w_idx] = True

                # Compute normal via finite differences
                pos_world = camera_pos + best_depth * ray_dir_world
                eps = voxel_size
                tsdf_center = _sample_tsdf(
                    tsdf_volume, pos_world, tsdf_origin, voxel_size
                )
                tsdf_x = _sample_tsdf(
                    tsdf_volume,
                    pos_world + torch.tensor([eps, 0, 0], device=device, dtype=dtype),
                    tsdf_origin,
                    voxel_size,
                )
                tsdf_y = _sample_tsdf(
                    tsdf_volume,
                    pos_world + torch.tensor([0, eps, 0], device=device, dtype=dtype),
                    tsdf_origin,
                    voxel_size,
                )
                tsdf_z = _sample_tsdf(
                    tsdf_volume,
                    pos_world + torch.tensor([0, 0, eps], device=device, dtype=dtype),
                    tsdf_origin,
                    voxel_size,
                )

                if tsdf_x is not None and tsdf_y is not None and tsdf_z is not None:
                    grad = torch.tensor(
                        [(tsdf_x - tsdf_center) / eps,
                         (tsdf_y - tsdf_center) / eps,
                         (tsdf_z - tsdf_center) / eps],
                        device=device,
                        dtype=dtype,
                    )
                    norm = torch.norm(grad)
                    if norm > 1e-6:
                        normal_map[h_idx, w_idx] = grad / norm

    return RenderedFrame(depth=depth_map, normal=normal_map, mask=mask)


def _sample_tsdf(
    tsdf_volume: torch.Tensor,
    pos_world: torch.Tensor,
    tsdf_origin: torch.Tensor,
    voxel_size: float,
) -> float | None:
    """Sample TSDF at world position via trilinear interpolation.

    Args:
        tsdf_volume: TSDF tensor [nx, ny, nz].
        pos_world: World position [3].
        tsdf_origin: Volume origin [3].
        voxel_size: Voxel size.

    Returns:
        TSDF value or None if out of bounds.
    """
    # World to voxel coordinates
    pos_voxel = (pos_world - tsdf_origin) / voxel_size
    nx, ny, nz = tsdf_volume.shape

    # Trilinear interpolation
    x, y, z = pos_voxel[0], pos_voxel[1], pos_voxel[2]
    x_i, y_i, z_i = int(x), int(y), int(z)

    # Bounds check
    if x_i < 0 or x_i >= nx - 1 or y_i < 0 or y_i >= ny - 1 or z_i < 0 or z_i >= nz - 1:
        return None

    # Interpolation weights
    x_w, y_w, z_w = x - x_i, y - y_i, z - z_i

    # Eight corner values
    v000 = tsdf_volume[x_i, y_i, z_i].item()
    v100 = tsdf_volume[x_i + 1, y_i, z_i].item()
    v010 = tsdf_volume[x_i, y_i + 1, z_i].item()
    v110 = tsdf_volume[x_i + 1, y_i + 1, z_i].item()
    v001 = tsdf_volume[x_i, y_i, z_i + 1].item()
    v101 = tsdf_volume[x_i + 1, y_i, z_i + 1].item()
    v011 = tsdf_volume[x_i, y_i + 1, z_i + 1].item()
    v111 = tsdf_volume[x_i + 1, y_i + 1, z_i + 1].item()

    # Trilinear interpolation
    v00 = v000 * (1 - x_w) + v100 * x_w
    v10 = v010 * (1 - x_w) + v110 * x_w
    v01 = v001 * (1 - x_w) + v101 * x_w
    v11 = v011 * (1 - x_w) + v111 * x_w

    v0 = v00 * (1 - y_w) + v10 * y_w
    v1 = v01 * (1 - y_w) + v11 * y_w

    v = v0 * (1 - z_w) + v1 * z_w

    return v
