"""Truncated Signed Distance Function (TSDF) fusion for 3D reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TSDFConfig:
    """TSDF volume configuration.

    Attributes:
        voxel_size: Size of each voxel in meters.
        truncation_margin_voxels: Truncation margin in voxel counts.
        fuse_color: Whether to fuse color alongside geometry.
        dtype: Data type for TSDF values.
    """

    voxel_size: float = 0.02
    truncation_margin_voxels: int = 3
    fuse_color: bool = False
    dtype: torch.dtype = torch.float32


class TSDFVolume(torch.nn.Module):
    """TSDF volume for incremental dense reconstruction.

    Stores signed distance and weight for each voxel, updated via frame-wise fusion.
    """

    def __init__(
        self,
        voxel_dim: torch.Tensor,
        origin: torch.Tensor,
        config: TSDFConfig = None,
        device: torch.device | None = None,
    ):
        """Initialize TSDF volume.

        Args:
            voxel_dim: [3] shape of voxel grid (nx, ny, nz).
            origin: [3] world-space origin of volume.
            config: TSDFConfig instance.
            device: Target device.
        """
        super().__init__()
        self.config = config or TSDFConfig()
        self.device = device or torch.device("cpu")

        voxel_dim = voxel_dim.to(device=self.device, dtype=torch.int64)
        origin = origin.to(device=self.device, dtype=self.config.dtype)

        self.register_buffer("_voxel_dim", voxel_dim)
        self.register_buffer("_origin", origin)
        self.register_buffer(
            "tsdf",
            torch.ones(
                voxel_dim[0], voxel_dim[1], voxel_dim[2],
                device=self.device,
                dtype=self.config.dtype,
            ),
        )
        self.register_buffer(
            "weight",
            torch.zeros(
                voxel_dim[0], voxel_dim[1], voxel_dim[2],
                device=self.device,
                dtype=torch.float32,
            ),
        )

    @torch.no_grad()
    def integrate(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        T_world_camera: torch.Tensor,
        obs_weight: float = 1.0,
        color: torch.Tensor | None = None,
    ) -> None:
        """Integrate a depth frame into the TSDF volume.

        Args:
            depth: Depth image [H, W] in meters.
            intrinsics: Camera intrinsics [3, 3] (K matrix).
            T_world_camera: Camera pose in world frame [4, 4].
            obs_weight: Weight of this observation. Default: 1.0.
            color: Optional color image [H, W, 3] in [0, 1] or [0, 255].
        """
        H, W = depth.shape
        device = depth.device
        dtype = self.config.dtype

        # Inverse camera transform (world to camera)
        T_camera_world = torch.linalg.inv(T_world_camera)
        R = T_camera_world[:3, :3]
        t = T_camera_world[:3, 3]

        # Voxel centers in world coordinates
        voxel_dim_float = self._voxel_dim.float()
        voxel_coords = self._voxel_coordinates(device, dtype)  # [N, 3]
        voxel_world = voxel_coords * self.config.voxel_size + self._origin

        # Transform to camera frame
        voxel_camera = torch.matmul(voxel_world, R.t()) + t  # [N, 3]

        # Project to image
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        u = (voxel_camera[:, 0] * fx / voxel_camera[:, 2] + cx).long()
        v = (voxel_camera[:, 1] * fy / voxel_camera[:, 2] + cy).long()

        # Frustum check
        valid = (
            (voxel_camera[:, 2] > 0)
            & (u >= 0)
            & (u < W)
            & (v >= 0)
            & (v < H)
        )

        if valid.sum() == 0:
            return

        # Valid voxel positions
        voxel_coords_valid = voxel_coords[valid]
        u_valid = u[valid]
        v_valid = v[valid]
        voxel_camera_valid = voxel_camera[valid]

        # Sample depth at projected pixel positions
        depth_sampled = depth[v_valid, u_valid]

        # Keep only voxels with valid (non-zero) depth readings
        depth_mask = depth_sampled > 0
        if depth_mask.sum() == 0:
            return

        voxel_coords_valid = voxel_coords_valid[depth_mask]
        voxel_camera_valid = voxel_camera_valid[depth_mask]
        depth_sampled = depth_sampled[depth_mask]

        # Signed distance and truncation
        signed_dist = depth_sampled - voxel_camera_valid[:, 2]
        trunc = self.config.truncation_margin_voxels * self.config.voxel_size
        truncated_dist = torch.clamp(signed_dist, -trunc, trunc)

        # Linearize 3D voxel coordinates to flat indices (row-major / C order)
        nx, ny, nz = self._voxel_dim[0], self._voxel_dim[1], self._voxel_dim[2]
        xi, yi, zi = voxel_coords_valid[:, 0].long(), voxel_coords_valid[:, 1].long(), voxel_coords_valid[:, 2].long()
        flat_idx = xi * (ny * nz) + yi * nz + zi  # [M]

        # Vectorized running-average TSDF update
        tsdf_flat = self.tsdf.reshape(-1)
        weight_flat = self.weight.reshape(-1)

        old_w = weight_flat[flat_idx]
        old_tsdf = tsdf_flat[flat_idx]
        new_w = old_w + obs_weight
        new_tsdf = (old_tsdf * old_w + truncated_dist * obs_weight) / new_w

        tsdf_flat.scatter_(0, flat_idx, new_tsdf)
        weight_flat.scatter_(0, flat_idx, new_w)

    def _voxel_coordinates(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Generate all voxel coordinates.

        Returns:
            [N, 3] voxel coordinates where N = nx*ny*nz.
        """
        nx, ny, nz = self._voxel_dim.cpu().numpy()
        x = torch.arange(nx, device=device, dtype=dtype)
        y = torch.arange(ny, device=device, dtype=dtype)
        z = torch.arange(nz, device=device, dtype=dtype)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
        return coords.reshape(-1, 3)
