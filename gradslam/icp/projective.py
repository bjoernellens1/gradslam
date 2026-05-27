"""Projective ICP tracker for RGB-D SLAM (KinectFusion-style)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry.se3utils import se3_exp
from .residuals import point_to_plane_projective
from .solvers import solve_lm_6x6


@dataclass
class ProjectiveICPConfig:
    """Configuration for projective ICP tracker.

    Attributes:
        n_pyramid_levels: Number of pyramid levels (coarse to fine).
        iterations: Iterations per level, e.g., (10, 5, 4).
        damping: Damping per level, e.g., (1e-2, 1e-3, 1e-4).
        max_depth_diff: Max depth difference for correspondence.
        max_normal_angle_deg: Max angle between normals (degrees).
        robust_loss: Robust loss function ("huber" or "none").
        huber_delta: Huber loss threshold (for robust_loss="huber").
    """

    n_pyramid_levels: int = 3
    iterations: tuple[int, ...] = (10, 5, 4)
    damping: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
    max_depth_diff: float = 0.10
    max_normal_angle_deg: float = 60.0
    robust_loss: str = "none"
    huber_delta: float = 0.03


class ProjectiveICPTracker(torch.nn.Module):
    """Projective ICP tracker for frame-to-model alignment in RGB-D SLAM.

    Uses a coarse-to-fine pyramid approach, computing point-to-plane ICP at each
    level to align live depth with rendered model depth.
    """

    def __init__(self, config: ProjectiveICPConfig = None):
        """Initialize tracker with config.

        Args:
            config: ProjectiveICPConfig instance. Defaults to ProjectiveICPConfig().
        """
        super().__init__()
        self.config = config or ProjectiveICPConfig()

        # Validate config
        assert len(self.config.iterations) == self.config.n_pyramid_levels
        assert len(self.config.damping) == self.config.n_pyramid_levels

    @torch.no_grad()
    def forward(
        self,
        live_depth: torch.Tensor,
        live_normal: torch.Tensor,
        model_depth: torch.Tensor,
        model_normal: torch.Tensor,
        intrinsics: torch.Tensor,
        init_T_model_live: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute rigid transform from live frame to model.

        Args:
            live_depth: Live frame depth, shape [H, W].
            live_normal: Live frame normals, shape [H, W, 3].
            model_depth: Model rendered depth, shape [H, W].
            model_normal: Model rendered normals, shape [H, W, 3].
            intrinsics: Camera intrinsics [3, 3] (K matrix).
            init_T_model_live: Initial guess (4x4 SE(3)). If None, uses identity.

        Returns:
            Tuple of:
            - T_model_live: Estimated transform (4x4 SE(3)).
            - quality_dict: Tracking quality metrics.

        Shapes:
            - live_depth, model_depth: [H, W]
            - live_normal, model_normal: [H, W, 3]
            - intrinsics: [3, 3]
            - init_T_model_live: [4, 4] or None
            - T_model_live: [4, 4]
        """
        device = live_depth.device
        dtype = live_depth.dtype

        # Initialize transform
        if init_T_model_live is None:
            T_model_live = torch.eye(4, device=device, dtype=dtype)
        else:
            T_model_live = init_T_model_live.clone().to(device=device, dtype=dtype)

        # Build depth/normal pyramids (coarsest first, index 0 = most downsampled)
        live_depths = self._build_pyramid(live_depth)
        live_normals = self._build_pyramid(live_normal, is_normal=True)
        model_depths = self._build_pyramid(model_depth)
        model_normals = self._build_pyramid(model_normal, is_normal=True)

        # Coarse-to-fine ICP: iterate from coarsest (index 0) to finest (index n-1)
        quality_metrics = {
            "num_valid": 0,
            "inlier_ratio": 0.0,
            "rmse": 0.0,
            "mean_abs_residual": 0.0,
            "update_norm": 0.0,
            "converged": False,
        }

        for level in range(self.config.n_pyramid_levels):
            # pyramid[0] = coarsest (most downsampled), pyramid[-1] = finest
            live_d = live_depths[level]   # [H_l, W_l]
            live_n = live_normals[level]  # [H_l, W_l, 3]
            model_d = model_depths[level]
            model_n = model_normals[level]
            # scale factor: coarsest level has highest downsampling
            scale = 2 ** (self.config.n_pyramid_levels - 1 - level)
            K = intrinsics / scale  # Scaled intrinsics for this level

            # Run ICP iterations at this level
            for it in range(self.config.iterations[level]):
                # Transform live vertices using current estimate
                live_vertex = self._depth_to_vertex(live_d, K, device, dtype)  # [H_l, W_l, 3]
                live_vertex_model = self._transform_points(live_vertex, T_model_live)

                # Find correspondences via projective lookup
                model_vertex = self._depth_to_vertex(model_d, K, device, dtype)
                assoc_vertex, assoc_normal, valid = self._find_correspondences(
                    live_vertex_model,
                    live_n,
                    model_vertex,
                    model_n,
                    model_d,
                    K=K,
                )

                if valid.sum() < 10:
                    break  # Too few correspondences

                # Extract valid points (valid is [H*W], flatten spatial dims first)
                H_l, W_l = live_vertex_model.shape[:2]
                live_v = live_vertex_model.reshape(-1, 3)[valid]      # [N, 3]
                model_v = assoc_vertex.reshape(-1, 3)[valid]          # [N, 3]
                model_n_valid = assoc_normal.reshape(-1, 3)[valid]    # [N, 3]

                # Compute residuals and Jacobian
                A, b = point_to_plane_projective(live_v, model_n_valid, model_v)

                # Solve for update: normal eqs are (A^T A) δξ = -A^T r
                damp = self.config.damping[level]
                xi = solve_lm_6x6(A, -b, damp=damp)  # [6, 1]

                # Compute quality metrics
                residual = b[:, 0]  # [N]
                rmse = torch.sqrt(torch.mean(residual**2)).item()
                quality_metrics = {
                    "num_valid": valid.sum().item(),
                    "inlier_ratio": (valid.sum().item() / valid.numel()),
                    "rmse": rmse,
                    "mean_abs_residual": torch.mean(torch.abs(residual)).item(),
                    "update_norm": torch.norm(xi).item(),
                    "converged": torch.norm(xi).item() < 1e-6,
                }

                # Apply update (left-multiply SE(3) exponential)
                dT = se3_exp(xi.squeeze())  # [4, 4]
                T_model_live = dT @ T_model_live

        return T_model_live, quality_metrics

    def _build_pyramid(
        self, tensor: torch.Tensor, is_normal: bool = False
    ) -> list[torch.Tensor]:
        """Build Gaussian pyramid coarsest-first.

        Args:
            tensor: Full-resolution input (depth [H,W] or normal [H,W,3]).
            is_normal: If True, renormalize after averaging.

        Returns:
            List of n_pyramid_levels tensors, index 0 = most downsampled (coarsest),
            index -1 = original resolution (finest).
        """
        levels = [tensor]
        for _ in range(self.config.n_pyramid_levels - 1):
            if is_normal:
                downed = torch.nn.functional.avg_pool2d(
                    tensor.unsqueeze(0).permute(0, 3, 1, 2), kernel_size=2, stride=2
                )
                downed = downed.permute(0, 2, 3, 1).squeeze(0)
                downed = torch.nn.functional.normalize(downed, dim=-1)
            else:
                downed = torch.nn.functional.avg_pool2d(
                    tensor.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2
                ).squeeze(0).squeeze(0)
            levels.append(downed)
            tensor = downed

        # Reverse so index 0 = coarsest (most downsampled)
        return list(reversed(levels))

    @staticmethod
    def _depth_to_vertex(
        depth: torch.Tensor, K: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert depth map to vertex map (camera coordinates).

        Args:
            depth: Depth image [H, W].
            K: Camera intrinsics [3, 3].
            device: Target device.
            dtype: Target dtype.

        Returns:
            Vertex map [H, W, 3].
        """
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Pixel coordinates
        u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device),
                              indexing="xy")
        u, v = u.float(), v.float()

        # Back-project
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        vertex = torch.stack([x, y, z], dim=-1).to(dtype=dtype)
        return vertex

    @staticmethod
    def _transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Transform points via rigid transform.

        Args:
            points: Points [H, W, 3] or [N, 3].
            T: Transform [4, 4].

        Returns:
            Transformed points, same shape.
        """
        was_batched = points.dim() == 3
        if was_batched:
            H, W, _ = points.shape
            points = points.reshape(-1, 3)

        # Apply transform
        points_h = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
        points_t = torch.matmul(T, points_h.t()).t()[:, :3]

        if was_batched:
            points_t = points_t.reshape(H, W, 3)

        return points_t

    def _find_correspondences(
        self,
        live_vertex: torch.Tensor,
        live_normal: torch.Tensor,
        model_vertex: torch.Tensor,
        model_normal: torch.Tensor,
        model_depth: torch.Tensor,
        K: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find correspondences via projective lookup and validity checks.

        Projects the (already-transformed) live vertices into image space to
        look up model vertices at their projected pixel positions.

        Args:
            live_vertex: Live vertices in model frame [H, W, 3].
            live_normal: Live normals (in live frame) [H, W, 3].
            model_vertex: Model vertices [H, W, 3].
            model_normal: Model normals [H, W, 3].
            model_depth: Model depth [H, W].
            K: Camera intrinsics [3, 3] (used for projective lookup when provided).

        Returns:
            Tuple of:
            - assoc_vertex: Associated model vertices [H, W, 3] (same layout as live).
            - assoc_normal: Associated model normals [H, W, 3].
            - valid: Validity mask [H*W].
        """
        H, W = live_vertex.shape[:2]
        device = live_vertex.device

        if K is not None:
            # Projective correspondence: project live vertices in model frame to pixels
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            z = live_vertex[:, :, 2].clamp(min=1e-6)  # [H, W]
            u = (live_vertex[:, :, 0] * fx / z + cx).long()  # [H, W]
            v = (live_vertex[:, :, 1] * fy / z + cy).long()  # [H, W]

            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u_safe = u.clamp(0, W - 1)
            v_safe = v.clamp(0, H - 1)

            # Look up model at projected pixel (u, v)
            assoc_vertex = model_vertex[v_safe, u_safe]    # [H, W, 3]
            assoc_normal = model_normal[v_safe, u_safe]    # [H, W, 3]
            assoc_depth = model_depth[v_safe, u_safe]      # [H, W]
        else:
            # Fallback: same-pixel correspondence (valid only for near-identity T)
            in_bounds = torch.ones(H, W, dtype=torch.bool, device=device)
            assoc_vertex = model_vertex
            assoc_normal = model_normal
            assoc_depth = model_depth

        # Validity: live depth > 0, model depth > 0, projected in bounds, depth diff, angle
        valid_live = live_vertex[:, :, 2] > 0
        valid_model = (assoc_depth > 0) & in_bounds
        valid_angle = self._check_normal_angle(live_normal, assoc_normal)
        valid_depth = torch.abs(live_vertex[:, :, 2] - assoc_depth) < self.config.max_depth_diff

        valid = (valid_live & valid_model & valid_angle & valid_depth).reshape(-1)

        return assoc_vertex, assoc_normal, valid

    @staticmethod
    def _check_normal_angle(n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
        """Check if normal angle is below threshold.

        Args:
            n1: Normals [H, W, 3].
            n2: Normals [H, W, 3].

        Returns:
            Mask [H, W] indicating acceptable angles.
        """
        dot = torch.sum(n1 * n2, dim=-1)  # [H, W]
        dot = torch.clamp(dot, -1.0, 1.0)
        angle = torch.acos(dot)  # radians
        angle_deg = angle * 180.0 / 3.14159265359
        return angle_deg < 60.0
