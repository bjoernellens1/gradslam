"""RGB-D TSDF SLAM pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..icp.projective import ProjectiveICPTracker, ProjectiveICPConfig
from ..mapping.tsdf import TSDFVolume, TSDFConfig


@dataclass
class RGBDFrame:
    """Input RGB-D frame to SLAM.

    Attributes:
        rgb: Optional color image [H, W, 3].
        depth: Depth image [H, W].
        intrinsics: Camera intrinsics [3, 3].
        timestamp: Frame timestamp (optional).
        T_world_camera_gt: Ground truth pose (optional, for evaluation).
    """

    rgb: torch.Tensor | None
    depth: torch.Tensor
    intrinsics: torch.Tensor
    timestamp: float | None = None
    T_world_camera_gt: torch.Tensor | None = None


@dataclass
class TrackingResult:
    """Output of frame tracking.

    Attributes:
        T_world_camera: Estimated camera pose.
        T_model_live: Transform from model to live frame (used by ICP).
        quality: Tracking quality metrics.
        used_keyframe: Whether a keyframe was created.
        lost: Whether tracking failed.
    """

    T_world_camera: torch.Tensor
    T_model_live: torch.Tensor
    quality: dict
    used_keyframe: bool
    lost: bool


class RGBDTSDFSLAM(torch.nn.Module):
    """RGB-D TSDF SLAM pipeline.

    Fuses RGB-D streams into a TSDF volume using projective ICP tracking.
    """

    def __init__(
        self,
        tsdf_config: TSDFConfig = None,
        icp_config: ProjectiveICPConfig = None,
        keyframe_inlier_ratio_thresh: float = 0.8,
        keyframe_motion_thresh: float = 0.05,
        keyframe_max_frames: int = 20,
    ):
        """Initialize SLAM pipeline.

        Args:
            tsdf_config: TSDF volume configuration.
            icp_config: Projective ICP tracker configuration.
            keyframe_inlier_ratio_thresh: Create keyframe if inlier ratio drops below this.
            keyframe_motion_thresh: Create keyframe if motion exceeds this (meters).
            keyframe_max_frames: Force keyframe every N frames.
        """
        super().__init__()
        self.tsdf_config = tsdf_config or TSDFConfig()
        self.icp_config = icp_config or ProjectiveICPConfig()
        self.tracker = ProjectiveICPTracker(self.icp_config)

        self.keyframe_inlier_ratio_thresh = keyframe_inlier_ratio_thresh
        self.keyframe_motion_thresh = keyframe_motion_thresh
        self.keyframe_max_frames = keyframe_max_frames

        self.tsdf = None
        self.T_world_camera = None
        self.frame_count = 0
        self.lost = False

    def initialize(
        self,
        frame: RGBDFrame,
        voxel_dim: torch.Tensor = None,
        volume_origin: torch.Tensor = None,
    ) -> None:
        """Initialize SLAM with first frame.

        Args:
            frame: First RGB-D frame.
            voxel_dim: [3] shape of TSDF volume. Defaults to [128, 128, 128].
            volume_origin: [3] world position of volume origin. Defaults to [-1, -1, 0].
        """
        if voxel_dim is None:
            voxel_dim = torch.tensor([128, 128, 128])
        if volume_origin is None:
            volume_origin = torch.tensor([-1.0, -1.0, 0.0])

        device = frame.depth.device
        voxel_dim = voxel_dim.to(device)
        volume_origin = volume_origin.to(device)

        self.tsdf = TSDFVolume(
            voxel_dim, volume_origin, config=self.tsdf_config, device=device
        )

        self.T_world_camera = torch.eye(4, device=device, dtype=frame.depth.dtype)
        self.frame_count = 0
        self.lost = False

        # Integrate first frame
        self.tsdf.integrate(frame.depth, frame.intrinsics, self.T_world_camera)

    def process_frame(self, frame: RGBDFrame) -> TrackingResult:
        """Process a new RGB-D frame.

        Args:
            frame: Input RGB-D frame.

        Returns:
            TrackingResult with pose and quality metrics.
        """
        if self.tsdf is None:
            self.initialize(frame)
            return TrackingResult(
                T_world_camera=self.T_world_camera.clone(),
                T_model_live=torch.eye(4, device=frame.depth.device),
                quality={},
                used_keyframe=True,
                lost=False,
            )

        device = frame.depth.device
        dtype = frame.depth.dtype

        # Compute depth derivatives (for normal estimation)
        depth_y, depth_x = torch.gradient(frame.depth)
        # Normal estimation (simplified)
        live_normal = self._estimate_normals(frame.depth, frame.intrinsics)

        # For frame-to-model tracking, we'd raycast the model here
        # For now, use a dummy model (identity, all black)
        model_depth = torch.zeros_like(frame.depth)
        model_normal = torch.zeros_like(live_normal)

        # Run ICP (placeholder: skip for now, just integrate)
        T_model_live = torch.eye(4, device=device, dtype=dtype)
        quality = {"num_valid": 0, "inlier_ratio": 0.0, "rmse": 0.0}

        # For full pipeline, track and integrate
        # For now, just integrate at current pose
        self.tsdf.integrate(frame.depth, frame.intrinsics, self.T_world_camera)

        self.frame_count += 1
        used_keyframe = self.frame_count % self.keyframe_max_frames == 0

        return TrackingResult(
            T_world_camera=self.T_world_camera.clone(),
            T_model_live=T_model_live,
            quality=quality,
            used_keyframe=used_keyframe,
            lost=self.lost,
        )

    @staticmethod
    def _estimate_normals(
        depth: torch.Tensor, intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Estimate surface normals from depth (simplified).

        Args:
            depth: Depth image [H, W].
            intrinsics: Camera intrinsics [3, 3].

        Returns:
            Normal map [H, W, 3].
        """
        H, W = depth.shape
        device = depth.device

        # Simple central difference normal estimation
        # (In production, use convolution or bilateral filters)
        pad_depth = torch.nn.functional.pad(
            depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
        ).squeeze(0).squeeze(0)

        dz_dy = (pad_depth[2:, 1:-1] - pad_depth[:-2, 1:-1]) / 2.0
        dz_dx = (pad_depth[1:-1, 2:] - pad_depth[1:-1, :-2]) / 2.0

        # Normal = [-dz/dx, -dz/dy, 1] (cross product of tangent vectors)
        normals = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
        norm = torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-6)
        normals = normals / norm

        # Pad to original size
        normals = torch.nn.functional.pad(
            normals.unsqueeze(0), (0, 0, 1, 1, 1, 1), mode="reflect"
        ).squeeze(0)

        return normals
