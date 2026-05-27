"""RGB-D TSDF SLAM pipeline (KinectFusion-style frame-to-model tracking)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..icp.projective import ProjectiveICPConfig, ProjectiveICPTracker
from ..mapping.tsdf import TSDFConfig, TSDFVolume
from ..rendering.tsdf_raycast import RenderedFrame, raycast_tsdf


@dataclass
class RGBDFrame:
    """Input RGB-D frame to SLAM.

    Attributes:
        rgb: Optional color image [H, W, 3].
        depth: Depth image [H, W] in meters.
        intrinsics: Camera intrinsics [3, 3] or [4, 4].
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
        T_world_camera: Estimated camera pose [4, 4].
        quality: Tracking quality metrics dict.
        used_keyframe: Whether this frame triggered a keyframe.
        lost: Whether tracking failed (too few correspondences).
    """

    T_world_camera: torch.Tensor
    quality: dict
    used_keyframe: bool
    lost: bool


class RGBDTSDFSLAM(torch.nn.Module):
    """RGB-D TSDF SLAM pipeline (KinectFusion-style).

    Frame-to-model tracking via projective ICP against a TSDF volume.
    The model is rendered (raycasted) from the current pose estimate before
    each ICP step, enabling drift correction relative to the fused map.
    """

    def __init__(
        self,
        tsdf_config: TSDFConfig = None,
        icp_config: ProjectiveICPConfig = None,
        keyframe_inlier_ratio_thresh: float = 0.30,
        keyframe_motion_thresh: float = 0.10,
        keyframe_max_frames: int = 30,
        voxel_dim: tuple[int, int, int] = (256, 256, 256),
        volume_origin: tuple[float, float, float] = (-2.0, -2.0, 0.0),
        near: float = 0.1,
        far: float = 6.0,
    ):
        """Initialize SLAM pipeline.

        Args:
            tsdf_config: TSDF volume configuration.
            icp_config: Projective ICP tracker configuration.
            keyframe_inlier_ratio_thresh: Force keyframe below this inlier ratio.
            keyframe_motion_thresh: Force keyframe above this translation (meters).
            keyframe_max_frames: Force keyframe every N frames.
            voxel_dim: (nx, ny, nz) voxel grid dimensions.
            volume_origin: (x, y, z) world-space TSDF volume origin in meters.
            near: Raycast near plane (meters).
            far: Raycast far plane (meters).
        """
        super().__init__()
        self.tsdf_config = tsdf_config or TSDFConfig()
        self.icp_config = icp_config or ProjectiveICPConfig()
        self.tracker = ProjectiveICPTracker(self.icp_config)

        self.keyframe_inlier_ratio_thresh = keyframe_inlier_ratio_thresh
        self.keyframe_motion_thresh = keyframe_motion_thresh
        self.keyframe_max_frames = keyframe_max_frames

        self._voxel_dim = voxel_dim
        self._volume_origin = volume_origin
        self.near = near
        self.far = far

        self.tsdf: TSDFVolume | None = None
        self.T_world_camera: torch.Tensor | None = None
        self.frame_count: int = 0
        self.lost: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the SLAM system (clear map and pose)."""
        self.tsdf = None
        self.T_world_camera = None
        self.frame_count = 0
        self.lost = False

    @torch.no_grad()
    def process_frame(self, frame: RGBDFrame) -> TrackingResult:
        """Process a new RGB-D frame.

        On the first call, initializes the TSDF volume and sets pose to identity.
        On subsequent calls:
          1. Raycast current model → model depth + normals.
          2. Estimate live normals from live depth.
          3. Run projective ICP to align live to model.
          4. Update pose, integrate live depth into TSDF.

        Args:
            frame: Input RGB-D frame (depth in meters, intrinsics [3,3] or [4,4]).

        Returns:
            TrackingResult with updated pose and quality metrics.
        """
        depth = frame.depth
        # Accept both 3×3 and 4×4 intrinsics; normalize to 3×3 float
        K = frame.intrinsics[:3, :3].to(dtype=depth.dtype)
        device = depth.device

        if self.tsdf is None:
            return self._initialize(depth, K, device)

        # --- Estimate live normals ---
        live_normal = _estimate_normals(depth)  # [H, W, 3]

        H, W = depth.shape

        # --- Raycast model at current pose ---
        rendered = raycast_tsdf(
            tsdf_volume=self.tsdf.tsdf,
            tsdf_origin=self.tsdf._origin,
            voxel_size=self.tsdf_config.voxel_size,
            T_world_camera=self.T_world_camera,
            intrinsics=K,
            height=H,
            width=W,
            near=self.near,
            far=self.far,
            n_samples=128,
        )

        model_depth = rendered.depth      # [H, W]
        model_normal = rendered.normal    # [H, W, 3]

        # --- Run ICP ---
        T_model_live, quality = self.tracker(
            live_depth=depth,
            live_normal=live_normal,
            model_depth=model_depth,
            model_normal=model_normal,
            intrinsics=K,
        )

        # Tracking quality gate — fall back to identity if no valid correspondences
        num_valid = quality.get("num_valid", 0)
        if num_valid < 100:
            # Not enough overlap; integrate at current pose without updating
            self.lost = True
            self.tsdf.integrate(depth, K, self.T_world_camera)
            self.frame_count += 1
            return TrackingResult(
                T_world_camera=self.T_world_camera.clone(),
                quality=quality,
                used_keyframe=False,
                lost=True,
            )

        self.lost = False

        # --- Update pose ---
        # T_model_live maps live camera → model camera (= previous world camera space).
        # New absolute pose = T_world_camera_prev @ T_model_live
        self.T_world_camera = self.T_world_camera @ T_model_live

        # --- Integrate live frame into TSDF ---
        self.tsdf.integrate(depth, K, self.T_world_camera)
        self.frame_count += 1

        # --- Keyframe policy ---
        translation = T_model_live[:3, 3].norm().item()
        inlier_ratio = quality.get("inlier_ratio", 1.0)
        used_keyframe = (
            self.frame_count % self.keyframe_max_frames == 0
            or translation > self.keyframe_motion_thresh
            or inlier_ratio < self.keyframe_inlier_ratio_thresh
        )

        return TrackingResult(
            T_world_camera=self.T_world_camera.clone(),
            quality=quality,
            used_keyframe=used_keyframe,
            lost=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize(
        self, depth: torch.Tensor, K: torch.Tensor, device: torch.device
    ) -> TrackingResult:
        """Set up TSDF volume and integrate first frame at identity pose."""
        vd = torch.tensor(list(self._voxel_dim), device=device)
        vo = torch.tensor(list(self._volume_origin), device=device)
        self.tsdf = TSDFVolume(vd, vo, config=self.tsdf_config, device=device)
        self.T_world_camera = torch.eye(4, device=device, dtype=depth.dtype)
        self.frame_count = 0
        self.lost = False

        self.tsdf.integrate(depth, K, self.T_world_camera)
        self.frame_count += 1

        return TrackingResult(
            T_world_camera=self.T_world_camera.clone(),
            quality={"num_valid": 0, "inlier_ratio": 1.0, "rmse": 0.0},
            used_keyframe=True,
            lost=False,
        )


# ------------------------------------------------------------------
# Module-level helpers (also usable standalone)
# ------------------------------------------------------------------

def _estimate_normals(depth: torch.Tensor) -> torch.Tensor:
    """Estimate surface normals from depth via cross-product of tangent vectors.

    Args:
        depth: Depth image [H, W] in meters.

    Returns:
        Normal map [H, W, 3] in camera space, unit-length.
    """
    # Pad to handle borders
    d = F.pad(depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    d = d.squeeze(0).squeeze(0)  # [H+2, W+2]

    # Central differences
    dz_dx = (d[1:-1, 2:] - d[1:-1, :-2]) * 0.5   # [H, W]
    dz_dy = (d[2:, 1:-1] - d[:-2, 1:-1]) * 0.5   # [H, W]

    ones = torch.ones_like(dz_dx)
    n = torch.stack([-dz_dx, -dz_dy, ones], dim=-1)  # [H, W, 3]
    return F.normalize(n, dim=-1)
