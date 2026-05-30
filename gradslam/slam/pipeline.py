"""RGB-D TSDF SLAM pipeline (KinectFusion-style frame-to-model tracking)."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..icp.projective import ProjectiveICPConfig, ProjectiveICPTracker
from ..mapping.tsdf import TSDFConfig, TSDFVolume
from ..rendering.tsdf_raycast import RenderedFrame, raycast_tsdf
from .keyframe_database import KeyframeDatabase
from .pose_graph import SlidingWindowPoseGraph

_logger = logging.getLogger(__name__)


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


@dataclass
class _LocalReference:
    depth: torch.Tensor
    normal: torch.Tensor
    gray: torch.Tensor | None
    intrinsics: torch.Tensor
    T_world_camera: torch.Tensor
    frame_idx: int
    is_keyframe: bool


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
        raycast_normal_mode: str = "gradient",
        process_scale: float = 1.0,
        autocast_dtype: str | None = None,
        tracking_mode: str = "hybrid",
        max_keyframes: int = 8,
        mapping_interval: int = 5,
        enable_mapping: bool = True,
        feature_interval: int = 5,
        keyframe_tracking_interval: int = 10,
        min_track_inliers: int = 100,
        lost_inlier_ratio_thresh: float = 0.02,
        borderline_inlier_ratio: float = 0.08,
        photometric_max_diff: float = 0.20,
        local_map_candidates: int = 1,
        max_frame_translation: float = 0.12,
        max_frame_rotation_deg: float = 30.0,
        candidate_disagreement_penalty: float = 1.0,
        scale_veto_ratio: float = 3.0,
        max_velocity_translation: float = 0.18,
        max_velocity_rotation: float = 0.30,
        pose_graph_enabled: bool = False,
        pose_graph_window: int = 8,
        relocalization_enabled: bool = False,
        loop_closure_enabled: bool = False,
        keyframe_db_size: int = 30,
        loop_closure_min_inliers: int = 30,
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
            raycast_normal_mode: Normal mode for raycast_tsdf; when ``"image"``,
                model normals are re-derived from the rendered depth map via
                ``_estimate_normals`` instead of being returned by the raycast.
            process_scale: Spatial downscale factor applied to depth and
                intrinsics before processing (e.g. ``0.5`` for half resolution).
                ``1.0`` disables scaling.
            autocast_dtype: Enable mixed-precision autocast for the raycast + ICP
                step.  Accepted values: ``"bf16"``, ``"fp16"``, or ``None``
                (disabled).  TSDF integration always runs in fp32.
            tracking_mode: ``"fast_rgbd"`` tracks against cached RGB-D
                references only; ``"hybrid"`` tracks against references and
                uses the TSDF raycast as fallback; ``"tsdf"`` keeps the
                original frame-to-model path.
            max_keyframes: Number of local keyframes retained for hybrid tracking.
            mapping_interval: Integrate non-keyframe poses every N frames when
                confidence is high. Keyframes are always integrated.
            enable_mapping: Enable TSDF allocation/integration. Defaults to
                true for compatibility, but should be false for fast tracking
                benchmarks.
            keyframe_tracking_interval: In fast RGB-D mode, add one local
                keyframe candidate every N frames to suppress drift while
                keeping the normal path previous-frame-only. Set 0 to disable.
            min_track_inliers: Minimum valid correspondences before a pose is
                marked lost.
            lost_inlier_ratio_thresh: Minimum valid-correspondence ratio.
            borderline_inlier_ratio: Inlier ratio below which TSDF fallback is
                attempted even if local tracking produced a pose.
            photometric_max_diff: Mean grayscale reprojection residual threshold
                used for keyframe quality reporting.
            local_map_candidates: Number of nearby keyframes tested every frame
                by ``tracking_mode="local_map"``.
            max_frame_translation: Reject frame-to-frame updates larger than
                this many meters. Set <= 0 to disable.
            max_frame_rotation_deg: Reject frame-to-frame updates larger than
                this many degrees. Set <= 0 to disable.
            candidate_disagreement_penalty: Lambda that penalizes candidates
                whose translation disagrees with the velocity prediction.
                Set 0.0 to disable.
            scale_veto_ratio: Veto the winning candidate if its translation
                exceeds this ratio times the predicted translation and a more
                consistent candidate exists. Set <= 0 to disable.
            max_velocity_translation: Maximum predicted translation (m) from
                the constant-velocity model before falling back to the last
                pose without a velocity prediction.
            max_velocity_rotation: Maximum predicted rotation (rad) from the
                constant-velocity model before falling back to the last pose.
            pose_graph_enabled: Enable sliding-window pose graph optimizer.
                When ``True``, after each keyframe insertion a Gauss-Newton
                optimizer corrects the last ``pose_graph_window`` keyframes.
                Defaults to ``False`` (opt-in).
            pose_graph_window: Window size (number of keyframes) for the
                sliding-window pose graph.
            relocalization_enabled: Enable ORB-based relocalization after 5+
                consecutive lost frames.  Requires cv2.  Defaults to ``False``.
            loop_closure_enabled: Enable loop-closure detection on keyframe
                insertion.  Requires cv2.  Defaults to ``False``.
            keyframe_db_size: Maximum number of keyframes kept in the keyframe
                database for relocalization / loop closure.
            loop_closure_min_inliers: Minimum ORB feature matches required to
                trigger a loop-closure edge.
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

        # Change 1 — raycast normal mode
        self.raycast_normal_mode = raycast_normal_mode

        # Change 2 — camera-frame unit-ray cache
        self._ray_cam_cache: dict[tuple, torch.Tensor] = {}

        # Change 3 — half-resolution processing scale
        self.process_scale = process_scale

        # Change 4 — autocast dtype
        self.autocast_dtype = autocast_dtype

        if tracking_mode not in ("fast_rgbd", "local_map", "hybrid", "tsdf"):
            raise ValueError("tracking_mode must be 'fast_rgbd', 'local_map', 'hybrid', or 'tsdf'")
        self.tracking_mode = tracking_mode
        self.max_keyframes = max_keyframes
        self.mapping_interval = max(1, mapping_interval)
        self.enable_mapping = enable_mapping
        self.feature_interval = feature_interval
        self.keyframe_tracking_interval = max(0, keyframe_tracking_interval)
        self.min_track_inliers = min_track_inliers
        self.lost_inlier_ratio_thresh = lost_inlier_ratio_thresh
        self.borderline_inlier_ratio = borderline_inlier_ratio
        self.photometric_max_diff = photometric_max_diff
        self.local_map_candidates = max(0, int(local_map_candidates))
        self.max_frame_translation = float(max_frame_translation)
        self.max_frame_rotation_rad = (
            float(max_frame_rotation_deg) * torch.pi / 180.0
            if max_frame_rotation_deg > 0
            else 0.0
        )
        self.candidate_disagreement_penalty = float(candidate_disagreement_penalty)
        self.scale_veto_ratio = float(scale_veto_ratio)
        self.max_velocity_translation = float(max_velocity_translation)  # default 0.18
        self.max_velocity_rotation = float(max_velocity_rotation)        # default 0.30

        # Keyframes within this many of the most recent are excluded from loop
        # search (a loop can't close against very recent frames). For a loop
        # edge to ever attach, the pose-graph window must reach back PAST this
        # horizon, i.e. window_size must be > loop_exclude_last_n. See
        # _try_loop_closure / SlidingWindowPoseGraph.add_loop_edge.
        self.loop_exclude_last_n = 8

        # Sliding-window pose graph (opt-in). When loop closure is enabled the
        # window is widened so that loop matches (which come from keyframes
        # OLDER than loop_exclude_last_n) still live inside the window and can
        # be connected by a loop edge; otherwise add_loop_edge always fails and
        # the pose graph never corrects drift.
        self.pose_graph_enabled = pose_graph_enabled
        if pose_graph_enabled and loop_closure_enabled:
            effective_window = max(pose_graph_window, self.loop_exclude_last_n + 8)
        else:
            effective_window = pose_graph_window
        self._pose_graph: SlidingWindowPoseGraph | None = (
            SlidingWindowPoseGraph(window_size=effective_window)
            if pose_graph_enabled
            else None
        )

        # Keyframe database for relocalization / loop closure (opt-in)
        self.relocalization_enabled = relocalization_enabled
        self.loop_closure_enabled = loop_closure_enabled
        self.loop_closure_min_inliers = loop_closure_min_inliers
        self._keyframe_db: KeyframeDatabase | None = (
            KeyframeDatabase(max_keyframes=keyframe_db_size)
            if (relocalization_enabled or loop_closure_enabled)
            else None
        )
        self._consecutive_lost: int = 0
        self._velocity_fallback_count: int = 0

        self.tsdf: TSDFVolume | None = None
        self.T_world_camera: torch.Tensor | None = None
        self.frame_count: int = 0
        self.lost: bool = False
        self.keyframes: list[_LocalReference] = []
        self._last_reference: _LocalReference | None = None
        self._last_T_prev_curr: torch.Tensor | None = None
        self._last_rgb_cpu = None
        self._keyframe_rgb_cpu = None
        self._keyframe_depth_cpu = None
        self._keyframe_K_cpu = None
        self._keyframe_pose_cpu = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the SLAM system (clear map and pose)."""
        self.tsdf = None
        self.T_world_camera = None
        self.frame_count = 0
        self.lost = False
        # Change 2 — clear ray cache on reset
        self._ray_cam_cache.clear()
        self.keyframes.clear()
        self._last_reference = None
        self._last_T_prev_curr = None
        self._last_rgb_cpu = None
        self._keyframe_rgb_cpu = None
        self._keyframe_depth_cpu = None
        self._keyframe_K_cpu = None
        self._keyframe_pose_cpu = None
        # Reset pose graph (if enabled, re-create to clear accumulated state)
        if self._pose_graph is not None:
            self._pose_graph = SlidingWindowPoseGraph(
                window_size=self._pose_graph.window_size
            )
        # Reset keyframe database and lost counter
        if self._keyframe_db is not None:
            self._keyframe_db.clear()
        self._consecutive_lost = 0
        self._velocity_fallback_count = 0

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
        rgb = frame.rgb
        # Accept both 3×3 and 4×4 intrinsics; normalize to 3×3 float
        K = frame.intrinsics[:3, :3].to(dtype=depth.dtype)
        device = depth.device

        # Change 3 — apply process_scale BEFORE first-frame check so that the
        # scaled depth and K are used consistently for the first frame too.
        if self.process_scale != 1.0:
            s = self.process_scale
            ks = round(1.0 / s)
            # Masked depth downsampling: preserve zeros as invalid
            valid = (depth > 0).float()
            depth_sum = F.avg_pool2d(
                (depth * valid).unsqueeze(0).unsqueeze(0),
                kernel_size=ks,
                stride=ks,
                divisor_override=1,
            ).squeeze()
            count = F.avg_pool2d(
                valid.unsqueeze(0).unsqueeze(0),
                kernel_size=ks,
                stride=ks,
                divisor_override=1,
            ).squeeze()
            depth = torch.where(
                count > 0,
                depth_sum / count.clamp(min=1e-8),
                torch.zeros_like(depth_sum),
            )
            K = K.clone()
            K[0, 0] *= s
            K[1, 1] *= s
            K[0, 2] *= s
            K[1, 2] *= s
            if rgb is not None:
                rgb = F.interpolate(
                    rgb.permute(2, 0, 1).unsqueeze(0),
                    size=depth.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)

        if self.T_world_camera is None:
            return self._initialize(depth, K, device, rgb)

        H, W = depth.shape

        # Change 4 — build autocast context
        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
        ac_dtype = _dtype_map.get(self.autocast_dtype)
        if ac_dtype is not None:
            device_type = device.type if hasattr(device, "type") else "cuda"
            ctx = torch.autocast(device_type=device_type, dtype=ac_dtype)
        else:
            ctx = nullcontext()

        if self.tracking_mode in ("fast_rgbd", "local_map"):
            return self._process_frame_fast_rgbd(depth, rgb, K, ctx)

        if self.tracking_mode == "hybrid":
            # Change 2 — pre-compute cached camera-frame unit rays
            ray_cam_unit = self._get_ray_cam_unit(H, W, K, device, depth.dtype)
            return self._process_frame_hybrid(depth, rgb, K, device, ctx, ray_cam_unit)

        if self.tsdf is None:
            raise RuntimeError("TSDF tracking requires enable_mapping=True")

        # Change 2 — pre-compute cached camera-frame unit rays
        ray_cam_unit = self._get_ray_cam_unit(H, W, K, device, depth.dtype)

        with ctx:
            # --- Raycast model at current pose ---
            # Change 1: pass normal_mode; Change 2: pass precomputed_ray_cam_unit
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
                normal_mode=self.raycast_normal_mode,
                precomputed_ray_cam_unit=ray_cam_unit,
            )

            # --- Estimate live normals ---
            live_normal = _estimate_normals(depth, K)  # [H, W, 3]

            # Change 1 — derive model normals from rendered depth when mode is "image"
            if self.raycast_normal_mode == "image":
                model_normal = _estimate_normals(rendered.depth, K)
                # Zero invalid pixels: central-diff normals at depth=0 borders are garbage.
                model_normal = model_normal * rendered.mask.unsqueeze(-1).to(dtype=model_normal.dtype)
            else:
                model_normal = rendered.normal  # [H, W, 3]

            model_depth = rendered.depth  # [H, W]

            # --- Run ICP ---
            T_model_live, quality = self.tracker(
                live_depth=depth,
                live_normal=live_normal,
                model_depth=model_depth,
                model_normal=model_normal,
                intrinsics=K,
            )
            quality["tracking_source"] = "tsdf"

        # TSDF integration OUTSIDE autocast (needs fp32 dynamic range)
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

    def _get_ray_cam_unit(
        self,
        H: int,
        W: int,
        K: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return cached (or freshly computed) unit rays in camera space.

        Args:
            H: Image height.
            W: Image width.
            K: 3×3 intrinsics matrix.
            device: Target device.
            dtype: Target floating-point dtype.

        Returns:
            Unit ray directions [H*W, 3] in camera space.
        """
        key = (
            H,
            W,
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
            str(device),
            str(dtype),
        )
        if key not in self._ray_cam_cache:
            u = torch.arange(W, device=device, dtype=dtype)
            v = torch.arange(H, device=device, dtype=dtype)
            vv, uu = torch.meshgrid(v, u, indexing="ij")
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            ray_cam = torch.stack(
                [(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], dim=-1
            )
            self._ray_cam_cache[key] = F.normalize(ray_cam.reshape(-1, 3), dim=-1)
        return self._ray_cam_cache[key]

    def _initialize(
        self,
        depth: torch.Tensor,
        K: torch.Tensor,
        device: torch.device,
        rgb: torch.Tensor | None = None,
    ) -> TrackingResult:
        """Set up TSDF volume and integrate first frame at identity pose."""
        vd = torch.tensor(list(self._voxel_dim), device=device)
        vo = torch.tensor(list(self._volume_origin), device=device)
        if self.enable_mapping:
            self.tsdf = TSDFVolume(vd, vo, config=self.tsdf_config, device=device)
        else:
            self.tsdf = None
        self.T_world_camera = torch.eye(4, device=device, dtype=depth.dtype)
        self.frame_count = 0
        self.lost = False

        if self.tsdf is not None:
            self.tsdf.integrate(depth, K, self.T_world_camera)
        self.frame_count += 1
        ref = self._make_reference(
            depth=depth,
            K=K,
            T_world_camera=self.T_world_camera,
            frame_idx=0,
            rgb=rgb,
            is_keyframe=True,
        )
        self.keyframes = [ref]
        self._last_reference = ref
        self._last_T_prev_curr = torch.eye(4, device=device, dtype=depth.dtype)
        self._set_feature_keyframe(rgb, depth, K, self.T_world_camera)

        return TrackingResult(
            T_world_camera=self.T_world_camera.clone(),
            quality={
                "num_valid": 0,
                "inlier_ratio": 1.0,
                "rmse": 0.0,
                "tracking_source": "init",
                "integrated": self.tsdf is not None,
            },
            used_keyframe=True,
            lost=False,
        )

    def _process_frame_fast_rgbd(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor | None,
        K: torch.Tensor,
        ctx,
    ) -> TrackingResult:
        """Track against previous frame/keyframes without TSDF raycasting."""
        assert self.T_world_camera is not None
        live_normal = _estimate_normals(depth, K)
        live_gray = _to_gray(rgb) if rgb is not None else None
        predicted_pose = self._predict_pose()

        predicted_rel = torch.linalg.inv(self.T_world_camera) @ predicted_pose
        t_predicted = float(torch.norm(predicted_rel[:3, 3]).item())

        best_pose = predicted_pose
        best_rel = predicted_rel
        best_quality = {
            "num_valid": 0,
            "inlier_ratio": 0.0,
            "rmse": 0.0,
            "tracking_source": "prediction",
        }

        all_evaluated: list[tuple[torch.Tensor, dict]] = []

        tried_refs: list[int] = []
        for ref in self._fast_tracking_candidates(predicted_pose):
            tried_refs.append(id(ref))
            init_T_ref_live = torch.linalg.inv(ref.T_world_camera) @ predicted_pose
            with ctx:
                T_ref_live, quality = self.tracker(
                    live_depth=depth,
                    live_normal=live_normal,
                    model_depth=ref.depth,
                    model_normal=ref.normal,
                    intrinsics=K,
                    init_T_model_live=init_T_ref_live,
                    live_gray=live_gray,
                    ref_gray=ref.gray,
                )
            candidate_pose = ref.T_world_camera @ T_ref_live
            quality = dict(quality)
            quality["tracking_source"] = "keyframe" if ref.is_keyframe else "previous"
            quality["reference_frame_idx"] = ref.frame_idx
            self._annotate_motion_quality(candidate_pose, quality)
            if live_gray is not None and ref.gray is not None:
                photo = self._photometric_reprojection_error(
                    live_depth=depth,
                    live_gray=live_gray,
                    ref_gray=ref.gray,
                    K=K,
                    T_ref_live=T_ref_live,
                )
                quality["photometric_mean_abs"] = photo
                if photo > self.photometric_max_diff:
                    quality["photometric_gate"] = False
            all_evaluated.append((candidate_pose, quality))
            if self._quality_score(quality, t_predicted, self.candidate_disagreement_penalty) > self._quality_score(best_quality, t_predicted, self.candidate_disagreement_penalty):
                best_pose = candidate_pose
                best_rel = torch.linalg.inv(self.T_world_camera) @ best_pose
                best_quality = quality

        if (
            best_quality.get("inlier_ratio", 0.0) < self.borderline_inlier_ratio
            or best_quality.get("num_valid", 0) < self.min_track_inliers
            or best_quality.get("motion_gate", True) is False
        ):
            for ref in self._fast_recovery_candidates(predicted_pose, tried_refs):
                init_T_ref_live = torch.linalg.inv(ref.T_world_camera) @ predicted_pose
                with ctx:
                    T_ref_live, quality = self.tracker(
                        live_depth=depth,
                        live_normal=live_normal,
                        model_depth=ref.depth,
                        model_normal=ref.normal,
                        intrinsics=K,
                        init_T_model_live=init_T_ref_live,
                        live_gray=live_gray,
                        ref_gray=ref.gray,
                    )
                candidate_pose = ref.T_world_camera @ T_ref_live
                quality = dict(quality)
                quality["tracking_source"] = "recovery_keyframe"
                quality["reference_frame_idx"] = ref.frame_idx
                self._annotate_motion_quality(candidate_pose, quality)
                if live_gray is not None and ref.gray is not None:
                    photo = self._photometric_reprojection_error(
                        live_depth=depth,
                        live_gray=live_gray,
                        ref_gray=ref.gray,
                        K=K,
                        T_ref_live=T_ref_live,
                    )
                    quality["photometric_mean_abs"] = photo
                    if photo > self.photometric_max_diff:
                        quality["photometric_gate"] = False
                all_evaluated.append((candidate_pose, quality))
                if self._quality_score(quality, t_predicted, self.candidate_disagreement_penalty) > self._quality_score(best_quality, t_predicted, self.candidate_disagreement_penalty):
                    best_pose = candidate_pose
                    best_rel = torch.linalg.inv(self.T_world_camera) @ best_pose
                    best_quality = quality

        # A2: scale-consistency veto — apply before PnP
        if all_evaluated:
            all_candidates_sorted = sorted(
                all_evaluated,
                key=lambda c: self._quality_score(c[1], t_predicted, self.candidate_disagreement_penalty),
                reverse=True,
            )
            veto_result = self._scale_veto(all_candidates_sorted, t_predicted, self.scale_veto_ratio)
            if veto_result is not None:
                best_pose, best_quality = veto_result
                best_rel = torch.linalg.inv(self.T_world_camera) @ best_pose

        # Summarize all evaluated candidates into best_quality for diagnostics
        best_quality["candidates"] = [
            {
                "ref_idx": int(q.get("reference_frame_idx", -1)),
                "source": q.get("tracking_source", "unknown"),
                "inliers": int(q.get("num_valid", 0)),
                "inlier_ratio": float(q.get("inlier_ratio", 0.0)),
                "rmse": float(q.get("rmse", 0.0)),
                "photo": float(q.get("photometric_mean_abs", -1.0)),
                "disagreement": float(q.get("t_disagreement_norm", -1.0)),
                "motion_gate": bool(q.get("motion_gate", True)),
                "accepted": bool(q is best_quality),
            }
            for _pose, q in all_evaluated
        ]

        pnp_pose = None
        if (
            rgb is not None
            and self._keyframe_rgb_cpu is not None
            and self.feature_interval > 0
            and (
                self.frame_count % self.feature_interval == 0
                or best_quality.get("inlier_ratio", 0.0) < self.keyframe_inlier_ratio_thresh
                or best_quality.get("photometric_gate", True) is False
            )
        ):
            pnp_pose, pnp_quality = self._feature_pnp_pose(rgb, K, depth.device, depth.dtype)
            if pnp_pose is not None and (
                best_quality.get("num_valid", 0) == 0
                or pnp_quality.get("feature_inliers", 0) >= 20
            ):
                best_pose = pnp_pose
                best_rel = torch.linalg.inv(self.T_world_camera) @ best_pose
                best_quality.update(pnp_quality)
                best_quality["tracking_source"] = "feature_pnp"
                self._annotate_motion_quality(best_pose, best_quality)

        # Store t_disagreement_norm in best_quality after all candidate selection
        if self.candidate_disagreement_penalty > 0.0 and t_predicted > 0.0:
            t_c = float(best_quality.get("frame_translation", t_predicted))
            best_quality["t_disagreement_norm"] = abs(t_c - t_predicted) / max(0.02, t_predicted + 0.02)

        lost = self._is_lost(best_quality)
        if lost:
            self._consecutive_lost += 1
            # Attempt relocalization after 5+ consecutive lost frames
            if (
                self.relocalization_enabled
                and self._keyframe_db is not None
                and self._consecutive_lost >= 5
                and rgb is not None
            ):
                rgb_uint8 = _rgb_to_uint8_cpu(rgb)
                if rgb_uint8 is not None:
                    depth_np = depth.detach().cpu().numpy().astype("float32")
                    K_np = K.detach().cpu().numpy()
                    recovered_T_np, reloc_info = self._keyframe_db.relocalize(
                        rgb_uint8, depth_np, K_np, min_inliers=20
                    )
                    if recovered_T_np is not None:
                        import torch as _torch
                        best_pose = _torch.tensor(
                            recovered_T_np, dtype=depth.dtype, device=depth.device
                        )
                        best_rel = torch.linalg.inv(self.T_world_camera) @ best_pose
                        best_quality.update(reloc_info or {})
                        best_quality["tracking_source"] = "relocalization"
                        self._annotate_motion_quality(best_pose, best_quality)
                        self._consecutive_lost = 0
                        lost = False

        if lost:
            tracking_state = "lost"
        elif self._is_weak(best_quality):
            tracking_state = "weak"
        else:
            tracking_state = "ok"
        best_quality["tracking_state"] = tracking_state

        if lost:
            # Keep a transient reference so the next frame does not match stale
            # geometry, but do not train the velocity model, insert a keyframe,
            # or integrate a weak pose into the map.
            self.T_world_camera = predicted_pose
            self._last_reference = self._make_reference(
                depth=depth,
                K=K,
                T_world_camera=self.T_world_camera,
                frame_idx=self.frame_count,
                rgb=rgb,
                normal=live_normal,
                is_keyframe=False,
            )
            self._last_T_prev_curr = None
            self.lost = True
            self.frame_count += 1
            return TrackingResult(
                T_world_camera=self.T_world_camera.clone(),
                quality=best_quality,
                used_keyframe=False,
                lost=True,
            )

        self._consecutive_lost = 0
        prev_pose = self.T_world_camera
        self.T_world_camera = best_pose
        self._last_T_prev_curr = torch.linalg.inv(prev_pose) @ self.T_world_camera
        self.lost = False

        used_keyframe = self._should_insert_keyframe(best_rel, best_quality)
        integrate_frame = False
        map_update_allowed = best_quality.get("tracking_state", "ok") == "ok"
        best_quality["map_update_allowed"] = map_update_allowed
        if self.tsdf is not None and map_update_allowed and (
            used_keyframe or (self.frame_count % self.mapping_interval == 0)
        ):
            self.tsdf.integrate(depth, K, self.T_world_camera)
            integrate_frame = True

        ref = self._make_reference(
            depth=depth,
            K=K,
            T_world_camera=self.T_world_camera,
            frame_idx=self.frame_count,
            rgb=rgb,
            normal=live_normal,
            is_keyframe=used_keyframe,
        )
        self._last_reference = ref
        self._last_rgb_cpu = _rgb_to_uint8_cpu(rgb)
        if used_keyframe:
            self.keyframes.append(ref)
            if len(self.keyframes) > self.max_keyframes:
                self.keyframes = self.keyframes[-self.max_keyframes :]
            self._set_feature_keyframe(rgb, depth, K, self.T_world_camera)

            # Sliding-window pose graph correction (opt-in via pose_graph_enabled).
            # The current keyframe is added to the graph exactly once here.
            self._apply_pose_graph(depth, K, rgb, live_normal, best_quality)

            # Keyframe DB population + loop closure (opt-in). Loop edges are the
            # only source of drift correction in the pose graph.
            self._try_loop_closure(depth, K, rgb, live_normal, best_quality)

        best_quality["integrated"] = integrate_frame
        best_quality["velocity_fallback_count"] = self._velocity_fallback_count
        self.frame_count += 1
        return TrackingResult(
            T_world_camera=self.T_world_camera.clone(),
            quality=best_quality,
            used_keyframe=used_keyframe,
            lost=False,
        )

    def _fast_tracking_candidates(self, predicted_pose: torch.Tensor) -> list[_LocalReference]:
        candidates: list[_LocalReference] = []
        if self._last_reference is not None:
            candidates.append(self._last_reference)

        if not self.keyframes:
            return candidates

        if not candidates:
            candidates.append(self.keyframes[-1])

        if (
            self.keyframe_tracking_interval > 0
            and self.frame_count % self.keyframe_tracking_interval == 0
        ):
            scored = []
            for ref in self.keyframes:
                if any(id(ref) == id(candidate) for candidate in candidates):
                    continue
                rel = torch.linalg.inv(ref.T_world_camera) @ predicted_pose
                scored.append((torch.norm(rel[:3, 3]).item(), ref))
            if scored:
                candidates.append(min(scored, key=lambda item: item[0])[1])

        return candidates

    def _fast_recovery_candidates(
        self,
        predicted_pose: torch.Tensor,
        tried_ref_ids: list[int],
    ) -> list[_LocalReference]:
        scored = []
        for ref in self.keyframes:
            if id(ref) in tried_ref_ids:
                continue
            rel = torch.linalg.inv(ref.T_world_camera) @ predicted_pose
            scored.append((torch.norm(rel[:3, 3]).item(), ref))
        limit = self.local_map_candidates if self.tracking_mode == "local_map" else 3
        return [ref for _, ref in sorted(scored, key=lambda item: item[0])[:limit]]

    def _set_feature_keyframe(
        self,
        rgb: torch.Tensor | None,
        depth: torch.Tensor,
        K: torch.Tensor,
        T_world_camera: torch.Tensor,
    ) -> None:
        self._keyframe_rgb_cpu = _rgb_to_uint8_cpu(rgb)
        self._keyframe_depth_cpu = depth.detach().cpu().numpy()
        self._keyframe_K_cpu = K.detach().cpu().numpy()
        self._keyframe_pose_cpu = T_world_camera.detach().cpu().numpy()

    def _feature_pnp_pose(
        self,
        rgb: torch.Tensor,
        K: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, dict]:
        if self._keyframe_rgb_cpu is None or self._keyframe_depth_cpu is None:
            return None, {}
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None, {}

        live_img = _rgb_to_uint8_cpu(rgb)
        if live_img is None:
            return None, {}

        orb = cv2.ORB_create(nfeatures=600, fastThreshold=12)
        kp_ref, des_ref = orb.detectAndCompute(self._keyframe_rgb_cpu, None)
        kp_live, des_live = orb.detectAndCompute(live_img, None)
        if des_ref is None or des_live is None or len(kp_ref) < 12 or len(kp_live) < 12:
            return None, {"feature_inliers": 0}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_ref, des_live)
        if len(matches) < 12:
            return None, {"feature_inliers": 0}
        matches = sorted(matches, key=lambda m: m.distance)[:120]

        K_np = self._keyframe_K_cpu
        depth_ref = self._keyframe_depth_cpu
        pts3 = []
        pts2 = []
        fx, fy = K_np[0, 0], K_np[1, 1]
        cx, cy = K_np[0, 2], K_np[1, 2]
        H, W = depth_ref.shape
        for m in matches:
            u_ref, v_ref = kp_ref[m.queryIdx].pt
            ui = int(round(u_ref))
            vi = int(round(v_ref))
            if ui < 0 or ui >= W or vi < 0 or vi >= H:
                continue
            z = float(depth_ref[vi, ui])
            if not (0.1 < z < 6.0):
                continue
            x = (u_ref - cx) * z / fx
            y = (v_ref - cy) * z / fy
            pts3.append([x, y, z])
            pts2.append(kp_live[m.trainIdx].pt)

        if len(pts3) < 12:
            return None, {"feature_inliers": 0}
        pts3 = np.asarray(pts3, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3,
            pts2,
            K_np.astype(np.float64),
            None,
            iterationsCount=80,
            reprojectionError=3.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        n_inliers = 0 if inliers is None else int(len(inliers))
        if not ok or n_inliers < 12:
            return None, {"feature_inliers": n_inliers}

        R, _ = cv2.Rodrigues(rvec)
        T_live_ref = np.eye(4, dtype=np.float32)
        T_live_ref[:3, :3] = R.astype(np.float32)
        T_live_ref[:3, 3] = tvec[:, 0].astype(np.float32)
        T_ref_live = np.linalg.inv(T_live_ref)
        T_world_live = self._keyframe_pose_cpu @ T_ref_live
        pose = torch.from_numpy(T_world_live).to(device=device, dtype=dtype)
        return pose, {"feature_inliers": n_inliers}

    def _process_frame_hybrid(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor | None,
        K: torch.Tensor,
        device: torch.device,
        ctx,
        ray_cam_unit: torch.Tensor,
    ) -> TrackingResult:
        """Track against local RGB-D references before touching the TSDF map."""
        assert self.T_world_camera is not None
        assert self.tsdf is not None

        live_normal = _estimate_normals(depth, K)
        live_gray = _to_gray(rgb) if rgb is not None else None
        predicted_pose = self._predict_pose()

        predicted_rel_hybrid = torch.linalg.inv(self.T_world_camera) @ predicted_pose
        t_predicted_hybrid = float(torch.norm(predicted_rel_hybrid[:3, 3]).item())

        best_pose = predicted_pose
        best_rel = predicted_rel_hybrid
        best_quality = {
            "num_valid": 0,
            "inlier_ratio": 0.0,
            "rmse": 0.0,
            "tracking_source": "prediction",
        }

        all_evaluated: list[tuple[torch.Tensor, dict]] = []

        for ref in self._tracking_candidates(predicted_pose):
            init_T_ref_live = torch.linalg.inv(ref.T_world_camera) @ predicted_pose
            with ctx:
                T_ref_live, quality = self.tracker(
                    live_depth=depth,
                    live_normal=live_normal,
                    model_depth=ref.depth,
                    model_normal=ref.normal,
                    intrinsics=K,
                    init_T_model_live=init_T_ref_live,
                    live_gray=live_gray,
                    ref_gray=ref.gray,
                )
            candidate_pose = ref.T_world_camera @ T_ref_live
            quality = dict(quality)
            quality["tracking_source"] = "keyframe" if ref.is_keyframe else "previous"
            quality["reference_frame_idx"] = ref.frame_idx
            self._annotate_motion_quality(candidate_pose, quality)
            if live_gray is not None and ref.gray is not None:
                photo = self._photometric_reprojection_error(
                    live_depth=depth,
                    live_gray=live_gray,
                    ref_gray=ref.gray,
                    K=K,
                    T_ref_live=T_ref_live,
                )
                quality["photometric_mean_abs"] = photo
                if photo > self.photometric_max_diff:
                    quality["photometric_gate"] = False
            all_evaluated.append((candidate_pose, quality))
            if self._quality_score(quality, t_predicted_hybrid, self.candidate_disagreement_penalty) > self._quality_score(best_quality, t_predicted_hybrid, self.candidate_disagreement_penalty):
                best_pose = candidate_pose
                best_rel = torch.linalg.inv(self.T_world_camera) @ best_pose
                best_quality = quality

        needs_tsdf_recovery = (
            best_quality.get("inlier_ratio", 0.0) < self.borderline_inlier_ratio
            or best_quality.get("photometric_gate", True) is False
        )
        if needs_tsdf_recovery:
            tsdf_pose, tsdf_rel, tsdf_quality = self._track_tsdf_from_pose(
                depth=depth,
                K=K,
                live_normal=live_normal,
                pose_init=predicted_pose,
                ctx=ctx,
                ray_cam_unit=ray_cam_unit,
            )
            all_evaluated.append((tsdf_pose, tsdf_quality))
            if self._quality_score(tsdf_quality, t_predicted_hybrid, self.candidate_disagreement_penalty) > self._quality_score(best_quality, t_predicted_hybrid, self.candidate_disagreement_penalty):
                best_pose, best_rel, best_quality = tsdf_pose, tsdf_rel, tsdf_quality

        # Summarize all evaluated candidates into best_quality for diagnostics
        best_quality["candidates"] = [
            {
                "ref_idx": int(q.get("reference_frame_idx", -1)),
                "source": q.get("tracking_source", "unknown"),
                "inliers": int(q.get("num_valid", 0)),
                "inlier_ratio": float(q.get("inlier_ratio", 0.0)),
                "rmse": float(q.get("rmse", 0.0)),
                "photo": float(q.get("photometric_mean_abs", -1.0)),
                "disagreement": float(q.get("t_disagreement_norm", -1.0)),
                "motion_gate": bool(q.get("motion_gate", True)),
                "accepted": bool(q is best_quality),
            }
            for _pose, q in all_evaluated
        ]

        if self.candidate_disagreement_penalty > 0.0 and t_predicted_hybrid > 0.0:
            if "frame_translation" in best_quality:
                t_c = float(best_quality["frame_translation"])
                best_quality["t_disagreement_norm"] = abs(t_c - t_predicted_hybrid) / max(0.02, t_predicted_hybrid + 0.02)
            else:
                best_quality["t_disagreement_norm"] = -1.0

        lost = self._is_lost(best_quality)
        if lost:
            self._consecutive_lost += 1
            # Attempt relocalization after 5+ consecutive lost frames
            if (
                self.relocalization_enabled
                and self._keyframe_db is not None
                and self._consecutive_lost >= 5
                and rgb is not None
            ):
                rgb_uint8_h = _rgb_to_uint8_cpu(rgb)
                if rgb_uint8_h is not None:
                    depth_np_h = depth.detach().cpu().numpy().astype("float32")
                    K_np_h = K.detach().cpu().numpy()
                    recovered_T_np_h, reloc_info_h = self._keyframe_db.relocalize(
                        rgb_uint8_h, depth_np_h, K_np_h, min_inliers=20
                    )
                    if recovered_T_np_h is not None:
                        best_pose = torch.tensor(
                            recovered_T_np_h, dtype=depth.dtype, device=depth.device
                        )
                        best_rel = torch.linalg.inv(self.T_world_camera) @ best_pose
                        best_quality.update(reloc_info_h or {})
                        best_quality["tracking_source"] = "relocalization"
                        self._annotate_motion_quality(best_pose, best_quality)
                        self._consecutive_lost = 0
                        lost = False

        if lost:
            tracking_state = "lost"
        elif self._is_weak(best_quality):
            tracking_state = "weak"
        else:
            tracking_state = "ok"
        best_quality["tracking_state"] = tracking_state

        if lost:
            self.lost = True
            self.frame_count += 1
            return TrackingResult(
                T_world_camera=self.T_world_camera.clone(),
                quality=best_quality,
                used_keyframe=False,
                lost=True,
            )

        self._consecutive_lost = 0
        prev_pose = self.T_world_camera
        self.T_world_camera = best_pose
        self._last_T_prev_curr = torch.linalg.inv(prev_pose) @ self.T_world_camera
        self.lost = False

        used_keyframe = self._should_insert_keyframe(best_rel, best_quality)
        map_update_allowed = best_quality.get("tracking_state", "ok") == "ok"
        best_quality["map_update_allowed"] = map_update_allowed
        integrate_frame = map_update_allowed and (used_keyframe or (self.frame_count % self.mapping_interval == 0))
        if integrate_frame:
            self.tsdf.integrate(depth, K, self.T_world_camera)

        ref = self._make_reference(
            depth=depth,
            K=K,
            T_world_camera=self.T_world_camera,
            frame_idx=self.frame_count,
            rgb=rgb,
            normal=live_normal,
            is_keyframe=used_keyframe,
        )
        self._last_reference = ref
        if used_keyframe:
            self.keyframes.append(ref)
            if len(self.keyframes) > self.max_keyframes:
                self.keyframes = self.keyframes[-self.max_keyframes :]

            # Sliding-window pose graph correction (opt-in via pose_graph_enabled).
            # The current keyframe is added to the graph exactly once here.
            self._apply_pose_graph(depth, K, rgb, live_normal, best_quality)

            # Keyframe DB population + loop closure (opt-in). Loop edges are the
            # only source of drift correction in the pose graph.
            self._try_loop_closure(depth, K, rgb, live_normal, best_quality)

        best_quality["integrated"] = integrate_frame
        best_quality["velocity_fallback_count"] = self._velocity_fallback_count
        self.frame_count += 1
        return TrackingResult(
            T_world_camera=self.T_world_camera.clone(),
            quality=best_quality,
            used_keyframe=used_keyframe,
            lost=False,
        )

    def _track_tsdf_from_pose(
        self,
        depth: torch.Tensor,
        K: torch.Tensor,
        live_normal: torch.Tensor,
        pose_init: torch.Tensor,
        ctx,
        ray_cam_unit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Run the TSDF raycast tracker from a predicted pose."""
        assert self.tsdf is not None
        H, W = depth.shape
        with ctx:
            rendered = raycast_tsdf(
                tsdf_volume=self.tsdf.tsdf,
                tsdf_origin=self.tsdf._origin,
                voxel_size=self.tsdf_config.voxel_size,
                T_world_camera=pose_init,
                intrinsics=K,
                height=H,
                width=W,
                near=self.near,
                far=self.far,
                n_samples=128,
                normal_mode=self.raycast_normal_mode,
                precomputed_ray_cam_unit=ray_cam_unit,
            )
            if self.raycast_normal_mode == "image":
                model_normal = _estimate_normals(rendered.depth, K)
                model_normal = model_normal * rendered.mask.unsqueeze(-1).to(dtype=model_normal.dtype)
            else:
                model_normal = rendered.normal
            T_model_live, quality = self.tracker(
                live_depth=depth,
                live_normal=live_normal,
                model_depth=rendered.depth,
                model_normal=model_normal,
                intrinsics=K,
            )
        pose = pose_init @ T_model_live
        rel = torch.linalg.inv(self.T_world_camera) @ pose
        quality = dict(quality)
        quality["tracking_source"] = "tsdf"
        return pose, rel, quality

    def _write_back_corrected_poses(self, corrected: list[torch.Tensor]) -> None:
        """Write optimized node poses back into self.keyframes by node id.

        The pose graph uses keyframe ``frame_idx`` as node ids, so we match
        corrected poses to keyframes by that id. Keyframes whose frame_idx is
        not a current node (e.g. slid out of the window) are left untouched.
        """
        id_to_pose = {
            nid: pose
            for nid, pose in zip(self._pose_graph._ids, corrected)
        }
        for kf in self.keyframes:
            new_pose = id_to_pose.get(getattr(kf, "frame_idx", None))
            if new_pose is not None:
                kf.T_world_camera = new_pose.clone()

    def _apply_pose_graph(
        self,
        depth: torch.Tensor,
        K: torch.Tensor,
        rgb,
        live_normal,
        best_quality: dict,
    ) -> None:
        """Add the current keyframe to the pose graph and apply correction.

        Adds the new keyframe as a node (id = its frame index) with a
        sequential keyframe-to-keyframe edge whose measurement is derived from
        the absolute poses (a residual-free no-op; drift correction comes from
        loop edges added separately). Then optimizes and writes corrected
        poses back. Adds the node exactly once per keyframe.
        """
        if self._pose_graph is None:
            return
        inlier_w = float(best_quality.get("inlier_ratio", 0.5))
        rmse_w = 1.0 / max(float(best_quality.get("rmse", 0.01)), 1e-4)
        w = inlier_w * min(rmse_w, 100.0)
        # node_id = the keyframe's frame index; T_rel_measured=None derives the
        # correct keyframe-to-keyframe relative from the absolute poses.
        self._pose_graph.add_keyframe(
            self.T_world_camera,
            node_id=self.frame_count,
            T_rel_measured=None,
            weight=w,
        )
        if self._pose_graph.num_keyframes >= 2:
            corrected = self._pose_graph.apply_correction()
            self._write_back_corrected_poses(corrected)
            if corrected:
                self.T_world_camera = corrected[-1].clone()
            # Rebuild _last_reference so the next frame tracks corrected geometry.
            self._last_reference = self._make_reference(
                depth=depth,
                K=K,
                T_world_camera=self.T_world_camera,
                frame_idx=self.frame_count,
                rgb=rgb,
                normal=live_normal,
                is_keyframe=True,
            )

    def _try_loop_closure(
        self,
        depth: torch.Tensor,
        K: torch.Tensor,
        rgb,
        live_normal,
        best_quality: dict,
    ) -> None:
        """Populate the keyframe DB and inject a loop-closure edge if found.

        On a detected loop, adds a *loop edge* between the matched node and the
        current node (not a new keyframe node), then re-optimizes and writes
        back corrected poses.
        """
        if self._keyframe_db is None or rgb is None:
            return
        rgb_uint8_kf = _rgb_to_uint8_cpu(rgb)
        if rgb_uint8_kf is None:
            return
        depth_np_kf = depth.detach().cpu().numpy().astype("float32")
        K_np_kf = K.detach().cpu().numpy()
        T_np_kf = self.T_world_camera.detach().cpu().numpy()
        self._keyframe_db.add(
            rgb_uint8_kf, depth_np_kf, K_np_kf, T_np_kf, self.frame_count
        )

        if not (
            self.loop_closure_enabled
            and self._pose_graph is not None
            and len(self._keyframe_db) > self.loop_exclude_last_n
        ):
            return

        import cv2 as _cv2
        gray_lc = _cv2.cvtColor(rgb_uint8_kf, _cv2.COLOR_RGB2GRAY)
        kpts_cur, desc_cur = self._keyframe_db._orb.detectAndCompute(gray_lc, None)
        T_rel_np, match_idx, n_inliers = self._keyframe_db.find_loop(
            (kpts_cur, desc_cur),
            K_np_kf,
            exclude_last_n=self.loop_exclude_last_n,
            min_inliers=self.loop_closure_min_inliers,
        )
        if T_rel_np is None:
            return

        # Convention derivation (sign is load-bearing):
        # find_loop runs solvePnP(objectPoints=pts_3d in the MATCHED keyframe
        # frame, imagePoints=query pixels, K=query). The solvePnP contract
        # returns [R|t] s.t. p_query = M @ p_ref, i.e. M maps ref->query points,
        # so  M = inv(T_world_query) @ T_world_match.
        # The pose-graph edge convention is  E ≈ inv(T_world_a) @ T_world_b
        # with a = match (older) and b = current (query):
        #   E = inv(T_world_match) @ T_world_query = inv(M).
        # Hence we must pass the INVERSE of find_loop's returned transform.
        M = torch.tensor(T_rel_np, dtype=depth.dtype, device=depth.device)
        T_rel_loop = torch.linalg.inv(M)
        # Inlier-scaled weight so a strong loop counts comparably to the
        # sequential (odometry) edges, which carry weight up to ~100. A fixed
        # weight of 2.0 was negligible against that stiffness and barely moved
        # the trajectory. Scale by verified PnP inliers, clamped to a sane band.
        loop_weight = float(min(max(n_inliers, self.loop_closure_min_inliers), 200)) / 4.0
        added = self._pose_graph.add_loop_edge(
            int(match_idx), int(self.frame_count), T_rel_loop, weight=loop_weight
        )
        if not added:
            return
        corrected_lc = self._pose_graph.apply_correction()
        self._write_back_corrected_poses(corrected_lc)
        if corrected_lc:
            self.T_world_camera = corrected_lc[-1].clone()
            # The loop correction moved the current pose after _apply_pose_graph
            # already rebuilt _last_reference, so rebuild it again to keep the
            # next frame tracking against corrected geometry.
            self._last_reference = self._make_reference(
                depth=depth,
                K=K,
                T_world_camera=self.T_world_camera,
                frame_idx=self.frame_count,
                rgb=rgb,
                normal=live_normal,
                is_keyframe=True,
            )
        best_quality["loop_closure_frame_idx"] = match_idx

    def _make_reference(
        self,
        depth: torch.Tensor,
        K: torch.Tensor,
        T_world_camera: torch.Tensor,
        frame_idx: int,
        rgb: torch.Tensor | None = None,
        normal: torch.Tensor | None = None,
        is_keyframe: bool = False,
    ) -> _LocalReference:
        return _LocalReference(
            depth=depth.detach().clone(),
            normal=(
                normal.detach().clone()
                if normal is not None
                else _estimate_normals(depth, K).detach()
            ),
            gray=(_to_gray(rgb).detach().clone() if rgb is not None else None),
            intrinsics=K.detach().clone(),
            T_world_camera=T_world_camera.detach().clone(),
            frame_idx=frame_idx,
            is_keyframe=is_keyframe,
        )

    def _predict_pose(self) -> torch.Tensor:
        assert self.T_world_camera is not None
        if self._last_T_prev_curr is None:
            return self.T_world_camera.clone()
        translation = torch.norm(self._last_T_prev_curr[:3, 3])
        trace = torch.trace(self._last_T_prev_curr[:3, :3])
        cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        if translation > self.max_velocity_translation or angle > self.max_velocity_rotation:
            _logger.debug(
                "velocity gate fallback at frame %d: t=%.4fm rot=%.3frad (limits: %.3f, %.3f)",
                self.frame_count, translation.item(), angle.item(),
                self.max_velocity_translation, self.max_velocity_rotation
            )
            self._velocity_fallback_count += 1
            return self.T_world_camera.clone()
        return self.T_world_camera @ self._last_T_prev_curr

    def _tracking_candidates(self, predicted_pose: torch.Tensor) -> list[_LocalReference]:
        candidates: list[_LocalReference] = []
        if self._last_reference is not None:
            candidates.append(self._last_reference)

        if self.keyframes:
            scored = []
            for ref in self.keyframes:
                rel = torch.linalg.inv(ref.T_world_camera) @ predicted_pose
                score = torch.norm(rel[:3, 3]).item()
                scored.append((score, ref))
            for _, ref in sorted(scored, key=lambda item: item[0])[:3]:
                if not any(id(ref) == id(candidate) for candidate in candidates):
                    candidates.append(ref)
        return candidates

    def _should_insert_keyframe(self, T_prev_live: torch.Tensor, quality: dict) -> bool:
        if self._is_lost(quality):
            return False
        if not self.keyframes:
            return True
        last_keyframe = self.keyframes[-1]
        rel_key = torch.linalg.inv(last_keyframe.T_world_camera) @ self.T_world_camera
        key_translation = torch.norm(rel_key[:3, 3]).item()
        frame_gap = self.frame_count - last_keyframe.frame_idx
        frame_translation = torch.norm(T_prev_live[:3, 3]).item()
        inlier_ratio = quality.get("inlier_ratio", 1.0)
        return (
            frame_gap >= self.keyframe_max_frames
            or key_translation > self.keyframe_motion_thresh
            or frame_translation > self.keyframe_motion_thresh
            or (
                inlier_ratio < self.keyframe_inlier_ratio_thresh
                and quality.get("num_valid", 0) >= self.min_track_inliers
            )
        )

    def _is_lost(self, quality: dict) -> bool:
        return (
            quality.get("num_valid", 0) < self.min_track_inliers
            or quality.get("inlier_ratio", 0.0) < self.lost_inlier_ratio_thresh
            or quality.get("motion_gate", True) is False
        )

    def _is_weak(self, quality: dict) -> bool:
        """True if tracking is borderline — pose accepted but map update suppressed."""
        return (
            not self._is_lost(quality)
            and quality.get("inlier_ratio", 1.0) < self.borderline_inlier_ratio
        )

    @staticmethod
    def _quality_score(
        quality: dict,
        t_predicted: float = 0.0,
        lambda_disagree: float = 0.0,
    ) -> tuple[float, float, float, float, float]:
        photometric_ok = 1.0 if quality.get("photometric_gate", True) is not False else 0.0
        motion_ok = 1.0 if quality.get("motion_gate", True) is not False else 0.0
        inlier_ratio = float(quality.get("inlier_ratio", 0.0))
        if lambda_disagree > 0.0 and t_predicted > 0.0:
            t_candidate = float(quality.get("frame_translation", t_predicted))
            t_disagree = abs(t_candidate - t_predicted)
            t_disagree_norm = t_disagree / max(0.02, t_predicted + 0.02)
            inlier_ratio = inlier_ratio - lambda_disagree * t_disagree_norm
        return (
            motion_ok,
            photometric_ok,
            inlier_ratio,
            float(quality.get("num_valid", 0)),
            -float(quality.get("rmse", 1e9)),
        )

    @staticmethod
    def _scale_veto(
        candidates: list,
        t_predicted: float,
        scale_veto_ratio: float = 3.0,
    ):
        """If the top candidate has t > scale_veto_ratio * t_predicted and a more
        consistent candidate exists, return the more consistent one instead."""
        if not candidates or t_predicted <= 0.0 or scale_veto_ratio <= 0.0:
            return None
        # already sorted best-first
        winner_pose, winner_quality = candidates[0]
        t_winner = float(winner_quality.get("frame_translation", t_predicted))
        if t_winner <= scale_veto_ratio * t_predicted:
            return None  # winner is not over-scaled — no veto needed
        winner_inlier = float(winner_quality.get("inlier_ratio", 0.0))
        for pose, q in candidates[1:]:
            if q.get("motion_gate", True) is False:
                continue
            t_cand = float(q.get("frame_translation", t_predicted))
            if t_cand <= 1.5 * t_predicted and float(q.get("inlier_ratio", 0.0)) >= 0.7 * winner_inlier:
                return pose, q
        return None

    def _annotate_motion_quality(self, candidate_pose: torch.Tensor, quality: dict) -> None:
        assert self.T_world_camera is not None
        rel = torch.linalg.inv(self.T_world_camera) @ candidate_pose
        translation = torch.norm(rel[:3, 3])
        trace = torch.trace(rel[:3, :3])
        cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        quality["frame_translation"] = float(translation.item())
        quality["frame_rotation_deg"] = float((angle * 180.0 / torch.pi).item())
        translation_ok = (
            self.max_frame_translation <= 0.0
            or quality["frame_translation"] <= self.max_frame_translation
        )
        rotation_ok = (
            self.max_frame_rotation_rad <= 0.0
            or float(angle.item()) <= self.max_frame_rotation_rad
        )
        if not (translation_ok and rotation_ok):
            quality["motion_gate"] = False

    def _photometric_reprojection_error(
        self,
        live_depth: torch.Tensor,
        live_gray: torch.Tensor,
        ref_gray: torch.Tensor,
        K: torch.Tensor,
        T_ref_live: torch.Tensor,
    ) -> float:
        live_vertex = self.tracker._depth_to_vertex(live_depth, K, live_depth.device, live_depth.dtype)
        live_vertex_ref = self.tracker._transform_points(live_vertex, T_ref_live)
        H, W = live_depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        z = live_vertex_ref[:, :, 2].clamp(min=1e-6)
        u = (live_vertex_ref[:, :, 0] * fx / z + cx).long()
        v = (live_vertex_ref[:, :, 1] * fy / z + cy).long()
        valid = (live_depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if valid.sum() < 10:
            return float("inf")
        diff = torch.abs(live_gray[valid] - ref_gray[v[valid], u[valid]])
        return float(diff.mean().item())


# ------------------------------------------------------------------
# Module-level helpers (also usable standalone)
# ------------------------------------------------------------------

def _estimate_normals(depth: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
    """Estimate camera-space surface normals from depth.

    Args:
        depth: Depth image [H, W] in meters.

    Returns:
        Normal map [H, W, 3] in camera space, unit-length.
    """
    if K is not None:
        H, W = depth.shape
        dtype = depth.dtype
        device = depth.device
        u, v = torch.meshgrid(
            torch.arange(W, device=device, dtype=dtype),
            torch.arange(H, device=device, dtype=dtype),
            indexing="xy",
        )
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        vertex = torch.stack(
            [
                (u - cx) * depth / fx,
                (v - cy) * depth / fy,
                depth,
            ],
            dim=-1,
        )
        vp = F.pad(vertex.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        vp = vp.squeeze(0).permute(1, 2, 0)
        dx = vp[1:-1, 2:] - vp[1:-1, :-2]
        dy = vp[2:, 1:-1] - vp[:-2, 1:-1]
        n = torch.linalg.cross(dx, dy, dim=-1)
        n = F.normalize(n, dim=-1)
        valid = depth > 0
        return torch.where(valid.unsqueeze(-1), n, torch.zeros_like(n))

    # Fallback image-space normal estimate for legacy callers.
    # Pad to handle borders
    d = F.pad(depth.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    d = d.squeeze(0).squeeze(0)  # [H+2, W+2]

    # Central differences
    dz_dx = (d[1:-1, 2:] - d[1:-1, :-2]) * 0.5   # [H, W]
    dz_dy = (d[2:, 1:-1] - d[:-2, 1:-1]) * 0.5   # [H, W]

    ones = torch.ones_like(dz_dx)
    n = torch.stack([-dz_dx, -dz_dy, ones], dim=-1)  # [H, W, 3]
    return F.normalize(n, dim=-1)


def _to_gray(rgb: torch.Tensor | None) -> torch.Tensor | None:
    """Convert HWC RGB to normalized grayscale."""
    if rgb is None:
        return None
    rgb_f = rgb.to(dtype=torch.float32)
    if rgb_f.max() > 2.0:
        rgb_f = rgb_f / 255.0
    return (
        0.299 * rgb_f[..., 0]
        + 0.587 * rgb_f[..., 1]
        + 0.114 * rgb_f[..., 2]
    ).to(dtype=rgb.dtype if rgb.dtype.is_floating_point else torch.float32)


def _rgb_to_uint8_cpu(rgb: torch.Tensor | None):
    """Return a CPU uint8 RGB image for low-rate OpenCV feature correction."""
    if rgb is None:
        return None
    rgb_cpu = rgb.detach().cpu()
    if rgb_cpu.dtype.is_floating_point:
        if float(rgb_cpu.max()) <= 2.0:
            rgb_cpu = rgb_cpu * 255.0
        rgb_cpu = rgb_cpu.clamp(0, 255).to(torch.uint8)
    else:
        rgb_cpu = rgb_cpu.to(torch.uint8)
    return rgb_cpu.numpy()
