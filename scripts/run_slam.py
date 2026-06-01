#!/usr/bin/env python3
"""Unified RGBDTSDFSLAM runner with trajectory evaluation.

Supports TUM, Replica (NICE-SLAM), ScanNet, and normalized RealSense captures.
Automatically evaluates ATE/RPE against ground truth when available.

Usage examples:
  # TUM freiburg1_desk
  python scripts/run_slam.py tum \\
      --dataset-root /workspace/datasets/public/TUM/tum_rgbd \\
      --sequence freiburg1_desk

  # Replica room0
  python scripts/run_slam.py replica \\
      --scene-dir /workspace/datasets/public/Replica-NICE-SLAM/Replica/room0

  # ScanNet scene0011_00
  python scripts/run_slam.py scannet \\
      --scene-dir /workspace/datasets/public/ScanNet/scans/scene0011_00

  # Normalized RealSense capture
  python scripts/run_slam.py normalized \\
      --capture-dir /workspace/datasets/bjoern/realsense_handheld/my_capture
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gradslam.backend import accelerator_backend, backend_report, default_device
from gradslam.evaluation import (
    ATEResult,
    RPEResult,
    associate_by_index,
    associate_poses,
    compute_ate,
    compute_rpe,
    evaluate_tum,
    load_tum_poses,
    load_replica_traj,
)
from gradslam.slam import RGBDTSDFSLAM
from gradslam.slam.pipeline import RGBDFrame

_STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\nReceived signal {signum}; finishing the current frame and saving partial results.")


def _cleanup_accelerator(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Module-level collate / worker-init (must be at module level for pickling)
# ---------------------------------------------------------------------------


def _passthrough_collate(batch):
    return batch[0]


def _worker_init(worker_id):
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------


def load_tum_dataset(args):
    from gradslam.datasets import TUM

    root = Path(args.dataset_root)
    seq = args.sequence
    tum_rgbd_subdir = root / seq / f"rgbd_dataset_{seq}"
    if tum_rgbd_subdir.exists():
        basedir = str(root / seq)
        seq_name = f"rgbd_dataset_{seq}"
    else:
        basedir = str(root)
        seq_name = seq

    dataset = TUM(
        basedir=basedir,
        sequences=(seq_name,),
        seqlen=1,
        dilation=0,
        stride=args.stride,
        return_pose=False,
        return_transform=False,
        intrinsics_mode=getattr(args, "tum_intrinsics", "registered"),
    )
    # GT pose file
    gt_file = Path(basedir) / seq_name / "groundtruth.txt"
    gt_file = str(gt_file) if gt_file.exists() else None
    return dataset, gt_file, "tum"


def load_replica_dataset(args):
    from gradslam.datasets.replica import ReplicaNICESLAM

    dataset = ReplicaNICESLAM(
        basedir=args.scene_dir,
        seqlen=1,
        stride=args.stride,
    )
    gt_file = str(Path(args.scene_dir) / "traj.txt")
    return dataset, gt_file, "replica"


def load_scannet_dataset(args):
    from gradslam.datasets.scannet_simple import ScanNetSimple

    dataset = ScanNetSimple(
        scene_dir=args.scene_dir,
        seqlen=1,
        stride=args.stride,
    )
    return dataset, None, "scannet"


def load_normalized_dataset(args):
    from gradslam.datasets.normalized import NormalizedRGBD

    dataset = NormalizedRGBD(
        capture_dir=args.capture_dir,
        seqlen=1,
        stride=args.stride,
        gt_file=args.gt_file if hasattr(args, "gt_file") else None,
        return_pose=False,
        return_transform=False,
    )
    gt_file = str(dataset._resolve_gt_file(getattr(args, "gt_file", None)) or "")
    gt_file = gt_file or None
    return dataset, gt_file, "normalized"


# ---------------------------------------------------------------------------
# Frame extraction helpers (dataset-specific tensor shapes)
# ---------------------------------------------------------------------------


def extract_frame_tum(sample, device):
    colors = sample[0][0].to(device, non_blocking=True)     # (H, W, 3)
    depths = sample[1][0].to(device, non_blocking=True)     # (H, W, 1)
    intrinsics = sample[2][0].to(device, non_blocking=True) # (4, 4) or (3,3)
    gt_pose = None
    ts = None
    if len(sample) > 4:
        ts_str = sample[-1]
        try:
            parts = str(ts_str).split()
            # Format: "rgb <ts> depth <ts> pose <ts>". The estimated pose is
            # depth-driven, so use depth timestamp first; GT pose timestamps
            # are only for evaluation association and must not sparse-sample
            # the tracking stream.
            if "depth" in parts:
                ts = float(parts[parts.index("depth") + 1])
            elif "pose" in parts and parts[parts.index("pose") + 1] != "None":
                ts = float(parts[parts.index("pose") + 1])
            elif len(parts) >= 2:
                ts = float(parts[1])
        except Exception:
            ts = None
    return colors, depths.squeeze(-1), intrinsics, gt_pose, ts


def extract_frame_replica(sample, device):
    colors = sample[0][0].to(device, non_blocking=True)
    depths = sample[1][0].squeeze(-1).to(device, non_blocking=True) if len(sample) > 1 and torch.is_tensor(sample[1]) else None
    if depths is None:
        # depths is second element if return_depth=True
        depths = sample[1][0].to(device, non_blocking=True)
        if depths.ndim == 3:
            depths = depths.squeeze(-1)
    intrinsics = sample[2][0].to(device, non_blocking=True)
    gt_pose = sample[3][0].numpy() if len(sample) > 3 else None
    return colors, depths, intrinsics, gt_pose, None


def extract_frame_generic(sample, device):
    """Generic frame extractor: colors, depths, intrinsics are first 3 items.

    NormalizedRGBD returns: (colors, depths, intrinsics, [poses], [transforms], [names], [timestamps])
    Only poses are tensor objects that can be converted to numpy. Names and timestamps are lists/floats.
    """
    colors = sample[0][0].to(device, non_blocking=True)
    depths = sample[1][0].to(device, non_blocking=True) if len(sample) > 1 else None
    if depths is not None and depths.ndim == 3:
        depths = depths.squeeze(-1)
    intrinsics = sample[2][0].to(device, non_blocking=True) if len(sample) > 2 else None

    gt_pose = None
    ts = None
    if len(sample) > 3:
        tail = sample[-1]
        if isinstance(tail, (list, tuple)) and tail:
            try:
                ts = float(tail[0])
            except (TypeError, ValueError):
                ts = None
        else:
            try:
                ts = float(tail)
            except (TypeError, ValueError):
                ts = None

    return colors, depths, intrinsics, gt_pose, ts


def _timestamp_gap_stats(tracking_log: list[dict]) -> dict[str, float | int]:
    ts = [float(r["ts"]) for r in tracking_log if r.get("ts") is not None]
    if len(ts) < 2:
        return {"max_frame_gap_s": 0.0, "frame_gaps_over_0_2s": 0}
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    return {
        "max_frame_gap_s": float(max(gaps)),
        "frame_gaps_over_0_2s": int(sum(g > 0.2 for g in gaps)),
    }


# ---------------------------------------------------------------------------
# GT loading helpers
# ---------------------------------------------------------------------------


def load_gt_poses_tum(gt_file: str) -> dict[float, np.ndarray]:
    return load_tum_poses(gt_file)


def load_gt_poses_replica(gt_file: str) -> list[np.ndarray]:
    return load_replica_traj(gt_file)


def load_gt_poses_auto(gt_file: str, dataset_type: str):
    if dataset_type == "replica":
        return load_gt_poses_replica(gt_file), "indexed"
    else:
        try:
            poses = load_tum_poses(gt_file)
            return poses, "tum"
        except Exception:
            return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description="RGBDTSDFSLAM runner with evaluation")
    sub = parser.add_subparsers(dest="dataset_type", required=True)

    # --- TUM ---
    p_tum = sub.add_parser("tum", help="TUM RGB-D dataset")
    p_tum.add_argument("--dataset-root", default="/workspace/datasets/public/TUM/tum_rgbd")
    p_tum.add_argument("--sequence", default="freiburg1_desk")
    p_tum.add_argument("--tum-intrinsics", choices=["registered", "official"], default="registered",
                       help="TUM intrinsics convention: registered benchmark default or official camera calibration")

    # --- Replica ---
    p_rep = sub.add_parser("replica", help="Replica NICE-SLAM dataset")
    p_rep.add_argument("--scene-dir", required=True,
                       help="Path to scene dir (e.g. .../Replica/room0)")

    # --- ScanNet ---
    p_sn = sub.add_parser("scannet", help="ScanNet extracted scene")
    p_sn.add_argument("--scene-dir", required=True,
                      help="Path to scene dir (e.g. .../scans/scene0011_00)")

    # --- Normalized ---
    p_norm = sub.add_parser("normalized", help="Normalized RealSense/pseudo-GT capture")
    p_norm.add_argument("--capture-dir", required=True,
                        help="Path to normalized capture directory")
    p_norm.add_argument("--gt-file", default=None,
                        help="Override GT trajectory file (TUM format)")

    # --- Common ---
    # Per-dataset volume defaults: (voxel_size, dim, origin)
    _VOL_DEFAULTS = {
        "tum":        (0.02, [256, 256, 256], [-2.0, -2.0, 0.0], 0.5, 400_000),
        # Replica has 90° HFOV, up to 5m depth → need ~12m wide volume
        "replica":    (0.03, [384, 256, 384], [-5.8, -3.8, 0.0], 0.5, 400_000),
        "scannet":    (0.02, [256, 256, 256], [-2.0, -2.0, 0.0], 0.5, 400_000),
        "normalized": (0.02, [256, 256, 256], [-2.0, -2.0, 0.0], 0.25, 100_000),
    }

    for p, name in ((p_tum, "tum"), (p_rep, "replica"), (p_sn, "scannet"), (p_norm, "normalized")):
        vs, vd, vo, ps, max_pixels = _VOL_DEFAULTS[name]
        p.add_argument("--stride", type=int, default=1)
        p.add_argument("--max-frames", type=int, default=None)
        p.add_argument("--device", type=str, default=None)
        p.add_argument("--output", type=str, default=f"./outputs/{name}_slam_results")
        p.add_argument("--voxel-size", type=float, default=vs)
        p.add_argument("--volume-dim", type=int, nargs=3, default=vd)
        p.add_argument("--volume-origin", type=float, nargs=3, default=vo)
        p.add_argument("--no-eval", action="store_true", help="Skip trajectory evaluation")
        p.add_argument("--compile", action="store_true", help="Use torch.compile for faster execution")
        p.add_argument("--icp-iters", type=int, nargs="+", default=None,
                       metavar="N", help="ICP iterations per pyramid level (e.g. 6 3 2)")
        p.add_argument("--process-scale", type=float, default=ps,
                       help="Downsample factor applied to depth + intrinsics before SLAM (0.5 = half-res)")
        p.add_argument("--max-processed-pixels", type=int, default=max_pixels,
                       help="Refuse workloads above H*W*process_scale^2 pixels (0 disables guard)")
        p.add_argument("--autocast", choices=["off", "bf16", "fp16"], default="off",
                       help="Mixed precision autocast mode for raycast+ICP")
        p.add_argument("--raycast-normal-mode", choices=["gradient", "image"], default="gradient",
                       help="Raycast normal mode: gradient (TSDF-based) or image (faster, depth-derived)")
        p.add_argument("--slam-backend", choices=["rgbdtsdf", "fast_rgbd", "local_map", "icpslam", "pointfusion"],
                       default="fast_rgbd",
                       help="SLAM backend: current RGBDTSDFSLAM fast path or upstream-style ICPSLAM/PointFusion")
        p.add_argument("--tracking-mode", choices=["fast_rgbd", "local_map", "hybrid", "tsdf"], default="fast_rgbd",
                       help="Tracking core: local RGB-D keyframe tracker or TSDF-only tracker")
        p.add_argument("--upstream-odom", choices=["gradicp", "icp"], default="gradicp",
                       help="Odometry provider for upstream-style ICPSLAM/PointFusion backends")
        p.add_argument("--upstream-dsratio", type=int, default=4,
                       help="Downsampling ratio for upstream-style ICPSLAM/PointFusion")
        p.add_argument("--upstream-iters", type=int, default=20,
                       help="ICP iterations for upstream-style ICPSLAM/PointFusion")
        p.add_argument("--allow-slow-upstream", action="store_true",
                       help="Allow ICPSLAM/PointFusion diagnostics beyond the protected frame cap")
        p.add_argument("--enable-mapping", action="store_true",
                       help="Enable TSDF mapping/integration for fast_rgbd mode")
        p.add_argument("--max-keyframes", type=int, default=8,
                       help="Number of local RGB-D keyframes retained by hybrid tracking")
        p.add_argument("--mapping-interval", type=int, default=5,
                       help="Integrate non-keyframe poses every N frames in hybrid mode")
        p.add_argument("--feature-interval", type=int, default=0,
                       help="Run low-rate ORB+PnP feature correction every N frames in fast_rgbd (0 = off)")
        p.add_argument("--keyframe-tracking-interval", type=int, default=0,
                       help="Add a local keyframe tracking candidate every N fast_rgbd frames (0 = off)")
        p.add_argument("--local-map-candidates", type=int, default=1,
                       help="Nearby keyframes tested every frame by --slam-backend local_map")
        p.add_argument("--max-frame-translation", type=float, default=0.12,
                       help="Reject frame-to-frame tracking updates above this translation in meters (<=0 disables)")
        p.add_argument("--max-frame-rotation-deg", type=float, default=30.0,
                       help="Reject frame-to-frame tracking updates above this rotation in degrees (<=0 disables)")
        p.add_argument("--tracking-warmup-frames", type=int, default=10,
                       help="Warmup frames excluded from reported tracking FPS")
        p.add_argument("--min-track-inliers", type=int, default=100,
                       help="Minimum valid correspondences before tracking is marked lost")
        p.add_argument("--lost-inlier-ratio", type=float, default=0.02,
                       help="Minimum inlier ratio before tracking is marked lost")
        p.add_argument("--borderline-inlier-ratio", type=float, default=0.08,
                       help="Try TSDF fallback when local tracking is below this inlier ratio")
        p.add_argument("--robust-loss", choices=["none", "huber", "tukey"], default="huber",
                       help="Robust loss used by projective ICP")
        p.add_argument("--huber-delta", type=float, default=0.03,
                       help="Huber delta for geometric residuals in projective ICP (meters)")
        p.add_argument("--max-depth-diff", type=float, default=0.14,
                       help="Maximum projective correspondence depth difference in meters (used when --no-adaptive-depth-diff is set)")
        p.add_argument("--no-adaptive-depth-diff", action="store_true",
                       help="Disable adaptive depth difference threshold (use constant --max-depth-diff)")
        p.add_argument("--max-normal-angle-deg", type=float, default=75.0,
                       help="Maximum projective correspondence normal angle in degrees")
        p.add_argument("--no-depth-weighting", action="store_true",
                       help="Disable depth uncertainty weighting in projective ICP")
        p.add_argument("--num-workers", type=int, default=4,
                       help="DataLoader workers for async frame loading (0 = main "
                            "thread / synchronous). Default 4: PNG decode + H2D "
                            "overlaps with GPU tracking, lifting end-to-end fps "
                            "toward tracking fps. Set 0 to disable.")
        p.add_argument("--prefetch-factor", type=int, default=2,
                       help="DataLoader prefetch factor (only used when --num-workers > 0)")
        p.add_argument("--no-pin-memory", action="store_true",
                       help="Disable pinned memory in DataLoader")
        p.add_argument("--candidate-disagreement-penalty", type=float, default=1.0,
                       help="Lambda for penalizing candidate motion disagreement vs velocity prediction")
        p.add_argument("--scale-veto-ratio", type=float, default=3.0,
                       help="Veto winner if t > this ratio * t_predicted and a more consistent candidate exists")
        p.add_argument("--photometric-weight", type=float, default=0.25,
                       help="Weight of photometric vs geometric residuals (0 = disabled)")
        p.add_argument("--photometric-levels", type=int, default=1,
                       help="Apply photometric residuals at the finest N pyramid levels")
        p.add_argument("--no-photometric", action="store_true",
                       help="Disable photometric residuals entirely")
        p.add_argument("--max-velocity-translation", type=float, default=0.18,
                       help="Maximum predicted translation (m) before velocity model is rejected")
        p.add_argument("--max-velocity-rotation", type=float, default=0.30,
                       help="Maximum predicted rotation (rad) before velocity model is rejected")
        p.add_argument("--pose-graph", choices=["off", "on"], default="off",
                       help="Enable sliding-window pose graph optimization after keyframe insertion")
        p.add_argument("--pose-graph-window", type=int, default=8,
                       help="Sliding-window pose graph window size (number of keyframes)")
        p.add_argument("--relocalization", choices=["off", "on"], default="off",
                       help="Enable ORB-based relocalization after lost-frame stretches")
        p.add_argument("--loop-closure", choices=["off", "on"], default="off",
                       help="Enable loop closure detection on keyframe insertion")
        p.add_argument("--loop-closure-min-inliers", type=int, default=30,
                       help="Minimum ORB match inliers to trigger a loop closure edge")

    return parser


def _validate_workload(args, dataset) -> None:
    scale = float(getattr(args, "process_scale", 1.0))
    max_pixels = int(getattr(args, "max_processed_pixels", 0) or 0)
    height = int(getattr(dataset, "height", 0) or 0)
    width = int(getattr(dataset, "width", 0) or 0)
    if max_pixels <= 0 or height <= 0 or width <= 0:
        return
    processed_pixels = int(height * width * scale * scale)
    if processed_pixels > max_pixels:
        raise RuntimeError(
            f"Refusing workload {processed_pixels} processed pixels/frame "
            f"({width}x{height} at process_scale={scale}) above guard {max_pixels}. "
            "Lower --process-scale or pass --max-processed-pixels 0 to override."
        )


def run_slam(args, dataset, extractor, device):
    slam_backend = getattr(args, "slam_backend", "fast_rgbd")
    if slam_backend in ("icpslam", "pointfusion"):
        return run_upstream_slam(args, dataset, extractor, device, slam_backend)

    from gradslam.mapping.tsdf import TSDFConfig
    from gradslam.icp.projective import ProjectiveICPConfig

    tsdf_cfg = TSDFConfig(voxel_size=args.voxel_size)

    tracking_mode = getattr(args, 'tracking_mode', 'fast_rgbd')
    if slam_backend == "rgbdtsdf":
        tracking_mode = "hybrid"
    elif slam_backend == "local_map":
        tracking_mode = "local_map"
    _photo_weight = (
        0.0 if getattr(args, 'no_photometric', False)
        else getattr(args, 'photometric_weight', 0.25)
    )
    _photo_levels = getattr(args, 'photometric_levels', 1)

    icp_cfg = None
    if getattr(args, 'icp_iters', None):
        icp_cfg = ProjectiveICPConfig(
            n_pyramid_levels=len(args.icp_iters),
            iterations=tuple(args.icp_iters),
            damping=tuple(1e-2 / (10 ** i) for i in range(len(args.icp_iters))),
            max_depth_diff=getattr(args, 'max_depth_diff', 0.14),
            max_normal_angle_deg=getattr(args, 'max_normal_angle_deg', 75.0),
            robust_loss=getattr(args, 'robust_loss', 'huber'),
            huber_delta=getattr(args, 'huber_delta', 0.03),
            depth_weighting=not getattr(args, 'no_depth_weighting', False),
            adaptive_depth_diff=not getattr(args, 'no_adaptive_depth_diff', False),
            photometric_weight=_photo_weight,
            photometric_levels=_photo_levels,
        )
    elif tracking_mode in ('fast_rgbd', 'local_map'):
        icp_cfg = ProjectiveICPConfig(
            n_pyramid_levels=2,
            iterations=(5, 3),
            damping=(1e-2, 1e-3),
            max_depth_diff=getattr(args, 'max_depth_diff', 0.14),
            max_normal_angle_deg=getattr(args, 'max_normal_angle_deg', 75.0),
            robust_loss=getattr(args, 'robust_loss', 'huber'),
            huber_delta=getattr(args, 'huber_delta', 0.03),
            depth_weighting=not getattr(args, 'no_depth_weighting', False),
            adaptive_depth_diff=not getattr(args, 'no_adaptive_depth_diff', False),
            photometric_weight=_photo_weight,
            photometric_levels=_photo_levels,
        )
    else:
        icp_cfg = ProjectiveICPConfig(
            max_depth_diff=getattr(args, 'max_depth_diff', 0.14),
            max_normal_angle_deg=getattr(args, 'max_normal_angle_deg', 75.0),
            robust_loss=getattr(args, 'robust_loss', 'huber'),
            huber_delta=getattr(args, 'huber_delta', 0.03),
            depth_weighting=not getattr(args, 'no_depth_weighting', False),
            adaptive_depth_diff=not getattr(args, 'no_adaptive_depth_diff', False),
            photometric_weight=_photo_weight,
            photometric_levels=_photo_levels,
        )

    autocast_dtype = None if getattr(args, 'autocast', 'off') == 'off' else args.autocast

    slam = RGBDTSDFSLAM(
        tsdf_config=tsdf_cfg,
        icp_config=icp_cfg,
        voxel_dim=tuple(args.volume_dim),
        volume_origin=tuple(args.volume_origin),
        process_scale=getattr(args, 'process_scale', 1.0),
        raycast_normal_mode=getattr(args, 'raycast_normal_mode', 'gradient'),
        autocast_dtype=autocast_dtype,
        tracking_mode=tracking_mode,
        max_keyframes=getattr(args, 'max_keyframes', 8),
        mapping_interval=getattr(args, 'mapping_interval', 5),
        enable_mapping=getattr(args, 'enable_mapping', False) or tracking_mode not in ('fast_rgbd', 'local_map'),
        feature_interval=getattr(args, 'feature_interval', 5),
        keyframe_tracking_interval=getattr(args, 'keyframe_tracking_interval', 10),
        min_track_inliers=getattr(args, 'min_track_inliers', 100),
        lost_inlier_ratio_thresh=getattr(args, 'lost_inlier_ratio', 0.02),
        borderline_inlier_ratio=getattr(args, 'borderline_inlier_ratio', 0.08),
        local_map_candidates=getattr(args, 'local_map_candidates', 1),
        max_frame_translation=getattr(args, 'max_frame_translation', 0.12),
        max_frame_rotation_deg=getattr(args, 'max_frame_rotation_deg', 30.0),
        candidate_disagreement_penalty=getattr(args, 'candidate_disagreement_penalty', 1.0),
        scale_veto_ratio=getattr(args, 'scale_veto_ratio', 3.0),
        max_velocity_translation=getattr(args, 'max_velocity_translation', 0.18),
        max_velocity_rotation=getattr(args, 'max_velocity_rotation', 0.30),
        pose_graph_enabled=getattr(args, 'pose_graph', 'off') == 'on',
        pose_graph_window=getattr(args, 'pose_graph_window', 8),
        relocalization_enabled=getattr(args, 'relocalization', 'off') == 'on',
        loop_closure_enabled=getattr(args, 'loop_closure', 'off') == 'on',
        loop_closure_min_inliers=getattr(args, 'loop_closure_min_inliers', 30),
    ).to(device)

    # torch.compile: compiling the whole SLAM module graph-breaks on the
    # pipeline's Python control flow / .item() / cv2 calls, so it bought little.
    # Instead, --compile enables GRADSLAM_COMPILE so the shape-stable leaf
    # kernels (ICP residual + 6x6 solver) compile via compile_if_requested.
    if getattr(args, 'compile', False):
        os.environ["GRADSLAM_COMPILE"] = "1"
        print("torch.compile enabled for leaf kernels (GRADSLAM_COMPILE=1). "
              "Note: must be set before importing gradslam to take effect.")

    n_frames = min(args.max_frames or len(dataset), len(dataset))
    poses_est = []
    gt_poses_by_ts: list | dict = []
    tracking_log = []
    tracking_times_ms = []
    warmup_frames = getattr(args, 'tracking_warmup_frames', 10)

    num_workers = getattr(args, 'num_workers', 0)
    pin_memory = not getattr(args, 'no_pin_memory', False) and num_workers > 0
    prefetch_factor = getattr(args, 'prefetch_factor', 2)

    if num_workers > 0:
        from torch.utils.data import DataLoader, Subset
        subset = Subset(dataset, list(range(n_frames)))
        loader = DataLoader(
            subset,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
            collate_fn=_passthrough_collate,
            worker_init_fn=_worker_init,
        )
        data_iter = enumerate(loader)
    else:
        loader = None
        data_iter = ((idx, dataset[idx]) for idx in range(n_frames))

    # --- Startup sanity diagnostics (peek at frame 0 without consuming iter) ---
    try:
        from gradslam.slam.diagnostics import log_startup_sanity
        _first_sample = dataset[0]
        _colors0, _depths0, _intrinsics0, _gt_pose0, _ts0 = extractor(_first_sample, device)
        if _depths0 is not None and _intrinsics0 is not None:
            _process_scale = float(getattr(args, 'process_scale', 1.0))
            _label = (getattr(args, 'sequence', None) or getattr(args, 'scene_dir', None)
                      or getattr(args, 'capture_dir', None) or '')
            if isinstance(_label, str):
                _label = _label.rstrip('/').split('/')[-1]
            log_startup_sanity(
                first_frame_depth=_depths0,
                intrinsics=_intrinsics0,
                process_scale=_process_scale,
                rgb_ts=None,
                depth_ts=None,
                gt_poses=None,
                label=str(_label),
            )
    except Exception as _diag_exc:
        print(f"[diagnostics] skipped: {_diag_exc}")

    start_time = time.time()
    with torch.no_grad():
        for idx, sample in tqdm(data_iter, total=n_frames, desc="SLAM"):
            if _STOP_REQUESTED:
                break
            colors, depths, intrinsics, gt_pose, ts = extractor(sample, device)

            if depths is None or intrinsics is None:
                continue

            frame = RGBDFrame(rgb=colors, depth=depths, intrinsics=intrinsics)
            use_cuda_events = device.type == "cuda" and torch.cuda.is_available()
            if use_cuda_events:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
            else:
                start_wall = time.perf_counter()
            result = slam.process_frame(frame)
            if use_cuda_events:
                end_evt.record()
                torch.cuda.synchronize(device)
                track_ms = start_evt.elapsed_time(end_evt)
            else:
                track_ms = (time.perf_counter() - start_wall) * 1000.0
            tracking_times_ms.append(track_ms)

            pose_np = result.T_world_camera.cpu().numpy()
            poses_est.append(pose_np)
            if gt_pose is not None:
                gt_poses_by_ts.append((ts, gt_pose))

            q = result.quality
            tracking_log.append({
                "idx": idx, "ts": ts,
                "num_valid": q.get("num_valid", 0),
                "inlier_ratio": q.get("inlier_ratio", 0.0),
                "rmse": q.get("rmse", 0.0),
                "source": q.get("tracking_source", "unknown"),
                "integrated": q.get("integrated", False),
                "photometric_mean_abs": q.get("photometric_mean_abs", -1.0),
                "feature_inliers": q.get("feature_inliers", 0),
                "frame_translation": q.get("frame_translation", -1.0),
                "frame_rotation_deg": q.get("frame_rotation_deg", -1.0),
                "motion_gate": q.get("motion_gate", True),
                "reference_frame_idx": q.get("reference_frame_idx", -1),
                "t_disagreement_norm": q.get("t_disagreement_norm", -1.0),
                "tracking_ms": track_ms,
                "lost": result.lost,
                "candidates_json": __import__('json').dumps(q.get("candidates", [])),
                "tracking_state": q.get("tracking_state", "ok"),
                "map_update_allowed": q.get("map_update_allowed", True),
            })

    elapsed = time.time() - start_time
    fps = n_frames / elapsed
    timed = tracking_times_ms[min(warmup_frames, len(tracking_times_ms)):]
    tracking_fps = 1000.0 / (sum(timed) / len(timed)) if timed else 0.0
    return poses_est, gt_poses_by_ts, tracking_log, elapsed, fps, tracking_fps


def run_upstream_slam(args, dataset, extractor, device, backend: str):
    from gradslam.slam import ICPSLAM, PointFusion
    from gradslam.structures.pointclouds import Pointclouds
    from gradslam.structures.rgbdimages import RGBDImages

    n_frames = min(args.max_frames or len(dataset), len(dataset))
    protected_cap = 30
    if n_frames > protected_cap and not getattr(args, "allow_slow_upstream", False):
        raise RuntimeError(
            f"{backend} is an upstream accuracy diagnostic with a growing point map; "
            f"refusing {n_frames} frames without --allow-slow-upstream. "
            f"Use --max-frames {protected_cap} for a bounded comparison."
        )

    if backend == "icpslam":
        slam = ICPSLAM(
            odom=args.upstream_odom,
            dsratio=args.upstream_dsratio,
            numiters=args.upstream_iters,
            device=device,
        )
    else:
        slam = PointFusion(
            odom=args.upstream_odom,
            dsratio=args.upstream_dsratio,
            numiters=args.upstream_iters,
            device=device,
        )

    pointclouds = Pointclouds(device=device)
    prev_frame = None
    poses_est = []
    tracking_log = []
    tracking_times_ms = []
    warmup_frames = getattr(args, "tracking_warmup_frames", 10)
    current_pose = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4)

    data_iter = ((idx, dataset[idx]) for idx in range(n_frames))
    start_time = time.time()
    with torch.no_grad():
        for idx, sample in tqdm(data_iter, total=n_frames, desc=backend):
            if _STOP_REQUESTED:
                break
            colors, depths, intrinsics, _gt_pose, ts = extractor(sample, device)
            if depths is None or intrinsics is None:
                continue
            rgb = colors.unsqueeze(0).unsqueeze(0).float()
            depth = depths.unsqueeze(0).unsqueeze(0).unsqueeze(-1).float()
            K = intrinsics.unsqueeze(0).unsqueeze(0).float()
            frame = RGBDImages(rgb, depth, K, current_pose.clone(), device=device)

            use_cuda_events = device.type == "cuda" and torch.cuda.is_available()
            if use_cuda_events:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
            else:
                start_wall = time.perf_counter()
            pointclouds, pose = slam.step(pointclouds, frame, prev_frame, inplace=True)
            if use_cuda_events:
                end_evt.record()
                torch.cuda.synchronize(device)
                track_ms = start_evt.elapsed_time(end_evt)
            else:
                track_ms = (time.perf_counter() - start_wall) * 1000.0

            current_pose = pose.detach()
            prev_frame = frame
            prev_frame.poses = current_pose
            poses_est.append(current_pose[0, 0].detach().cpu().numpy())
            tracking_times_ms.append(track_ms)
            tracking_log.append({
                "idx": idx, "ts": ts,
                "num_valid": 0,
                "inlier_ratio": 0.0,
                "rmse": 0.0,
                "source": backend,
                "integrated": True,
                "photometric_mean_abs": -1.0,
                "feature_inliers": 0,
                "tracking_ms": track_ms,
                "lost": False,
            })

    elapsed = time.time() - start_time
    fps = n_frames / elapsed
    timed = tracking_times_ms[min(warmup_frames, len(tracking_times_ms)):]
    tracking_fps = 1000.0 / (sum(timed) / len(timed)) if timed else 0.0
    return poses_est, [], tracking_log, elapsed, fps, tracking_fps


def save_results(output_dir: Path, poses_est, tracking_log, dataset_type,
                 gt_file=None, gt_format=None, gt_poses_inline=None,
                 no_eval=False, tracking_fps=None, elapsed=None, args=None):
    import csv
    import json
    from scipy.spatial.transform import Rotation

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper: write TUM-format pose lines to an open file handle
    # ------------------------------------------------------------------
    def _write_tum_poses(fh):
        for i, T in enumerate(poses_est):
            ts = tracking_log[i].get("ts") if i < len(tracking_log) else None
            ts_out = float(ts) if ts is not None else float(i)
            t = T[:3, 3]
            R = T[:3, :3]
            q = Rotation.from_matrix(R).as_quat()  # qx qy qz qw
            fh.write(f"{ts_out:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                     f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    # Save estimated poses (TUM format) — keep existing filename
    est_file = output_dir / "estimated_poses.txt"
    with open(est_file, "w") as f:
        _write_tum_poses(f)
    print(f"✓ Estimated poses → {est_file}")

    # Also write trajectory.txt (same content, canonical name)
    traj_file = output_dir / "trajectory.txt"
    with open(traj_file, "w") as f:
        _write_tum_poses(f)
    print(f"✓ Trajectory      → {traj_file}")

    # Save tracking metrics (existing space-separated format)
    metrics_file = output_dir / "tracking_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(
            "idx num_valid inlier_ratio rmse source integrated photometric_mean_abs "
            "feature_inliers frame_translation frame_rotation_deg motion_gate "
            "reference_frame_idx t_disagreement_norm tracking_ms lost\n"
        )
        for r in tracking_log:
            f.write(
                f"{r['idx']:6d} {r['num_valid']:8d} {r['inlier_ratio']:.4f} "
                f"{r['rmse']:.6f} {r['source']} {int(r['integrated'])} "
                f"{r['photometric_mean_abs']:.6f} {r['feature_inliers']:4d} "
                f"{r.get('frame_translation', -1.0):.6f} "
                f"{r.get('frame_rotation_deg', -1.0):.6f} "
                f"{int(r.get('motion_gate', True))} "
                f"{int(r.get('reference_frame_idx', -1)):6d} "
                f"{r.get('t_disagreement_norm', -1.0):.6f} "
                f"{r['tracking_ms']:.3f} {int(r['lost'])}\n"
            )
    print(f"✓ Tracking metrics → {metrics_file}")

    # Save tracking_debug.csv (CSV version of per-frame data)
    csv_file = output_dir / "tracking_debug.csv"
    _CSV_FIELDS = [
        "idx", "num_valid", "inlier_ratio", "rmse", "source", "integrated",
        "photometric_mean_abs", "feature_inliers", "frame_translation",
        "frame_rotation_deg", "motion_gate", "reference_frame_idx",
        "t_disagreement_norm", "tracking_ms", "lost", "candidates_json",
        "tracking_state", "map_update_allowed",
    ]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in tracking_log:
            row = {
                "idx": r["idx"],
                "num_valid": r["num_valid"],
                "inlier_ratio": r["inlier_ratio"],
                "rmse": r["rmse"],
                "source": r["source"],
                "integrated": int(r["integrated"]),
                "photometric_mean_abs": r["photometric_mean_abs"],
                "feature_inliers": r["feature_inliers"],
                "frame_translation": r.get("frame_translation", -1.0),
                "frame_rotation_deg": r.get("frame_rotation_deg", -1.0),
                "motion_gate": int(r.get("motion_gate", True)),
                "reference_frame_idx": r.get("reference_frame_idx", -1),
                "t_disagreement_norm": r.get("t_disagreement_norm", -1.0),
                "tracking_ms": r["tracking_ms"],
                "lost": int(r["lost"]),
                "candidates_json": r.get("candidates_json", "[]"),
                "tracking_state": r.get("tracking_state", "ok"),
                "map_update_allowed": int(r.get("map_update_allowed", True)),
            }
            writer.writerow(row)
    print(f"✓ Tracking debug CSV → {csv_file}")

    summary_file = output_dir / "tracking_summary.txt"
    lost_count = sum(1 for r in tracking_log if r["lost"])
    with open(summary_file, "w") as f:
        f.write(f"frames {len(tracking_log)}\n")
        if elapsed is not None:
            f.write(f"elapsed_s {elapsed:.6f}\n")
            f.write(f"end_to_end_fps {len(tracking_log) / elapsed:.6f}\n")
        if tracking_fps is not None:
            f.write(f"tracking_fps_warmup_excluded {tracking_fps:.6f}\n")
        f.write(f"lost_count {lost_count}\n")
        for key, value in _timestamp_gap_stats(tracking_log).items():
            f.write(f"{key} {value}\n")
    print(f"✓ Tracking summary → {summary_file}")

    # ------------------------------------------------------------------
    # Compute summary statistics for metrics.json (filter -1.0 sentinels)
    # ------------------------------------------------------------------
    def _mean_valid(key, sentinel=-1.0):
        vals = [r.get(key, sentinel) for r in tracking_log]
        valid = [v for v in vals if v != sentinel]
        return float(sum(valid) / len(valid)) if valid else None

    n_keyframes = sum(
        1 for r in tracking_log
        if "kf" in str(r.get("source", "")).lower() or r.get("source") == "keyframe"
        or r.get("integrated", False)
    )

    end_to_end_fps = (len(tracking_log) / elapsed) if elapsed else None

    eval_metrics = None  # will be filled below if eval runs

    if not no_eval:
        if gt_file and Path(gt_file).exists():
            print(f"\nEvaluating against GT: {gt_file}")
            eval_metrics = _evaluate_and_print(
                poses_est, gt_file, gt_format, output_dir, tracking_log
            )
        elif gt_poses_inline:
            eval_metrics = _evaluate_inline(poses_est, gt_poses_inline, output_dir)

    # ------------------------------------------------------------------
    # Write metrics.json (always, even if no_eval=True)
    # ------------------------------------------------------------------
    sequence_name = getattr(args, "sequence", None)
    if sequence_name is None and args is not None:
        # Derive from scene_dir or capture_dir if available
        scene_dir = getattr(args, "scene_dir", None) or getattr(args, "capture_dir", None)
        if scene_dir:
            sequence_name = Path(scene_dir).name

    metrics_json = {
        "n_frames": len(tracking_log),
        "lost_frames": lost_count,
        "n_keyframes": n_keyframes,
        "mean_inlier_ratio": _mean_valid("inlier_ratio", sentinel=-1.0),
        "mean_depth_residual": _mean_valid("rmse", sentinel=-1.0),
        "mean_update_norm": _mean_valid("t_disagreement_norm", sentinel=-1.0),
        "tracking_fps": tracking_fps,
        "end_to_end_fps": end_to_end_fps,
        "ate_rmse_m": eval_metrics.get("ate_rmse_m") if eval_metrics else None,
        "ate_mean_m": eval_metrics.get("ate_mean_m") if eval_metrics else None,
        "rpe_rmse_m": eval_metrics.get("rpe_rmse_m") if eval_metrics else None,
        "rpe_rmse_deg": eval_metrics.get("rpe_rmse_deg") if eval_metrics else None,
        "sequence": sequence_name,
        "dataset": dataset_type,
    }
    json_file = output_dir / "metrics.json"
    with open(json_file, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"✓ Metrics JSON    → {json_file}")

    # ------------------------------------------------------------------
    # Debug plots
    # ------------------------------------------------------------------
    _save_debug_plots(
        output_dir,
        tracking_log,
        ate_per_frame=(eval_metrics or {}).get("ate_per_frame"),
    )

    # ------------------------------------------------------------------
    # Write config_resolved.yaml (best-effort)
    # ------------------------------------------------------------------
    if args is not None:
        config_file = output_dir / "config_resolved.yaml"
        try:
            import yaml
            config_data = vars(args)
            with open(config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
            print(f"✓ Config resolved → {config_file}")
        except ImportError:
            # YAML not available — fall back to JSON with stringified values
            try:
                import json as _json
                config_data = {}
                for k, v in vars(args).items():
                    try:
                        _json.dumps(v)
                        config_data[k] = v
                    except (TypeError, ValueError):
                        config_data[k] = str(v)
                config_file_json = output_dir / "config_resolved.json"
                with open(config_file_json, "w") as f:
                    _json.dump(config_data, f, indent=2)
                print(f"✓ Config resolved → {config_file_json} (YAML unavailable)")
            except Exception:
                pass  # best-effort: skip silently
        except Exception:
            pass  # best-effort: skip silently


STATE_COLORS = {"ok": "#2ecc71", "weak": "#e67e22", "lost": "#e74c3c", "unknown": "#95a5a6"}
_STATE_INT = {"ok": 2, "weak": 1, "lost": 0, "unknown": -1}


def _save_debug_plots(
    output_dir: Path,
    tracking_log: list[dict],
    ate_per_frame: list[float] | None = None,
) -> None:
    """Generate and save tracking debug plots to output_dir/tracking_plots.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping debug plots (install with pip install matplotlib)")
        return

    if not tracking_log:
        print("tracking_log is empty — skipping debug plots")
        return

    frames = [r["idx"] for r in tracking_log]
    inlier_ratios = [r.get("inlier_ratio", 0.0) for r in tracking_log]
    rmse_vals = [r.get("rmse", 0.0) for r in tracking_log]
    translations = [r.get("frame_translation", -1.0) for r in tracking_log]
    rotations = [r.get("frame_rotation_deg", -1.0) for r in tracking_log]
    states = [r.get("tracking_state", "unknown") for r in tracking_log]
    point_colors = [STATE_COLORS.get(s, STATE_COLORS["unknown"]) for s in states]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    ax = axes.flatten()

    # --- Panel 1: Inlier ratio over time ---
    ax[0].scatter(frames, inlier_ratios, c=point_colors, s=10, zorder=3)
    ax[0].axhline(0.15, color="green", linestyle="--", linewidth=1.0, label="ok (0.15)")
    ax[0].axhline(0.08, color="orange", linestyle="--", linewidth=1.0, label="weak (0.08)")
    ax[0].set_xlabel("Frame index")
    ax[0].set_ylabel("Inlier ratio")
    ax[0].set_title("Inlier ratio over time")
    ax[0].legend(fontsize=8)

    # --- Panel 2: ICP RMSE over time ---
    ax[1].scatter(frames, rmse_vals, c=point_colors, s=10, zorder=3)
    ax[1].set_xlabel("Frame index")
    ax[1].set_ylabel("RMSE (m)")
    ax[1].set_title("ICP RMSE over time")

    # --- Panel 3: Frame-to-frame translation ---
    valid_t_idx = [f for f, t in zip(frames, translations) if t != -1.0]
    valid_t_val = [t for t in translations if t != -1.0]
    if valid_t_idx:
        ax[2].plot(valid_t_idx, valid_t_val, linewidth=0.8, color="#3498db")
    ax[2].set_xlabel("Frame index")
    ax[2].set_ylabel("Translation (m)")
    ax[2].set_title("Frame-to-frame translation")

    # --- Panel 4: Frame-to-frame rotation ---
    valid_r_idx = [f for f, r in zip(frames, rotations) if r != -1.0]
    valid_r_val = [r for r in rotations if r != -1.0]
    if valid_r_idx:
        ax[3].plot(valid_r_idx, valid_r_val, linewidth=0.8, color="#9b59b6")
    ax[3].set_xlabel("Frame index")
    ax[3].set_ylabel("Rotation (deg)")
    ax[3].set_title("Frame-to-frame rotation")

    # --- Panel 5: Tracking state over time ---
    state_ints = [_STATE_INT.get(s, -1) for s in states]
    ax[4].step(frames, state_ints, where="post", linewidth=1.0, color="#2c3e50")
    ax[4].set_xlabel("Frame index")
    ax[4].set_ylabel("State")
    ax[4].set_yticks([0, 1, 2])
    ax[4].set_yticklabels(["lost", "weak", "ok"])
    ax[4].set_title("Tracking state over time")

    # --- Panel 6: ATE per frame (if available) ---
    if ate_per_frame is not None and len(ate_per_frame) > 0:
        ate_frames = list(range(len(ate_per_frame)))
        ax[5].plot(ate_frames, ate_per_frame, linewidth=0.8, color="#e74c3c")
        ax[5].set_xlabel("Frame index")
        ax[5].set_ylabel("ATE (m)")
        ax[5].set_title("ATE per frame")
    else:
        ax[5].text(
            0.5, 0.5,
            "ATE per-frame not available\n(requires GT)",
            ha="center", va="center",
            fontsize=10, color="#7f8c8d",
            transform=ax[5].transAxes,
        )
        ax[5].set_title("ATE per frame")
        ax[5].set_xticks([])
        ax[5].set_yticks([])

    fig.tight_layout()
    plot_path = output_dir / "tracking_plots.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Debug plots → {plot_path}")


def _evaluate_and_print(poses_est, gt_file, gt_format, output_dir, tracking_log=None):
    from gradslam.evaluation import (
        load_tum_poses, load_replica_traj,
        associate_by_index, associate_poses,
        compute_ate, compute_rpe,
    )

    if gt_format == "indexed":
        gt_list = load_replica_traj(gt_file)
        pairs = associate_by_index(poses_est, gt_list)
    else:
        gt_dict = load_tum_poses(gt_file)
        if tracking_log and any(r.get("ts") is not None for r in tracking_log):
            est_dict = {
                float(r["ts"]): p
                for r, p in zip(tracking_log, poses_est)
                if r.get("ts") is not None
            }
        else:
            est_dict = {float(i): p for i, p in enumerate(poses_est)}
        pairs = associate_poses(est_dict, gt_dict, max_dt=0.1)
        if len(pairs) < 2:
            # Fallback: index-based
            gt_list = [gt_dict[k] for k in sorted(gt_dict.keys())]
            pairs = associate_by_index(poses_est, gt_list)

    if len(pairs) < 2:
        print("  Warning: too few matched pairs for evaluation")
        return None

    ate = compute_ate(pairs)
    rpe1 = compute_rpe(pairs, delta=1)
    rpe10 = compute_rpe(pairs, delta=min(10, len(pairs) - 1))

    # Per-pair aligned ATE (translation error after Umeyama alignment) so the
    # debug plot's ATE-over-time panel can render instead of "(requires GT)".
    import numpy as _np
    est_t = _np.array([p[0][:3, 3] for p in pairs])
    gt_t = _np.array([p[1][:3, 3] for p in pairs])
    est_aligned = (ate.T_align[:3, :3] @ est_t.T).T + ate.T_align[:3, 3]
    ate_per_frame = _np.linalg.norm(est_aligned - gt_t, axis=1).tolist()

    print(f"\n{ate}")
    print(f"{rpe1}")
    print(f"{rpe10}")

    eval_file = output_dir / "evaluation.txt"
    with open(eval_file, "w") as f:
        f.write(f"gt_assoc_count {len(pairs)}\n")
        f.write(f"eval_assoc_ratio {len(pairs) / max(1, len(poses_est)):.6f}\n")
        f.write(f"{ate}\n{rpe1}\n{rpe10}\n")
    print(f"\n✓ Evaluation results → {eval_file}")

    return {
        "ate_rmse_m": ate.rmse,
        "ate_mean_m": ate.mean,
        "rpe_rmse_m": rpe1.rmse_t,
        "rpe_rmse_deg": rpe1.rmse_r,
        "ate_per_frame": ate_per_frame,
    }


def _evaluate_inline(poses_est, gt_poses_inline, output_dir):
    """Evaluate using GT poses collected during SLAM (with timestamps)."""
    from gradslam.evaluation import associate_by_index, compute_ate, compute_rpe

    gt_list = [p for _, p in gt_poses_inline]
    pairs = associate_by_index(poses_est, gt_list)
    if len(pairs) < 2:
        return None
    ate = compute_ate(pairs)
    rpe1 = compute_rpe(pairs, delta=1)
    print(f"\n{ate}")
    print(f"{rpe1}")
    rpe10 = compute_rpe(pairs, delta=min(10, len(pairs) - 1))
    print(f"{rpe10}")
    eval_file = output_dir / "evaluation.txt"
    with open(eval_file, "w") as f:
        f.write(f"{ate}\n{rpe1}\n{rpe10}\n")
    print(f"\n✓ Evaluation results → {eval_file}")

    return {
        "ate_rmse_m": ate.rmse,
        "ate_mean_m": ate.mean,
        "rpe_rmse_m": rpe1.rmse_t,
        "rpe_rmse_deg": rpe1.rmse_r,
    }


def main():
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else default_device()

    print(f"\n{'=' * 70}")
    print(f"RGBDTSDFSLAM — {args.dataset_type.upper()}")
    print(f"{'=' * 70}\n")
    print(backend_report())
    print(f"Device: {device}\n")

    # Load dataset
    if args.dataset_type == "tum":
        dataset, gt_file, ds_type = load_tum_dataset(args)
        extractor = extract_frame_tum
        gt_format = "tum"
    elif args.dataset_type == "replica":
        dataset, gt_file, ds_type = load_replica_dataset(args)
        extractor = extract_frame_replica
        gt_format = "indexed"
    elif args.dataset_type == "scannet":
        dataset, gt_file, ds_type = load_scannet_dataset(args)
        extractor = extract_frame_generic
        gt_format = "scannet"
    elif args.dataset_type == "normalized":
        dataset, gt_file, ds_type = load_normalized_dataset(args)
        extractor = extract_frame_generic
        gt_format = "tum"

    n = min(args.max_frames or len(dataset), len(dataset))
    _validate_workload(args, dataset)
    print(f"Dataset: {ds_type}  frames: {n}  stride: {args.stride}")
    if gt_file:
        print(f"GT file: {gt_file}")
    print()

    # Run SLAM
    poses_est, gt_poses_inline, tracking_log, elapsed, fps, tracking_fps = run_slam(
        args, dataset, extractor, device
    )

    print(f"\n{'=' * 70}")
    print(f"Complete: {len(poses_est)} frames in {elapsed:.1f}s  ({fps:.1f} fps)")
    print(f"Tracking FPS (warmup-excluded): {tracking_fps:.1f}")
    print(f"{'=' * 70}")

    output_dir = Path(args.output)
    save_results(
        output_dir, poses_est, tracking_log, ds_type,
        gt_file=gt_file, gt_format=gt_format,
        gt_poses_inline=gt_poses_inline if gt_poses_inline else None,
        no_eval=args.no_eval,
        tracking_fps=tracking_fps,
        elapsed=elapsed,
        args=args,
    )
    _cleanup_accelerator(device)


if __name__ == "__main__":
    main()
