#!/usr/bin/env python3
"""Run the RGBDTSDFSLAM pipeline on TUM RGB-D sequences."""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from gradslam.backend import accelerator_backend, backend_report, default_device
from gradslam.datasets import TUM
from gradslam.slam import RGBDTSDFSLAM
from gradslam.slam.pipeline import RGBDFrame


def main():
    parser = argparse.ArgumentParser(
        description="Run RGBDTSDFSLAM on TUM RGB-D sequences"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/workspace/datasets/public/TUM/tum_rgbd",
        help="Root directory containing TUM sequences",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="freiburg1_desk",
        help="Sequence name. For tum_rgbd layout use short name (e.g. freiburg1_desk); "
             "for groundtruth layout use full name (e.g. rgbd_dataset_freiburg1_desk)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Skip frames (stride=2 processes every 2nd frame)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda:0, etc); auto-detected if None",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/tum_slam_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for performance optimization",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit processing to N frames (useful for testing)",
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = default_device()

    print(f"\n{'=' * 70}")
    print(f"gradslam RGBDTSDFSLAM on TUM RGB-D Dataset")
    print(f"{'=' * 70}\n")
    print(backend_report())
    print(f"Using device: {device}\n")

    # Resolve basedir and sequence name.
    # tum_rgbd layout: <root>/<short>/ contains rgbd_dataset_<short>/
    # groundtruth layout: <root>/ directly contains rgbd_dataset_<full>/
    root = Path(args.dataset_root)
    seq = args.sequence
    tum_rgbd_subdir = root / seq / f"rgbd_dataset_{seq}"
    if tum_rgbd_subdir.exists():
        basedir = str(root / seq)
        seq_name = f"rgbd_dataset_{seq}"
    else:
        basedir = str(root)
        seq_name = seq

    print(f"Loading TUM sequence: {seq_name} from {basedir}")
    dataset = TUM(
        basedir=basedir,
        sequences=(seq_name,),
        seqlen=1,
        dilation=0,
        stride=args.stride,
    )
    print(f"Loaded {len(dataset)} samples at {dataset.height}x{dataset.width}\n")

    # Initialize SLAM
    slam = RGBDTSDFSLAM()
    slam = slam.to(device)
    if args.compile:
        print("Enabling torch.compile...\n")
        # slam = torch.compile(slam)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run SLAM
    max_frames = args.max_frames or len(dataset)
    keyframe_count = 0
    start_time = time.time()

    print(f"Processing {min(max_frames, len(dataset))} frames...\n")

    tracking_results = []
    poses = []

    with torch.no_grad():
        for idx in tqdm(range(min(max_frames, len(dataset))), desc="SLAM"):
            sample = dataset[idx]

            # TUM dataset returns: (colors, depths, intrinsics, pose, T_world_camera, seq_name, rgb_name)
            # Shapes: colors=(B, H, W, C), depths=(B, H, W, 1), intrinsics=(B, 3, 3)
            colors = sample[0][0]          # (B, H, W, C) -> (H, W, C)
            depths = sample[1][0]          # (B, H, W, 1) -> (H, W, 1)
            intrinsics = sample[2][0]      # (B, 3, 3) -> (3, 3)

            # Move to device
            colors = colors.to(device)
            depths = depths.to(device)
            intrinsics = intrinsics.to(device)

            # Create RGBDFrame (rgb should be H, W, C; depth should be H, W)
            frame = RGBDFrame(
                rgb=colors,                 # (H, W, C)
                depth=depths.squeeze(-1),   # (H, W, 1) -> (H, W)
                intrinsics=intrinsics,
            )

            # Process frame
            result = slam.process_frame(frame)

            quality = result.quality
            tracking_results.append(
                {
                    "frame_idx": idx,
                    "num_valid": quality.get("num_valid", 0),
                    "inlier_ratio": quality.get("inlier_ratio", 0.0),
                    "rmse": quality.get("rmse", 0.0),
                    "is_keyframe": result.used_keyframe,
                }
            )

            poses.append(result.T_world_camera.cpu().numpy())

            if result.used_keyframe:
                keyframe_count += 1

    elapsed = time.time() - start_time
    fps = len(dataset) / elapsed

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SLAM Pipeline Complete")
    print(f"{'=' * 70}")
    print(f"Processed:     {min(max_frames, len(dataset))} frames")
    print(f"Keyframes:     {keyframe_count}")
    print(f"Time:          {elapsed:.2f} seconds")
    print(f"Throughput:    {fps:.2f} fps")
    print(f"Device:        {device}")
    print(f"Output dir:    {output_dir}\n")

    # Save results
    tracking_metrics = {
        "frames": [r["frame_idx"] for r in tracking_results],
        "num_valid": [r["num_valid"] for r in tracking_results],
        "inlier_ratio": [r["inlier_ratio"] for r in tracking_results],
        "rmse": [r["rmse"] for r in tracking_results],
        "is_keyframe": [r["is_keyframe"] for r in tracking_results],
    }

    # Save poses as TUM format
    pose_file = output_dir / "estimated_poses.txt"
    with open(pose_file, "w") as f:
        for idx, pose in enumerate(poses):
            # Convert 4x4 pose to TUM format: timestamp tx ty tz qx qy qz qw
            # We use frame index as timestamp
            t = pose[:3, 3]
            R = pose[:3, :3]
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
            f.write(f"{idx:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    print(f"✓ Saved estimated poses to {pose_file}")

    # Save tracking metrics
    metrics_file = output_dir / "tracking_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("frame_idx num_valid inlier_ratio rmse is_keyframe\n")
        for r in tracking_results:
            f.write(
                f"{r['frame_idx']:6d} {r['num_valid']:8d} {r['inlier_ratio']:.6f} {r['rmse']:.6f} {int(r['is_keyframe']):1d}\n"
            )

    print(f"✓ Saved tracking metrics to {metrics_file}\n")


if __name__ == "__main__":
    main()
