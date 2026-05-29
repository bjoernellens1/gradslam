#!/usr/bin/env python3
"""Extract RealSense bag file to normalized RGB-D format.

Uses the RealSense Python SDK to read .bag files and extract color/depth
frames into the normalized format expected by NormalizedRGBD loader.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover - depends on host/container extra
    raise SystemExit(
        "pyrealsense2 is required for RealSense .bag extraction. "
        "Install the project with the 'realsense' extra or rebuild the ROCm container."
    ) from exc


def _depth_stats_m(depth_raw: np.ndarray, depth_scale: float) -> dict[str, float]:
    valid = depth_raw[depth_raw > 0].astype(np.float32) * float(depth_scale)
    if valid.size == 0:
        return {"valid_ratio": 0.0, "median_m": 0.0, "p95_m": 0.0, "max_m": 0.0}
    return {
        "valid_ratio": float(valid.size / depth_raw.size),
        "median_m": float(np.median(valid)),
        "p95_m": float(np.percentile(valid, 95)),
        "max_m": float(valid.max()),
    }


def extract_bag(bag_path: str, output_dir: str, skip_frames: int = 1, max_frames: int | None = None) -> None:
    """Extract RealSense bag to normalized RGB-D format.

    Args:
        bag_path: Path to RealSense .bag file
        output_dir: Output directory for normalized dataset
        skip_frames: Process every Nth frame (1 = no skipping)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    images_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    print(f"Opening RealSense bag: {bag_path}")
    started_at = time.time()

    config = rs.config()
    config.enable_device_from_file(bag_path)
    config.enable_all_streams()

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")

    # Get stream profiles to extract intrinsics
    color_profile = None
    depth_profile = None
    for stream in profile.get_streams():
        if stream.stream_type() == rs.stream.color:
            color_profile = stream.as_video_stream_profile()
        elif stream.stream_type() == rs.stream.depth:
            depth_profile = stream.as_video_stream_profile()

    if not color_profile or not depth_profile:
        raise RuntimeError("Could not find color or depth stream in bag file")

    color_intr = color_profile.get_intrinsics()
    depth_intr = depth_profile.get_intrinsics()

    print(f"Color resolution: {color_intr.width}x{color_intr.height}")
    print(f"Depth resolution: {depth_intr.width}x{depth_intr.height}")
    print(f"Color intrinsics: fx={color_intr.fx}, fy={color_intr.fy}, cx={color_intr.ppx}, cy={color_intr.ppy}")

    # Create alignment object
    align = rs.align(rs.stream.color)

    frames_data = []
    frame_count = 0
    saved_frame_count = 0
    start_time = None
    last_progress_time = None
    last_progress_frame = 0

    print("Extracting frames...")
    while True:
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError:
            # End of bag file
            break

        if start_time is None:
            start_time = time.time()

        # Skip frames if requested
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Align depth to color
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            frame_count += 1
            continue

        # Extract numpy arrays. RealSense depth frames are raw uint16 depth
        # units; metric depth is raw * depth_scale. Keep the raw units in PNG
        # and encode the matching depth_factor in camera_info.json so the
        # normalized loader recovers meters exactly.
        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        if depth_raw.dtype != np.uint16:
            depth_raw = depth_raw.astype(np.uint16)

        # Save images
        frame_idx = saved_frame_count
        rgb_filename = f"frame_{frame_idx:06d}.png"
        depth_filename = f"frame_{frame_idx:06d}.png"

        # Convert BGR to RGB for OpenCV saving
        color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_dir / rgb_filename), color_bgr)
        cv2.imwrite(str(depth_dir / depth_filename), depth_raw)

        # Record frame metadata
        timestamp = color_frame.get_timestamp() / 1000.0  # Convert ms to seconds
        frames_data.append({
            "index": frame_idx,
            "timestamp": timestamp,
            "rgb_file": f"images/{rgb_filename}",
            "depth_file": f"depth/{depth_filename}",
        })

        saved_frame_count += 1
        frame_count += 1

        # Progress reporting every 500 frames or every 10 seconds
        if max_frames is not None and saved_frame_count >= max_frames:
            break

        now = time.time()
        if last_progress_time is None or (now - last_progress_time >= 10.0 or saved_frame_count % 500 == 0):
            elapsed = now - start_time
            frames_since_last = saved_frame_count - last_progress_frame
            time_since_last = now - last_progress_time if last_progress_time else elapsed

            if time_since_last > 0:
                fps = frames_since_last / time_since_last
                # Estimate total frames (rough guess based on typical RealSense rates)
                # This will be accurate once we reach the end
                eta_seconds = int((saved_frame_count * 1.15 - saved_frame_count) / fps) if fps > 0 else 0
                eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds > 0 else "?"

                mins, secs = divmod(int(elapsed), 60)
                elapsed_str = f"{mins}m {secs}s"

                print(f"  {saved_frame_count:,} frames | {fps:.1f} fps | Elapsed: {elapsed_str} | ETA: {eta_str}")

                last_progress_time = now
                last_progress_frame = saved_frame_count

    pipeline.stop()

    # Write frames.csv
    frames_csv_path = output_dir / "frames.csv"
    with open(frames_csv_path, "w") as f:
        f.write("index,timestamp,rgb_file,depth_file\n")
        for frame in frames_data:
            f.write(f"{frame['index']},{frame['timestamp']},{frame['rgb_file']},{frame['depth_file']}\n")

    # Write camera_info.json. NormalizedRGBD computes depth_m = raw / depth_factor.
    depth_factor = 1.0 / float(depth_scale)
    camera_info = {
        "width": color_intr.width,
        "height": color_intr.height,
        "fx": float(color_intr.fx),
        "fy": float(color_intr.fy),
        "cx": float(color_intr.ppx),
        "cy": float(color_intr.ppy),
        "depth_factor": depth_factor,
        "depth_scale": float(depth_scale),
        "d": list(color_intr.coeffs),
    }

    camera_info_path = output_dir / "camera_info.json"
    with open(camera_info_path, "w") as f:
        json.dump(camera_info, f, indent=2)

    stats = {}
    if frames_data:
        sample_ids = sorted(set([0, len(frames_data) // 2, len(frames_data) - 1]))
        medians = []
        valid_ratios = []
        for sample_id in sample_ids:
            depth_path = output_dir / frames_data[sample_id]["depth_file"]
            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                continue
            s = _depth_stats_m(depth_raw, depth_scale)
            medians.append(s["median_m"])
            valid_ratios.append(s["valid_ratio"])
        if medians:
            stats = {
                "median_depth_median_m": float(np.median(medians)),
                "median_valid_ratio": float(np.median(valid_ratios)),
            }

    manifest = {
        "bag_path": str(bag_path),
        "output_dir": str(output_dir),
        "started_at_unix": started_at,
        "elapsed_s": time.time() - started_at,
        "skip_frames": skip_frames,
        "max_frames": max_frames,
        "frames_processed": frame_count,
        "frames_extracted": saved_frame_count,
        "depth_scale": float(depth_scale),
        "depth_factor": depth_factor,
        "depth_stats": stats,
    }
    with open(output_dir / "extraction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\nExtraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames extracted: {saved_frame_count}")
    print(f"Output directory: {output_dir}")
    print(f"Camera info:")
    print(f"  Resolution: {camera_info['width']}x{camera_info['height']}")
    print(f"  fx={camera_info['fx']:.2f}, fy={camera_info['fy']:.2f}")
    print(f"  cx={camera_info['cx']:.2f}, cy={camera_info['cy']:.2f}")
    if stats:
        print(f"  median depth: {stats['median_depth_median_m']:.3f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RealSense bag file to normalized RGB-D format"
    )
    parser.add_argument(
        "bag_path",
        type=str,
        help="Path to RealSense .bag file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <bag_parent>/<bag_stem>_extracted)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1 = no skipping)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N extracted frames (default: all)",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag_path)
    if not bag_path.exists():
        print(f"Error: Bag file not found: {bag_path}")
        exit(1)

    output_dir = args.output or str(bag_path.parent / f"{bag_path.stem}_extracted")
    extract_bag(str(bag_path), output_dir, skip_frames=args.skip_frames, max_frames=args.max_frames)
