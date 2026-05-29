#!/usr/bin/env python3
"""Finalize the partially extracted RealSense dataset.

Generates frames.csv and camera_info.json for already extracted frames.
"""

import json
import os
from pathlib import Path
import pyrealsense2 as rs

bag_path = Path("/mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity/20260522_115050.bag")
output_dir = Path("/mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity_extracted")

images_dir = output_dir / "images"
depth_dir = output_dir / "depth"

print(f"Counting files in {images_dir}...")
if not images_dir.exists():
    print(f"Error: {images_dir} does not exist.")
    exit(1)

images = sorted(list(images_dir.glob("frame_*.png")))
num_images = len(images)
print(f"Found {num_images} extracted images.")

if num_images == 0:
    print("No images found. Exiting.")
    exit(1)

# Open bag to get intrinsics
print(f"Opening RealSense bag to read intrinsics and timestamps: {bag_path}")
config = rs.config()
config.enable_device_from_file(str(bag_path))
config.enable_all_streams()

pipeline = rs.pipeline()
profile = pipeline.start(config)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Get stream profiles to extract intrinsics
color_profile = None
for stream in profile.get_streams():
    if stream.stream_type() == rs.stream.color:
        color_profile = stream.as_video_stream_profile()

if not color_profile:
    raise RuntimeError("Could not find color stream in bag file")

color_intr = color_profile.get_intrinsics()

# Write camera_info.json
camera_info = {
    "width": color_intr.width,
    "height": color_intr.height,
    "fx": float(color_intr.fx),
    "fy": float(color_intr.fy),
    "cx": float(color_intr.ppx),
    "cy": float(color_intr.ppy),
    "depth_factor": 1000.0,  # Depth in mm, scale to meters
    "d": list(color_intr.coeffs),
}

camera_info_path = output_dir / "camera_info.json"
with open(camera_info_path, "w") as f:
    json.dump(camera_info, f, indent=2)
print(f"✓ Wrote camera_info.json to {camera_info_path}")

# Collect timestamps from bag for the first num_images frames
print("Reading frame timestamps from bag...")
align = rs.align(rs.stream.color)
frames_data = []
saved_frame_count = 0

while saved_frame_count < num_images:
    try:
        frames = pipeline.wait_for_frames()
    except RuntimeError:
        break

    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    # Record frame metadata
    timestamp = color_frame.get_timestamp() / 1000.0  # Convert ms to seconds
    rgb_filename = f"frame_{saved_frame_count:06d}.png"
    depth_filename = f"frame_{saved_frame_count:06d}.png"
    
    frames_data.append({
        "index": saved_frame_count,
        "timestamp": timestamp,
        "rgb_file": f"images/{rgb_filename}",
        "depth_file": f"depth/{depth_filename}",
    })
    saved_frame_count += 1
    
    if saved_frame_count % 1000 == 0:
        print(f"  Processed {saved_frame_count}/{num_images} timestamps...")

pipeline.stop()

# Write frames.csv
frames_csv_path = output_dir / "frames.csv"
with open(frames_csv_path, "w") as f:
    f.write("index,timestamp,rgb_file,depth_file\n")
    for frame in frames_data:
        f.write(f"{frame['index']},{frame['timestamp']},{frame['rgb_file']},{frame['depth_file']}\n")

print(f"✓ Wrote frames.csv with {len(frames_data)} entries to {frames_csv_path}")
print("Finalization complete! You can now run SLAM.")
