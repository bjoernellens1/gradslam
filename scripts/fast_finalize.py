#!/usr/bin/env python3
"""Instantly finalize the extracted dataset without reading the .bag file.

Uses the ground truth start timestamp and 30 fps interpolation.
"""

from pathlib import Path

target_dir = Path("/mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity_extracted")
images_dir = target_dir / "images"
depth_dir = target_dir / "depth"
csv_file = target_dir / "frames.csv"
gt_file = Path("/mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity/groundtruth_tum.txt")

print(f"Counting images in {images_dir}...")
if not images_dir.exists():
    print(f"Error: {images_dir} does not exist.")
    exit(1)

images = list(images_dir.glob("frame_*.png"))
num_images = len(images)
print(f"Found {num_images} extracted images.")

if num_images == 0:
    print("No images found. Exiting.")
    exit(1)

# Read start timestamp from groundtruth
start_ts = 0.0
if gt_file.exists():
    try:
        with open(gt_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    start_ts = float(line.split()[0])
                    break
        print(f"✓ Found start timestamp in ground truth: {start_ts}")
    except Exception as e:
        print(f"Warning: Could not read ground truth file: {e}")
else:
    print("Warning: groundtruth_tum.txt not found. Using 0.0 as start timestamp.")

# Write frames.csv with generated timestamps (30 fps)
print("Generating frames.csv...")
with open(csv_file, "w") as f:
    f.write("index,timestamp,rgb_file,depth_file\n")
    for i in range(num_images):
        timestamp = start_ts + i * (1.0 / 30.0)
        rgb_file = f"images/frame_{i:06d}.png"
        depth_file = f"depth/frame_{i:06d}.png"
        f.write(f"{i},{timestamp:.6f},{rgb_file},{depth_file}\n")

print(f"✓ Successfully generated frames.csv with {num_images} entries in milliseconds!")
