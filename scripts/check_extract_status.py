#!/usr/bin/env python3
import os
import psutil
from pathlib import Path

target_dir = Path("/mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/cps_1stfloor_hallway_mediumdensity_extracted")
images_dir = target_dir / "images"
depth_dir = target_dir / "depth"
csv_file = target_dir / "frames.csv"

print("=" * 60)
print("EXTRACTION STATUS DIAGNOSTIC REPORT")
print("=" * 60)

# Check active extraction processes
print("Checking active extract_realsense_bag.py processes...")
found_process = False
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = proc.info['cmdline']
        if cmd and any('extract_realsense_bag.py' in part for part in cmd):
            print(f"Found active process: PID {proc.info['pid']} - {' '.join(cmd)}")
            found_process = True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
if not found_process:
    print("No active extract_realsense_bag.py process found.")

# Check directories and file counts
print("\nChecking directories:")
if target_dir.exists():
    print(f"✓ Target directory exists: {target_dir}")
    if csv_file.exists():
        print(f"✓ frames.csv exists! Size: {csv_file.stat().st_size} bytes")
    else:
        print("✗ frames.csv does NOT exist.")

    if images_dir.exists():
        img_count = len(os.listdir(images_dir))
        print(f"  - Images directory contains {img_count:,} files.")
    else:
        print("  - Images directory does not exist.")

    if depth_dir.exists():
        depth_count = len(os.listdir(depth_dir))
        print(f"  - Depth directory contains {depth_count:,} files.")
    else:
        print("  - Depth directory does not exist.")
else:
    print(f"✗ Target directory does not exist: {target_dir}")

print("=" * 60)
