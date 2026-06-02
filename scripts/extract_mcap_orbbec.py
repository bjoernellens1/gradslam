#!/usr/bin/env python3
"""Extract Orbbec MCAP (ROS2 bag) to normalized RGB-D format."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def extract(bag_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)

    store = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    cam_info_written = False

    with Reader(str(bag_dir)) as reader:
        # Collect color and depth by timestamp
        color_msgs = {}   # ts -> compressed bytes
        depth_msgs = {}   # ts -> compressed bytes
        K_color = None

        color_topic  = "/camera/color/image_raw/compressed"
        depth_topic  = "/camera/depth/image_raw/compressed"
        color_info_t = "/camera/color/camera_info"

        conns_color = [c for c in reader.connections if c.topic == color_topic]
        conns_depth = [c for c in reader.connections if c.topic == depth_topic]
        conns_info  = [c for c in reader.connections if c.topic == color_info_t]
        all_conns   = conns_color + conns_depth + conns_info

        for conn, ts, raw in reader.messages(connections=all_conns):
            msg = store.deserialize_cdr(raw, conn.msgtype)
            ts_sec = ts * 1e-9

            if conn.topic == color_info_t and K_color is None:
                K_color = {
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "fx": float(msg.k[0]),
                    "fy": float(msg.k[4]),
                    "cx": float(msg.k[2]),
                    "cy": float(msg.k[5]),
                    "depth_factor": 1000.0,
                    "d": [float(x) for x in msg.d],
                }
            elif conn.topic == color_topic:
                color_msgs[ts] = (ts_sec, bytes(msg.data))
            elif conn.topic == depth_topic:
                depth_msgs[ts] = (ts_sec, bytes(msg.data))

    if K_color is None:
        sys.exit("No camera_info found")

    # Match color to nearest depth by timestamp
    depth_ts = sorted(depth_msgs.keys())
    idx = 0
    for i, cts in enumerate(sorted(color_msgs.keys())):
        # Find nearest depth
        while idx + 1 < len(depth_ts) and abs(depth_ts[idx+1] - cts) < abs(depth_ts[idx] - cts):
            idx += 1
        dts = depth_ts[idx]
        if abs(dts - cts) * 1e-9 > 0.05:  # > 50ms gap → skip
            continue

        ts_sec, color_bytes = color_msgs[cts]
        _, depth_bytes = depth_msgs[dts]

        # Decode color
        color_arr = np.frombuffer(color_bytes, dtype=np.uint8)
        color_img = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
        if color_img is None:
            continue
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Decode depth (16UC1 PNG)
        depth_arr = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            continue

        fname = f"frame_{i:06d}.png"
        cv2.imwrite(str(out_dir / "images" / fname), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "depth" / fname), depth_img)
        rows.append((i, ts_sec, f"images/{fname}", f"depth/{fname}"))

    # Write camera_info.json
    (out_dir / "camera_info.json").write_text(json.dumps(K_color, indent=2))

    # Write frames.csv
    with open(out_dir / "frames.csv", "w") as f:
        f.write("index,timestamp,rgb_file,depth_file\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]},{r[3]}\n")

    print(f"Extracted {len(rows)} frames → {out_dir}")
    print(f"Camera: {K_color['width']}x{K_color['height']}  fx={K_color['fx']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    extract(Path(args.bag_dir), Path(args.out_dir))
