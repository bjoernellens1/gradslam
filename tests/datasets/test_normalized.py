from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from gradslam.datasets.normalized import NormalizedRGBD


def _write_minimal_normalized_capture(root: Path) -> Path:
    capture_dir = root / "dataset"
    (capture_dir / "images").mkdir(parents=True)
    (capture_dir / "depth").mkdir(parents=True)

    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.full((2, 3), 1000, dtype=np.uint16)
    assert cv2.imwrite(str(capture_dir / "images" / "frame_000000.png"), rgb)
    assert cv2.imwrite(str(capture_dir / "depth" / "frame_000000.png"), depth)

    (capture_dir / "frames.csv").write_text(
        "index,timestamp,rgb_file,depth_file\n"
        "0,0.0,images/frame_000000.png,depth/frame_000000.png\n",
        encoding="utf-8",
    )
    (capture_dir / "camera_info.json").write_text(
        json.dumps(
            {
                "width": 3,
                "height": 2,
                "fx": 1.0,
                "fy": 1.0,
                "cx": 1.0,
                "cy": 1.0,
                "depth_factor": 1000.0,
                "d": [0.0] * 5,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "best_pseudo_gt_tum.csv").write_text(
        "timestamp,tx,ty,tz,qx,qy,qz,qw\n"
        "0.0,1.0,2.0,3.0,0.0,0.0,0.0,1.0\n",
        encoding="utf-8",
    )
    return capture_dir


def test_normalized_rgbd_prefers_sibling_pseudo_gt(tmp_path: Path) -> None:
    capture_dir = _write_minimal_normalized_capture(tmp_path)

    dataset = NormalizedRGBD(str(capture_dir), seqlen=1, stride=1)

    gt_path = dataset._resolve_gt_file(None)
    assert gt_path == tmp_path / "best_pseudo_gt_tum.csv"

    sample = dataset[0]
    pose = sample[3][0].numpy()
    assert np.allclose(pose[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_normalized_rgbd_prefers_parent_pseudo_gt_over_local_groundtruth(
    tmp_path: Path,
) -> None:
    capture_dir = _write_minimal_normalized_capture(tmp_path)
    (capture_dir / "groundtruth.txt").write_text(
        "0.0 9.0 9.0 9.0 0.0 0.0 0.0 1.0\n",
        encoding="utf-8",
    )

    dataset = NormalizedRGBD(str(capture_dir), seqlen=1, stride=1)

    gt_path = dataset._resolve_gt_file(None)
    assert gt_path == tmp_path / "best_pseudo_gt_tum.csv"

    sample = dataset[0]
    pose = sample[3][0].numpy()
    assert np.allclose(pose[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_normalized_rgbd_skips_invalid_zero_quaternion_gt(tmp_path: Path) -> None:
    capture_dir = _write_minimal_normalized_capture(tmp_path)
    (tmp_path / "best_pseudo_gt_tum.csv").write_text(
        "timestamp,tx,ty,tz,qx,qy,qz,qw\n"
        "0.0,9.0,9.0,9.0,0.0,0.0,0.0,0.0\n"
        "0.0,1.0,2.0,3.0,0.0,0.0,0.0,1.0\n",
        encoding="utf-8",
    )

    dataset = NormalizedRGBD(str(capture_dir), seqlen=1, stride=1)

    sample = dataset[0]
    pose = sample[3][0].numpy()
    assert np.allclose(pose[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))
