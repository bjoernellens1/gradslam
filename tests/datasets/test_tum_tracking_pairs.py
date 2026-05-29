from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gradslam.datasets.tum import TUM


def _write_tum_sequence(root: Path) -> Path:
    seq = root / "rgbd_dataset_freiburg2_test"
    (seq / "rgb").mkdir(parents=True)
    (seq / "depth").mkdir()

    rgb_lines = []
    depth_lines = []
    for i in range(5):
        ts = float(i)
        rgb_name = f"rgb/{ts:.6f}.png"
        depth_name = f"depth/{ts:.6f}.png"
        rgb = np.full((4, 4, 3), i, dtype=np.uint8)
        depth = np.full((4, 4), 1000, dtype=np.uint16)
        assert cv2.imwrite(str(seq / rgb_name), rgb)
        assert cv2.imwrite(str(seq / depth_name), depth)
        rgb_lines.append(f"{ts:.6f} {rgb_name}\n")
        depth_lines.append(f"{ts:.6f} {depth_name}\n")

    (seq / "rgb.txt").write_text("".join(rgb_lines), encoding="utf-8")
    (seq / "depth.txt").write_text("".join(depth_lines), encoding="utf-8")
    (seq / "groundtruth.txt").write_text(
        "0.000000 0 0 0 0 0 0 1\n"
        "4.000000 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    return seq


def test_tum_can_track_all_rgbd_pairs_without_sparse_gt_sampling(tmp_path: Path) -> None:
    _write_tum_sequence(tmp_path)

    track_dataset = TUM(
        basedir=str(tmp_path),
        sequences=("rgbd_dataset_freiburg2_test",),
        seqlen=1,
        stride=1,
        return_pose=False,
        return_transform=False,
    )
    pose_dataset = TUM(
        basedir=str(tmp_path),
        sequences=("rgbd_dataset_freiburg2_test",),
        seqlen=1,
        stride=1,
        return_pose=True,
        return_transform=False,
    )

    assert len(track_dataset) == 5
    assert len(pose_dataset) == 2
    assert "pose None" in track_dataset[1][-1]


def test_tum_official_intrinsics_selects_freiburg_family(tmp_path: Path) -> None:
    _write_tum_sequence(tmp_path)

    dataset = TUM(
        basedir=str(tmp_path),
        sequences=("rgbd_dataset_freiburg2_test",),
        seqlen=1,
        stride=1,
        return_pose=False,
        return_transform=False,
        intrinsics_mode="official",
    )

    K = dataset.intrinsics[0].numpy()
    assert np.isclose(K[0, 0], 520.9)
    assert np.isclose(K[1, 1], 521.0)
    assert np.isclose(K[0, 2], 325.1)
    assert np.isclose(K[1, 2], 249.7)
