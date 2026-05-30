"""Unit tests for gradslam.slam.diagnostics.log_startup_sanity."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gradslam.slam.diagnostics import log_startup_sanity


def _make_depth(mean_m: float, h: int = 32, w: int = 32) -> torch.Tensor:
    """Create a synthetic [H, W] depth map with given mean (a few zeros for realism)."""
    depth = torch.full((h, w), mean_m)
    depth[0, 0] = 0.0  # one invalid pixel
    return depth


def _make_intrinsics(fx: float = 525.0, fy: float = 525.0,
                     cx: float = 319.5, cy: float = 239.5) -> torch.Tensor:
    K = torch.eye(4)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


# ---------------------------------------------------------------------------
# Normal case
# ---------------------------------------------------------------------------

def test_normal_depth_no_warning(capsys):
    depth = _make_depth(1.5)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
        label="test_seq",
    )

    assert result["warning"] is None
    assert abs(result["depth_mean"] - 1.5) < 0.1
    assert result["depth_min"] >= 0.0
    assert result["depth_max"] <= 2.0
    assert 0.0 < result["valid_ratio"] <= 1.0

    # Check intrinsics pass-through at scale=1
    assert result["fx"] == pytest.approx(525.0)
    assert result["fy"] == pytest.approx(525.0)
    assert result["cx"] == pytest.approx(319.5)
    assert result["cy"] == pytest.approx(239.5)

    # Timestamp delta should be None (both ts were None)
    assert result["dt"] is None

    # GT motions should be None (not provided)
    assert result["gt_motions_cm"] is None

    out = capsys.readouterr().out
    assert "Startup Sanity" in out
    assert "test_seq" in out
    assert "pose convention" in out


# ---------------------------------------------------------------------------
# Millimeter-scale depth warning
# ---------------------------------------------------------------------------

def test_millimeter_depth_triggers_warning(capsys):
    # mean depth ~0.001 m → likely mm data
    depth = _make_depth(0.001)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
    )

    assert result["warning"] is not None
    assert "millimeter" in result["warning"].lower() or "small" in result["warning"].lower()

    out = capsys.readouterr().out
    assert "WARNING" in out


# ---------------------------------------------------------------------------
# Very large depth warning
# ---------------------------------------------------------------------------

def test_large_depth_triggers_warning(capsys):
    depth = _make_depth(50.0)  # 50 m mean — suspicious
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
    )

    assert result["warning"] is not None
    out = capsys.readouterr().out
    assert "WARNING" in out


# ---------------------------------------------------------------------------
# process_scale applied to intrinsics
# ---------------------------------------------------------------------------

def test_process_scale_applied_to_intrinsics(capsys):
    K = _make_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    depth = _make_depth(2.0)

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=0.5,
        rgb_ts=None,
        depth_ts=None,
    )

    assert result["fx"] == pytest.approx(250.0)
    assert result["fy"] == pytest.approx(250.0)
    assert result["cx"] == pytest.approx(160.0)
    assert result["cy"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Timestamp delta: OK
# ---------------------------------------------------------------------------

def test_timestamp_delta_ok(capsys):
    depth = _make_depth(1.5)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=1.000,
        depth_ts=1.003,
    )

    assert result["dt"] == pytest.approx(0.003, abs=1e-9)
    out = capsys.readouterr().out
    assert "RGB/depth dt" in out
    assert "[OK]" in out
    assert "WARNING" not in out


# ---------------------------------------------------------------------------
# Timestamp delta: warning
# ---------------------------------------------------------------------------

def test_timestamp_delta_warning(capsys):
    depth = _make_depth(1.5)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=0.0,
        depth_ts=0.1,  # 100 ms gap → warning
    )

    assert result["dt"] == pytest.approx(0.1, abs=1e-9)
    out = capsys.readouterr().out
    assert "WARNING" in out


# ---------------------------------------------------------------------------
# Timestamp delta: skipped when either is None
# ---------------------------------------------------------------------------

def test_timestamp_none_skips_dt(capsys):
    depth = _make_depth(1.5)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=1.0,
        depth_ts=None,
    )

    assert result["dt"] is None
    out = capsys.readouterr().out
    assert "RGB/depth dt" not in out


# ---------------------------------------------------------------------------
# GT relative motions
# ---------------------------------------------------------------------------

def test_gt_relative_motions(capsys):
    # Build 5 poses with 10 cm step in X direction
    poses = []
    for i in range(5):
        T = np.eye(4)
        T[0, 3] = i * 0.1  # 10 cm per step
        poses.append(T)

    depth = _make_depth(1.5)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
        gt_poses=poses,
    )

    motions = result["gt_motions_cm"]
    assert motions is not None
    assert len(motions) == 4  # 5 poses → 4 pairs
    for d in motions:
        assert abs(d - 10.0) < 0.01  # each step ~10 cm

    out = capsys.readouterr().out
    assert "GT frame motions" in out


def test_gt_relative_motions_capped_at_10(capsys):
    """More than 10 pairs are capped to the first 10."""
    poses = [np.eye(4) for _ in range(15)]
    for i, T in enumerate(poses):
        T[1, 3] = i * 0.05  # 5 cm steps

    depth = _make_depth(1.5)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
        gt_poses=poses,
    )

    assert result["gt_motions_cm"] is not None
    assert len(result["gt_motions_cm"]) == 10


def test_gt_single_pose_skipped(capsys):
    """Only one GT pose → not enough for motions, skipped gracefully."""
    poses = [np.eye(4)]
    depth = _make_depth(1.5)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
        gt_poses=poses,
    )

    assert result["gt_motions_cm"] is None


# ---------------------------------------------------------------------------
# All-zero depth (no valid pixels)
# ---------------------------------------------------------------------------

def test_all_zero_depth(capsys):
    depth = torch.zeros(32, 32)
    K = _make_intrinsics()

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
    )

    assert result["valid_ratio"] == pytest.approx(0.0)
    assert result["depth_mean"] is None
    out = capsys.readouterr().out
    assert "no valid" in out


# ---------------------------------------------------------------------------
# 3x3 intrinsics also accepted
# ---------------------------------------------------------------------------

def test_3x3_intrinsics(capsys):
    K = torch.eye(3)
    K[0, 0] = 600.0
    K[1, 1] = 600.0
    K[0, 2] = 400.0
    K[1, 2] = 300.0
    depth = _make_depth(2.0)

    result = log_startup_sanity(
        first_frame_depth=depth,
        intrinsics=K,
        process_scale=1.0,
        rgb_ts=None,
        depth_ts=None,
    )

    assert result["fx"] == pytest.approx(600.0)
    assert result["fy"] == pytest.approx(600.0)
