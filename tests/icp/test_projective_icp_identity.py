"""Test that projective ICP returns near-identity transform for zero motion."""

import torch
import pytest
from gradslam.icp.projective import ProjectiveICPConfig, ProjectiveICPTracker
from gradslam.geometry.normals import estimate_normals


@pytest.fixture
def tracker():
    cfg = ProjectiveICPConfig(
        n_pyramid_levels=2,
        iterations=(5, 5),
        damping=(1e-2, 1e-3),
    )
    return ProjectiveICPTracker(cfg)


@pytest.fixture
def simple_depth():
    """Slanted plane depth field (non-constant so normals are valid)."""
    H, W = 60, 80
    y, x = torch.meshgrid(
        torch.linspace(0, 1, H),
        torch.linspace(0, 1, W),
        indexing="ij",
    )
    depth = 2.0 + 0.1 * x + 0.05 * y  # non-flat so normals vary
    return depth


@pytest.fixture
def intrinsics():
    return torch.tensor([
        [80.0,  0.0, 40.0],
        [ 0.0, 80.0, 30.0],
        [ 0.0,  0.0,  1.0],
    ])


def test_identity_motion(tracker, simple_depth, intrinsics):
    """When live == model, ICP should return near-identity transform."""
    normals = estimate_normals(simple_depth)

    T, quality = tracker(
        live_depth=simple_depth,
        live_normal=normals,
        model_depth=simple_depth,
        model_normal=normals,
        intrinsics=intrinsics,
    )

    assert T.shape == (4, 4)
    eye = torch.eye(4)
    # Translation should be near zero
    assert torch.norm(T[:3, 3]) < 0.05
    # Rotation should be near identity
    assert torch.norm(T[:3, :3] - eye[:3, :3]) < 0.05


def test_quality_metrics_present(tracker, simple_depth, intrinsics):
    normals = estimate_normals(simple_depth)
    _, quality = tracker(
        live_depth=simple_depth,
        live_normal=normals,
        model_depth=simple_depth,
        model_normal=normals,
        intrinsics=intrinsics,
    )
    for key in ("num_valid", "inlier_ratio", "rmse", "update_norm", "converged"):
        assert key in quality


def test_valid_correspondences_nonzero(tracker, simple_depth, intrinsics):
    normals = estimate_normals(simple_depth)
    _, quality = tracker(
        live_depth=simple_depth,
        live_normal=normals,
        model_depth=simple_depth,
        model_normal=normals,
        intrinsics=intrinsics,
    )
    assert quality["num_valid"] > 100
