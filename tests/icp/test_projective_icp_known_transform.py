"""Test that projective ICP recovers a small known rigid-body motion."""

import torch
import pytest
from gradslam.icp.projective import ProjectiveICPConfig, ProjectiveICPTracker
from gradslam.geometry.normals import estimate_normals


def _plane_depth(H, W, K, A, B, C):
    """Render depth of infinite plane Z = A*X + B*Y + C in camera space.

    Closed-form: for each pixel (u,v),
        Z = C / (1 - A*(u-cx)/fx - B*(v-cy)/fy)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v_idx, u_idx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    denom = 1.0 - A * (u_idx - cx) / fx - B * (v_idx - cy) / fy
    denom = denom.clamp(min=1e-4)
    return C / denom


@pytest.fixture
def intrinsics():
    return torch.tensor([
        [80.0,  0.0, 40.0],
        [ 0.0, 80.0, 30.0],
        [ 0.0,  0.0,  1.0],
    ])


def test_recover_small_translation(intrinsics):
    """ICP should approximately recover a 2 cm x-translation.

    A slanted plane Z = 0.02*X + 0.01*Y + 2.0 renders differently from
    two cameras related by a 2 cm x-translation.  The difference in depth
    at the model intrinsics provides a clean, closed-form ground truth.
    """
    H, W = 60, 80
    cfg = ProjectiveICPConfig(
        n_pyramid_levels=2,
        iterations=(15, 15),
        damping=(1e-2, 1e-3),
        max_depth_diff=0.5,
        max_normal_angle_deg=80.0,
    )
    tracker = ProjectiveICPTracker(cfg)

    # Plane coefficients: Z = A*X + B*Y + C (in camera space)
    A, B, C = 0.02, 0.01, 2.0

    # Model depth: camera at origin
    model_depth = _plane_depth(H, W, intrinsics, A, B, C)
    model_normal = estimate_normals(model_depth)

    # Live depth: camera shifted by [dx, 0, 0] in world/camera frame
    # New plane equation in live cam: Z_live = A*(X_live + dx) + B*Y_live + C
    #                                         = A*X_live + B*Y_live + (C + A*dx)
    dx = 0.02  # 2 cm
    live_depth = _plane_depth(H, W, intrinsics, A, B, C + A * dx)
    live_normal = estimate_normals(live_depth)

    T_est, quality = tracker(
        live_depth=live_depth,
        live_normal=live_normal,
        model_depth=model_depth,
        model_normal=model_normal,
        intrinsics=intrinsics,
    )

    # The estimated translation should be close to [dx, 0, 0]
    T_true = torch.eye(4)
    T_true[0, 3] = dx

    t_err = torch.norm(T_est[:3, 3] - T_true[:3, 3]).item()
    # Allow up to 2.5 cm error (ICP linearization error on a small slanted plane)
    assert t_err < 0.025, f"Translation error {t_err:.4f} >= 0.025 m"
