"""Tests for RGBDTSDFSLAM two-frame pipeline."""

import torch
import pytest
from gradslam.slam.pipeline import RGBDTSDFSLAM, RGBDFrame
from gradslam.mapping.tsdf import TSDFConfig
from gradslam.icp.projective import ProjectiveICPConfig


@pytest.fixture
def slam():
    tsdf_cfg = TSDFConfig(voxel_size=0.02)
    icp_cfg = ProjectiveICPConfig(
        n_pyramid_levels=2,
        iterations=(5, 5),
        damping=(1e-2, 1e-3),
    )
    return RGBDTSDFSLAM(
        tsdf_config=tsdf_cfg,
        icp_config=icp_cfg,
        voxel_dim=(32, 32, 32),
        volume_origin=(-0.32, -0.32, 0.0),
        near=0.1,
        far=1.5,
    )


@pytest.fixture
def intrinsics_4x4():
    return torch.tensor([
        [64.0,  0.0, 32.0, 0.0],
        [ 0.0, 64.0, 24.0, 0.0],
        [ 0.0,  0.0,  1.0, 0.0],
        [ 0.0,  0.0,  0.0, 1.0],
    ])


def _make_depth(H=48, W=64, z=0.5, noise_std=0.0):
    depth = torch.full((H, W), z)
    if noise_std > 0:
        depth = depth + torch.randn_like(depth) * noise_std
    return depth.clamp(min=0.01)


def test_first_frame_pose_is_identity(slam, intrinsics_4x4):
    depth = _make_depth()
    frame = RGBDFrame(rgb=None, depth=depth, intrinsics=intrinsics_4x4)
    result = slam.process_frame(frame)
    eye = torch.eye(4)
    assert torch.allclose(result.T_world_camera, eye, atol=1e-5)
    assert result.lost is False


def test_second_frame_returns_pose(slam, intrinsics_4x4):
    depth = _make_depth()
    frame = RGBDFrame(rgb=None, depth=depth, intrinsics=intrinsics_4x4)
    slam.process_frame(frame)
    result2 = slam.process_frame(frame)
    assert result2.T_world_camera.shape == (4, 4)
    # Pose should be a valid SE(3): det(R) ≈ 1
    R = result2.T_world_camera[:3, :3]
    det = torch.linalg.det(R)
    assert abs(det.item() - 1.0) < 0.05


def test_reset_clears_state(slam, intrinsics_4x4):
    depth = _make_depth()
    frame = RGBDFrame(rgb=None, depth=depth, intrinsics=intrinsics_4x4)
    slam.process_frame(frame)
    slam.process_frame(frame)
    slam.reset()
    assert slam.tsdf is None
    assert slam.T_world_camera is None
    assert slam.frame_count == 0


def test_quality_dict_populated(slam, intrinsics_4x4):
    depth1 = _make_depth(z=0.50)
    depth2 = _make_depth(z=0.52, noise_std=0.002)
    K = intrinsics_4x4

    slam.process_frame(RGBDFrame(rgb=None, depth=depth1, intrinsics=K))
    result = slam.process_frame(RGBDFrame(rgb=None, depth=depth2, intrinsics=K))

    q = result.quality
    assert "num_valid" in q
    assert "rmse" in q


def test_no_grad_in_forward(slam, intrinsics_4x4):
    """process_frame must not build autograd graph."""
    depth = _make_depth()
    frame = RGBDFrame(rgb=None, depth=depth, intrinsics=intrinsics_4x4)
    slam.process_frame(frame)
    result = slam.process_frame(frame)
    assert not result.T_world_camera.requires_grad
