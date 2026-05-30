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


def test_hybrid_tracker_uses_local_reference(intrinsics_4x4):
    tsdf_cfg = TSDFConfig(voxel_size=0.02)
    icp_cfg = ProjectiveICPConfig(
        n_pyramid_levels=2,
        iterations=(5, 5),
        damping=(1e-2, 1e-3),
        robust_loss="huber",
    )
    slam = RGBDTSDFSLAM(
        tsdf_config=tsdf_cfg,
        icp_config=icp_cfg,
        voxel_dim=(32, 32, 32),
        volume_origin=(-0.32, -0.32, 0.0),
        near=0.1,
        far=1.5,
        tracking_mode="hybrid",
        mapping_interval=10,
    )

    depth = _make_depth(z=0.5)
    rgb = torch.ones(48, 64, 3)
    frame = RGBDFrame(rgb=rgb, depth=depth, intrinsics=intrinsics_4x4)
    slam.process_frame(frame)
    result = slam.process_frame(frame)

    assert result.quality["tracking_source"] in {"previous", "keyframe"}
    assert result.lost is False
    assert result.quality["integrated"] is False


def test_pose_graph_and_loop_closure_enabled_short_run(intrinsics_4x4):
    """Enabling pose_graph + loop_closure must not raise and must leave the
    trajectory valid over a short synthetic run."""
    cv2 = pytest.importorskip("cv2")
    tsdf_cfg = TSDFConfig(voxel_size=0.02)
    icp_cfg = ProjectiveICPConfig(
        n_pyramid_levels=2,
        iterations=(5, 5),
        damping=(1e-2, 1e-3),
    )
    slam = RGBDTSDFSLAM(
        tsdf_config=tsdf_cfg,
        icp_config=icp_cfg,
        voxel_dim=(64, 64, 64),
        volume_origin=(-0.64, -0.64, 0.0),
        near=0.1,
        far=2.0,
        keyframe_max_frames=1,  # force a keyframe every frame
        pose_graph_enabled=True,
        pose_graph_window=8,
        loop_closure_enabled=True,
        keyframe_db_size=30,
        loop_closure_min_inliers=5,
    )

    H, W = 48, 64
    rgb = torch.zeros(H, W, 3)
    for i in range(0, H, 6):
        for j in range(0, W, 6):
            if (i + j) % 12 == 0:
                rgb[i:i + 3, j:j + 3, :] = 1.0
    depth = _make_depth(H=H, W=W, z=0.6)

    for _ in range(15):
        result = slam.process_frame(
            RGBDFrame(rgb=rgb.clone(), depth=depth.clone(), intrinsics=intrinsics_4x4)
        )
        assert result.lost is False
        T = result.T_world_camera
        assert torch.isfinite(T).all()
        R = T[:3, :3]
        assert abs(torch.linalg.det(R).item() - 1.0) < 0.05


def test_tsdf_tracking_mode_still_available(intrinsics_4x4):
    tsdf_cfg = TSDFConfig(voxel_size=0.02)
    icp_cfg = ProjectiveICPConfig(
        n_pyramid_levels=2,
        iterations=(5, 5),
        damping=(1e-2, 1e-3),
    )
    slam = RGBDTSDFSLAM(
        tsdf_config=tsdf_cfg,
        icp_config=icp_cfg,
        voxel_dim=(32, 32, 32),
        volume_origin=(-0.32, -0.32, 0.0),
        near=0.1,
        far=1.5,
        tracking_mode="tsdf",
    )

    depth = _make_depth()
    frame = RGBDFrame(rgb=None, depth=depth, intrinsics=intrinsics_4x4)
    slam.process_frame(frame)
    result = slam.process_frame(frame)

    assert result.quality["tracking_source"] == "tsdf"
