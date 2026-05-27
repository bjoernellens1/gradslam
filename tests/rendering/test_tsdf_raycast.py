"""Tests for TSDF raycasting."""

import torch
import pytest
from gradslam.mapping.tsdf import TSDFConfig, TSDFVolume
from gradslam.rendering.tsdf_raycast import raycast_tsdf, RenderedFrame


@pytest.fixture
def integrated_tsdf():
    """A small TSDF volume with one integrated depth frame."""
    vd = torch.tensor([32, 32, 32])
    vo = torch.tensor([-0.32, -0.32, 0.0])
    cfg = TSDFConfig(voxel_size=0.02)
    tsdf = TSDFVolume(vd, vo, config=cfg, device=torch.device("cpu"))

    depth = torch.full((48, 64), 0.5)
    K = torch.tensor([
        [64.0,  0.0, 32.0],
        [ 0.0, 64.0, 24.0],
        [ 0.0,  0.0,  1.0],
    ])
    T = torch.eye(4)
    tsdf.integrate(depth, K, T)
    return tsdf, K


def test_raycast_returns_rendered_frame(integrated_tsdf):
    tsdf, K = integrated_tsdf
    result = raycast_tsdf(
        tsdf_volume=tsdf.tsdf,
        tsdf_origin=tsdf._origin,
        voxel_size=tsdf.config.voxel_size,
        T_world_camera=torch.eye(4),
        intrinsics=K,
        height=48,
        width=64,
        near=0.1,
        far=1.5,
        n_samples=64,
    )
    assert isinstance(result, RenderedFrame)
    assert result.depth.shape == (48, 64)
    assert result.normal.shape == (48, 64, 3)
    assert result.mask.shape == (48, 64)


def test_raycast_has_valid_pixels(integrated_tsdf):
    """After integrating a depth plane, raycasting should find some valid pixels."""
    tsdf, K = integrated_tsdf
    result = raycast_tsdf(
        tsdf_volume=tsdf.tsdf,
        tsdf_origin=tsdf._origin,
        voxel_size=tsdf.config.voxel_size,
        T_world_camera=torch.eye(4),
        intrinsics=K,
        height=48,
        width=64,
        near=0.1,
        far=1.5,
        n_samples=64,
    )
    assert result.mask.any(), "Expected at least some valid raycasted pixels"


def test_raycast_depth_positive_where_valid(integrated_tsdf):
    tsdf, K = integrated_tsdf
    result = raycast_tsdf(
        tsdf_volume=tsdf.tsdf,
        tsdf_origin=tsdf._origin,
        voxel_size=tsdf.config.voxel_size,
        T_world_camera=torch.eye(4),
        intrinsics=K,
        height=48,
        width=64,
        near=0.1,
        far=1.5,
        n_samples=64,
    )
    valid_depth = result.depth[result.mask]
    assert (valid_depth > 0).all()


def test_raycast_normals_unit_length_where_valid(integrated_tsdf):
    tsdf, K = integrated_tsdf
    result = raycast_tsdf(
        tsdf_volume=tsdf.tsdf,
        tsdf_origin=tsdf._origin,
        voxel_size=tsdf.config.voxel_size,
        T_world_camera=torch.eye(4),
        intrinsics=K,
        height=48,
        width=64,
        near=0.1,
        far=1.5,
        n_samples=64,
    )
    valid_normals = result.normal[result.mask]
    norms = torch.linalg.norm(valid_normals, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
