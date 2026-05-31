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


def test_batched_gradient_matches_six_call_reference(integrated_tsdf):
    """The batched-into-1 gradient grid_sample must produce normals identical to
    the original 6-separate-call central-difference computation."""
    import torch.nn.functional as F

    tsdf, K = integrated_tsdf
    result = raycast_tsdf(
        tsdf_volume=tsdf.tsdf,
        tsdf_origin=tsdf._origin,
        voxel_size=tsdf.config.voxel_size,
        T_world_camera=torch.eye(4),
        intrinsics=K,
        height=48, width=64, near=0.1, far=1.5, n_samples=64,
        normal_mode="gradient",
    )

    # Reference: recompute normals with 6 independent grid_sample calls at the
    # rendered surface points, mirroring the pre-optimization code exactly.
    vol = tsdf.tsdf
    nx, ny, nz = vol.shape
    origin = tsdf._origin
    vsize = tsdf.config.voxel_size
    tsdf_for_gs = vol.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
    vol_size = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=vol.dtype)

    # Recover surface points from the rendered depth (z-depth -> world point).
    H, W = 48, 64
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vv, uu = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    z = result.depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    surf = torch.stack([x, y, z], dim=-1).reshape(-1, 3)  # [N,3] world (identity pose)
    surf_vox = (surf - origin[None, :]) / vsize
    eps = 1.0

    def sample_at(off):
        pts = surf_vox + torch.tensor(off, dtype=vol.dtype)[None, :]
        pn = (2.0 * pts / vol_size[None, :] - 1.0).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return F.grid_sample(tsdf_for_gs, pn, mode="bilinear",
                             align_corners=True, padding_mode="border").reshape(-1)

    gx = sample_at([eps, 0, 0]) - sample_at([-eps, 0, 0])
    gy = sample_at([0, eps, 0]) - sample_at([0, -eps, 0])
    gz = sample_at([0, 0, eps]) - sample_at([0, 0, -eps])
    grad = torch.stack([gx, gy, gz], dim=-1)
    ref = F.normalize(grad, dim=-1)
    flip = (ref[:, 2] > 0).float() * 2 - 1
    ref = (ref * flip[:, None]).reshape(H, W, 3)

    m = result.mask
    assert torch.allclose(result.normal[m], ref[m], atol=1e-4)
