"""Tests for TSDFVolume.integrate."""

import torch
import pytest
from gradslam.mapping.tsdf import TSDFConfig, TSDFVolume


@pytest.fixture
def small_tsdf():
    vd = torch.tensor([32, 32, 32])
    vo = torch.tensor([-0.32, -0.32, 0.0])
    cfg = TSDFConfig(voxel_size=0.02)
    return TSDFVolume(vd, vo, config=cfg, device=torch.device("cpu"))


@pytest.fixture
def simple_depth():
    """Flat depth plane at z = 0.5 m."""
    return torch.full((48, 64), 0.5)


@pytest.fixture
def intrinsics():
    return torch.tensor([
        [64.0,  0.0, 32.0],
        [ 0.0, 64.0, 24.0],
        [ 0.0,  0.0,  1.0],
    ])


def test_integrate_does_not_raise(small_tsdf, simple_depth, intrinsics):
    T = torch.eye(4)
    small_tsdf.integrate(simple_depth, intrinsics, T)


def _reference_integrate(tsdf, depth, K, T_world_camera, obs_weight=1.0):
    """Reference integrate that transforms ALL voxels (pre-optimization path),
    used to prove the frustum pre-cull is behavior-identical."""
    from gradslam.geometry.se3utils import se3_inv
    H, W = depth.shape
    Tcw = se3_inv(T_world_camera)
    R, t = Tcw[:3, :3], Tcw[:3, 3]
    vc = tsdf._cached_voxel_coords
    vw = tsdf._cached_voxel_world
    cam = vw @ R.t() + t
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = (cam[:, 0] * fx / cam[:, 2] + cx).long()
    v = (cam[:, 1] * fy / cam[:, 2] + cy).long()
    valid = (cam[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vc, u, v, cam = vc[valid], u[valid], v[valid], cam[valid]
    ds = depth[v, u]
    m = ds > 0
    vc, cam, ds = vc[m], cam[m], ds[m]
    trunc = tsdf.config.truncation_margin_voxels * tsdf.config.voxel_size
    td = torch.clamp(ds - cam[:, 2], -trunc, trunc)
    xi, yi, zi = vc[:, 0].long(), vc[:, 1].long(), vc[:, 2].long()
    fi = xi * tsdf._ny_nz + yi * tsdf._nz + zi
    tf, wf = tsdf.tsdf.reshape(-1), tsdf.weight.reshape(-1)
    ow = wf[fi]
    nw = ow + obs_weight
    tf.scatter_(0, fi, (tf[fi] * ow + td * obs_weight) / nw)
    wf.scatter_(0, fi, nw)


def test_frustum_precull_is_map_identical(small_tsdf, simple_depth, intrinsics):
    """The frustum depth pre-cull must yield a bit-identical TSDF + weights to
    the all-voxel reference, including a pose with voxels behind the camera."""
    from gradslam.geometry.se3utils import se3_exp
    # A rotated+translated pose so a chunk of the volume is behind the camera.
    T = se3_exp(torch.tensor([0.05, -0.03, 0.1, 0.2, -0.15, 0.1]))

    ref = TSDFVolume(torch.tensor([32, 32, 32]), torch.tensor([-0.32, -0.32, 0.0]),
                     config=small_tsdf.config, device=torch.device("cpu"))
    _reference_integrate(ref, simple_depth, intrinsics, T)
    small_tsdf.integrate(simple_depth, intrinsics, T)

    assert torch.equal(small_tsdf.weight, ref.weight)
    assert torch.allclose(small_tsdf.tsdf, ref.tsdf, atol=1e-6)
    assert (small_tsdf.weight > 0).any()  # the test actually exercised updates


def test_integrate_modifies_tsdf(small_tsdf, simple_depth, intrinsics):
    """After integration, some TSDF values should differ from initial +1."""
    T = torch.eye(4)
    small_tsdf.integrate(simple_depth, intrinsics, T)
    # Some voxels near the depth surface should be updated (weight > 0)
    assert (small_tsdf.weight > 0).any()


def test_integrate_tsdf_values_in_range(small_tsdf, simple_depth, intrinsics):
    T = torch.eye(4)
    small_tsdf.integrate(simple_depth, intrinsics, T)
    updated = small_tsdf.weight > 0
    tsdf_updated = small_tsdf.tsdf[updated]
    trunc = small_tsdf.config.truncation_margin_voxels * small_tsdf.config.voxel_size
    assert (tsdf_updated >= -trunc - 1e-5).all()
    assert (tsdf_updated <= trunc + 1e-5).all()


def test_integrate_zero_depth_ignored(small_tsdf, intrinsics):
    """Pixels with zero depth should not contribute to TSDF."""
    depth = torch.zeros(48, 64)
    T = torch.eye(4)
    small_tsdf.integrate(depth, intrinsics, T)
    assert (small_tsdf.weight == 0).all()


def test_integrate_weight_increases_with_frames(small_tsdf, simple_depth, intrinsics):
    """Integrating the same frame twice should increase weights."""
    T = torch.eye(4)
    small_tsdf.integrate(simple_depth, intrinsics, T)
    w1 = small_tsdf.weight.clone()
    small_tsdf.integrate(simple_depth, intrinsics, T)
    w2 = small_tsdf.weight.clone()
    # Weight should be >= what it was (never decrease)
    assert (w2 >= w1).all()
    assert (w2 > w1).any()


def test_tsdf_fuse_color_raises():
    """TSDFConfig.fuse_color=True should raise NotImplementedError."""
    config = TSDFConfig(fuse_color=True)
    vd = torch.tensor([16, 16, 16])
    vo = torch.tensor([-0.16, -0.16, 0.0])
    with pytest.raises(NotImplementedError, match="color fusion"):
        TSDFVolume(vd, vo, config=config)


def test_tsdf_integrate_color_arg_raises(small_tsdf, simple_depth, intrinsics):
    """Passing color to integrate() should raise NotImplementedError."""
    T = torch.eye(4)
    color = torch.zeros(48, 64, 3)
    with pytest.raises(NotImplementedError, match="color fusion"):
        small_tsdf.integrate(simple_depth, intrinsics, T, color=color)
