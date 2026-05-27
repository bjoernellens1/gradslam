"""Tests for gradslam.geometry.image_pyramid."""

import torch
import pytest
from gradslam.geometry.image_pyramid import (
    build_depth_pyramid,
    build_normal_pyramid,
    scale_intrinsics,
)


@pytest.fixture
def depth():
    return torch.rand(64, 96) + 0.5  # positive depth


@pytest.fixture
def normal():
    n = torch.randn(64, 96, 3)
    import torch.nn.functional as F
    return F.normalize(n, dim=-1)


def test_depth_pyramid_length(depth):
    pyr = build_depth_pyramid(depth, n_levels=3)
    assert len(pyr) == 3


def test_depth_pyramid_coarsest_first(depth):
    pyr = build_depth_pyramid(depth, n_levels=3)
    # Level 0 = coarsest (most downsampled)
    assert pyr[0].shape == (16, 24)  # 64/4, 96/4
    assert pyr[1].shape == (32, 48)  # 64/2, 96/2
    assert pyr[2].shape == (64, 96)  # full res


def test_depth_pyramid_single_level(depth):
    pyr = build_depth_pyramid(depth, n_levels=1)
    assert len(pyr) == 1
    assert pyr[0].shape == depth.shape


def test_normal_pyramid_length(normal):
    pyr = build_normal_pyramid(normal, n_levels=3)
    assert len(pyr) == 3


def test_normal_pyramid_coarsest_first(normal):
    pyr = build_normal_pyramid(normal, n_levels=3)
    assert pyr[0].shape == (16, 24, 3)
    assert pyr[2].shape == (64, 96, 3)


def test_normal_pyramid_unit_length(normal):
    pyr = build_normal_pyramid(normal, n_levels=3)
    for level in pyr:
        norms = torch.linalg.norm(level, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_scale_intrinsics():
    K = torch.tensor([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])
    K2 = scale_intrinsics(K, scale=2.0)
    assert abs(K2[0, 0].item() - 250.0) < 1e-5
    assert abs(K2[1, 1].item() - 250.0) < 1e-5
    assert abs(K2[0, 2].item() - 160.0) < 1e-5
    assert abs(K2[1, 2].item() - 120.0) < 1e-5
    # Off-diagonal and last row unchanged
    assert K2[2, 2].item() == 1.0


def test_scale_intrinsics_does_not_modify_input():
    K = torch.tensor([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])
    K_orig = K.clone()
    _ = scale_intrinsics(K, scale=4.0)
    assert torch.allclose(K, K_orig)
