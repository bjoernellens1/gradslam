"""Tests for gradslam.geometry.normals."""

import torch
import pytest
from gradslam.geometry.normals import depth_to_vertex, estimate_normals, vertex_to_normal


@pytest.fixture
def flat_depth():
    """Flat depth plane at z=2.0."""
    return torch.full((60, 80), 2.0)


@pytest.fixture
def simple_intrinsics():
    K = torch.tensor([
        [80.0,  0.0, 40.0],
        [ 0.0, 80.0, 30.0],
        [ 0.0,  0.0,  1.0],
    ])
    return K


def test_depth_to_vertex_shape(flat_depth, simple_intrinsics):
    v = depth_to_vertex(flat_depth, simple_intrinsics)
    assert v.shape == (60, 80, 3)


def test_depth_to_vertex_z_equals_depth(flat_depth, simple_intrinsics):
    v = depth_to_vertex(flat_depth, simple_intrinsics)
    assert torch.allclose(v[..., 2], flat_depth)


def test_depth_to_vertex_center_pixel(simple_intrinsics):
    """Center pixel should project to (0, 0, depth)."""
    depth = torch.zeros(60, 80)
    depth[30, 40] = 3.0
    v = depth_to_vertex(depth, simple_intrinsics)
    center = v[30, 40]
    assert abs(center[0].item()) < 1e-4
    assert abs(center[1].item()) < 1e-4
    assert abs(center[2].item() - 3.0) < 1e-4


def test_estimate_normals_shape(flat_depth):
    n = estimate_normals(flat_depth)
    assert n.shape == (60, 80, 3)


def test_estimate_normals_flat_depth_points_z(flat_depth):
    """For a flat depth image, normals should point in -z direction."""
    n = estimate_normals(flat_depth)
    # Central pixels: dz/dx = 0, dz/dy = 0 → n ∝ (0, 0, 1)
    interior = n[5:-5, 5:-5]
    assert torch.allclose(interior[..., 0], torch.zeros_like(interior[..., 0]), atol=1e-5)
    assert torch.allclose(interior[..., 1], torch.zeros_like(interior[..., 1]), atol=1e-5)
    assert torch.allclose(interior[..., 2], torch.ones_like(interior[..., 2]), atol=1e-5)


def test_estimate_normals_unit_length(flat_depth):
    n = estimate_normals(flat_depth)
    norms = torch.linalg.norm(n, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_vertex_to_normal_shape(flat_depth, simple_intrinsics):
    v = depth_to_vertex(flat_depth, simple_intrinsics)
    n = vertex_to_normal(v)
    assert n.shape == (60, 80, 3)


def test_vertex_to_normal_unit_length(flat_depth, simple_intrinsics):
    v = depth_to_vertex(flat_depth, simple_intrinsics)
    n = vertex_to_normal(v)
    norms = torch.linalg.norm(n[5:-5, 5:-5], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
