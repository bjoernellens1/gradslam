"""Tests for SE(3) exp/log and pose convention."""

import torch
import math
import pytest
from gradslam.geometry.se3utils import se3_exp, se3_log, se3_inv


def test_se3_inv_matches_linalg_inv():
    """Analytic se3_inv equals torch.linalg.inv for proper rigid transforms,
    and satisfies T @ inv(T) = I."""
    torch.manual_seed(0)
    for scale in (0.05, 0.5, 1.2):
        for _ in range(5):
            xi = torch.randn(6) * scale
            xi[3:] = xi[3:].clamp(-1.0, 1.0)
            T = se3_exp(xi)
            Ti = se3_inv(T)
            assert torch.allclose(Ti, torch.linalg.inv(T), atol=1e-5), (
                f"se3_inv != linalg.inv at scale {scale}"
            )
            assert torch.allclose(T @ Ti, torch.eye(4), atol=1e-5)
            assert torch.allclose(Ti @ T, torch.eye(4), atol=1e-5)


def test_se3_inv_preserves_dtype_device():
    T = se3_exp(torch.randn(6) * 0.3)
    Ti = se3_inv(T)
    assert Ti.dtype == T.dtype and Ti.device == T.device
    assert Ti.shape == (4, 4)


def test_exp_log_roundtrip():
    """exp(log(T)) ≈ T and log(exp(ξ)) ≈ ξ for various magnitudes."""
    torch.manual_seed(42)

    # Small twists
    for _ in range(5):
        xi = torch.randn(6) * 0.05
        T = se3_exp(xi)
        xi_back = se3_log(T)
        assert (xi - xi_back).norm() < 1e-4, f"log(exp(xi)) failed: {xi} -> {xi_back}"

    # Medium twists (up to ~1 rad)
    for _ in range(5):
        xi = torch.randn(6) * 0.3
        # Clamp rotation part to stay below pi
        xi[3:] = xi[3:].clamp(-1.0, 1.0)
        T = se3_exp(xi)
        xi_back = se3_log(T)
        assert (xi - xi_back).norm() < 1e-4, f"log(exp(xi)) failed: {xi} -> {xi_back}"

    # exp(log(T)) ≈ T
    for _ in range(5):
        xi = torch.randn(6) * 0.2
        xi[3:] = xi[3:].clamp(-1.0, 1.0)
        T = se3_exp(xi)
        T_back = se3_exp(se3_log(T))
        assert (T - T_back).norm() < 1e-4, f"exp(log(T)) failed"


def test_inverse():
    """T @ inv(T) ≈ I for various SE(3) matrices."""
    torch.manual_seed(7)
    for _ in range(10):
        xi = torch.randn(6) * 0.3
        xi[3:] = xi[3:].clamp(-1.0, 1.0)
        T = se3_exp(xi)
        T_inv = torch.linalg.inv(T)
        product = T @ T_inv
        assert (product - torch.eye(4)).norm() < 1e-5, "T @ inv(T) != I"

        product2 = T_inv @ T
        assert (product2 - torch.eye(4)).norm() < 1e-5, "inv(T) @ T != I"


def test_pose_convention():
    """T_world_camera maps camera points to world frame; inv gives T_camera_world."""
    # Create a simple T_world_camera: camera translated 1m along world x-axis,
    # no rotation (camera frame aligned with world frame).
    T_world_camera = torch.eye(4)
    T_world_camera[0, 3] = 1.0  # camera origin is at x=1 in world

    # A point at the camera origin (0,0,0 in camera frame)
    p_camera = torch.tensor([0.0, 0.0, 0.0, 1.0])

    # Apply T_world_camera to get world coordinates
    p_world = T_world_camera @ p_camera
    # Camera origin should be at world position (1, 0, 0)
    assert abs(p_world[0].item() - 1.0) < 1e-6
    assert abs(p_world[1].item() - 0.0) < 1e-6
    assert abs(p_world[2].item() - 0.0) < 1e-6

    # Round-trip: T_camera_world = inv(T_world_camera)
    T_camera_world = torch.linalg.inv(T_world_camera)
    p_camera_back = T_camera_world @ p_world
    assert (p_camera_back - p_camera).norm() < 1e-6, "Round-trip failed"

    # More general: random pose, random point, verify round-trip
    torch.manual_seed(99)
    xi = torch.tensor([0.3, -0.1, 0.2, 0.2, -0.3, 0.1])
    T_wc = se3_exp(xi)
    T_cw = torch.linalg.inv(T_wc)

    p_cam = torch.tensor([1.0, 2.0, 3.0, 1.0])
    p_world_2 = T_wc @ p_cam
    p_cam_back = T_cw @ p_world_2
    assert (p_cam_back - p_cam).norm() < 1e-5, "General round-trip failed"
