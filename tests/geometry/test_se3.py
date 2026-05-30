"""Tests for SE(3) exp/log and pose convention."""

import torch
import math
import pytest
from gradslam.geometry.se3utils import se3_exp, se3_log


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
