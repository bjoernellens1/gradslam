import torch
import math
from gradslam.slam.pose_graph import se3_log
from gradslam.geometry.se3utils import se3_exp


def test_se3_log_identity():
    assert se3_log(torch.eye(4)).norm() < 1e-6


def test_se3_log_pure_translation():
    T = torch.eye(4)
    T[0, 3] = 0.5
    xi = se3_log(T)
    assert abs(xi[0].item() - 0.5) < 1e-5
    assert xi[3:].norm() < 1e-5


def test_se3_log_roundtrip_various():
    """Round-trip log/exp on several random twists."""
    for _ in range(5):
        xi = torch.randn(6) * 0.1  # small twists
        T = se3_exp(xi)
        xi_back = se3_log(T)
        assert (xi - xi_back).norm() < 1e-4, f"Round-trip failed: {xi} -> {xi_back}"


def test_se3_log_roundtrip_moderate_rotation():
    """Round-trip at moderate (~0.5 rad) rotation exercising V_inv formula."""
    xi = torch.tensor([0.1, -0.05, 0.08, 0.3, -0.4, 0.2])
    T = se3_exp(xi)
    xi_back = se3_log(T)
    assert (xi - xi_back).norm() < 1e-4, f"Round-trip failed: {xi} -> {xi_back}"


def test_se3_log_roundtrip_1radian():
    """Round-trip at ~1 radian rotation to stress-test V_inv coefficient."""
    xi = torch.tensor([0.05, 0.03, -0.02, 0.6, 0.5, -0.4])
    T = se3_exp(xi)
    xi_back = se3_log(T)
    assert (xi - xi_back).norm() < 1e-4, f"Round-trip failed at 1 rad: {xi} -> {xi_back}"
