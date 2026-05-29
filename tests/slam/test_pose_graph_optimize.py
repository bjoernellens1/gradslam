import torch
import pytest
from gradslam.slam.pose_graph import SlidingWindowPoseGraph, se3_log
from gradslam.geometry.se3utils import se3_exp


def test_se3_log_identity():
    T = torch.eye(4)
    xi = se3_log(T)
    assert xi.norm().item() < 1e-5


def test_se3_log_roundtrip():
    """se3_log(se3_exp(xi)) ≈ xi for small twists."""
    xi = torch.tensor([0.01, -0.02, 0.03, 0.05, -0.03, 0.02])
    T = se3_exp(xi)
    xi_recovered = se3_log(T)
    assert (xi - xi_recovered).norm().item() < 1e-4, (
        f"Round-trip failed: {xi} vs {xi_recovered}"
    )


def test_se3_log_roundtrip_large_rotation():
    """Round-trip at ~1 radian rotation to exercise the V_inv formula."""
    xi = torch.tensor([0.05, -0.03, 0.02, 0.6, -0.5, 0.4])
    T = se3_exp(xi)
    xi_recovered = se3_log(T)
    assert (xi - xi_recovered).norm().item() < 1e-4, (
        f"Round-trip failed at large rotation: {xi} vs {xi_recovered}"
    )


def test_pose_graph_3_keyframe_convergence():
    """3-keyframe ring: optimizer recovers consistent poses."""
    T0 = torch.eye(4)
    T1 = torch.eye(4); T1[0, 3] = 0.1
    T2 = torch.eye(4); T2[1, 3] = 0.1

    # Use measured relative poses (ICP odometry), not derived from absolutes
    T_rel_01 = torch.linalg.inv(T0) @ T1  # inv(T0) @ T1
    T_rel_12 = torch.linalg.inv(T1) @ T2  # inv(T1) @ T2

    pg = SlidingWindowPoseGraph(window_size=8, n_iterations=10)
    pg.add_keyframe(T0)
    pg.add_keyframe(T1, T_rel_measured=T_rel_01)
    pg.add_keyframe(T2, T_rel_measured=T_rel_12)

    corrected = pg.optimize()
    assert len(corrected) == 3

    # First pose is fixed (anchor)
    assert torch.allclose(corrected[0], T0, atol=1e-6)

    # Other poses should be close to original (already consistent)
    assert (corrected[1] - T1).norm() < 0.05
    assert (corrected[2] - T2).norm() < 0.05


def test_pose_graph_corrects_drift():
    """Optimizer reduces residual when poses have accumulated error."""
    T0 = torch.eye(4)
    T1_true = torch.eye(4); T1_true[0, 3] = 0.1
    T2_true = torch.eye(4); T2_true[0, 3] = 0.2

    # Measured relative poses reflect true motion
    T_rel_01 = torch.linalg.inv(T0) @ T1_true
    T_rel_12 = torch.linalg.inv(T1_true) @ T2_true

    # But the absolute pose for KF2 is drifted by 5cm
    T2_noisy = T2_true.clone(); T2_noisy[0, 3] += 0.05

    pg = SlidingWindowPoseGraph(window_size=8, n_iterations=20, damping=1e-6)
    pg.add_keyframe(T0)
    pg.add_keyframe(T1_true, T_rel_measured=T_rel_01)
    pg.add_keyframe(T2_noisy, T_rel_measured=T_rel_12)

    corrected = pg.optimize()
    # Corrected T2 should be closer to T2_true than the noisy version
    err_before = (T2_noisy[0, 3] - T2_true[0, 3]).abs().item()
    err_after = (corrected[2][0, 3] - T2_true[0, 3]).abs().item()
    assert err_after < err_before, (
        f"Pose graph did not reduce drift: before={err_before:.4f}, after={err_after:.4f}"
    )
