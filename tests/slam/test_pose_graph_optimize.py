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


def test_loop_edge_reduces_drift():
    """The key regression: a single loop edge between an early node and the
    last node, measured from the GROUND-TRUTH relative, must reduce the
    last-node position error.  Validates the generalized optimizer AND the
    edge sign/convention (T_rel_meas ≈ inv(T_world_a) @ T_world_b)."""
    n = 6
    # Ground-truth: pure translation along +x, 0.1 m per step.
    true_poses = []
    for i in range(n):
        T = torch.eye(4)
        T[0, 3] = 0.1 * i
        true_poses.append(T)

    # Chained (drifted) absolute poses: each step is over-estimated, so error
    # accumulates and the last pose is wrong by a known amount.
    drift_step = 0.02  # 2 cm of over-shoot per chained step
    drifted = [true_poses[0].clone()]
    for i in range(1, n):
        rel = torch.linalg.inv(true_poses[i - 1]) @ true_poses[i]
        rel_drift = rel.clone()
        rel_drift[0, 3] += drift_step
        drifted.append(drifted[-1] @ rel_drift)

    pg = SlidingWindowPoseGraph(window_size=16, n_iterations=30, damping=1e-6)
    # Sequential no-op edges (measurement derived from the drifted absolutes).
    for i in range(n):
        pg.add_keyframe(drifted[i], node_id=i)

    err_before = (pg._poses[-1][0, 3] - true_poses[-1][0, 3]).abs().item()
    assert err_before > 0.05, "test setup: drift should be significant"

    # One loop edge between node 0 (anchor) and the last node, measured from
    # the GROUND-TRUTH relative pose.
    M = torch.linalg.inv(true_poses[0]) @ true_poses[n - 1]
    ok = pg.add_loop_edge(0, n - 1, M, weight=5.0)
    assert ok is True

    corrected = pg.optimize()
    err_after = (corrected[-1][0, 3] - true_poses[-1][0, 3]).abs().item()
    assert err_after < err_before, (
        f"Loop edge did not reduce drift: before={err_before:.4f}, after={err_after:.4f}"
    )
    # Anchor stays fixed.
    assert torch.allclose(corrected[0], true_poses[0], atol=1e-6)


def test_apply_correction_commits_get_corrected_does_not():
    """apply_correction() must persist optimized poses into self._poses;
    get_corrected_poses() must not mutate. Regression for the bug where a
    loop correction was discarded and re-injected as fake odometry on the
    next keyframe."""
    n = 5
    true_poses = [torch.eye(4) for _ in range(n)]
    for i in range(n):
        true_poses[i][0, 3] = 0.1 * i
    drifted = [true_poses[0].clone()]
    for i in range(1, n):
        rel = torch.linalg.inv(true_poses[i - 1]) @ true_poses[i]
        rel[0, 3] += 0.02
        drifted.append(drifted[-1] @ rel)

    pg = SlidingWindowPoseGraph(window_size=16, n_iterations=30, damping=1e-6)
    for i in range(n):
        pg.add_keyframe(drifted[i], node_id=i)
    pg.add_loop_edge(0, n - 1, torch.linalg.inv(true_poses[0]) @ true_poses[n - 1], weight=5.0)

    before = [p.clone() for p in pg._poses]
    # Read-only path leaves internal state untouched.
    _ = pg.get_corrected_poses()
    assert all(torch.allclose(a, b) for a, b in zip(pg._poses, before))

    # Commit path persists the correction.
    corrected = pg.apply_correction()
    assert all(torch.allclose(a, b) for a, b in zip(pg._poses, corrected))
    assert not torch.allclose(pg._poses[-1], before[-1])

    # The next keyframe's auto-derived sequential edge is now measured from the
    # CORRECTED last pose, so adding a consistent next pose introduces no spurious
    # residual: optimizing again must not move the committed poses appreciably.
    next_true = torch.eye(4); next_true[0, 3] = 0.1 * n
    rel_next = torch.linalg.inv(pg._poses[-1]) @ (pg._poses[-1] @ (torch.linalg.inv(true_poses[n - 2]) @ next_true))
    pg.add_keyframe(pg._poses[-1] @ rel_next, node_id=n)
    committed = [p.clone() for p in pg._poses]
    re_opt = pg.optimize()
    assert all((a - b).norm().item() < 1e-3 for a, b in zip(re_opt, committed))


def test_try_commit_rejects_divergent_loop():
    """try_commit_correction must reject (return None, leave state unchanged) a
    loop edge whose correction is non-finite or jumps implausibly far, so a bad
    PnP loop measurement cannot corrupt the trajectory or accumulate into
    numerical divergence. Regression for the 1e24 blow-up observed on real data."""
    n = 10
    pg = SlidingWindowPoseGraph(window_size=16, n_iterations=5, damping=1e-4)
    for i in range(n):
        T = torch.eye(4)
        T[0, 3] = 0.1 * i
        pg.add_keyframe(T, node_id=i)
        if pg.num_keyframes >= 2:
            pg.try_commit_correction()

    before = [p.clone() for p in pg._poses]
    n_edges_before = len(pg._edges)
    # A garbage loop measurement (huge, geometrically wrong) that passed an
    # inlier gate upstream.
    bad = se3_exp(torch.tensor([5.0, -4.0, 6.0, 1.0, -1.0, 2.0]))
    assert pg.add_loop_edge(0, n - 1, bad, weight=2.0) is True
    result = pg.try_commit_correction()
    assert result is None, "divergent loop correction must be rejected"
    pg.drop_last_edge()

    assert len(pg._edges) == n_edges_before
    assert all(torch.allclose(a, b) for a, b in zip(before, pg._poses))
    assert all(torch.isfinite(p).all() for p in pg._poses)


def test_add_loop_edge_out_of_window_returns_false():
    """add_loop_edge must return False when a referenced node id is not in
    the current window, and must not corrupt the edge list."""
    pg = SlidingWindowPoseGraph(window_size=8)
    for i in range(3):
        T = torch.eye(4)
        T[0, 3] = 0.1 * i
        pg.add_keyframe(T, node_id=i)

    M = torch.eye(4)
    assert pg.add_loop_edge(0, 2, M, weight=1.0) is True  # both in window
    assert pg.add_loop_edge(0, 99, M, weight=1.0) is False  # 99 not present
    assert pg.add_loop_edge(99, 2, M, weight=1.0) is False  # 99 not present


def test_trimming_drops_edges_referencing_dropped_nodes():
    """When the window slides, nodes and any edges referencing dropped node
    ids are removed.  A loop edge to a node that later slides out is gone."""
    pg = SlidingWindowPoseGraph(window_size=3)
    for i in range(3):
        T = torch.eye(4)
        T[0, 3] = 0.1 * i
        pg.add_keyframe(T, node_id=i)
    # Loop edge between node 0 and node 2.
    assert pg.add_loop_edge(0, 2, torch.eye(4), weight=1.0) is True

    # Adding two more keyframes slides node 0 (and 1) out of the window.
    for i in range(3, 5):
        T = torch.eye(4)
        T[0, 3] = 0.1 * i
        pg.add_keyframe(T, node_id=i)

    node_ids = pg.node_ids()
    assert 0 not in node_ids
    # No remaining edge may reference the dropped node 0.
    for a, b, _M, _w in pg._edges:
        assert a in node_ids and b in node_ids


def test_auto_assigned_node_ids_backward_compat():
    """Callers that don't pass node_id keep working (auto-incrementing ids)."""
    pg = SlidingWindowPoseGraph(window_size=8)
    T0 = torch.eye(4)
    T1 = torch.eye(4); T1[0, 3] = 0.1
    pg.add_keyframe(T0)
    pg.add_keyframe(T1, T_rel_measured=torch.linalg.inv(T0) @ T1)
    assert pg.num_keyframes == 2
    corrected = pg.optimize()
    assert len(corrected) == 2
