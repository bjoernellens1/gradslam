"""Tests for the pypose-backed GlobalPoseGraph (Workstream B).

pypose is only installed in the ROCm container, so these tests skip when it is
unavailable (e.g. the CPU host). They lock in the mechanism + the two invariants
that the B-i shipped-path test exposed: immutable sequential edges, and a
finalize() that corrects the trajectory tail.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("pypose")

from gradslam.geometry.se3utils import se3_exp, se3_inv
from gradslam.slam.global_pose_graph import GlobalPoseGraph


def _drift_trajectory(n=12, drift=0.03):
    """Ground-truth straight-line + a chained estimate that over-shoots, so the
    last node is wrong by a known amount. Returns (gt[list 4x4], est[list 4x4])."""
    gt = []
    for i in range(n):
        T = torch.eye(4, dtype=torch.float64)
        T[0, 3] = 0.1 * i
        gt.append(T)
    est = [gt[0].clone()]
    for i in range(1, n):
        rel = se3_inv(gt[i - 1]) @ gt[i]
        rel = rel @ se3_exp(torch.tensor([drift, 0, 0, 0.0, 0, 0]).double())
        est.append(est[-1] @ rel)
    return gt, est


def test_sequential_edges_immutable_across_commits():
    """The sequential-edge measurements must NOT change because a loop commit
    mutated the optimization state in between (the incremental-vs-batch bug)."""
    gt, est = _drift_trajectory()
    pg = GlobalPoseGraph()
    for j in range(5):
        pg.add_keyframe(est[j], node_id=j)
    edges_before = [(a, b, M.clone()) for a, b, M, _ in pg._edges]
    # Add a loop edge between node 0 and 4 and commit (mutates _poses).
    M = se3_inv(gt[0]) @ gt[4]
    pg.add_loop_edge(0, 4, M, weight=10.0)
    pg.try_commit_correction(max_translation_step=1e9)
    # Now add another keyframe — its sequential edge must derive from RAW poses.
    pg.add_keyframe(est[5], node_id=5)
    # Find the new (4->5) sequential edge by its node ids (not position: a loop
    # edge was inserted before it).
    seq_edge = next(e for e in pg._edges if e[0] == 4 and e[1] == 5)
    expected = se3_inv(est[4]) @ est[5]   # raw odometry, NOT corrected
    assert torch.allclose(seq_edge[2], expected, atol=1e-9)
    # And the earlier sequential edges are untouched by the commit.
    for (a0, b0, M0), (a1, b1, M1, _) in zip(edges_before, pg._edges[:4]):
        assert a0 == a1 and b0 == b1 and torch.allclose(M0, M1, atol=1e-9)


def test_loop_plus_finalize_reduces_drift():
    """A single GT loop edge + finalize() reduces the last-node error well below
    the accumulated drift (the core B mechanism)."""
    gt, est = _drift_trajectory(n=12, drift=0.04)
    pg = GlobalPoseGraph(n_iterations=30)
    for j in range(12):
        pg.add_keyframe(est[j], node_id=j)
    err_before = float((pg._poses[-1][:3, 3] - gt[-1][:3, 3]).norm())
    assert err_before > 0.1, "test setup: drift should be significant"
    M = se3_inv(gt[0]) @ gt[11]
    assert pg.add_loop_edge(0, 11, M, weight=20.0)
    pg.finalize()
    err_after = float((pg._poses[-1][:3, 3] - gt[-1][:3, 3]).norm())
    assert err_after < 0.3 * err_before, f"{err_before:.3f} -> {err_after:.3f}"


def test_try_commit_rejects_divergent_loop():
    """A garbage loop edge must be rejected (None, state unchanged), mirroring
    the SlidingWindowPoseGraph safety contract."""
    gt, est = _drift_trajectory()
    pg = GlobalPoseGraph()
    for j in range(8):
        pg.add_keyframe(est[j], node_id=j)
    before = [p.clone() for p in pg._poses]
    bad = se3_exp(torch.tensor([5.0, -4.0, 6.0, 1.0, -1.0, 2.0]).double())
    assert pg.add_loop_edge(0, 7, bad, weight=20.0)
    res = pg.try_commit_correction(max_translation_step=2.0)
    assert res is None
    pg.drop_last_edge()
    assert all(torch.allclose(a, b) for a, b in zip(before, pg._poses))


def test_add_loop_edge_unknown_id_returns_false():
    pg = GlobalPoseGraph()
    for j in range(3):
        pg.add_keyframe(torch.eye(4, dtype=torch.float64), node_id=j)
    assert pg.add_loop_edge(0, 2, torch.eye(4, dtype=torch.float64)) is True
    assert pg.add_loop_edge(0, 99, torch.eye(4, dtype=torch.float64)) is False
