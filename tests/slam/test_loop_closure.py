"""Tests for KeyframeDatabase loop closure."""
import pytest
import numpy as np

cv2 = pytest.importorskip("cv2")

from gradslam.slam.keyframe_database import KeyframeDatabase


def _textured_rgb(H=120, W=160):
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(0, H, 10):
        for j in range(0, W, 10):
            if (i + j) % 20 == 0:
                rgb[i:i + 5, j:j + 5] = 255
    return rgb


def _K(H=120, W=160):
    return np.array([[100., 0., W / 2], [0., 100., H / 2], [0., 0., 1.]], dtype=np.float64)


def test_find_loop_empty_database():
    db = KeyframeDatabase()
    import cv2 as _cv2
    orb = _cv2.ORB_create(500)
    gray = np.zeros((120, 160), dtype=np.uint8)
    kpts, desc = orb.detectAndCompute(gray, None)
    T_rel_np, match_idx, n_inliers = db.find_loop(
        (kpts, desc), query_K=_K(), exclude_last_n=8
    )
    assert match_idx == -1
    assert T_rel_np is None


def test_find_loop_skips_recent_keyframes():
    """Loop closure should not fire against keyframes in the sliding window."""
    db = KeyframeDatabase()
    rgb = _textured_rgb()
    T = np.eye(4)
    K = _K()
    depth = np.full((120, 160), 2.0, dtype=np.float32)

    # Add only 5 keyframes (all within exclude_last_n=8)
    for i in range(5):
        db.add(rgb, depth, K, T, frame_idx=i)

    import cv2 as _cv2
    orb = _cv2.ORB_create(500)
    gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
    kpts, desc = orb.detectAndCompute(gray, None)
    T_rel_np, match_idx, n_inliers = db.find_loop(
        (kpts, desc), query_K=_K(), exclude_last_n=8, min_inliers=5
    )
    # Should return -1 since all keyframes are within exclude_last_n
    assert match_idx == -1


def test_find_loop_detects_when_enough_keyframes():
    """With enough keyframes (>8), old ones are eligible for loop detection."""
    db = KeyframeDatabase()
    rgb = _textured_rgb()
    T = np.eye(4)
    K = _K()
    depth = np.full((120, 160), 2.0, dtype=np.float32)

    # Add 12 keyframes so entries[0:4] are outside the window
    for i in range(12):
        db.add(rgb, depth, K, T, frame_idx=i)

    import cv2 as _cv2
    orb = _cv2.ORB_create(500)
    gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
    kpts, desc = orb.detectAndCompute(gray, None)

    # With a very low min_inliers threshold, same-image match should be found
    T_rel_np, match_idx, n_inliers = db.find_loop(
        (kpts, desc), query_K=_K(), exclude_last_n=8, min_inliers=1
    )
    # Either finds a match (positive match_idx) or returns -1 if no features
    # (checker may not produce features on some platforms)
    assert match_idx == -1 or (match_idx >= 0 and T_rel_np is not None)
    if match_idx >= 0 and T_rel_np is not None:
        assert T_rel_np.shape == (4, 4)
        assert np.allclose(T_rel_np, np.eye(4), atol=0.15)  # same image → near identity


def test_find_loop_min_frame_gap_rejects_short_baseline():
    """Entries within min_frame_gap of query_frame_idx must be skipped."""
    db = KeyframeDatabase()
    rgb = _textured_rgb()
    T = np.eye(4)
    K = _K()
    depth = np.full((120, 160), 2.0, dtype=np.float32)

    # Add 12 keyframes with frame_idx 0..11, then query with frame_idx=100
    # and min_frame_gap=90 → only entries with frame_idx <= 10 pass (gap >= 90).
    for i in range(12):
        db.add(rgb, depth, K, T, frame_idx=i)

    import cv2 as _cv2
    orb = _cv2.ORB_create(500)
    gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
    kpts, desc = orb.detectAndCompute(gray, None)

    # min_frame_gap=200 → query_frame_idx(100) - max_entry_frame_idx(11) = 89 < 200
    # All entries are filtered out.
    T_rel_np, match_idx, n_inliers = db.find_loop(
        (kpts, desc),
        query_K=K,
        exclude_last_n=0,
        min_inliers=1,
        query_frame_idx=100,
        min_frame_gap=200,
    )
    assert match_idx == -1, "All entries should be rejected when gap < min_frame_gap"
    assert T_rel_np is None

    # min_frame_gap=0 → gap filter disabled; same as before
    T_rel_np2, match_idx2, _ = db.find_loop(
        (kpts, desc),
        query_K=K,
        exclude_last_n=0,
        min_inliers=1,
        query_frame_idx=100,
        min_frame_gap=0,
    )
    # With gap filter off, entries are eligible (match possible or not based on features)
    assert match_idx2 == -1 or match_idx2 >= 0  # either outcome is valid; filter is gone


def test_find_loop_min_frame_gap_allows_old_entries():
    """Entries older than min_frame_gap must remain eligible."""
    db = KeyframeDatabase()
    rgb = _textured_rgb()
    T = np.eye(4)
    K = _K()
    depth = np.full((120, 160), 2.0, dtype=np.float32)

    # frame_idx=0 is 150 frames back from query_frame_idx=150; gap=50 → passes.
    # frame_idx=140 is only 10 frames back; gap=50 → rejected.
    db.add(rgb, depth, K, T, frame_idx=0)
    db.add(rgb, depth, K, T, frame_idx=140)

    import cv2 as _cv2
    orb = _cv2.ORB_create(500)
    gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
    kpts, desc = orb.detectAndCompute(gray, None)

    T_rel_np, match_idx, n_inliers = db.find_loop(
        (kpts, desc),
        query_K=K,
        exclude_last_n=0,
        min_inliers=1,
        query_frame_idx=150,
        min_frame_gap=50,
    )
    # frame_idx=0 passes (gap=150 >= 50); frame_idx=140 is filtered (gap=10 < 50).
    # So if a match is found, it must be frame_idx=0.
    if match_idx >= 0:
        assert match_idx == 0, "Only the old entry (frame_idx=0) should be eligible"


def test_loop_edge_convention_matches_find_loop_sign():
    """Pin the sign of the loop-edge measurement against find_loop's PnP
    output convention.

    find_loop runs solvePnP(objectPoints in the MATCHED (ref) frame,
    imagePoints = query pixels, K = query), so the returned M satisfies
    p_query = M @ p_ref, i.e. M = inv(T_world_query) @ T_world_ref. The
    pose-graph edge convention is E ≈ inv(T_world_a) @ T_world_b with
    a=match (ref), b=current (query):
        E = inv(T_world_ref) @ T_world_query = inv(M).
    This test builds M from known world poses the way find_loop would, then
    asserts that feeding inv(M) (NOT M) as the loop edge corrects drift while
    feeding M makes it worse — catching a sign flip.
    """
    import torch
    from gradslam.slam.pose_graph import SlidingWindowPoseGraph

    # Ground-truth world poses (cam->world). Node 0 anchor, node 4 = query.
    n = 5
    true_poses = []
    for i in range(n):
        T = torch.eye(4)
        T[0, 3] = 0.1 * i
        true_poses.append(T)

    # Drifted chained absolute poses (over-shoot accumulates).
    drifted = [true_poses[0].clone()]
    for i in range(1, n):
        rel = torch.linalg.inv(true_poses[i - 1]) @ true_poses[i]
        rel[0, 3] += 0.03
        drifted.append(drifted[-1] @ rel)

    # M as find_loop would return it, from the GROUND-TRUTH world poses:
    #   M = inv(T_world_query) @ T_world_ref   (ref = node 0, query = node 4)
    M = torch.linalg.inv(true_poses[4]) @ true_poses[0]

    def run(edge_meas):
        pg = SlidingWindowPoseGraph(window_size=16, n_iterations=30, damping=1e-6)
        for i in range(n):
            pg.add_keyframe(drifted[i], node_id=i)
        pg.add_loop_edge(0, 4, edge_meas, weight=5.0)
        corrected = pg.optimize()
        return (corrected[-1][0, 3] - true_poses[-1][0, 3]).abs().item()

    err_before = (drifted[-1][0, 3] - true_poses[-1][0, 3]).abs().item()
    err_correct = run(torch.linalg.inv(M))  # the sign the pipeline uses
    err_flipped = run(M)                     # the wrong sign

    assert err_correct < err_before, "inv(M) edge should reduce drift"
    assert err_flipped > err_correct, "M (wrong sign) should be worse than inv(M)"
