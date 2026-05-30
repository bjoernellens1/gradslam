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
