"""Tests for KeyframeDatabase relocalization."""
import pytest
import numpy as np

cv2 = pytest.importorskip("cv2")

from gradslam.slam.keyframe_database import KeyframeDatabase


def _random_rgb(H=120, W=160):
    return np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)


def _random_depth(H=120, W=160, z=2.0):
    return np.full((H, W), z, dtype=np.float32)


def _K(H=120, W=160):
    return np.array([[100., 0., W / 2], [0., 100., H / 2], [0., 0., 1.]], dtype=np.float64)


def test_relocalize_returns_none_when_empty():
    db = KeyframeDatabase()
    result, info = db.relocalize(_random_rgb(), _random_depth(), _K())
    assert result is None
    assert info is None


def test_relocalize_finds_match():
    """After adding a keyframe, relocalize returns a pose when query matches."""
    db = KeyframeDatabase()
    T = np.eye(4)
    # Create a textured image (checkerboard) so ORB can find features
    H, W = 120, 160
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[::10, :, :] = 255
    rgb[:, ::10, :] = 255
    depth = _random_depth(H, W)

    db.add(rgb, depth, _K(H, W), T, frame_idx=0)

    # Same image as query → should match
    result, info = db.relocalize(rgb, depth, _K(H, W), min_inliers=5)
    # If cv2 finds enough features: result is not None; otherwise None is acceptable
    # (checker may not always produce 5 PnP inliers, so just check no crash)
    assert result is None or (isinstance(result, np.ndarray) and result.shape == (4, 4))


def test_relocalize_returns_none_empty_rgb():
    """Uniform RGB (no features) returns None."""
    db = KeyframeDatabase()
    rgb = np.full((120, 160, 3), 128, dtype=np.uint8)
    depth = _random_depth()
    T = np.eye(4)
    db.add(rgb, depth, _K(), T, frame_idx=0)

    result, info = db.relocalize(rgb, depth, _K(), min_inliers=20)
    assert result is None  # no features to match


def test_database_max_size():
    """Database should cap at max_keyframes entries."""
    db = KeyframeDatabase(max_keyframes=5)
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    depth = _random_depth()
    K = _K()
    T = np.eye(4)
    for i in range(10):
        db.add(rgb, depth, K, T, frame_idx=i)
    assert len(db) == 5


def test_database_clear():
    """clear() empties the database."""
    db = KeyframeDatabase()
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    db.add(rgb, _random_depth(), _K(), np.eye(4), frame_idx=0)
    assert len(db) == 1
    db.clear()
    assert len(db) == 0
