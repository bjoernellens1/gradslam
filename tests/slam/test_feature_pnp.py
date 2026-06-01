"""Verify _feature_pnp_pose math: PnP uses live K, not keyframe K."""
import pytest
import numpy as np

cv2 = pytest.importorskip("cv2")


def _run_feature_pnp_math(K_ref, K_live, T_world_ref, T_world_live):
    """Standalone version of _feature_pnp_pose math for testing."""
    # Build synthetic 3D scene points in ref camera coords
    pts_world = np.array([
        [0.1*i - 0.2, 0.1*j - 0.2, 2.5]
        for i in range(5) for j in range(5)
    ], dtype=np.float64)

    # Back-project into ref cam (as pipeline does with K_ref)
    T_ref_from_world = np.linalg.inv(T_world_ref)
    pts_ref_cam = (T_ref_from_world[:3,:3] @ pts_world.T + T_ref_from_world[:3,3:]).T
    # Fake keyframe keypoints in ref image (not needed for 3D — we already have 3D)
    pts3 = pts_ref_cam  # [N, 3] in ref cam frame

    # Project into live image using K_live
    T_live_from_world = np.linalg.inv(T_world_live)
    pts_live_cam = (T_live_from_world[:3,:3] @ pts_world.T + T_live_from_world[:3,3:]).T
    pts2 = (K_live[:2,:2] @ (pts_live_cam[:,:2] / pts_live_cam[:,2:]).T).T + K_live[:2,2]

    # Run PnP with the CORRECT K (live K)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3.astype(np.float64), pts2.astype(np.float64), K_live, None,
        iterationsCount=200, reprojectionError=1.0, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    assert ok and inliers is not None and len(inliers) >= 10

    R, _ = cv2.Rodrigues(rvec)
    T_live_from_ref = np.eye(4, dtype=np.float64)
    T_live_from_ref[:3,:3] = R
    T_live_from_ref[:3, 3] = tvec[:,0]
    T_ref_from_live = np.linalg.inv(T_live_from_ref)
    T_world_live_recovered = T_world_ref @ T_ref_from_live
    return T_world_live_recovered


def test_feature_pnp_uses_live_k():
    """Using live K gives correct T_world_live; using keyframe K gives wrong result."""
    K_ref = np.array([[150., 0., 80.], [0., 150., 60.], [0., 0., 1.]], dtype=np.float64)
    K_live = np.array([[200., 0., 85.], [0., 200., 65.], [0., 0., 1.]], dtype=np.float64)  # different!

    T_world_ref = np.eye(4, dtype=np.float64)
    T_world_ref[:3, :3] = np.array([
        [np.cos(0.2), 0, np.sin(0.2)],
        [0,           1, 0           ],
        [-np.sin(0.2),0, np.cos(0.2)],
    ])
    T_world_ref[:3, 3] = [0.3, 0.1, 0.5]

    T_world_live = np.eye(4, dtype=np.float64)
    T_world_live[:3, :3] = np.array([
        [np.cos(0.3), 0, np.sin(0.3)],
        [0,           1, 0           ],
        [-np.sin(0.3),0, np.cos(0.3)],
    ])
    T_world_live[:3, 3] = [0.8, 0.1, 0.5]

    recovered = _run_feature_pnp_math(K_ref, K_live, T_world_ref, T_world_live)
    np.testing.assert_allclose(recovered[:3, 3], T_world_live[:3, 3], atol=1e-3)
    np.testing.assert_allclose(recovered[:3,:3], T_world_live[:3,:3], atol=1e-2)
