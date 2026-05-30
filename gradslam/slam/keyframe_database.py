"""Keyframe database for relocalization and loop closure.

Stores keyframe images + poses + ORB descriptors for feature matching.
"""

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

import numpy as np


class KeyframeDatabase:
    """Cap at max_keyframes entries. Each entry stores CPU numpy data."""

    def __init__(self, max_keyframes: int = 30):
        if not _CV2_AVAILABLE:
            raise ImportError(
                "KeyframeDatabase requires OpenCV (cv2). "
                "Install it with: pip install opencv-python-headless"
            )
        self.max_keyframes = max_keyframes
        self._entries: list[dict] = []
        self._orb = cv2.ORB_create(nfeatures=500)

    def add(
        self,
        rgb_uint8: np.ndarray,     # [H, W, 3] uint8
        depth: np.ndarray,          # [H, W] float32
        K: np.ndarray,              # [3, 3] float64
        T_world_camera: np.ndarray, # [4, 4]
        frame_idx: int,
    ) -> None:
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        kpts, desc = self._orb.detectAndCompute(gray, None)
        entry = {
            "rgb": rgb_uint8.copy(),
            "depth": depth.copy(),
            "K": K.copy(),
            "T_world_camera": T_world_camera.copy(),
            "frame_idx": frame_idx,
            "gray": gray,
            "kpts": kpts,
            "desc": desc,  # may be None if no features found
        }
        self._entries.append(entry)
        if len(self._entries) > self.max_keyframes:
            self._entries = self._entries[-self.max_keyframes:]

    def relocalize(
        self,
        query_rgb: np.ndarray,
        query_depth: np.ndarray,
        K: np.ndarray,
        min_inliers: int = 20,
    ) -> tuple:
        """Find best matching keyframe and return (T_world_camera_4x4, match_info).

        Returns (None, None) if no match found.
        """
        if not self._entries:
            return None, None

        gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
        kpts_q, desc_q = self._orb.detectAndCompute(gray, None)
        if desc_q is None or len(kpts_q) < 10:
            return None, None

        best_result = None
        best_inliers = min_inliers - 1

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for entry in self._entries:
            if entry["desc"] is None:
                continue
            try:
                matches = bf.match(desc_q, entry["desc"])
            except cv2.error:
                continue

            if len(matches) < 10:
                continue

            # Get 3D-2D correspondences for PnP
            pts_3d = []
            pts_2d = []
            depth_map = entry["depth"]
            H, W = depth_map.shape
            K_ref = entry["K"]
            kpts_ref = entry["kpts"]

            for m in matches:
                ref_kpt = kpts_ref[m.trainIdx].pt  # (u, v) in ref frame
                u, v = int(round(ref_kpt[0])), int(round(ref_kpt[1]))
                if not (0 <= u < W and 0 <= v < H):
                    continue
                z = float(depth_map[v, u])
                if z <= 0:
                    continue
                x = (u - K_ref[0, 2]) * z / K_ref[0, 0]
                y = (v - K_ref[1, 2]) * z / K_ref[1, 1]
                pts_3d.append([x, y, z])
                query_pt = kpts_q[m.queryIdx].pt
                pts_2d.append(query_pt)

            if len(pts_3d) < 8:
                continue

            pts_3d_np = np.array(pts_3d, dtype=np.float64)
            pts_2d_np = np.array(pts_2d, dtype=np.float64)
            K_q = K.astype(np.float64)

            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d_np, pts_2d_np, K_q, None,
                    iterationsCount=100, reprojectionError=8.0,
                    confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                continue

            if not success or inliers is None or len(inliers) < min_inliers:
                continue

            n_inliers = len(inliers)
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                # Build T_query_ref (camera-to-camera transform)
                R_mat, _ = cv2.Rodrigues(rvec)
                T_ref_query = np.eye(4)
                T_ref_query[:3, :3] = R_mat
                T_ref_query[:3, 3] = tvec[:, 0]
                # T_world_query = T_world_ref @ T_ref_query
                T_world_query = entry["T_world_camera"] @ T_ref_query
                best_result = (
                    T_world_query,
                    {"feature_inliers": n_inliers, "ref_frame_idx": entry["frame_idx"]},
                )

        if best_result is None:
            return None, None
        return best_result

    def find_loop(
        self,
        query_desc: tuple,      # (kpts, desc) already computed
        query_K: np.ndarray,    # camera intrinsics for PnP
        exclude_last_n: int = 8,
        min_inliers: int = 30,
    ) -> tuple:
        """Find loop closure: match query against entries older than exclude_last_n.

        Runs PnP-RANSAC geometric verification on appearance candidates.

        Returns (T_rel_np, match_frame_idx, num_inliers) or (None, -1, 0).
        T_rel is the PnP-estimated camera-to-camera transform from the matched
        keyframe to the query frame (the loop edge measurement).
        """
        kpts_q, desc_q = query_desc
        if desc_q is None or len(kpts_q) < 10:
            return None, -1, 0

        entries_to_check = (
            self._entries[:-exclude_last_n]
            if exclude_last_n < len(self._entries)
            else []
        )
        if not entries_to_check:
            return None, -1, 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_inliers = min_inliers - 1
        best = (None, -1, 0)

        K_q = query_K.astype(np.float64)

        for entry in entries_to_check:
            if entry["desc"] is None:
                continue
            try:
                matches = bf.match(desc_q, entry["desc"])
            except cv2.error:
                continue
            if len(matches) < 10:
                continue

            # Build 3D-2D correspondences using ref keyframe depth
            pts_3d = []
            pts_2d = []
            depth_map = entry["depth"]
            H, W = depth_map.shape
            K_ref = entry["K"]
            kpts_ref = entry["kpts"]

            for m in matches:
                ref_kpt = kpts_ref[m.trainIdx].pt  # (u, v) in ref frame
                u, v = int(round(ref_kpt[0])), int(round(ref_kpt[1]))
                if not (0 <= u < W and 0 <= v < H):
                    continue
                z = float(depth_map[v, u])
                if z <= 0:
                    continue
                x = (u - K_ref[0, 2]) * z / K_ref[0, 0]
                y = (v - K_ref[1, 2]) * z / K_ref[1, 1]
                pts_3d.append([x, y, z])
                query_pt = kpts_q[m.queryIdx].pt
                pts_2d.append(query_pt)

            if len(pts_3d) < 8:
                continue

            pts_3d_np = np.array(pts_3d, dtype=np.float64)
            pts_2d_np = np.array(pts_2d, dtype=np.float64)

            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d_np, pts_2d_np, K_q, None,
                    iterationsCount=100, reprojectionError=8.0,
                    confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                continue

            if not success or inliers is None or len(inliers) < min_inliers:
                continue

            n_inliers = len(inliers)
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                R_mat, _ = cv2.Rodrigues(rvec)
                T_ref_query = np.eye(4)
                T_ref_query[:3, :3] = R_mat
                T_ref_query[:3, 3] = tvec[:, 0]
                # T_rel is the PnP-estimated transform from the matched KF to the query
                T_rel = T_ref_query
                best = (T_rel, entry["frame_idx"], n_inliers)

        if best[0] is None:
            return None, -1, 0

        return best

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
