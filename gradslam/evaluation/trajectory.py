"""Trajectory evaluation metrics: ATE and RPE.

Implements the standard TUM RGB-D benchmark metrics:
- ATE (Absolute Trajectory Error): global drift after Umeyama SE(3) alignment.
- RPE (Relative Pose Error): local odometry accuracy over fixed time intervals.

Reference: Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
           IROS 2012 (https://vision.in.tum.de/data/datasets/rgbd-dataset).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class ATEResult:
    """Absolute Trajectory Error result.

    Attributes:
        rmse: Root-mean-square translation error (m) after alignment.
        mean: Mean translation error (m).
        median: Median translation error (m).
        std: Standard deviation of translation errors (m).
        min: Minimum translation error (m).
        max: Maximum translation error (m).
        n_pairs: Number of matched pose pairs used.
        T_align: SE(3) alignment transform [4, 4] that maps estimated → GT frame.
    """

    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    n_pairs: int
    T_align: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __str__(self) -> str:
        return (
            f"ATE(n={self.n_pairs}): "
            f"RMSE={self.rmse:.4f}m  mean={self.mean:.4f}m  "
            f"median={self.median:.4f}m  std={self.std:.4f}m  "
            f"[{self.min:.4f}, {self.max:.4f}]"
        )


@dataclass
class RPEResult:
    """Relative Pose Error result.

    Attributes:
        rmse_t: RMSE of translational errors (m).
        rmse_r: RMSE of rotational errors (deg).
        mean_t: Mean translation error (m).
        mean_r: Mean rotation error (deg).
        median_t: Median translation error (m).
        median_r: Median rotation error (deg).
        n_pairs: Number of pose pairs evaluated.
        delta: Frame interval used.
    """

    rmse_t: float
    rmse_r: float
    mean_t: float
    mean_r: float
    median_t: float
    median_r: float
    n_pairs: int
    delta: int = 1

    def __str__(self) -> str:
        return (
            f"RPE(delta={self.delta}, n={self.n_pairs}): "
            f"trans RMSE={self.rmse_t:.4f}m  rot RMSE={self.rmse_r:.3f}°  "
            f"trans mean={self.mean_t:.4f}m  rot mean={self.mean_r:.3f}°"
        )


# ---------------------------------------------------------------------------
# Pose I/O helpers
# ---------------------------------------------------------------------------


def load_tum_poses(path: str) -> dict[float, np.ndarray]:
    """Load poses from a TUM-format trajectory file.

    Each line: ``timestamp tx ty tz qx qy qz qw``

    Args:
        path: Path to the trajectory file.

    Returns:
        Dict mapping timestamp (float) → T_world_camera [4, 4] numpy array.
    """
    from scipy.spatial.transform import Rotation

    poses = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            ts = float(vals[0])
            t = np.array([float(v) for v in vals[1:4]])
            q = np.array([float(v) for v in vals[4:8]])  # qx qy qz qw
            R = Rotation.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses[ts] = T
    return poses


def load_replica_traj(path: str) -> list[np.ndarray]:
    """Load poses from Replica traj.txt format (one 4×4 per line, 16 floats).

    Args:
        path: Path to traj.txt.

    Returns:
        List of T_world_camera [4, 4] numpy arrays (index-ordered).
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return [row.reshape(4, 4) for row in data]


def poses_from_tensors(pose_list: list[np.ndarray]) -> dict[float, np.ndarray]:
    """Convert a list of 4×4 pose arrays to a timestamp-keyed dict (index as timestamp)."""
    return {float(i): p for i, p in enumerate(pose_list)}


# ---------------------------------------------------------------------------
# Timestamp association
# ---------------------------------------------------------------------------


def associate_poses(
    poses_est: dict[float, np.ndarray],
    poses_gt: dict[float, np.ndarray],
    max_dt: float = 0.02,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Associate estimated and GT poses by closest timestamp.

    Args:
        poses_est: Estimated poses keyed by timestamp.
        poses_gt: GT poses keyed by timestamp.
        max_dt: Maximum timestamp difference to consider a match (seconds).

    Returns:
        List of (T_est, T_gt) matched pairs.
    """
    ts_gt = sorted(poses_gt.keys())
    ts_gt_arr = np.array(ts_gt)

    pairs = []
    for ts_e, T_e in sorted(poses_est.items()):
        diffs = np.abs(ts_gt_arr - ts_e)
        nearest_idx = diffs.argmin()
        if diffs[nearest_idx] <= max_dt:
            pairs.append((T_e, poses_gt[ts_gt[nearest_idx]]))
    return pairs


def associate_by_index(
    poses_est: list[np.ndarray],
    poses_gt: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair estimated and GT poses by index (no timestamp matching).

    Args:
        poses_est: List of estimated T_world_camera [4, 4].
        poses_gt: List of GT T_world_camera [4, 4].

    Returns:
        List of (T_est, T_gt) pairs, trimmed to shorter list length.
    """
    n = min(len(poses_est), len(poses_gt))
    return [(poses_est[i], poses_gt[i]) for i in range(n)]


# ---------------------------------------------------------------------------
# SE(3) Umeyama alignment (no scale)
# ---------------------------------------------------------------------------


def align_umeyama(
    est_positions: np.ndarray,
    gt_positions: np.ndarray,
) -> np.ndarray:
    """Compute the rigid-body SE(3) transform that best aligns estimated to GT.

    Uses the Umeyama method (no scale): minimizes sum ||T @ p_est[i] - p_gt[i]||².

    Args:
        est_positions: Estimated positions [N, 3].
        gt_positions: GT positions [N, 3].

    Returns:
        T_align [4, 4] such that T_align @ est ≈ gt.
    """
    assert est_positions.shape == gt_positions.shape
    N = est_positions.shape[0]

    mu_est = est_positions.mean(axis=0)
    mu_gt = gt_positions.mean(axis=0)

    est_c = est_positions - mu_est
    gt_c = gt_positions - mu_gt

    W = gt_c.T @ est_c / N  # [3, 3]
    U, S, Vt = np.linalg.svd(W)

    # Correct for reflection
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = mu_gt - R @ mu_est

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# ATE
# ---------------------------------------------------------------------------


def compute_ate(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    align: bool = True,
) -> ATEResult:
    """Compute Absolute Trajectory Error (ATE).

    Args:
        pairs: List of (T_est, T_gt) matched pose pairs.
        align: If True, Umeyama-align estimated trajectory to GT first.
              Set False if trajectories are already in the same reference frame.

    Returns:
        ATEResult with RMSE and other statistics.
    """
    if len(pairs) == 0:
        raise ValueError("No pose pairs to evaluate")

    est_traj = np.array([p[0][:3, 3] for p in pairs])  # [N, 3]
    gt_traj = np.array([p[1][:3, 3] for p in pairs])   # [N, 3]

    if align:
        T_align = align_umeyama(est_traj, gt_traj)
        est_aligned = (T_align[:3, :3] @ est_traj.T).T + T_align[:3, 3]
    else:
        T_align = np.eye(4)
        est_aligned = est_traj

    errors = np.linalg.norm(est_aligned - gt_traj, axis=1)  # [N]

    return ATEResult(
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        min=float(np.min(errors)),
        max=float(np.max(errors)),
        n_pairs=len(pairs),
        T_align=T_align,
    )


# ---------------------------------------------------------------------------
# RPE
# ---------------------------------------------------------------------------


def compute_rpe(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    delta: int = 1,
) -> RPEResult:
    """Compute Relative Pose Error (RPE).

    For each pair of poses separated by `delta` frames, compare the relative
    motion in the estimated and GT trajectories.

    Args:
        pairs: List of (T_est, T_gt) matched pose pairs (ordered by time).
        delta: Frame interval for computing relative motions. Default: 1.

    Returns:
        RPEResult with translational and rotational errors.
    """
    if len(pairs) < delta + 1:
        raise ValueError(f"Need at least {delta + 1} pairs for RPE with delta={delta}")

    t_errors, r_errors = [], []

    for i in range(len(pairs) - delta):
        T_est_i, T_gt_i = pairs[i]
        T_est_j, T_gt_j = pairs[i + delta]

        # Relative motions: T_i^{-1} T_j
        rel_est = np.linalg.inv(T_est_i) @ T_est_j
        rel_gt = np.linalg.inv(T_gt_i) @ T_gt_j

        # Error: difference between relative motions
        err = np.linalg.inv(rel_gt) @ rel_est  # should be I if perfect

        t_err = np.linalg.norm(err[:3, 3])
        # Rotation angle from rotation matrix
        R_err = err[:3, :3]
        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        r_err_rad = np.arccos(cos_angle)
        r_err_deg = np.degrees(r_err_rad)

        t_errors.append(t_err)
        r_errors.append(r_err_deg)

    t_errors = np.array(t_errors)
    r_errors = np.array(r_errors)

    return RPEResult(
        rmse_t=float(np.sqrt(np.mean(t_errors ** 2))),
        rmse_r=float(np.sqrt(np.mean(r_errors ** 2))),
        mean_t=float(np.mean(t_errors)),
        mean_r=float(np.mean(r_errors)),
        median_t=float(np.median(t_errors)),
        median_r=float(np.median(r_errors)),
        n_pairs=len(t_errors),
        delta=delta,
    )


# ---------------------------------------------------------------------------
# Convenience: evaluate from files
# ---------------------------------------------------------------------------


def evaluate_tum(
    estimated_traj_path: str,
    groundtruth_traj_path: str,
    max_dt: float = 0.02,
    align: bool = True,
    rpe_deltas: tuple[int, ...] = (1, 10, 100),
) -> dict:
    """Evaluate estimated trajectory against TUM groundtruth.

    Args:
        estimated_traj_path: Path to estimated poses (TUM format).
        groundtruth_traj_path: Path to GT poses (TUM format).
        max_dt: Max timestamp difference for matching (s).
        align: Umeyama-align before ATE.
        rpe_deltas: Frame intervals for RPE.

    Returns:
        Dict with "ate" (ATEResult) and "rpe" (dict of delta → RPEResult).
    """
    est = load_tum_poses(estimated_traj_path)
    gt = load_tum_poses(groundtruth_traj_path)
    pairs = associate_poses(est, gt, max_dt=max_dt)

    ate = compute_ate(pairs, align=align)
    rpe = {d: compute_rpe(pairs, delta=d) for d in rpe_deltas if d < len(pairs)}

    return {"ate": ate, "rpe": rpe}


def evaluate_by_index(
    estimated_poses: list[np.ndarray],
    gt_poses: list[np.ndarray],
    align: bool = True,
    rpe_deltas: tuple[int, ...] = (1, 10),
) -> dict:
    """Evaluate by index (no timestamp matching).

    Args:
        estimated_poses: List of T_world_camera [4, 4].
        gt_poses: List of T_world_camera [4, 4].
        align: Umeyama-align before ATE.
        rpe_deltas: Frame intervals for RPE.

    Returns:
        Dict with "ate" (ATEResult) and "rpe" (dict of delta → RPEResult).
    """
    pairs = associate_by_index(estimated_poses, gt_poses)
    ate = compute_ate(pairs, align=align)
    rpe = {d: compute_rpe(pairs, delta=d) for d in rpe_deltas if d < len(pairs)}
    return {"ate": ate, "rpe": rpe}
