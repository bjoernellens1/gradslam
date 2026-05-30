"""Sliding-window pose graph optimizer for SLAM keyframe correction.

Hand-rolled Gauss-Newton on se(3). Maintains last N keyframe poses as nodes
with relative pose edges. Runs inline after keyframe insertion.
"""

import torch

from ..geometry.se3utils import se3_exp
from ..geometry.se3utils import se3_log  # noqa: F401 (re-exported for backward compat)


@torch.no_grad()
def _skew(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix from a 3-vector."""
    x, y, z = v[0], v[1], v[2]
    O = torch.zeros_like(x)
    return torch.stack([
        O, -z, y,
        z, O, -x,
        -y, x, O,
    ]).reshape(3, 3)


class SlidingWindowPoseGraph:
    """Gauss-Newton sliding-window pose graph over keyframes.

    Maintains last ``window_size`` keyframe absolute poses T_world_cam[i].
    Edges store **independently measured** relative poses (ICP odometry), so
    the system can actually detect and correct drift.

    Usage in SLAM pipeline::

        pg.add_keyframe(T_world_camera, T_rel_measured=best_rel, weight=w)
        corrected = pg.get_corrected_poses()

    where ``best_rel`` is the ICP-estimated relative pose from the current
    frame to the previous absolute pose (i.e. ``inv(T_w_prev) @ T_w_curr``
    as estimated by ICP, not recomputed from the stored absolute poses).
    """

    def __init__(
        self,
        window_size: int = 8,
        n_iterations: int = 5,
        damping: float = 1e-4,
    ):
        self.window_size = window_size
        self.n_iterations = n_iterations
        self.damping = damping

        # Stored as list of 4x4 tensors (world frame)
        self._poses: list[torch.Tensor] = []
        # Relative pose edges: (T_rel_measured, weight)
        # edge[i]: measured relative pose from pose i to pose i+1
        # i.e. inv(T_w_i) @ T_w_{i+1}  as measured by ICP
        self._edges: list[tuple[torch.Tensor, float]] = []

    @torch.no_grad()
    def add_keyframe(
        self,
        T_world_camera: torch.Tensor,
        T_rel_measured: torch.Tensor | None = None,
        weight: float = 1.0,
    ) -> None:
        """Add a new keyframe pose.

        Args:
            T_world_camera: Absolute pose [4, 4] in world frame.
            T_rel_measured: Independent relative-pose measurement from the
                previous keyframe to this one (ICP estimate).  If ``None``,
                the relative pose is derived from the stored absolute poses
                (no independent measurement → optimizer will be a no-op for
                this edge since the residual is zero by construction).
            weight: Edge weight (higher = more trusted measurement).
        """
        if self._poses:
            if T_rel_measured is not None:
                # Use the independently measured edge
                T_rel = T_rel_measured.clone()
            else:
                # Fallback: derive from absolute poses (no independent info)
                T_rel = torch.linalg.inv(self._poses[-1]) @ T_world_camera
            self._edges.append((T_rel.clone(), weight))

        self._poses.append(T_world_camera.clone())

        # Trim to window size
        if len(self._poses) > self.window_size:
            self._poses = self._poses[-self.window_size:]
            self._edges = self._edges[-(self.window_size - 1):]

    @torch.no_grad()
    def optimize(self) -> list[torch.Tensor]:
        """Run Gauss-Newton optimization and return corrected poses.

        Returns the updated list of poses (same length as current window).
        The first pose is held fixed as an anchor.
        """
        n = len(self._poses)
        if n < 2:
            return list(self._poses)

        device = self._poses[0].device
        dtype = self._poses[0].dtype

        poses = [p.clone() for p in self._poses]

        for _ in range(self.n_iterations):
            n_free = n - 1  # first pose is fixed
            # Rows: 6 per edge; cols: 6 per free pose
            J_total = torch.zeros(6 * (n - 1), 6 * n_free, device=device, dtype=dtype)
            r_total = torch.zeros(6 * (n - 1), device=device, dtype=dtype)

            for i, (T_rel_meas, w) in enumerate(self._edges):
                # Edge: measured rel pose from node i to node i+1
                # T_rel_meas ≈ inv(T_w_i) @ T_w_{i+1}
                # Predicted: inv(T_w_i_curr) @ T_w_{i+1_curr}
                T_predicted = torch.linalg.inv(poses[i]) @ poses[i + 1]
                # Error: measured vs predicted
                T_err = torch.linalg.inv(T_rel_meas) @ T_predicted
                residual = se3_log(T_err)  # [6]

                r_total[i * 6:(i + 1) * 6] = w * residual

                # Jacobian blocks (linearized identity approximation):
                # d_residual / d_xi_i   = -I  (if i > 0, i.e., not fixed)
                # d_residual / d_xi_{i+1} = +I  (always free since i+1 >= 1)
                if i > 0:  # pose i is free (column index = i-1)
                    J_total[i * 6:(i + 1) * 6, (i - 1) * 6:i * 6] = -w * torch.eye(6, device=device, dtype=dtype)
                # pose i+1 is always free (column index = i)
                J_total[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = w * torch.eye(6, device=device, dtype=dtype)

            # Normal equations with Levenberg-Marquardt damping.
            # GN: minimize ||r + J*delta||^2  => (J^T J) delta = -J^T r
            JtJ = J_total.t() @ J_total
            Jtr = J_total.t() @ r_total

            lhs = JtJ + self.damping * torch.eye(6 * n_free, device=device, dtype=dtype)

            try:
                delta = torch.linalg.solve(lhs, -Jtr)  # [6*n_free]
            except Exception:
                break

            # Apply left-multiplication updates on SE(3)
            for i in range(1, n):
                xi = delta[(i - 1) * 6:i * 6]
                dT = se3_exp(xi)
                poses[i] = dT @ poses[i]

            if delta.norm() < 1e-6:
                break

        return poses

    def get_corrected_poses(self) -> list[torch.Tensor]:
        """Return optimized poses without modifying internal state."""
        return self.optimize()

    @property
    def num_keyframes(self) -> int:
        return len(self._poses)
