"""Sliding-window pose graph optimizer for SLAM keyframe correction.

Hand-rolled Gauss-Newton on se(3). Maintains last N keyframe poses as nodes
(each with a stable integer id) connected by relative pose edges. Sequential
edges chain consecutive keyframes; non-sequential *loop* edges connect
arbitrary node pairs and are what actually let the optimizer correct drift.
Runs inline after keyframe insertion.
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

    Maintains the last ``window_size`` keyframe absolute poses T_world_cam as
    nodes, each carrying a stable integer id (caller-supplied or
    auto-assigned). Edges store **independently measured** relative poses with
    the convention ``T_rel_meas ≈ inv(T_world_a) @ T_world_b`` for an edge from
    node ``a`` to node ``b``. Sequential edges (consecutive keyframes) whose
    measurement is derived from the chained absolute poses are residual-free
    no-ops; drift correction comes from *loop* edges between non-adjacent nodes.

    Usage in SLAM pipeline::

        pg.add_keyframe(T_world_camera, node_id=frame_idx)
        pg.add_loop_edge(match_idx, frame_idx, T_rel_meas, weight=2.0)
        corrected = pg.get_corrected_poses()
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

        # Node ids parallel to poses (window order).
        self._ids: list[int] = []
        # Absolute poses (world frame) as 4x4 tensors, parallel to _ids.
        self._poses: list[torch.Tensor] = []
        # Edges: (a_id, b_id, T_rel_measured, weight) with
        # T_rel_measured ≈ inv(T_world_a) @ T_world_b.
        self._edges: list[tuple[int, int, torch.Tensor, float]] = []
        # Next auto-assigned id when caller omits node_id.
        self._next_auto_id: int = 0

    def node_ids(self) -> list[int]:
        """Return the current window's node ids in window order."""
        return list(self._ids)

    def _trim(self) -> None:
        """Drop nodes outside the window and edges referencing dropped ids."""
        if len(self._poses) <= self.window_size:
            return
        self._ids = self._ids[-self.window_size:]
        self._poses = self._poses[-self.window_size:]
        live = set(self._ids)
        self._edges = [
            e for e in self._edges if e[0] in live and e[1] in live
        ]

    @torch.no_grad()
    def add_keyframe(
        self,
        T_world_camera: torch.Tensor,
        node_id: int | None = None,
        T_rel_measured: torch.Tensor | None = None,
        weight: float = 1.0,
    ) -> int:
        """Add a new keyframe pose as a node, plus a sequential edge.

        Args:
            T_world_camera: Absolute pose [4, 4] in world frame.
            node_id: Stable integer id for this node. If ``None``, an
                auto-incrementing id is assigned (keeps legacy callers working).
            T_rel_measured: Independent relative-pose measurement from the
                previous keyframe node to this one (ICP estimate), with the
                convention ``inv(T_world_prev) @ T_world_this``. If ``None``,
                the relative pose is derived from the stored absolute poses (no
                independent info → the sequential edge is a residual-free
                no-op).
            weight: Edge weight (higher = more trusted measurement).

        Returns:
            The node id assigned to this keyframe.
        """
        if node_id is None:
            node_id = self._next_auto_id
            self._next_auto_id += 1
        else:
            # Keep auto ids from colliding with caller-supplied ids.
            self._next_auto_id = max(self._next_auto_id, node_id + 1)

        if self._poses:
            prev_id = self._ids[-1]
            if T_rel_measured is not None:
                T_rel = T_rel_measured.clone()
            else:
                # Fallback: derive from absolute poses (no independent info).
                T_rel = torch.linalg.inv(self._poses[-1]) @ T_world_camera
            self._edges.append((prev_id, node_id, T_rel.clone(), weight))

        self._ids.append(node_id)
        self._poses.append(T_world_camera.clone())

        self._trim()
        return node_id

    @torch.no_grad()
    def add_loop_edge(
        self,
        a_id: int,
        b_id: int,
        T_rel_meas: torch.Tensor,
        weight: float = 1.0,
    ) -> bool:
        """Add a non-sequential loop edge between two nodes in the window.

        The measurement convention is ``T_rel_meas ≈ inv(T_world_a) @ T_world_b``.

        Returns:
            True if both ``a_id`` and ``b_id`` are currently in the window and
            the edge was added; False otherwise (e.g. a node slid out after
            trimming).
        """
        live = set(self._ids)
        if a_id not in live or b_id not in live:
            return False
        self._edges.append((a_id, b_id, T_rel_meas.clone(), weight))
        return True

    @torch.no_grad()
    def drop_last_edge(self) -> None:
        """Remove the most recently added edge (used to reject a loop edge
        whose correction failed the safety guard)."""
        if self._edges:
            self._edges.pop()

    @torch.no_grad()
    def optimize(self) -> list[torch.Tensor]:
        """Run Gauss-Newton optimization and return corrected poses.

        Returns the updated list of poses (same length / order as the current
        window). The first node in the window is held fixed as the anchor.
        """
        n = len(self._poses)
        if n < 2 or not self._edges:
            return list(self._poses)

        device = self._poses[0].device
        dtype = self._poses[0].dtype

        poses = [p.clone() for p in self._poses]
        # Map node id -> window position.
        pos_of = {nid: idx for idx, nid in enumerate(self._ids)}

        n_free = n - 1  # first node (window position 0) is the fixed anchor
        if n_free == 0:
            return poses

        n_edges = len(self._edges)
        I6 = torch.eye(6, device=device, dtype=dtype)

        for _ in range(self.n_iterations):
            J_total = torch.zeros(6 * n_edges, 6 * n_free, device=device, dtype=dtype)
            r_total = torch.zeros(6 * n_edges, device=device, dtype=dtype)

            for e, (a_id, b_id, M, w) in enumerate(self._edges):
                a = pos_of[a_id]
                b = pos_of[b_id]
                # Convention: M ≈ inv(T_world_a) @ T_world_b
                T_pred = torch.linalg.inv(poses[a]) @ poses[b]
                T_err = torch.linalg.inv(M) @ T_pred
                residual = se3_log(T_err)  # [6]

                r_total[e * 6:(e + 1) * 6] = w * residual

                # Jacobian (linearized identity approximation):
                #   d_residual / d_xi_a = -I  (only if a is free)
                #   d_residual / d_xi_b = +I  (only if b is free)
                # Free node window position p maps to column block (p - 1)
                # since window position 0 is the fixed anchor.
                if a > 0:
                    J_total[e * 6:(e + 1) * 6, (a - 1) * 6:a * 6] = -w * I6
                if b > 0:
                    J_total[e * 6:(e + 1) * 6, (b - 1) * 6:b * 6] = w * I6

            # Normal equations with Levenberg-Marquardt damping.
            # GN: minimize ||r + J*delta||^2  => (J^T J) delta = -J^T r
            JtJ = J_total.t() @ J_total
            Jtr = J_total.t() @ r_total

            lhs = JtJ + self.damping * torch.eye(6 * n_free, device=device, dtype=dtype)

            try:
                delta = torch.linalg.solve(lhs, -Jtr)  # [6*n_free]
            except Exception:
                break

            # Apply left-multiplication updates on SE(3) to free nodes.
            for idx in range(1, n):
                xi = delta[(idx - 1) * 6:idx * 6]
                dT = se3_exp(xi)
                poses[idx] = dT @ poses[idx]

            if delta.norm() < 1e-6:
                break

        return poses

    def get_corrected_poses(self) -> list[torch.Tensor]:
        """Return optimized poses without modifying internal state."""
        return self.optimize()

    @torch.no_grad()
    def apply_correction(self) -> list[torch.Tensor]:
        """Optimize and COMMIT the corrected poses into the graph state.

        Unlike ``get_corrected_poses`` (read-only), this persists the optimized
        absolute poses back into ``self._poses``. This is required so the next
        keyframe's sequential edge (derived from the absolute poses when
        ``T_rel_measured=None``) is measured relative to the *corrected* last
        pose rather than the stale pre-correction one — otherwise a loop
        correction's discontinuity is re-injected as fake odometry on the
        following keyframe and compounds into divergence.
        """
        corrected = self.optimize()
        self._poses = [p.clone() for p in corrected]
        return corrected

    @torch.no_grad()
    def try_commit_correction(
        self, max_translation_step: float = 2.0
    ) -> list[torch.Tensor] | None:
        """Optimize and commit the result ONLY if it is safe.

        A correction is rejected (state left unchanged, ``None`` returned) when
        any optimized pose is non-finite or any node moves more than
        ``max_translation_step`` metres from its pre-optimization estimate.
        This is the safety boundary that prevents an inconsistent constraint
        (e.g. a bad PnP loop measurement that passes the inlier gate but is
        geometrically wrong) from corrupting the trajectory or accumulating
        across keyframes into numerical divergence. The non-robust optimizer
        has no outlier kernel, so callers must reject-and-drop on ``None``.
        """
        pre = self._poses
        corrected = self.optimize()
        for p in corrected:
            if not torch.isfinite(p).all():
                return None
        for a, b in zip(pre, corrected):
            if float((a[:3, 3] - b[:3, 3]).norm()) > max_translation_step:
                return None
        self._poses = [p.clone() for p in corrected]
        return corrected

    @property
    def num_keyframes(self) -> int:
        return len(self._poses)
