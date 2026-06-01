"""Global pose-graph optimizer backed by pypose SE(3) Levenberg-Marquardt.

Unlike :class:`SlidingWindowPoseGraph`, this keeps **all** keyframe nodes (no
window trim) so that a loop edge between an early keyframe and the current one
constrains the whole accumulated chain — the prerequisite for reducing ATE on
loopy sequences.

It uses pypose's Lie-group SE(3) with an LBFGS/strong-Wolfe step on the se3
tangent (node 0 fixed as the gauge anchor). The repo's hand-rolled
``SlidingWindowPoseGraph`` uses an identity-approximation Jacobian that diverges
on real loop edges (oracle: 0.135 -> 13 m); the same oracle with this pypose
backend converges (0.135 -> 0.045 m), which is why Workstream B routes
``--pose-graph on`` here.

API mirrors the parts of ``SlidingWindowPoseGraph`` the pipeline uses:
``add_keyframe``, ``add_loop_edge``, ``drop_last_edge``, ``try_commit_correction``,
``node_ids``, ``num_keyframes`` — so the pipeline integration is a drop-in swap.
"""

from __future__ import annotations

import torch

from ..geometry.se3utils import se3_inv


def _mat_to_SE3(T: torch.Tensor):
    import pypose as pp

    return pp.mat2SE3(T[:3, :4].unsqueeze(0)).tensor().squeeze(0)


class GlobalPoseGraph:
    """Global SE(3) pose graph (pypose LM), node 0 fixed as anchor.

    Conventions match ``SlidingWindowPoseGraph``: an edge ``(a, b, M, w)`` asserts
    ``M ≈ inv(T_world_a) @ T_world_b``. Sequential edges derived from the absolute
    poses are residual-free; loop edges carry the drift-correcting information.
    """

    def __init__(self, n_iterations: int = 25, damping: float = 1e-4):
        self.n_iterations = n_iterations
        self.damping = damping
        self._ids: list[int] = []
        self._poses: list[torch.Tensor] = []          # 4x4 absolute, float64 (OPTIMIZATION STATE)
        # Immutable raw tracking poses, parallel to _ids. Sequential edges are
        # odometry MEASUREMENTS and must derive only from raw tracking, never
        # from corrected _poses — otherwise a keyframe added after a loop commit
        # mixes a corrected world frame with a raw one and the edge is garbage
        # (incremental ATE 0.171 vs batch 0.053). _poses is warm-start only.
        self._raw_poses: list[torch.Tensor] = []
        self._edges: list[tuple[int, int, torch.Tensor, float]] = []
        self._next_auto_id: int = 0

    # -- introspection -------------------------------------------------------
    def node_ids(self) -> list[int]:
        return list(self._ids)

    @property
    def num_keyframes(self) -> int:
        return len(self._poses)

    # -- graph construction --------------------------------------------------
    @torch.no_grad()
    def add_keyframe(
        self,
        T_world_camera: torch.Tensor,
        node_id: int | None = None,
        T_rel_measured: torch.Tensor | None = None,
        weight: float = 1.0,
    ) -> int:
        """Append a node + a sequential edge from the previous node.

        ``T_rel_measured=None`` derives the sequential edge from the absolute
        poses (a residual-free no-op), so by default the optimizer only moves
        when a loop edge is later added.
        """
        if node_id is None:
            node_id = self._next_auto_id
        self._next_auto_id = max(self._next_auto_id, node_id + 1)

        T = T_world_camera.detach().to(torch.float64).clone()
        if self._poses:
            prev_id = self._ids[-1]
            if T_rel_measured is None:
                # Derive from RAW tracking poses (immutable), not corrected state.
                rel = se3_inv(self._raw_poses[-1]) @ T
            else:
                rel = T_rel_measured.detach().to(torch.float64).clone()
            self._edges.append((prev_id, node_id, rel, float(weight)))
        self._ids.append(node_id)
        self._poses.append(T.clone())        # optimization state (mutated by commit)
        self._raw_poses.append(T)            # immutable odometry record
        return node_id

    @torch.no_grad()
    def add_loop_edge(
        self, a_id: int, b_id: int, T_rel_meas: torch.Tensor, weight: float = 1.0
    ) -> bool:
        """Add a non-sequential loop edge. Both nodes must exist (never trimmed,
        so this only fails for an unknown id)."""
        live = set(self._ids)
        if a_id not in live or b_id not in live:
            return False
        self._edges.append(
            (a_id, b_id, T_rel_meas.detach().to(torch.float64).clone(), float(weight))
        )
        return True

    @torch.no_grad()
    def drop_last_edge(self) -> None:
        if self._edges:
            self._edges.pop()

    # -- optimization --------------------------------------------------------
    def optimize(self) -> list[torch.Tensor]:
        """Return globally optimized poses (does not mutate state)."""
        n = len(self._poses)
        if n < 2 or not self._edges:
            return [p.clone() for p in self._poses]

        import pypose as pp

        pos_of = {nid: i for i, nid in enumerate(self._ids)}
        nodes = pp.SE3(torch.stack([_mat_to_SE3(p) for p in self._poses]))  # [n,7]

        ei = torch.tensor([pos_of[a] for a, _, _, _ in self._edges])
        ej = torch.tensor([pos_of[b] for _, b, _, _ in self._edges])
        meas = pp.SE3(torch.stack([_mat_to_SE3(M) for _, _, M, _ in self._edges]))
        w = torch.tensor([wt for *_, wt in self._edges], dtype=torch.float64)
        wsqrt = w.sqrt().unsqueeze(1)

        # Free tangent params for every node; node 0 is held at identity (anchor).
        xi = torch.zeros(n, 6, dtype=torch.float64, requires_grad=True)

        def residual():
            P = pp.se3(xi).Exp() @ nodes              # left-perturbed poses
            pred = P[ei].Inv() @ P[ej]                # inv(Ti) @ Tj
            err = (meas.Inv() @ pred).Log().tensor()  # [E,6]
            return err * wsqrt

        opt = torch.optim.LBFGS(
            [xi], max_iter=self.n_iterations, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            r = residual()
            loss = (r ** 2).sum() + self.damping * (xi[1:] ** 2).sum()
            loss.backward()
            with torch.no_grad():           # keep anchor fixed
                if xi.grad is not None:
                    xi.grad[0].zero_()
                xi.data[0].zero_()
            return loss

        for _ in range(5):
            opt.step(closure)

        with torch.no_grad():
            P = pp.se3(xi).Exp() @ nodes
            return [P[i].matrix().to(torch.float64) for i in range(n)]

    @torch.no_grad()
    def finalize(self) -> list[torch.Tensor]:
        """Run one final global optimization over the FULL graph and commit it.

        Keyframes added after the last accepted loop commit never triggered a
        re-optimization, so the trajectory tail still carries raw drift. Calling
        this once at stream end re-optimizes every node against all accumulated
        loop edges (still single-pass — no re-tracking). No-op if no loop edges
        were ever added (sequential-only graph optimizes to itself).
        """
        corrected = self.optimize()
        if all(torch.isfinite(p).all() for p in corrected):
            self._poses = [p.clone() for p in corrected]
        return self._poses

    @torch.no_grad()
    def try_commit_correction(
        self, max_translation_step: float = 2.0
    ) -> list[torch.Tensor] | None:
        """Optimize and commit ONLY if every pose stays finite and no node moves
        more than ``max_translation_step`` metres. On rejection, state is left
        unchanged and ``None`` is returned (caller should ``drop_last_edge``).

        This is the same safety contract as ``SlidingWindowPoseGraph`` — a single
        bad (e.g. outlier-PnP) loop edge must not corrupt the trajectory.
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
