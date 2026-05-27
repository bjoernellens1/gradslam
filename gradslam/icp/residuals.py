"""Residual computation for point-to-plane ICP."""

from __future__ import annotations

import torch


def point_to_plane_projective(
    vertex_live: torch.Tensor,
    normal_model: torch.Tensor,
    vertex_model: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute point-to-plane residuals and Jacobian for projective ICP.

    Given live vertices and model vertices/normals, compute the residual and
    Jacobian matrix for point-to-plane (point-to-surface) alignment.

    Args:
        vertex_live: Live vertices in model frame, shape [N, 3].
        normal_model: Surface normals of model vertices, shape [N, 3].
        vertex_model: Model vertices (associations), shape [N, 3].

    Returns:
        Tuple of:
        - A: Jacobian matrix, shape [N, 6].
        - b: Residual vector, shape [N, 1].

    Shapes:
        - vertex_live: [N, 3]
        - normal_model: [N, 3]
        - vertex_model: [N, 3]
        - A: [N, 6]
        - b: [N, 1]
    """
    # Point-to-plane residual: (p_live - p_model) · n
    diff = vertex_live - vertex_model  # [N, 3]
    residual = torch.sum(diff * normal_model, dim=1, keepdim=True)  # [N, 1]

    # Jacobian w.r.t. Lie algebra (twist) coordinates
    # J = [n^T, (p_live - origin) x n]  where x is cross product
    n = normal_model  # [N, 3]
    p = vertex_live  # [N, 3]

    # Translational part: n^T
    J_trans = n  # [N, 3]

    # Rotational part: cross(p, n) = [p x n]
    # cross(a, b) = [a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z, a_x*b_y - a_y*b_x]
    cross = torch.stack(
        [
            p[:, 1] * n[:, 2] - p[:, 2] * n[:, 1],
            p[:, 2] * n[:, 0] - p[:, 0] * n[:, 2],
            p[:, 0] * n[:, 1] - p[:, 1] * n[:, 0],
        ],
        dim=1,
    )  # [N, 3]

    # Combine: J = [n | cross]
    A = torch.cat([J_trans, cross], dim=1)  # [N, 6]
    b = residual  # [N, 1]

    return A, b
