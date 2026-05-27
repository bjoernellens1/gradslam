"""Linear solvers for ICP optimization (LM, damped normal equations)."""

from __future__ import annotations

import torch


def solve_lm_6x6(A: torch.Tensor, b: torch.Tensor, damp: float = 1e-4) -> torch.Tensor:
    """Solve damped normal equations for 6-DOF SE(3) optimization.

    Solves: (A^T A + damp * I) x = A^T b for the 6-vector x (Lie algebra twist).

    Args:
        A: Constraint matrix of shape [N, 6] (rows = correspondences, cols = DOF).
        b: Residual vector of shape [N, 1].
        damp: Damping coefficient (Levenberg-Marquardt lambda). Default: 1e-4.

    Returns:
        Solution vector x of shape [6, 1].

    Shapes:
        - A: [N, 6]
        - b: [N, 1]
        - Output: [6, 1]
    """
    device = A.device
    dtype = A.dtype

    # Compute normal equations: (A^T A + damp*I)
    AtA = torch.matmul(A.t(), A)  # [6, 6]
    Atb = torch.matmul(A.t(), b)  # [6, 1]

    # Add damping to diagonal
    I = torch.eye(6, device=device, dtype=dtype)
    H = AtA + damp * I

    # Solve H @ x = Atb via LU decomposition
    try:
        x = torch.linalg.solve(H, Atb)
    except RuntimeError:
        # Fallback: use lstsq if solve fails (singular or ill-conditioned)
        x, _ = torch.linalg.lstsq(H, Atb)

    return x
