"""Linear solvers for ICP optimization (LM, damped normal equations)."""

from __future__ import annotations

import torch


# Patch 4: Pre-allocated buffers for normal equations (thread-local cache)
_ATA_BUFFER = {}
_ATB_BUFFER = {}
_I_BUFFER = {}


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
    out_dtype = A.dtype

    # The 6x6 normal-equation solve always runs in float32: it is cheap, more
    # numerically stable, and dtype-safe under mixed-precision autocast. Without
    # this, A/b can arrive as bfloat16/float16 and torch.linalg.solve/lstsq raise
    # (e.g. "Expected input and other to have the same dtype, but got Float and
    # BFloat16") or lose the rotation precision the SE(3) update needs.
    A = A.float()
    b = b.float()

    # Patch 4: Use pre-allocated buffers for normal equations
    device_key = str(device)
    if device_key not in _ATA_BUFFER:
        _ATA_BUFFER[device_key] = torch.empty((6, 6), device=device, dtype=torch.float32)
        _ATB_BUFFER[device_key] = torch.empty((6, 1), device=device, dtype=torch.float32)
        _I_BUFFER[device_key] = torch.eye(6, device=device, dtype=torch.float32)
    
    AtA = _ATA_BUFFER[device_key]
    Atb = _ATB_BUFFER[device_key]
    I = _I_BUFFER[device_key]

    # Compute normal equations: (A^T A + damp*I) using in-place operations
    torch.matmul(A.t(), A, out=AtA)  # [6, 6]
    torch.matmul(A.t(), b, out=Atb)  # [6, 1]

    # Add damping to diagonal (in-place)
    H = AtA + damp * I

    # Solve H @ x = Atb via LU decomposition
    try:
        x = torch.linalg.solve(H, Atb)
    except RuntimeError:
        # Fallback: use lstsq if solve fails (singular or ill-conditioned)
        x, _ = torch.linalg.lstsq(H, Atb)

    return x.to(out_dtype)
