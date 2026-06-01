"""
Operations over the Lie Group SE(3), for rigid-body transformations in 3D
"""

import torch

# Threshold to determine if a quantity can be considered 'small'
_eps = 1e-6


def so3_hat(omega: torch.Tensor) -> torch.Tensor:
    """Implements the hat operator for SO(3), given an input axis-angle
    vector omega.

    """
    assert torch.is_tensor(omega), "Input must be of type torch.tensor."

    omega_hat = torch.zeros(3, 3).type(omega.dtype).to(omega.device)
    omega_hat[0, 1] = -omega[2]
    omega_hat[1, 0] = omega[2]
    omega_hat[0, 2] = omega[1]
    omega_hat[2, 0] = -omega[1]
    omega_hat[1, 2] = -omega[0]
    omega_hat[2, 1] = omega[0]

    return omega_hat


def se3_hat(xi: torch.Tensor) -> torch.Tensor:
    """Implements the SE(3) hat operator, given a vector of twist
    (exponential) coordinates.
    """

    assert torch.is_tensor(xi), "Input must be of type torch.tensor."

    v = xi[:3]
    omega = xi[3:]
    omega_hat = so3_hat(omega)

    xi_hat = torch.zeros(4, 4).type(xi.dtype).to(xi.device)
    xi_hat[0:3, 0:3] = omega_hat
    xi_hat[0:3, 3] = v

    return xi_hat


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Computes the exponential map for the coordinate-vector omega.
    Returns a 3 x 3 SO(3) matrix.

    """

    assert torch.is_tensor(omega), "Input must be of type torch.Tensor."

    omega_hat = so3_hat(omega)

    if omega.norm() < _eps:
        R = torch.eye(3, 3).type(omega.dtype).to(omega.device) + omega_hat
    else:
        theta = omega.norm()
        s = theta.sin()
        c = theta.cos()
        omega_hat_sq = omega_hat.mm(omega_hat)
        # Coefficients of the Rodrigues formula
        A = s / theta
        B = (1 - c) / torch.pow(theta, 2)
        C = (theta - s) / torch.pow(theta, 3)
        R = (
            torch.eye(3, 3).type(omega.dtype).to(omega.device)
            + A * omega_hat
            + B * omega_hat_sq
        )

    return R


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """Computes the exponential map for the coordinate-vector xi.
    Returns a 4 x 4 SE(3) matrix.

    """

    assert torch.is_tensor(xi), "Input must be of type torch.tensor."

    v = xi[:3]
    omega = xi[3:]
    omega_hat = so3_hat(omega)

    if omega.norm() < _eps:
        R = torch.eye(3, 3).type(omega.dtype).to(omega.device) + omega_hat
        V = torch.eye(3, 3).type(omega.dtype).to(omega.device) + omega_hat
    else:
        theta = omega.norm()
        s = theta.sin()
        c = theta.cos()
        omega_hat_sq = omega_hat.mm(omega_hat)
        # Coefficients of the Rodrigues formula
        A = s / theta
        B = (1 - c) / torch.pow(theta, 2)
        C = (theta - s) / torch.pow(theta, 3)
        R = (
            torch.eye(3, 3).type(omega.dtype).to(omega.device)
            + A * omega_hat
            + B * omega_hat_sq
        )
        V = (
            torch.eye(3, 3).type(omega.dtype).to(omega.device)
            + B * omega_hat
            + C * omega_hat_sq
        )

    t = torch.mm(V, v.view(3, 1))
    last_row = torch.tensor([0, 0, 0, 1]).type(omega.dtype).to(omega.device)

    return torch.cat((torch.cat((R, t), dim=1), last_row.unsqueeze(0)), dim=0)


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm of SE(3) matrix -> se(3) twist [6].

    Returns [v; omega] in R^6 where v is the translation part and omega is
    the rotation part.  Uses the Rodrigues formula with a small-angle Taylor
    branch for numerical stability near the identity.
    """
    assert torch.is_tensor(T), "Input must be of type torch.Tensor."
    assert T.shape == (4, 4), f"Expected 4x4 SE(3) matrix, got {T.shape}."
    R = T[:3, :3]
    t = T[:3, 3]

    trace = torch.clamp(R.trace(), -1.0 + 1e-7, 3.0 - 1e-7)
    cos_angle = (trace - 1.0) * 0.5
    angle = torch.acos(cos_angle.clamp(-1.0, 1.0))

    if angle < 1e-6:
        # Small-angle: omega ≈ vex(R - R^T) / 2, V_inv ≈ I
        omega = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) * 0.5
        v = t
    else:
        sin_angle = torch.sin(angle)
        # omega_hat = skew(omega_unit) = (R - R^T) / (2 sin θ)
        omega_hat = (R - R.t()) / (2.0 * sin_angle)
        omega = torch.stack([
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0],
        ]) * angle

        # Rodrigues coefficients
        A = sin_angle / angle
        B = (1.0 - torch.cos(angle)) / (angle * angle)

        # Inverse left Jacobian V_inv:
        # V_inv = I - (θ/2) * Ω_hat + (1 - A/(2B)) * Ω_hat²
        # where Ω_hat = skew(omega) / θ  (unit-axis skew matrix)
        # so Ω_hat² = omega_hat @ omega_hat  (already unit-axis scaled)
        # Correct coefficient on Ω_hat² is (1 - A/(2B)), NOT * θ²
        V_inv = (
            torch.eye(3, device=T.device, dtype=T.dtype)
            - 0.5 * omega_hat * angle
            + (1.0 - A / (2.0 * B)) * (omega_hat @ omega_hat)
        )
        v = V_inv @ t

    return torch.cat([v, omega])


def se3_inv(T: torch.Tensor, reorthonormalize: bool = True) -> torch.Tensor:
    """Inverse of a 4x4 SE(3) matrix via the analytic rigid-transform formula.

    For ``T = [[R, t], [0, 1]]`` the inverse is ``[[R^-1, -R^-1 t], [0, 1]]``,
    and for a rotation ``R^-1 == R^T``. This avoids a general LU solve.

    DO NOT use this on the tracking hot path. Camera poses there are inverted by
    ``torch.linalg.inv`` for a reason: using ``R^T`` instead feeds a slightly
    wrong inverse back into the ``T = T @ dT`` chain, which then accelerates R
    away from SO(3) (det(R) measured collapsing 1.0 -> 0.89 in ~16 frames) and
    destroys tracking (fr1_desk ATE 0.135 -> 0.73/0.87, 538/573 frames lost).
    The control pipeline with ``linalg.inv`` keeps det(R) ~= 1, so this is a
    feedback failure of the analytic form, not a latent pipeline bug. Also note
    a 4x4 inverse is a GPU kernel, not a host sync — it is not a bottleneck, so
    there is no speed reason to replace it.

    ``reorthonormalize=True`` (default) projects R onto SO(3) with one
    Gram-Schmidt pass before transposing. This makes the inverse self-consistent
    for the SINGLE call but does NOT rescue the hot path (the rest of the
    pipeline still consumes the degraded pose). Its only intended use is
    short-lived relative transforms built from two clean poses — e.g. pose-graph
    edge construction in Workstream B — where ``reorthonormalize=False`` is also
    safe because the inputs are freshly composed ``se3_exp`` outputs.

    Args:
        T: SE(3) matrix, shape [4, 4].
        reorthonormalize: Project R onto SO(3) (Gram-Schmidt) before inverting.

    Returns:
        The inverse SE(3) matrix, shape [4, 4].
    """
    R = T[:3, :3]
    t = T[:3, 3]
    if reorthonormalize:
        # One modified Gram-Schmidt pass on the columns of R -> nearest SO(3).
        c0 = R[:, 0]
        c0 = c0 / c0.norm().clamp_min(1e-12)
        c1 = R[:, 1] - (c0 @ R[:, 1]) * c0
        c1 = c1 / c1.norm().clamp_min(1e-12)
        c2 = torch.linalg.cross(c0, c1)
        R = torch.stack([c0, c1, c2], dim=1)  # [3,3], orthonormal, det +1
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[:3, :3] = Rt
    out[:3, 3] = -(Rt @ t)
    out[3, 3] = 1.0
    return out
