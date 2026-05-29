"""Photometric residual for joint geometric+photometric ICP.

Computes per-correspondence intensity residuals and their 6-DOF Jacobians.
"""

import torch
import torch.nn.functional as F


def sobel_gradients(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Sobel image gradients.

    Args:
        gray: Grayscale image [H, W], values in [0, 1].

    Returns:
        (dI_dx, dI_dy): Gradient images [H, W] each.
    """
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3) / 8.0
    g = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    dI_dx = F.conv2d(g, kx, padding=1).squeeze()  # [H, W]
    dI_dy = F.conv2d(g, ky, padding=1).squeeze()  # [H, W]
    return dI_dx, dI_dy


def photometric_residuals_and_jacobian(
    live_vertex: torch.Tensor,
    live_gray_pixels: torch.Tensor,
    ref_gray: torch.Tensor,
    ref_dI_dx: torch.Tensor,
    ref_dI_dy: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute photometric residuals and 6-DOF Jacobians for valid correspondences.

    Args:
        live_vertex: Live vertices in ref frame [N, 3] (already transformed).
        live_gray_pixels: Intensity of live pixels at correspondence locations [N].
        ref_gray: Reference grayscale image [H, W].
        ref_dI_dx: Reference x-gradient [H, W].
        ref_dI_dy: Reference y-gradient [H, W].
        K: Scaled camera intrinsics [3, 3].

    Returns:
        (A_photo, b_photo, valid_mask): Jacobian [M, 6], residual [M, 1], valid [N] bool.
        M <= N (only valid projected correspondences).
    """
    N = live_vertex.shape[0]
    H, W = ref_gray.shape
    device = live_vertex.device
    dtype = live_vertex.dtype

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = live_vertex[:, 0]  # [N]
    Y = live_vertex[:, 1]  # [N]
    Z = live_vertex[:, 2].clamp(min=1e-6)  # [N]

    # Project to pixel coordinates
    u = X * fx / Z + cx  # [N]
    v = Y * fy / Z + cy  # [N]

    u_int = u.long()
    v_int = v.long()
    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

    if in_bounds.sum() < 1:
        empty = torch.zeros(0, 6, device=device, dtype=dtype)
        return empty, torch.zeros(0, 1, device=device, dtype=dtype), in_bounds

    u_safe = u_int.clamp(0, W - 1)
    v_safe = v_int.clamp(0, H - 1)

    # Sample reference intensity and gradients at projected pixel
    ref_intensity = ref_gray[v_safe, u_safe]   # [N]
    gx = ref_dI_dx[v_safe, u_safe]             # [N]
    gy = ref_dI_dy[v_safe, u_safe]             # [N]

    # Photometric residual: r = ref_sampled - live
    residual = (ref_intensity - live_gray_pixels)  # [N]

    # Jacobian of projection: d(u,v)/d(X,Y,Z)
    # du/dX = fx/Z,  du/dY = 0,      du/dZ = -fx*X/Z^2
    # dv/dX = 0,     dv/dY = fy/Z,   dv/dZ = -fy*Y/Z^2
    inv_Z = 1.0 / Z
    inv_Z2 = inv_Z * inv_Z

    du_dX = fx * inv_Z
    du_dZ = -fx * X * inv_Z2
    dv_dY = fy * inv_Z
    dv_dZ = -fy * Y * inv_Z2

    # dI/d(X,Y,Z) = gx * d(u)/d(XYZ) + gy * d(v)/d(XYZ)
    dI_dX = gx * du_dX                    # [N]
    dI_dY = gy * dv_dY                    # [N]
    dI_dZ = gx * du_dZ + gy * dv_dZ      # [N]

    # Chain rule through SE(3): Jacobian of X_ref w.r.t. xi (twist)
    # For left-multiply: J_xi = [I | -[X]_x] * d_xyz
    # d(X,Y,Z)/d_xi = [I | -[X]x] where [X]x is skew-symmetric
    # Row of J_photo = [dI_dX, dI_dY, dI_dZ] @ J_xi_xyz
    # J_xi_xyz for SE3 left-mult (order: v1,v2,v3,w1,w2,w3):
    #   [1, 0, 0, 0,  Z, -Y]
    #   [0, 1, 0, -Z, 0,  X]
    #   [0, 0, 1, Y, -X,  0]
    # J_photo_xi = dI_dX * [1, 0, 0, 0, Z, -Y]
    #            + dI_dY * [0, 1, 0, -Z, 0, X]
    #            + dI_dZ * [0, 0, 1, Y, -X, 0]

    J = torch.stack([
        dI_dX,                          # d/dv1
        dI_dY,                          # d/dv2
        dI_dZ,                          # d/dv3
        dI_dY * (-Z) + dI_dZ * Y,      # d/dw1
        dI_dX * Z    + dI_dZ * (-X),   # d/dw2
        dI_dX * (-Y) + dI_dY * X,      # d/dw3
    ], dim=-1)  # [N, 6]

    return J[in_bounds], residual[in_bounds].unsqueeze(-1), in_bounds
