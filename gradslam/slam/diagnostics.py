"""Startup sanity-check diagnostics for GradSLAM.

Call ``log_startup_sanity`` once before the main SLAM loop to validate
depth units, intrinsics, timestamp alignment, and GT motion scale.
"""
from __future__ import annotations

import numpy as np
import torch


def log_startup_sanity(
    first_frame_depth: torch.Tensor,       # [H, W] in meters
    intrinsics: torch.Tensor,              # [3,3] or [4,4]
    process_scale: float,                  # scaling factor applied
    rgb_ts: float | None,                  # RGB timestamp (or None)
    depth_ts: float | None,                # depth timestamp (or None)
    gt_poses: list[np.ndarray] | None = None,  # first N GT poses as list of [4,4]
    label: str = "",                       # e.g., sequence name
) -> dict:
    """Run startup sanity checks and print a diagnostic block.

    Prints a formatted block to stdout. Also returns a dict of the computed
    stats for testing/logging purposes.
    """
    header_label = f" Startup Sanity: {label} " if label else " Startup Sanity "
    width = 70
    line_char = "─"  # ─

    header = f"{line_char * 3}{header_label}{line_char * (width - 3 - len(header_label))}"
    footer = line_char * width
    print(header)

    result: dict = {}

    # ------------------------------------------------------------------
    # 1. Depth units check
    # ------------------------------------------------------------------
    depth = first_frame_depth.float()
    valid_mask = depth > 0
    valid_ratio = float(valid_mask.float().mean().item())
    result["valid_ratio"] = valid_ratio

    warning: str | None = None
    if valid_mask.any():
        valid_depth = depth[valid_mask]
        depth_mean = float(valid_depth.mean().item())
        depth_min = float(valid_depth.min().item())
        depth_max = float(valid_depth.max().item())
        result["depth_mean"] = depth_mean
        result["depth_min"] = depth_min
        result["depth_max"] = depth_max

        if depth_mean < 0.05:
            warning = f"depth mean {depth_mean:.4f}m is very small — possibly in millimeters?"
        elif depth_mean > 20.0:
            warning = f"depth mean {depth_mean:.2f}m is very large — check units"
        result["warning"] = warning

        warn_str = f"  ** WARNING: {warning} **" if warning else ""
        print(
            f"  depth range (m):   mean={depth_mean:.2f}  "
            f"min={depth_min:.2f}  max={depth_max:.2f}  "
            f"valid={valid_ratio * 100:.1f}%"
            + warn_str
        )
    else:
        result["depth_mean"] = None
        result["depth_min"] = None
        result["depth_max"] = None
        result["warning"] = "no valid (>0) depth pixels found"
        print(f"  depth range (m):   ** no valid pixels ** valid={valid_ratio * 100:.1f}%")

    # ------------------------------------------------------------------
    # 2. Intrinsics after process_scale
    # ------------------------------------------------------------------
    K3 = intrinsics[:3, :3].float()
    K_scaled = K3 * process_scale
    K_scaled[2, 2] = 1.0

    fx = float(K_scaled[0, 0].item())
    fy = float(K_scaled[1, 1].item())
    cx = float(K_scaled[0, 2].item())
    cy = float(K_scaled[1, 2].item())
    result["fx"] = fx
    result["fy"] = fy
    result["cx"] = cx
    result["cy"] = cy

    print(f"  intrinsics (scaled): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # ------------------------------------------------------------------
    # 3. Convention note
    # ------------------------------------------------------------------
    print("  pose convention:   T_world_camera (camera→world)")

    # ------------------------------------------------------------------
    # 4. RGB/depth timestamp delta
    # ------------------------------------------------------------------
    if rgb_ts is not None and depth_ts is not None:
        dt = abs(float(rgb_ts) - float(depth_ts))
        result["dt"] = dt
        dt_str = f"{dt:.3f}s"
        dt_flag = "  ** WARNING: large RGB/depth gap **" if dt > 0.05 else "  [OK]"
        print(f"  RGB/depth dt:      {dt_str}{dt_flag}")
    else:
        result["dt"] = None
        # Skip per spec when either is None

    # ------------------------------------------------------------------
    # 5. (valid-pixel ratio already printed above in depth range line)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 6. GT relative motions
    # ------------------------------------------------------------------
    if gt_poses is not None and len(gt_poses) >= 2:
        n_pairs = min(10, len(gt_poses) - 1)
        motions_cm: list[float] = []
        for i in range(n_pairs):
            T0 = np.asarray(gt_poses[i], dtype=np.float64)
            T1 = np.asarray(gt_poses[i + 1], dtype=np.float64)
            # Translation part of relative transform T0^{-1} T1
            t_diff = T1[:3, 3] - T0[:3, 3]
            dist_cm = float(np.linalg.norm(t_diff) * 100.0)
            motions_cm.append(dist_cm)
        result["gt_motions_cm"] = motions_cm
        motion_str = " ".join(f"{d:.1f}" for d in motions_cm)
        print(f"  GT frame motions (cm): {motion_str}")
    else:
        result["gt_motions_cm"] = None

    print(footer)
    return result
