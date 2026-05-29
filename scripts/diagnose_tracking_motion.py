#!/usr/bin/env python3
"""Diagnose relative-motion failures in TUM-format trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gradslam.evaluation import compute_ate, compute_rpe, load_tum_poses


def _associate_with_timestamps(
    poses_est: dict[float, np.ndarray],
    poses_gt: dict[float, np.ndarray],
    max_dt: float,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    gt_ts = np.array(sorted(poses_gt.keys()), dtype=np.float64)
    associated = []
    for ts_est, T_est in sorted(poses_est.items()):
        nearest = int(np.argmin(np.abs(gt_ts - ts_est)))
        if abs(float(gt_ts[nearest]) - float(ts_est)) <= max_dt:
            associated.append((float(ts_est), T_est, poses_gt[float(gt_ts[nearest])]))
    return associated


def _rotation_angle_deg(T: np.ndarray) -> float:
    cos_angle = (np.trace(T[:3, :3]) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _relative_rows(
    associated: list[tuple[float, np.ndarray, np.ndarray]],
    delta: int,
) -> list[dict[str, float]]:
    rows = []
    for i in range(len(associated) - delta):
        ts_i, T_est_i, T_gt_i = associated[i]
        ts_j, T_est_j, T_gt_j = associated[i + delta]
        rel_est = np.linalg.inv(T_est_i) @ T_est_j
        rel_gt = np.linalg.inv(T_gt_i) @ T_gt_j
        err = np.linalg.inv(rel_gt) @ rel_est
        est_t = float(np.linalg.norm(rel_est[:3, 3]))
        gt_t = float(np.linalg.norm(rel_gt[:3, 3]))
        rows.append(
            {
                "i": i,
                "j": i + delta,
                "ts_i": ts_i,
                "ts_j": ts_j,
                "dt": ts_j - ts_i,
                "est_translation": est_t,
                "gt_translation": gt_t,
                "translation_scale_ratio": est_t / gt_t if gt_t > 1e-8 else 0.0,
                "translation_error": float(np.linalg.norm(err[:3, 3])),
                "est_rotation_deg": _rotation_angle_deg(rel_est),
                "gt_rotation_deg": _rotation_angle_deg(rel_gt),
                "rotation_error_deg": _rotation_angle_deg(err),
            }
        )
    return rows


def _summary_for(
    poses_est: dict[float, np.ndarray],
    poses_gt: dict[float, np.ndarray],
    max_dt: float,
    delta: int,
) -> tuple[dict, list[dict[str, float]]]:
    associated = _associate_with_timestamps(poses_est, poses_gt, max_dt=max_dt)
    pairs = [(T_est, T_gt) for _, T_est, T_gt in associated]
    summary = {
        "associated_pairs": len(pairs),
        "delta": delta,
    }
    if len(pairs) >= 2:
        ate = compute_ate(pairs, align=True)
        summary.update(
            {
                "ate_rmse_m": ate.rmse,
                "ate_mean_m": ate.mean,
                "ate_max_m": ate.max,
            }
        )
    if len(pairs) >= delta + 1:
        rpe = compute_rpe(pairs, delta=delta)
        summary.update(
            {
                "rpe_translation_rmse_m": rpe.rmse_t,
                "rpe_rotation_rmse_deg": rpe.rmse_r,
                "rpe_translation_mean_m": rpe.mean_t,
                "rpe_rotation_mean_deg": rpe.mean_r,
            }
        )
    rows = _relative_rows(associated, delta=delta) if len(pairs) >= delta + 1 else []
    if rows:
        trans_errors = np.array([r["translation_error"] for r in rows], dtype=np.float64)
        scale_ratios = np.array([r["translation_scale_ratio"] for r in rows], dtype=np.float64)
        first_bad = next((r for r in rows if r["translation_error"] > 0.10), None)
        summary.update(
            {
                "relative_translation_error_p50_m": float(np.percentile(trans_errors, 50)),
                "relative_translation_error_p90_m": float(np.percentile(trans_errors, 90)),
                "relative_translation_scale_ratio_median": float(np.median(scale_ratios)),
                "first_translation_error_gt_10cm": first_bad,
            }
        )
    return summary, rows


def _parse_tracking_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split()
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    if not rows:
        return {}
    lost = sum(int(r.get("lost", "0")) for r in rows)
    motion_fail = sum(1 for r in rows if r.get("motion_gate") == "0")
    inliers = [float(r.get("inlier_ratio", "0")) for r in rows]
    return {
        "tracking_rows": len(rows),
        "lost_count": lost,
        "motion_gate_fail_count": motion_fail,
        "mean_inlier_ratio": float(np.mean(inliers)) if inliers else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimated", required=True, help="Estimated TUM trajectory")
    parser.add_argument("--gt", required=True, help="Ground-truth TUM trajectory")
    parser.add_argument("--tracking-metrics", default=None, help="Optional tracking_metrics.txt")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON/CSV diagnostics")
    parser.add_argument("--max-dt", type=float, default=0.1, help="Max timestamp association delta")
    parser.add_argument("--delta", type=int, default=1, help="Relative-motion interval")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poses_est = load_tum_poses(args.estimated)
    poses_gt = load_tum_poses(args.gt)
    direct_summary, rows = _summary_for(poses_est, poses_gt, args.max_dt, args.delta)
    inverted_summary, _ = _summary_for(
        {ts: np.linalg.inv(T) for ts, T in poses_est.items()},
        poses_gt,
        args.max_dt,
        args.delta,
    )

    with open(output_dir / "relative_motion.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "i",
            "j",
            "ts_i",
            "ts_j",
            "dt",
            "est_translation",
            "gt_translation",
            "translation_scale_ratio",
            "translation_error",
            "est_rotation_deg",
            "gt_rotation_deg",
            "rotation_error_deg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "estimated": str(args.estimated),
        "groundtruth": str(args.gt),
        "max_dt": args.max_dt,
        "direct": direct_summary,
        "inverted_estimate": inverted_summary,
        "tracking_metrics": (
            _parse_tracking_metrics(Path(args.tracking_metrics))
            if args.tracking_metrics
            else {}
        ),
    }
    (output_dir / "motion_diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
