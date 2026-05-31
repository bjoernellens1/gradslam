#!/usr/bin/env python3
"""Analyze ablation results: feasibility (ATE<10cm), fps/ATE Pareto front, and a
ranked recommendation.

Objective (user-selected): among configs feasible on the target scenes
(ATE < ATE_MAX on every target), prefer the largest ATE margin (lowest worst-case
ATE), breaking ties by tracking FPS. Also reports the speed-optimal feasible
config and the full fps-vs-ATE Pareto front so the trade-off is visible.

Usage: python scripts/ablation/analyze.py [--ate-max 0.10] [--targets fr1_xyz,fr1_desk]
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path("outputs/ablation/results.csv")


def load(results=RESULTS):
    rows = []
    if not results.exists():
        return rows
    for r in csv.DictReader(open(results)):
        def f(k):
            try:
                return float(r[k])
            except (TypeError, ValueError):
                return None
        rows.append({
            "id": r["id"], "seq": r["seq"], "ate": f("ate_rmse_m"),
            "rpe": f("rpe_rmse_m"), "tfps": f("tracking_fps"),
            "e2efps": f("end_to_end_fps"), "lost": f("lost_frames"),
            "status": r.get("status", ""), "flags": r.get("flags", ""),
        })
    return rows


def pareto(points):
    """points: list of (fps, ate, label). Non-dominated = higher fps AND lower ate."""
    front = []
    for p in points:
        if p[0] is None or p[1] is None:
            continue
        dominated = any(
            (q[0] >= p[0] and q[1] <= p[1] and (q[0] > p[0] or q[1] < p[1]))
            for q in points if q[0] is not None and q[1] is not None
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda x: -x[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ate-max", type=float, default=0.10)
    ap.add_argument("--targets", default="freiburg1_xyz,freiburg1_desk")
    ap.add_argument("--metric", choices=["tfps", "e2efps"], default="tfps")
    args = ap.parse_args()
    targets = args.targets.split(",")

    rows = load()
    if not rows:
        print("No results yet (outputs/ablation/results.csv missing/empty).")
        return

    by_id = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["seq"]] = r

    # Per-sequence Pareto + table
    seqs = sorted({r["seq"] for r in rows})
    print(f"\n{'='*78}\nABLATION RESULTS  (ATE_MAX={args.ate_max} m, metric={args.metric})\n{'='*78}")
    for s in seqs:
        pts = [(by_id[i][s]["tfps"], by_id[i][s]["ate"], i) for i in by_id if s in by_id[i]]
        print(f"\n--- {s} ---  (sorted by {args.metric} desc)")
        recs = sorted(
            [by_id[i][s] for i in by_id if s in by_id[i]],
            key=lambda r: (-(r[args.metric] or -1)),
        )
        print(f"  {'config':14s} {'ATE(m)':>8} {'tFPS':>6} {'e2eFPS':>7} {'lost':>5}  feasible")
        for r in recs:
            feas = "✓" if (r["ate"] is not None and r["ate"] < args.ate_max) else "✗"
            print(f"  {r['id']:14s} {('%.4f'%r['ate']) if r['ate'] is not None else 'NA':>8} "
                  f"{('%.1f'%r['tfps']) if r['tfps'] else 'NA':>6} "
                  f"{('%.1f'%r['e2efps']) if r['e2efps'] else 'NA':>7} "
                  f"{('%d'%r['lost']) if r['lost'] is not None else 'NA':>5}  {feas}")
        front = pareto(pts)
        print(f"  Pareto front ({args.metric} vs ATE): " +
              ", ".join(f"{l}({f:.1f}fps,{a:.3f}m)" for f, a, l in front))

    # Cross-target feasibility + recommendation
    print(f"\n{'='*78}\nFEASIBLE ACROSS TARGETS {targets} (ATE<{args.ate_max} on every target)\n{'='*78}")
    feasible = []
    for i, perseq in by_id.items():
        if not all(t in perseq for t in targets):
            continue
        ates = [perseq[t]["ate"] for t in targets]
        fpss = [perseq[t][args.metric] for t in targets]
        if any(a is None for a in ates) or any(perseq[t]["status"] != "ok" for t in targets):
            continue
        worst_ate = max(ates)
        min_fps = min(f for f in fpss if f is not None) if all(fpss) else None
        if worst_ate < args.ate_max:
            feasible.append((i, worst_ate, min_fps, ates, fpss))

    if not feasible:
        print("  (none yet — waiting for more results, or no config is feasible on all targets)")
    else:
        # User objective: best ATE margin (worst_ate asc) then FPS desc
        by_margin = sorted(feasible, key=lambda x: (x[1], -(x[2] or 0)))
        by_speed = sorted(feasible, key=lambda x: (-(x[2] or 0), x[1]))
        print(f"  {'config':14s} {'worst_ATE':>9} {'min_'+args.metric:>9}   per-target ATE / FPS")
        for i, wa, mf, ates, fpss in by_margin:
            det = "  ".join(f"{t.split('_')[-1]}:{a:.3f}/{(f or 0):.0f}" for t, a, f in zip(targets, ates, fpss))
            print(f"  {i:14s} {wa:9.4f} {(mf or 0):9.1f}   {det}")
        print(f"\n  >>> RECOMMENDED (best ATE margin, tie→FPS): {by_margin[0][0]}  "
              f"(worst ATE {by_margin[0][1]:.4f} m, {args.metric} {by_margin[0][2]:.1f})")
        print(f"  >>> FASTEST feasible (max FPS s.t. ATE<{args.ate_max}): {by_speed[0][0]}  "
              f"({args.metric} {by_speed[0][2]:.1f}, worst ATE {by_speed[0][1]:.4f} m)")
        # flags for the two picks
        for label, pick in (("RECOMMENDED", by_margin[0][0]), ("FASTEST", by_speed[0][0])):
            fl = next((r["flags"] for r in rows if r["id"] == pick), "")
            print(f"     {label} flags: {fl}")


if __name__ == "__main__":
    main()
