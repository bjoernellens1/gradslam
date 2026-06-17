#!/usr/bin/env python3
"""Collect metrics.json from each host's output dir and print a side-by-side
table. Usage: python scripts/collect_fps_results.py <host1_dir> <host2_dir> ..."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

FIELDS = [
    ("frames", "n_frames"),
    ("tracking_fps", "track_fps"),
    ("end_to_end_fps", "e2e_fps"),
    ("lost", "lost"),
    ("ate_rmse", "ate"),
    ("duration_s", "dur"),
]


def collect(root: Path) -> list[dict]:
    out = []
    for metrics in sorted(root.rglob("metrics.json")):
        try:
            data = json.loads(metrics.read_text())
        except Exception as e:
            print(f"!! could not read {metrics}: {e}", file=sys.stderr)
            continue
        out.append(
            {
                "host": root.name,
                "run": metrics.parent.name,
                "data": data,
            }
        )
    return out


def fmt(x: Optional[float], spec: str = ".2f") -> str:
    if x is None:
        return "-"
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    roots = [Path(a) for a in argv[1:]]
    rows = []
    for r in roots:
        rows.extend(collect(r))
    if not rows:
        print("no metrics.json found", file=sys.stderr)
        return 1

    # group by run name
    by_run: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], {})[r["host"]] = r["data"]

    # print markdown table
    cols = [h.name for h in roots]
    print("# Per-run metrics")
    print()
    for run, host_data in sorted(by_run.items()):
        print(f"## {run}")
        print()
        print("| host | " + " | ".join(FIELDS_NAME) + " |")
        print("|---" * (1 + len(FIELDS)) + "|")
        for h in cols:
            d = host_data.get(h, {})
            line = [h]
            for k, _ in FIELDS:
                line.append(fmt(d.get(k)))
            print("| " + " | ".join(line) + " |")
        print()
    return 0


FIELDS_NAME = [name for _, name in FIELDS]

if __name__ == "__main__":
    sys.exit(main(sys.argv))
