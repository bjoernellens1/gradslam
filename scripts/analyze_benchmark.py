#!/usr/bin/env python3
"""
Analyze benchmark results and generate summary report.
"""
import json
import csv
from pathlib import Path
from typing import Optional

def load_benchmark_csv(csv_path: str) -> list[dict]:
    """Load benchmark CSV."""
    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)
    return results

def analyze_results(csv_path: str, output_path: Optional[str] = None) -> dict:
    """Analyze benchmark results and generate report."""
    results = load_benchmark_csv(csv_path)

    # Parse results
    successful = []
    failed = []

    for r in results:
        if r["Status"] == "OK" and r["ATE (m)"] and r["ATE (m)"] != "":
            try:
                ate = float(r["ATE (m)"])
                fps = float(r["FPS"]) if r["FPS"] else None
                lost = int(r["Lost Frames"]) if r["Lost Frames"] else None
                total = int(r["Total Frames"]) if r["Total Frames"] else None

                successful.append({
                    "dataset": r["Dataset"],
                    "sequence": r["Sequence"],
                    "ate": ate,
                    "fps": fps,
                    "lost": lost,
                    "total": total,
                })
            except (ValueError, TypeError):
                failed.append(r)
        else:
            failed.append(r)

    # Compute statistics
    summary = {
        "total": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
    }

    if successful:
        ates = [s["ate"] for s in successful]
        summary["ate_min"] = min(ates)
        summary["ate_max"] = max(ates)
        summary["ate_mean"] = sum(ates) / len(ates)
        summary["ate_median"] = sorted(ates)[len(ates) // 2]

        fps_values = [s["fps"] for s in successful if s["fps"] is not None]
        if fps_values:
            summary["fps_mean"] = sum(fps_values) / len(fps_values)
            summary["fps_min"] = min(fps_values)
            summary["fps_max"] = max(fps_values)

    # Print report
    print("=" * 80)
    print("BENCHMARK ANALYSIS")
    print("=" * 80)
    print(f"\nOverall: {summary['successful']}/{summary['total']} successful ({summary['success_rate']*100:.1f}%)")

    if successful:
        print(f"\nATE Statistics (meters):")
        print(f"  Min:    {summary['ate_min']:.4f} m")
        print(f"  Max:    {summary['ate_max']:.4f} m")
        print(f"  Mean:   {summary['ate_mean']:.4f} m")
        print(f"  Median: {summary['ate_median']:.4f} m")

        if "fps_mean" in summary:
            print(f"\nFPS Statistics:")
            print(f"  Min:  {summary['fps_min']:.1f} fps")
            print(f"  Max:  {summary['fps_max']:.1f} fps")
            print(f"  Mean: {summary['fps_mean']:.1f} fps")

    print(f"\nDetailed Results:")
    print(f"{'Dataset':<30} {'Sequence':<35} {'ATE (m)':<10} {'FPS':<8}")
    print("-" * 83)

    for s in sorted(successful, key=lambda x: x["ate"]):
        ate_str = f"{s['ate']:.4f}"
        fps_str = f"{s['fps']:.1f}" if s["fps"] else "N/A"
        print(f"{s['dataset']:<30} {s['sequence']:<35} {ate_str:<10} {fps_str:<8}")

    if failed:
        print(f"\nFailed/Skipped ({len(failed)}):")
        for f in failed:
            print(f"  {f['Dataset']}/{f['Sequence']}: {f['Status']} — {f.get('Error', 'N/A')}")

    # Write JSON report
    if output_path:
        report = {
            "summary": summary,
            "successful": successful,
            "failed": [r for r in results if r["Status"] != "OK"],
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n✓ Report saved to {output_path}")

    return summary

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "runs/benchmark_report.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "runs/benchmark_analysis.json"

    if Path(csv_path).exists():
        analyze_results(csv_path, output_path)
    else:
        print(f"Benchmark report not found: {csv_path}")
        sys.exit(1)
