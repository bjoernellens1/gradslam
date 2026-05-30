#!/usr/bin/env python3
"""
Benchmark Phase D+E implementation across TUM and RealSense datasets.

Runs SLAM on all available sequences, collects metrics, and generates a summary report.
"""
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv

@dataclass
class BenchmarkResult:
    dataset: str
    sequence: str
    status: str  # "OK", "FAILED", "TIMEOUT"
    ate_m: Optional[float] = None
    rpe_m: Optional[float] = None
    fps: Optional[float] = None
    lost_frames: Optional[int] = None
    total_frames: Optional[int] = None
    error: Optional[str] = None

def run_benchmark(dataset_type: str, sequence_path: str, dataset_name: str) -> BenchmarkResult:
    """Run SLAM on a single sequence and collect results."""
    sequence_name = Path(sequence_path).name
    print(f"\n{'='*70}")
    print(f"Running: {dataset_name} / {sequence_name}")
    print(f"Path: {sequence_path}")
    print(f"{'='*70}")

    result = BenchmarkResult(dataset=dataset_name, sequence=sequence_name, status="RUNNING")

    try:
        output_dir = Path("runs") / dataset_name / sequence_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run SLAM
        cmd = [
            "python", "scripts/run_slam.py",
            dataset_type,
            "--output", str(output_dir),
        ]

        if dataset_type == "tum":
            cmd.extend([
                "--dataset-root", "/mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/",
                "--sequence", sequence_name,
            ])
        elif dataset_type == "normalized":
            cmd.extend([
                "--dataset-root", sequence_path,
            ])

        proc = subprocess.run(
            cmd,
            timeout=1800,  # 30 min timeout
            capture_output=True,
            text=True
        )

        if proc.returncode != 0:
            result.status = "FAILED"
            result.error = proc.stderr[-500:] if proc.stderr else "Unknown error"
            print(f"❌ FAILED: {result.error}")
            return result

        # Load metrics.json
        metrics_file = output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                result.ate_m = metrics.get("ATE_m", None)
                result.rpe_m = metrics.get("RPE_m", None)
                result.fps = metrics.get("fps", None)
                result.lost_frames = metrics.get("lost_frames", None)
                result.total_frames = metrics.get("total_frames", None)

        result.status = "OK"
        print(f"✓ OK")
        if result.ate_m:
            print(f"  ATE: {result.ate_m:.4f} m")
        if result.fps:
            print(f"  FPS: {result.fps:.1f}")
        if result.lost_frames is not None:
            print(f"  Lost frames: {result.lost_frames}/{result.total_frames}")

        return result

    except subprocess.TimeoutExpired:
        result.status = "TIMEOUT"
        result.error = "30 minute timeout exceeded"
        print(f"⏱ TIMEOUT: {result.error}")
        return result

    except Exception as e:
        result.status = "FAILED"
        result.error = str(e)
        print(f"❌ FAILED: {e}")
        return result

def main():
    # Define benchmark sets
    benchmarks = []

    # TUM sequences
    tum_root = Path("/mnt/cps_persistent1_shared/datasets/public/TUM/tum_rgbd/")
    for seq_dir in tum_root.iterdir():
        if seq_dir.is_dir() and (seq_dir / f"rgbd_dataset_{seq_dir.name}").exists():
            benchmarks.append(("tum", str(seq_dir), f"TUM/{seq_dir.name}"))

    # RealSense sequences (extracted only)
    rs_root = Path("/mnt/cps_persistent1_shared/datasets/bjoern/realsense_handheld/")
    for seq_dir in rs_root.iterdir():
        if seq_dir.is_dir() and "_extracted" in seq_dir.name:
            benchmarks.append(("normalized", str(seq_dir), f"RealSense/{seq_dir.name}"))

    # Run benchmarks
    results = []
    for dataset_type, seq_path, dataset_name in sorted(benchmarks):
        result = run_benchmark(dataset_type, seq_path, dataset_name)
        results.append(result)

    # Generate report
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Dataset':<30} {'Sequence':<35} {'Status':<8} {'ATE (m)':<10} {'FPS':<8} {'Lost':<10}")
    print("-" * 100)

    for r in results:
        ate_str = f"{r.ate_m:.4f}" if r.ate_m else "N/A"
        fps_str = f"{r.fps:.1f}" if r.fps else "N/A"
        lost_str = f"{r.lost_frames}/{r.total_frames}" if r.lost_frames is not None else "N/A"
        print(f"{r.dataset:<30} {r.sequence:<35} {r.status:<8} {ate_str:<10} {fps_str:<8} {lost_str:<10}")

    # Write CSV report
    report_file = Path("runs/benchmark_report.csv")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Sequence", "Status", "ATE (m)", "RPE (m)", "FPS", "Lost Frames", "Total Frames", "Error"])
        for r in results:
            writer.writerow([
                r.dataset,
                r.sequence,
                r.status,
                r.ate_m or "",
                r.rpe_m or "",
                r.fps or "",
                r.lost_frames or "",
                r.total_frames or "",
                r.error or "",
            ])

    print(f"\n✓ Report saved to {report_file}")

    # Summary stats
    ok_count = sum(1 for r in results if r.status == "OK")
    failed_count = sum(1 for r in results if r.status == "FAILED")
    timeout_count = sum(1 for r in results if r.status == "TIMEOUT")

    print(f"\nRuns: {ok_count} OK, {failed_count} FAILED, {timeout_count} TIMEOUT")

    if ok_count > 0:
        ate_values = [r.ate_m for r in results if r.ate_m is not None]
        if ate_values:
            print(f"ATE: min={min(ate_values):.4f}m, max={max(ate_values):.4f}m, mean={sum(ate_values)/len(ate_values):.4f}m")

    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
