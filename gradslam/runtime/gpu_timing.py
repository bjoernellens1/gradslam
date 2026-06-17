"""GPU timing infrastructure for SLAM profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GPUTimer:
    """GPU-accelerated timer using CUDA events."""

    name: str
    device: torch.device

    def __post_init__(self):
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)
        self._elapsed_ms: Optional[float] = None

    def __enter__(self):
        self.start_evt.record()
        return self

    def __exit__(self, *args):
        self.end_evt.record()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (forces sync)."""
        if self._elapsed_ms is None:
            torch.cuda.synchronize(self.device)
            self._elapsed_ms = self.start_evt.elapsed_time(self.end_evt)
        return self._elapsed_ms

    def reset(self):
        self._elapsed_ms = None


@dataclass
class SLAMProfiler:
    """Fine-grained per-stage profiler for SLAM pipeline."""

    device: torch.device
    enabled: bool = True

    def __post_init__(self):
        if not self.enabled:
            return

        self.timers = {
            "io_wait": GPUTimer("io_wait", self.device),
            "h2d": GPUTimer("h2d", self.device),
            "preprocess": GPUTimer("preprocess", self.device),
            "pyramid": GPUTimer("pyramid", self.device),
            "track": GPUTimer("track", self.device),
            "raycast": GPUTimer("raycast", self.device),
            "integrate": GPUTimer("integrate", self.device),
            "feature": GPUTimer("feature", self.device),
            "keyframe": GPUTimer("keyframe", self.device),
            "sync": GPUTimer("sync", self.device),
        }

        self.history: list[dict[str, float]] = []
        self.gpu_mem_allocated: list[int] = []
        self.gpu_mem_reserved: list[int] = []

    def record_frame(self, queue_depth: int = -1):
        """Record timing for current frame."""
        if not self.enabled:
            return

        entry = {
            "queue_depth": queue_depth,
            "gpu_mem_allocated": torch.cuda.memory_allocated(self.device),
            "gpu_mem_reserved": torch.cuda.memory_reserved(self.device),
        }

        for name, timer in self.timers.items():
            entry[name] = timer.elapsed_ms()
            timer.reset()

        self.history.append(entry)

    def summary(self, warmup_frames: int = 10) -> dict[str, float]:
        """Compute average timing breakdown (warmup-excluded)."""
        if not self.enabled or len(self.history) <= warmup_frames:
            return {}

        timed = self.history[warmup_frames:]
        summary = {}

        for key in timed[0].keys():
            if key == "queue_depth":
                depths = [t[key] for t in timed if t[key] >= 0]
                summary[f"avg_{key}"] = sum(depths) / len(depths) if depths else -1
            elif key.startswith("gpu_mem"):
                summary[f"avg_{key}"] = sum(t[key] for t in timed) / len(timed)
            else:
                summary[f"avg_{key}_ms"] = sum(t[key] for t in timed) / len(timed)

        return summary

    def print_summary(self, warmup_frames: int = 10):
        """Print human-readable timing breakdown."""
        if not self.enabled:
            return

        summary = self.summary(warmup_frames)
        if not summary:
            return

        print("\n=== SLAM Profiler Summary (warmup-excluded) ===")
        for key, val in summary.items():
            if "ms" in key:
                print(f"  {key:25s}: {val:8.3f} ms")
            elif "gpu_mem" in key:
                print(f"  {key:25s}: {val / 1e9:8.3f} GB")
            else:
                print(f"  {key:25s}: {val:8.1f}")
        print("=" * 50)


@dataclass
class CPUTimer:
    """CPU timer for non-GPU operations."""

    name: str

    def __post_init__(self):
        self._start: Optional[float] = None
        self._elapsed_ms: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000.0

    def elapsed_ms(self) -> float:
        if self._elapsed_ms is None:
            return 0.0
        return self._elapsed_ms

    def reset(self):
        self._elapsed_ms = None
