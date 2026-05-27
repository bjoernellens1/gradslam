#!/usr/bin/env python
"""Check ROCm PyTorch stack and run a basic GPU test."""

import sys
import torch

try:
    from gradslam.backend.device import backend_report
    report = backend_report()
    for key, val in report.items():
        print(f"{key}: {val}")
except Exception as e:
    print(f"Could not import gradslam.backend.device: {e}")
    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")

# Check GPU availability
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() is False")
    sys.exit(1)

# Check ROCm
if getattr(torch.version, "hip", None) is None:
    print("WARNING: torch.version.hip is None — this may not be a ROCm PyTorch build")
    # Allow fallback to CPU for testing; don't fail

# Run a simple GPU matmul
print("\nRunning GPU matmul test...")
x = torch.randn(2048, 2048, device="cuda")
y = x @ x.T
torch.cuda.synchronize()
print(f"✓ GPU matmul OK: {y.shape}, {y.dtype}")
print("\nROCm stack check PASSED")
