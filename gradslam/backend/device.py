"""Device detection and reporting for PyTorch backends (CUDA, ROCm, CPU)."""

from __future__ import annotations

import torch


def accelerator_backend() -> str:
    """Detect the active PyTorch backend.

    Returns:
        One of: "rocm", "cuda", "cpu", "unknown".
    """
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "unknown"


def default_device() -> torch.device:
    """Return the default device for tensor operations.

    Returns:
        torch.device("cuda") if available, else torch.device("cpu").
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backend_report() -> dict:
    """Generate a diagnostic report of the PyTorch/accelerator stack.

    Returns:
        dict with keys: torch, torch_cuda, torch_hip, cuda_available_api,
        backend, device_count, device_name.
    """
    return {
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available_api": torch.cuda.is_available(),
        "backend": accelerator_backend(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
