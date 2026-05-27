"""Backend utilities for device management and torch.compile."""

from .device import accelerator_backend, backend_report, default_device
from .compile import compile_if_requested, compile_debug

__all__ = [
    "accelerator_backend",
    "backend_report",
    "default_device",
    "compile_if_requested",
    "compile_debug",
]
