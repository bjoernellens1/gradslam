"""Backend utilities for device management and torch.compile."""

from .compile import compile_debug, compile_if_requested
from .device import accelerator_backend, backend_report, default_device

__all__ = [
    "accelerator_backend",
    "backend_report",
    "default_device",
    "compile_if_requested",
    "compile_debug",
]
