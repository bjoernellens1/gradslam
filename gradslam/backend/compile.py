"""torch.compile utilities, gated by environment flag."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch


def compile_if_requested(fn: Callable, *, fullgraph: bool = False) -> Callable:
    """Conditionally compile a function via torch.compile.

    Respects GRADSLAM_COMPILE environment variable (0/1/true/yes).
    When disabled, returns the original function unchanged.

    Args:
        fn: Function to compile.
        fullgraph: If True, force fullgraph=True in torch.compile (for fixed-shape tests).
                   If False, use fullgraph=False (default, allows dynamic shapes).

    Returns:
        Compiled function or original function if compilation is disabled.
    """
    mode = os.environ.get("GRADSLAM_COMPILE", "0")
    if mode not in {"1", "true", "yes"}:
        return fn
    return torch.compile(fn, mode="reduce-overhead", fullgraph=fullgraph)


def compile_debug(fn: Callable) -> Callable:
    """Compile with fullgraph=True (useful for debugging graph breaks in tests)."""
    return torch.compile(fn, fullgraph=True)
