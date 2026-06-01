"""torch.compile utilities, gated by environment flag."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch


def compile_if_requested(fn: Callable, *, fullgraph: bool = False) -> Callable:
    """Conditionally compile a function via torch.compile.

    Respects GRADSLAM_COMPILE environment variable (0/1/true/yes), checked
    LAZILY at first call (not at import). Module-level functions get wrapped
    during import, but the flag may be set afterward (e.g. by a --compile CLI
    flag), so a call-time check keeps the flag order-independent. When disabled
    the wrapper just forwards to the eager function with negligible overhead.

    Args:
        fn: Function to compile.
        fullgraph: If True, force fullgraph=True in torch.compile (for fixed-shape tests).
                   If False, use fullgraph=False (default, allows dynamic shapes).

    Returns:
        A wrapper that runs eager or compiled depending on GRADSLAM_COMPILE at
        call time. The compiled callable is built once and cached.
    """
    compiled_holder: list = []

    def _maybe_compiled(*args, **kwargs):
        mode = os.environ.get("GRADSLAM_COMPILE", "0")
        if mode not in {"1", "true", "yes"}:
            return fn(*args, **kwargs)
        if not compiled_holder:
            compiled_holder.append(
                torch.compile(fn, mode="reduce-overhead", fullgraph=fullgraph)
            )
        return compiled_holder[0](*args, **kwargs)

    return _maybe_compiled


def compile_debug(fn: Callable) -> Callable:
    """Compile with fullgraph=True (useful for debugging graph breaks in tests)."""
    return torch.compile(fn, fullgraph=True)
