"""Optional torch.compile tests for ICP hot paths (CPU, fixed shapes)."""

import os
import pytest
import torch
from gradslam.icp.solvers import solve_lm_6x6
from gradslam.icp.residuals import point_to_plane_projective
from gradslam.backend.compile import compile_if_requested

# Only run when GRADSLAM_COMPILE=1 is set
skip_unless_compile = pytest.mark.skipif(
    os.environ.get("GRADSLAM_COMPILE", "0") != "1",
    reason="Set GRADSLAM_COMPILE=1 to run compile tests",
)


def test_compile_if_requested_eager_by_default(monkeypatch):
    """With GRADSLAM_COMPILE unset/0, the wrapper forwards to eager and matches."""
    monkeypatch.delenv("GRADSLAM_COMPILE", raising=False)
    wrapped = compile_if_requested(point_to_plane_projective, fullgraph=True)
    vl = torch.randn(50, 3)
    nm = torch.nn.functional.normalize(torch.randn(50, 3), dim=-1)
    vm = torch.randn(50, 3)
    A_e, b_e = point_to_plane_projective(vl, nm, vm)
    A_w, b_w = wrapped(vl, nm, vm)
    assert torch.equal(A_e, A_w) and torch.equal(b_e, b_w)


def test_compile_if_requested_lazy_flag_disabled_is_eager(monkeypatch):
    """The flag is read at CALL time, not wrap time. Wrapped while OFF and left
    OFF, the wrapper must forward to eager (no compilation, host-independent)."""
    monkeypatch.delenv("GRADSLAM_COMPILE", raising=False)
    wrapped = compile_if_requested(solve_lm_6x6, fullgraph=True)
    A, b = torch.randn(80, 6), torch.randn(80, 1)
    assert torch.equal(wrapped(A, b, damp=1e-3), solve_lm_6x6(A, b, damp=1e-3))


@skip_unless_compile
def test_compile_if_requested_lazy_flag_enabled(monkeypatch):
    """Wrapping with the flag OFF then enabling it still compiles+runs correctly
    (order-independent). Requires GRADSLAM_COMPILE=1 + a working inductor backend."""
    monkeypatch.delenv("GRADSLAM_COMPILE", raising=False)
    wrapped = compile_if_requested(solve_lm_6x6, fullgraph=True)  # wrapped while OFF
    A, b = torch.randn(80, 6), torch.randn(80, 1)
    x_ref = solve_lm_6x6(A, b, damp=1e-3)
    monkeypatch.setenv("GRADSLAM_COMPILE", "1")                   # enable after wrap
    assert torch.allclose(x_ref, wrapped(A, b, damp=1e-3), atol=1e-4)


@skip_unless_compile
def test_compile_solver():
    """solve_lm_6x6 should compile and produce the same result as eager."""
    A = torch.randn(200, 6)
    b = torch.randn(200, 1)

    compiled = torch.compile(solve_lm_6x6, fullgraph=True)
    x_eager = solve_lm_6x6(A, b, damp=1e-3)
    x_compiled = compiled(A, b, damp=1e-3)

    assert torch.allclose(x_eager, x_compiled, atol=1e-4)


@skip_unless_compile
def test_compile_residuals():
    """point_to_plane_projective should compile and produce the same result."""
    N = 1000
    vl = torch.randn(N, 3)
    nm = torch.randn(N, 3)
    nm = nm / nm.norm(dim=-1, keepdim=True)
    vm = torch.randn(N, 3)

    compiled = torch.compile(point_to_plane_projective, fullgraph=True)
    A_e, b_e = point_to_plane_projective(vl, nm, vm)
    A_c, b_c = compiled(vl, nm, vm)

    assert torch.allclose(A_e, A_c, atol=1e-5)
    assert torch.allclose(b_e, b_c, atol=1e-5)
