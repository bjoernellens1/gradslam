"""Optional torch.compile tests for ICP hot paths (CPU, fixed shapes)."""

import os
import pytest
import torch
from gradslam.icp.solvers import solve_lm_6x6
from gradslam.icp.residuals import point_to_plane_projective

# Only run when GRADSLAM_COMPILE=1 is set
skip_unless_compile = pytest.mark.skipif(
    os.environ.get("GRADSLAM_COMPILE", "0") != "1",
    reason="Set GRADSLAM_COMPILE=1 to run compile tests",
)


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
