"""Tests for gradslam.icp.solvers."""

import torch
import pytest
from gradslam.icp.solvers import solve_lm_6x6


def test_solve_lm_6x6_shape():
    A = torch.randn(100, 6)
    b = torch.randn(100, 1)
    x = solve_lm_6x6(A, b, damp=1e-3)
    assert x.shape == (6, 1)


def test_solve_lm_6x6_known_solution():
    """For a diagonal A^T A system, solution should be known."""
    # Construct A s.t. A^T A = I (orthonormal rows)
    A = torch.eye(6)                  # [6, 6] — columns are ortho-normal
    b = torch.arange(1, 7, dtype=torch.float32).unsqueeze(1)  # [6, 1]
    # With zero damping: A^T A x = A^T b → x = b (since A = I)
    x = solve_lm_6x6(A, b, damp=0.0)
    assert torch.allclose(x, b, atol=1e-4)


def test_solve_lm_6x6_damping_reduces_magnitude():
    """Larger damping should yield smaller-magnitude solution."""
    A = torch.randn(200, 6)
    b = torch.randn(200, 1)
    x_small = solve_lm_6x6(A, b, damp=1e-6)
    x_large = solve_lm_6x6(A, b, damp=1e2)
    assert torch.norm(x_large) < torch.norm(x_small)


def test_solve_lm_6x6_residual_decreases():
    """The solved x should reduce ||Ax - b||^2 vs x=0."""
    A = torch.randn(100, 6)
    b = torch.randn(100, 1)
    x = solve_lm_6x6(A, b, damp=1e-4)
    res_0 = torch.norm(b).item()
    res_x = torch.norm(A @ x - b).item()
    assert res_x < res_0
