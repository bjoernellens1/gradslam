"""Tests for RGBDTSDFSLAM._quality_score motion disagreement penalty.

_quality_score returns a tuple used for lexicographic comparison:
    (motion_ok, photometric_ok, inlier_ratio_adjusted, num_valid, -rmse)
Higher tuples are better.
"""

import pytest
import torch
from gradslam.slam.pipeline import RGBDTSDFSLAM


def test_quality_score_penalizes_disagreement():
    """Candidate with smaller motion disagreement should score higher when inliers are similar."""
    # Two candidates with similar inlier ratios but different translation magnitudes
    # t_predicted = 0.05 m (5 cm predicted motion)
    t_predicted = 0.05
    lambda_disagree = 1.0

    # Candidate A: matches predicted motion well (t=0.05), moderate inlier ratio
    quality_a = {
        "inlier_ratio": 0.60,
        "num_valid": 5000,
        "rmse": 0.003,
        "frame_translation": 0.05,  # agrees with prediction
        "motion_gate": True,
        "photometric_gate": True,
    }

    # Candidate B: over-scaled translation (t=0.35 = 7x predicted), slightly higher inlier ratio
    quality_b = {
        "inlier_ratio": 0.65,  # slightly better inliers
        "num_valid": 5200,
        "rmse": 0.003,
        "frame_translation": 0.35,  # 7x over-scaled
        "motion_gate": True,
        "photometric_gate": True,
    }

    score_a = RGBDTSDFSLAM._quality_score(quality_a, t_predicted=t_predicted, lambda_disagree=lambda_disagree)
    score_b = RGBDTSDFSLAM._quality_score(quality_b, t_predicted=t_predicted, lambda_disagree=lambda_disagree)

    # A should win despite lower raw inlier ratio because B has large disagreement
    assert score_a > score_b, (
        f"Expected score_a {score_a} > score_b {score_b}: "
        "candidate with small motion disagreement should beat over-scaled candidate"
    )


def test_quality_score_no_penalty_when_lambda_zero():
    """With lambda_disagree=0, score should be based purely on inlier_ratio."""
    quality_a = {"inlier_ratio": 0.60, "num_valid": 5000, "rmse": 0.003, "frame_translation": 0.05}
    quality_b = {"inlier_ratio": 0.65, "num_valid": 5200, "rmse": 0.003, "frame_translation": 0.35}

    score_a = RGBDTSDFSLAM._quality_score(quality_a, t_predicted=0.05, lambda_disagree=0.0)
    score_b = RGBDTSDFSLAM._quality_score(quality_b, t_predicted=0.05, lambda_disagree=0.0)

    # Without penalty, B wins because it has higher inlier_ratio (tuples compared lexicographically,
    # motion_ok and photometric_ok are equal, so inlier_ratio is the tiebreaker)
    assert score_b > score_a


def test_quality_score_no_penalty_when_t_predicted_zero():
    """With t_predicted=0 (no velocity model), disagreement penalty must not apply."""
    quality_a = {"inlier_ratio": 0.60, "num_valid": 5000, "rmse": 0.003, "frame_translation": 0.05}
    quality_b = {"inlier_ratio": 0.65, "num_valid": 5200, "rmse": 0.003, "frame_translation": 0.35}

    score_a = RGBDTSDFSLAM._quality_score(quality_a, t_predicted=0.0, lambda_disagree=1.0)
    score_b = RGBDTSDFSLAM._quality_score(quality_b, t_predicted=0.0, lambda_disagree=1.0)

    # With no velocity prediction, the penalty does not activate (guarded by t_predicted > 0),
    # so B still wins on raw inlier_ratio
    assert score_b > score_a


def test_quality_score_motion_gate_dominates():
    """A candidate with motion_gate=False should lose regardless of inlier_ratio."""
    quality_good = {"inlier_ratio": 0.90, "num_valid": 9000, "rmse": 0.001, "motion_gate": False}
    quality_poor = {"inlier_ratio": 0.10, "num_valid": 1000, "rmse": 0.01, "motion_gate": True}

    score_good = RGBDTSDFSLAM._quality_score(quality_good, t_predicted=0.0, lambda_disagree=0.0)
    score_poor = RGBDTSDFSLAM._quality_score(quality_poor, t_predicted=0.0, lambda_disagree=0.0)

    assert score_poor > score_good, (
        "motion_gate=False candidate should lose to any motion_gate=True candidate"
    )


def test_quality_score_returns_tuple():
    """_quality_score should return a tuple (for lexicographic comparison)."""
    quality = {"inlier_ratio": 0.5, "num_valid": 1000, "rmse": 0.005}
    result = RGBDTSDFSLAM._quality_score(quality, t_predicted=0.0, lambda_disagree=0.0)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 5, f"Expected 5-tuple, got {len(result)}-tuple"
