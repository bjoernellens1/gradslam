"""Tests for RGBDTSDFSLAM._scale_veto.

_scale_veto vetoes the top candidate when it is over-scaled relative to
the predicted translation and a more consistent alternative exists.
"""

import pytest
import torch
from gradslam.slam.pipeline import RGBDTSDFSLAM


def _make_candidate(t_translation: float, inlier_ratio: float, motion_gate: bool = True):
    pose = torch.eye(4)
    pose[0, 3] = t_translation  # simple translation along x
    quality = {
        "frame_translation": t_translation,
        "inlier_ratio": inlier_ratio,
        "motion_gate": motion_gate,
        "num_valid": int(inlier_ratio * 10000),
        "rmse": 0.003,
    }
    return pose, quality


def test_veto_returns_none_when_winner_not_overscaled():
    """No veto when winner's translation is within scale_veto_ratio of predicted."""
    candidates = [
        _make_candidate(0.05, 0.70),  # winner: t=0.05, predicted=0.05, ratio=1.0 < 3.0
        _make_candidate(0.04, 0.60),  # alt
    ]
    result = RGBDTSDFSLAM._scale_veto(candidates, t_predicted=0.05, scale_veto_ratio=3.0)
    assert result is None


def test_veto_returns_none_when_no_consistent_alternative():
    """No veto when winner is over-scaled but no consistent alternative exists."""
    candidates = [
        _make_candidate(0.40, 0.80),  # winner: t=0.40, predicted=0.05, 8x over-scaled
        _make_candidate(0.38, 0.50),  # alt: also over-scaled, t=0.38 > 1.5*0.05=0.075
    ]
    result = RGBDTSDFSLAM._scale_veto(candidates, t_predicted=0.05, scale_veto_ratio=3.0)
    assert result is None


def test_veto_switches_to_consistent_candidate():
    """Veto switches when winner is over-scaled and a consistent alternative exists."""
    t_pred = 0.05
    pose_winner, quality_winner = _make_candidate(0.40, 0.80)  # 8x over-scaled
    pose_alt, quality_alt = _make_candidate(0.06, 0.65)        # within 1.5x of predicted

    candidates = [(pose_winner, quality_winner), (pose_alt, quality_alt)]
    result = RGBDTSDFSLAM._scale_veto(candidates, t_predicted=t_pred, scale_veto_ratio=3.0)

    assert result is not None, "Expected veto to activate"
    _, result_quality = result
    assert result_quality["frame_translation"] == pytest.approx(0.06), (
        "Expected alternative candidate to be selected"
    )


def test_veto_requires_sufficient_inlier_ratio_in_alternative():
    """Alternative must have >= 70% of winner's inlier_ratio to be selected."""
    # winner inlier_ratio = 0.80; 70% threshold = 0.56
    candidates = [
        _make_candidate(0.40, 0.80),   # winner: over-scaled
        _make_candidate(0.06, 0.55),   # consistent scale, but inlier_ratio < 0.56 threshold
    ]
    result = RGBDTSDFSLAM._scale_veto(candidates, t_predicted=0.05, scale_veto_ratio=3.0)
    assert result is None, "Alternative below 70% inlier threshold should not trigger veto"


def test_veto_skips_motion_gated_alternatives():
    """Veto must not return a candidate with motion_gate=False."""
    candidates = [
        _make_candidate(0.40, 0.80),                           # winner: over-scaled
        _make_candidate(0.04, 0.65, motion_gate=False),        # consistent but gated-out
    ]
    result = RGBDTSDFSLAM._scale_veto(candidates, t_predicted=0.05, scale_veto_ratio=3.0)
    assert result is None, "Veto must not select a motion-gated candidate"


def test_veto_returns_none_empty_candidates():
    assert RGBDTSDFSLAM._scale_veto([], t_predicted=0.05) is None


def test_veto_returns_none_t_predicted_zero():
    candidates = [_make_candidate(0.40, 0.80), _make_candidate(0.04, 0.65)]
    assert RGBDTSDFSLAM._scale_veto(candidates, t_predicted=0.0) is None


def test_veto_returns_none_scale_veto_ratio_zero():
    """scale_veto_ratio <= 0 disables the veto entirely."""
    candidates = [
        _make_candidate(0.40, 0.80),  # would be over-scaled at ratio=3.0
        _make_candidate(0.04, 0.65),  # consistent alternative
    ]
    result = RGBDTSDFSLAM._scale_veto(candidates, t_predicted=0.05, scale_veto_ratio=0.0)
    assert result is None, "scale_veto_ratio <= 0 should disable veto"
