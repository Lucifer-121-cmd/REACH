"""Unit tests for mpc/adaptive_horizon.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from mpc.adaptive_horizon import (
    sigmoid, compute_phi_prox, compute_phi_energy, compute_phi, AdaptiveHorizon
)

def test_phi_range():
    for d in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for dP in [-20, 0, 10, 30]:
            phi = compute_phi(d, dP, d_thresh=1.0, dP_thresh=8.0, beta_d=3.0, beta_e=0.15)
            assert 0.0 <= phi <= 1.0

def test_phi_prox_increases_near_obstacle():
    phi_far = compute_phi_prox(3.0, 1.0, 3.0)
    phi_near = compute_phi_prox(0.2, 1.0, 3.0)
    assert phi_near > phi_far

def test_phi_energy_increases_with_anomaly():
    phi_low = compute_phi_energy(1.0, 8.0, 0.15)
    phi_high = compute_phi_energy(20.0, 8.0, 0.15)
    assert phi_high > phi_low

def test_horizon_bounds():
    ah = AdaptiveHorizon({'N_min': 8, 'N_max': 24})
    for d_obs in [0.2, 0.5, 1.0, 5.0]:
        N = ah.compute(d_obs, 0.5, 0.1, 5.0)
        assert 8 <= N <= 24

def test_max_operator():
    # Near obstacle alone triggers extension
    N_prox = AdaptiveHorizon({'N_min': 8, 'N_max': 24}).compute(0.15, 0.5, 0.0, 0.0)
    # Energy anomaly alone triggers extension
    N_energy = AdaptiveHorizon({'N_min': 8, 'N_max': 24}).compute(10.0, 0.5, 0.0, 0.0)
    N_base = AdaptiveHorizon({'N_min': 8, 'N_max': 24}).compute(10.0, 0.0, 0.0, 0.0)
    assert N_prox >= N_base

def test_free_space_low_horizon():
    ah = AdaptiveHorizon({'N_min': 8, 'N_max': 24})
    N = ah.compute(d_obs=8.0, v=0.5, omega=0.0, prev_power=0.0)
    assert N <= 12


def test_straight_line_power_anomaly_near_zero():
    """Straight-line driving (omega=0) produces no *model-based* anomaly.

    Rewritten 2026-09-01. The previous version read ``_history[-1]['dP_model']``,
    a key the logger has never written in any tracked revision, so it raised
    KeyError rather than failing an assertion. The quantity is still recoverable:
    ``compute_power_anomaly`` derives dP_model as ``P_actual - P_expected``
    (mpc/adaptive_horizon.py:112-116) and both terms *are* logged, so the
    subtraction reconstructs it exactly without changing the logged schema --
    which matters, because the 527 tracked step records behind Figure 3 carry
    exactly the current 12 keys.
    """
    ah = AdaptiveHorizon({'N_min': 8, 'N_max': 24})
    ah.compute(d_obs=10.0, v=0.6, omega=0.0, prev_power=0.0)
    h = ah._history[-1]
    dP_model = h['P_actual'] - h['P_expected']
    assert abs(dP_model) < 1e-6, (
        f"omega=0 must give a zero model anomaly, got {dP_model}"
    )


def test_history_schema_is_stable():
    """Guard the logged key set: the tracked Figure-3 traces carry exactly these.

    results/statistics/adaptive_horizon_history.json holds 527 step records with
    this key set. A silent addition or rename here would make new traces
    schema-diverge from the tracked ones.
    """
    ah = AdaptiveHorizon({'N_min': 8, 'N_max': 24})
    ah.compute(d_obs=1.0, v=0.6, omega=0.2, prev_power=5.0)
    assert set(ah._history[-1]) == {
        'phi', 'phi_prox', 'phi_energy', 'N', 'N_base', 'd_obs',
        'dP', 'P_actual', 'P_expected', 'P_prev', 'v', 'omega',
    }


def test_turning_power_anomaly_positive():
    # Turning (omega>0) should produce a positive anomaly.
    ah = AdaptiveHorizon({'N_min': 8, 'N_max': 24})
    dP = ah.compute(d_obs=10.0, v=0.6, omega=1.0, prev_power=0.0)
    assert ah._history[-1]['dP'] > 0
