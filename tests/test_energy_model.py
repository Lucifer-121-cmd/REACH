"""Unit tests for mpc/energy_model.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from mpc.energy_model import (
    compute_power, load_default_coefficients, optimal_velocity,
    compute_energy, generate_training_data, identify_coefficients
)

def test_power_function_positive():
    coeffs = load_default_coefficients()
    for v in np.linspace(0, 1.2, 10):
        for w in np.linspace(-3.14, 3.14, 10):
            assert compute_power(v, w, coeffs) >= 0

def test_optimal_velocity_exists():
    coeffs = load_default_coefficients()
    v_star = optimal_velocity(coeffs)
    assert 0.3 <= v_star <= 1.5  # physically reasonable range

def test_energy_calculation():
    coeffs = load_default_coefficients()
    v_seq = np.full(20, 0.5)
    w_seq = np.zeros(20)
    E = compute_energy(v_seq, w_seq, dt=0.1, coeffs=coeffs)
    assert len(E) == 21
    assert np.all(np.diff(E) >= 0), "Energy must be monotonically non-decreasing"

def test_coefficient_identification():
    v, w, p = generate_training_data(n_v=10, n_omega=10, noise_std=0.01, seed=42)
    true_coeffs = load_default_coefficients()
    fitted = identify_coefficients(v, w, p)
    for k in ['c1', 'c2', 'c4', 'c5']:
        assert abs(fitted[k] - true_coeffs[k]) / (abs(true_coeffs[k]) + 1e-6) < 0.2

def test_gate0_conditions():
    coeffs = load_default_coefficients()
    P_at_05 = compute_power(0.5, 0.0, coeffs)
    t_10m = 10.0 / 0.5
    E_10m = P_at_05 * t_10m
    assert 50 <= E_10m <= 200, f"Gate0: E={E_10m:.1f}J not in [50,200]J"
