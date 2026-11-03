"""Tests for Issue 1 solver status taxonomy + evaluator aggregation."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.ep2_mpc import EP2MPC
from mpc.mpc_params import MPCParams, DEFAULT_PARAMS
from evaluation.ep2_evaluator import EP2Evaluator, TrialResult


def test_normalize_solver_status_mappings():
    assert EP2MPC._normalize_solver_status({'success': True}) == 'success'
    assert EP2MPC._normalize_solver_status(
        {'success': False, 'return_status': 'Infeasible_Problem_Detected'}
    ) == 'infeasible'
    assert EP2MPC._normalize_solver_status(
        {'success': False, 'return_status': 'Maximum_Iterations_Exceeded'}
    ) == 'max_iter'
    assert EP2MPC._normalize_solver_status(
        {'success': False, 'return_status': 'Numeric_Divergence'}
    ) == 'numerical_error'
    assert EP2MPC._normalize_solver_status(None) == 'unknown'


def test_evaluator_aggregates_solver_status_counts():
    cfg = {
        'mpc': {'dt': 0.1},
        'evaluation': {'timeout_per_trial': 0.2},
    }
    evaluator = EP2Evaluator(cfg)
    params = MPCParams(**DEFAULT_PARAMS)

    statuses = iter(['success', 'infeasible', 'max_iter'])

    def fake_run_trial(_params, _scenario, seed):
        status = next(statuses)
        return TrialResult(
            energy=1.0,
            time=0.1,
            solver_ms=2.0,
            success=(status == 'success'),
            collision=False,
            path_distance=0.0,
            n_steps=1,
            solver_status=status,
        )

    evaluator._run_trial = fake_run_trial
    result = evaluator.evaluate(params, 'corridor', n_trials=3)

    assert result.n_trials == 3
    assert result.solver_status_counts == {
        'success': 1,
        'infeasible': 1,
        'max_iter': 1,
    }
    assert np.isclose(result.success_rate, 1.0 / 3.0)
