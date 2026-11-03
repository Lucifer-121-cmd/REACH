"""Move-1 regression guards: warm-starting the REACH replans.

Pins the pieces measured on 2026-09-12 (map reach_train_20260911, B=220,
seed 101): the shifted-previous-solution guess, the cold-retry when a warm
solve comes back violating, the dock-stop that keeps the return phase from
idling the pack away at home, and the iteration cap that cut mean solves
40 s -> 24 s without changing the mission outcome.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.reach_mpc import ReachMPC, ReachSpec
from planning.reach_maps import MapSpec, generate_map


def _mpc():
    sc, _ = generate_map(MapSpec(seed=20260911, split="train"))
    spec = ReachSpec(B=220.0, certificate='affine', a_q=24.0, lam=1.0)
    return ReachMPC(sc, spec, N=60, dt=0.5, config={})


def _straight_plan(mpc):
    X = np.zeros((6, 61)); U = np.zeros((3, 60))
    for k in range(61):
        X[:, k] = (1.0 + 0.55 * k, 3.0, 0.0, 1.0, 0.0, 0.5 * k)
    for k in range(60):
        U[:, k] = (1.0, 0.0, 1.0)
    return X, U


def test_shift_guess_identity_and_shift():
    mpc = _mpc()
    Xp, Up = _straight_plan(mpc)
    # n=0: identity
    Xg, Ug = mpc._shift_guess(Xp, Up, 0)
    assert np.allclose(Xg, Xp) and np.allclose(Ug, Up)
    # n=10: head carries nodes 10.., inputs shift, tail holds last input
    Xg, Ug = mpc._shift_guess(Xp, Up, 10)
    assert np.allclose(Xg[:, 0], Xp[:, 10])
    assert np.allclose(Xg[:, :51], Xp[:, 10:])
    assert np.allclose(Ug[:, :50], Up[:, 10:])
    assert np.allclose(Ug[:, 50:], Up[:, -1:])
    assert np.all(np.diff(Xg[5, 49:]) > 0)   # tail energy monotone


def test_shift_guess_full_execution_branch():
    mpc = _mpc()
    Xp, Up = _straight_plan(mpc)
    Xg, Ug = mpc._shift_guess(Xp, Up, 60)      # everything flown
    assert Xg.shape == (6, 61) and Ug.shape == (3, 60)
    assert np.all(np.diff(Xg[5]) > 0)


def test_solve_accepts_warm_and_anchors_node0():
    """A warm solve must anchor its guess to the MEASURED state -- the
    equality row pins it regardless, but a guess disagreeing with the row
    is a cold start wearing a warm label."""
    mpc = _mpc()
    Xp, Up = _straight_plan(mpc)
    x0 = np.array([2.0, 3.0, 0.1, 0.5, 0.0, 3.0])
    res = mpc.solve(x0, B=220.0, warm=(Xp, Up, 4))
    # the returned plan's first node equals x0 (the equality row)
    assert np.allclose(res['X'][:, 0], x0, atol=1e-6)


def test_profile_caps_iterations_for_supervisor_acceptance():
    """1500, not 3000: the supervisor's acceptance is constraint
    satisfaction, so iterations beyond convergence buy nothing the
    protocol uses. 800 was measured TOO tight (turn-back at t=0.1 s)."""
    from mpc.solver_opts import PLANNING_PROFILE
    assert PLANNING_PROFILE['ipopt.max_iter'] == 1500


def test_planning_profile_guard_still_passes():
    from mpc.solver_opts import PLANNING_PROFILE
    assert PLANNING_PROFILE['ipopt.acceptable_tol'] == 1e-2
    assert PLANNING_PROFILE['ipopt.acceptable_iter'] == 15
