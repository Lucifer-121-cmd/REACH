"""Unit tests for mpc/mpc_params.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from mpc.mpc_params import (
    MPCParams, DEFAULT_PARAMS, validate_params, params_to_array,
    array_to_params, get_bounds_array, PARAM_NAMES, PARAM_BOUNDS, D_SAFE_MAX
)

#: config['robot']['collision_radius'] in both config files. Hard-coded here on
#: purpose: this test asserts the *contract* between the two, so reading the
#: value from the same config the source reads would make it vacuous.
COLLISION_RADIUS = 0.25

def test_default_params_valid():
    params = MPCParams(**DEFAULT_PARAMS)
    valid, violations = validate_params(params)
    assert valid, f"Default params invalid: {violations}"

def test_d_safe_must_exceed_collision_radius():
    """d_safe is a *cost activation* margin and must fire before contact.

    Rewritten 2026-09-01. The previous version asserted that d_safe = 0.30 is
    INVALID, which encoded a superseded box of (0.05, 0.26) -- i.e. it demanded
    that values below the 0.25 m collision radius be permitted. That is exactly
    the defect PLAN.md §1 identifies as fatal in the Table I baseline: with
    d_safe < r_body the obstacle penalty is identically zero at every
    collision-free state, so the controller cannot see an obstacle until it is
    already inside it. The source is correct; the old test was not.
    """
    lo, hi = PARAM_BOUNDS['d_safe']
    assert lo > COLLISION_RADIUS, (
        f"d_safe lower bound {lo} must exceed the collision radius "
        f"{COLLISION_RADIUS}, or the obstacle penalty is dead at every "
        f"collision-free state"
    )
    # A value inside the box is valid...
    valid, violations = validate_params(MPCParams(**{**DEFAULT_PARAMS, 'd_safe': 0.30}))
    assert valid, f"d_safe=0.30 is inside [{lo}, {hi}] and must validate: {violations}"
    # ...and one at or below the collision radius is not.
    for bad in (COLLISION_RADIUS, 0.245, 0.23, 0.05):
        valid, violations = validate_params(
            MPCParams(**{**DEFAULT_PARAMS, 'd_safe': bad}))
        assert not valid, f"d_safe={bad} <= r_body must be rejected"
        assert any('d_safe' in v for v in violations)


def test_d_safe_box_matches_deployed_contract():
    """The box is [0.26, 0.38] and every other declaration must agree."""
    assert PARAM_BOUNDS['d_safe'] == (0.26, 0.38)
    assert PARAM_BOUNDS['d_safe'][1] <= D_SAFE_MAX


def test_d_safe_above_box_is_rejected():
    valid, violations = validate_params(
        MPCParams(**{**DEFAULT_PARAMS, 'd_safe': 0.39}))
    assert not valid
    assert any('d_safe' in v for v in violations)

def test_N_base_integer():
    params = MPCParams(**DEFAULT_PARAMS)
    assert params.N_base_int == int(round(params.N_base))

def test_bounds_array_shape():
    lbs, ubs = get_bounds_array()
    assert lbs.shape == (10,)
    assert ubs.shape == (10,)
    assert np.all(lbs < ubs)

def test_params_roundtrip():
    params = MPCParams(**DEFAULT_PARAMS)
    arr = params_to_array(params)
    params2 = array_to_params(arr)
    for name in PARAM_NAMES:
        assert abs(getattr(params, name) - getattr(params2, name)) < 1e-9
