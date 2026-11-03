"""Tests for planning/energy_dijkstra.py (C3 / F9).

Three things worth pinning:

1. **The field reproduces the audited values.** D*(start) = 19.000 / 20.698 /
   20.325 m and E_floor = 126.87 / 138.21 / 135.72 J. These are quoted in PLAN.md
   and used as the denominator for every ENVELOPE-normalised effect size, so a
   silent change to the grid or the metric must fail a test.
2. **Inflation is r_j + r_body and the failure mode is loud.** Adding d_safe to
   the WALL inset makes both endpoints unreachable; the builder raises rather
   than returning an all-INF field.
3. **The interpolant has bounded gradients.** A constant fill for unreachable
   cells produced |dD*/dp| ~ 2e7 one cell from an obstacle, which would destroy
   IPOPT's step computation. The extension keeps it near the penalty bound.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.scenarios import get_scenario
from mpc.energy_model import load_default_coefficients
from planning.energy_dijkstra import (
    build_field, build_interpolant, extend_into_blocked, geodesic_path,
    lambda_min, v_star, save_npz, load_npz, R_BODY, DEFAULT_H,
)

#: From PLAN.md §1 / the audited probe, to 3 dp.
EXPECTED_DSTAR = {'corridor': 19.000, 's_curve': 20.698, 'chicane': 20.325}
EXPECTED_FLOOR = {'corridor': 126.87, 's_curve': 138.21, 'chicane': 135.72}


def test_lambda_min_and_vstar_match_the_plan():
    C = load_default_coefficients()
    assert lambda_min(C) == pytest.approx(6.677226, abs=1e-6)
    assert v_star(C) == pytest.approx(1.095445, abs=1e-6)
    # the whole point: lambda is nearly flat, so there is no speed lever
    c1, c4, c5 = C['c1'], C['c4'], C['c5']
    lam_at_vmax = c1 * 1.2 + c4 + c5 / 1.2
    assert (lam_at_vmax / lambda_min(C) - 1.0) < 0.005      # < 0.5% spread


@pytest.mark.parametrize('name', sorted(EXPECTED_DSTAR))
def test_dstar_reproduces_the_audited_value(name):
    sc = get_scenario(name)
    f = build_field(sc)
    d = f.dstar(float(sc.start[0]), float(sc.start[1]))
    assert d == pytest.approx(EXPECTED_DSTAR[name], abs=5e-3)
    assert f.lambda_min * d == pytest.approx(EXPECTED_FLOOR[name], abs=0.02)


@pytest.mark.parametrize('name', sorted(EXPECTED_DSTAR))
def test_default_inflation_is_r_body_only(name):
    f = build_field(get_scenario(name))
    assert f.obs_inflation == pytest.approx(R_BODY)
    assert f.wall_inflation == pytest.approx(R_BODY)
    assert f.h == DEFAULT_H


@pytest.mark.parametrize('name', sorted(EXPECTED_DSTAR))
def test_wall_inset_by_d_safe_blocks_the_goal_loudly(name):
    """The endpoints sit 0.5 m from a wall; r_body + d_safe closes them off."""
    with pytest.raises(ValueError, match='not free'):
        build_field(get_scenario(name), wall_inflation=R_BODY + 0.2922)


@pytest.mark.parametrize('name', sorted(EXPECTED_DSTAR))
def test_obstacle_inflation_by_d_safe_only_lengthens_the_route(name):
    """Distinguishes the two knobs: obstacles are survivable, walls are not."""
    sc = get_scenario(name)
    base = build_field(sc).dstar(*sc.start[:2])
    infl = build_field(sc, extra_inflation=0.2922).dstar(*sc.start[:2])
    assert np.isfinite(infl), "obstacle inflation alone must stay solvable"
    assert infl >= base - 1e-9


def test_goal_cell_has_zero_distance():
    sc = get_scenario('corridor')
    f = build_field(sc)
    assert f.dstar(float(sc.goal[0]), float(sc.goal[1])) == pytest.approx(0.0)


def test_geodesic_path_arclength_matches_dstar():
    sc = get_scenario('s_curve')
    f = build_field(sc)
    path = geodesic_path(f, sc.start[:2])
    assert path is not None and len(path) > 10
    arclen = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    assert arclen == pytest.approx(f.dstar(*sc.start[:2]), rel=1e-3)


def test_geodesic_path_from_a_blocked_cell_returns_none():
    sc = get_scenario('s_curve')
    f = build_field(sc)
    cx, cy, _ = sc.obstacles[0]
    assert geodesic_path(f, (cx, cy)) is None


def test_npz_round_trip_is_lossless(tmp_path):
    f = build_field(get_scenario('chicane'))
    p = save_npz(f, str(tmp_path / 'dstar_chicane.npz'))
    g = load_npz(p)
    assert np.array_equal(f.D, g.D, equal_nan=True)
    assert np.array_equal(f.free, g.free)
    assert (g.lambda_min, g.h, g.name) == (f.lambda_min, f.h, f.name)
    assert np.array_equal(f.goal, g.goal)


def test_extension_preserves_reachable_cells_and_fills_everything():
    f = build_field(get_scenario('s_curve'))
    De = extend_into_blocked(f)
    reach = np.isfinite(f.D)
    assert np.isfinite(De).all(), "extension must leave no infinities"
    assert np.array_equal(De[reach], f.D[reach]), "reachable cells must not move"
    assert De[~reach].min() > 0.0


def test_interpolant_gradients_are_bounded_near_obstacles():
    """A constant fill gave |grad| ~ 2e7 here; the extension must not."""
    import casadi as ca
    f = build_field(get_scenario('s_curve'))
    interp = build_interpolant(f, penalty=5.0)
    x = ca.MX.sym('x', 2)
    J = ca.Function('J', [x], [ca.jacobian(interp(x), x)])
    # points chosen to straddle the gate obstacles at x=6 and x=14
    for p in [(5.0, 2.0), (6.0, 2.0), (6.0, 1.9), (14.0, 2.1), (10.0, 2.0)]:
        g = np.array(J(list(p))).ravel()
        assert np.all(np.isfinite(g))
        # bound is penalty * sqrt(2) for a diagonal step, plus grid slack
        assert np.hypot(*g) < 12.0, f"gradient blew up at {p}: {np.hypot(*g)}"


def test_interpolant_agrees_with_the_grid_at_cell_centres():
    import casadi as ca
    sc = get_scenario('corridor')
    f = build_field(sc)
    interp = build_interpolant(f)
    for p in [(sc.start[0], sc.start[1]), (10.0, 2.0), (15.0, 3.0)]:
        assert float(interp(list(p))) == pytest.approx(f.dstar(*p), abs=1e-9)


def test_E_floor_is_lambda_min_times_D():
    f = build_field(get_scenario('corridor'))
    reach = np.isfinite(f.D)
    assert np.allclose(f.E_floor[reach], f.lambda_min * f.D[reach])
