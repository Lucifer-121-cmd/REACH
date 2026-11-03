"""Guard: the hybrid geodesic-return arm (2026-09-16).

The threshold-baseline sweep (scripts/29, truth_paper.md §3.14)
measured that the pi_fb-on-geodesic return leg docks 84-90% while the
NLP return leg strands near-dock at 10% -- the return leg is execution,
not formulation. The hybrid arm keeps the constrained NLP OUTBOUND
(the contribution) and hands the return leg to the same tracker the
certificate model assumes.

These tests pin: the arm's registration, the tracker's pursuit math
(exact, no solver needed), and the supervisor's branch structure by
source inspection (the full solve path needs IPOPT; the branch order
is what must not silently regress -- the geodesic handover must bypass
NLP replanning, and the dock stop must stay FIRST).
"""
import inspect
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import reach_evaluator as ev


def test_hybrid_arm_registered():
    arm = ev.ARMS['reach_hybrid']
    assert arm.return_policy == 'geodesic'
    assert arm.turnback_rule == 'nlp'          # constrained outbound
    assert arm.use_constraint is True
    assert ev.ARMS['reach'].return_policy == 'nlp'   # default untouched


def test_tracker_pursuit_math():
    """_geodesic_track_step on a synthetic polyline: waypoint advance,
    pure-pursuit gains, w saturation, and the final straight-in rule."""
    cls = ev.ReachSupervisor
    fake = SimpleNamespace(
        **{k: getattr(cls, k) for k in
           ('_TRK_VSTAR', '_TRK_WMAX', '_TRK_KP', '_TRK_LA')},
        _geo_return={'path': np.array([[0.0, 0.0], [2.0, 0.0],
                                       [4.0, 0.0]]),
                     'idx': 0},
        dock=np.array([4.0, 0.0]))
    # robot at origin facing +x: must advance past waypoints within La
    # and command v* toward the pursued point
    v, w = ev.ReachSupervisor._geodesic_track_step(fake, np.array([0.0, 0.0, 0.0, 0, 0, 0]))
    assert v == pytest.approx(ev.ReachSupervisor._TRK_VSTAR)
    assert w == pytest.approx(0.0, abs=1e-12)
    assert fake._geo_return['idx'] == 1          # 0->1 (within La), not 2

    # misaligned heading: w = kp * wrapped(tangent - theta), capped at WMAX
    # tangent toward waypoint (2,0) is 0; heading is 2.0 rad; the wrapped
    # error is -2.0 -> kp * -2.0 = -4.0 saturates at -WMAX = -pi; the
    # cos of the error is negative, so v clamps to 0 (slow-while-realign)
    v, w = ev.ReachSupervisor._geodesic_track_step(fake, np.array([0.0, 0.0, 2.0, 0, 0, 0]))
    err = np.arctan2(np.sin(0.0 - 2.0), np.cos(0.0 - 2.0))   # = -2.0
    assert w == pytest.approx(max(-ev.ReachSupervisor._TRK_WMAX,
                                  min(ev.ReachSupervisor._TRK_WMAX,
                                      ev.ReachSupervisor._TRK_KP * err)))
    assert v == pytest.approx(ev.ReachSupervisor._TRK_VSTAR
                              * max(0.0, np.cos(err)))
    assert w < 0 and v == pytest.approx(0.0)

    # polyline exhausted within 1 m of the dock: head straight in.
    # bearing to dock = atan2(-0.3, 0.2) ~ -0.983; heading = pi/2;
    # wrapped error ~ -2.554 -> kp * error = -5.11 saturates at -WMAX,
    # and cos(error) < 0 clamps v to 0 (realign first, then approach)
    fake._geo_return['idx'] = len(fake._geo_return['path']) - 1
    v, w = ev.ReachSupervisor._geodesic_track_step(
        fake, np.array([3.8, 0.3, np.pi / 2, 0, 0, 0]))
    bearing = np.arctan2(0.0 - 0.3, 4.0 - 3.8)
    err = np.arctan2(np.sin(bearing - np.pi / 2), np.cos(bearing - np.pi / 2))
    want_w = max(-ev.ReachSupervisor._TRK_WMAX,
                 min(ev.ReachSupervisor._TRK_WMAX,
                     ev.ReachSupervisor._TRK_KP * err))
    assert w == pytest.approx(want_w)
    assert want_w == pytest.approx(-np.pi)          # saturated, as designed
    assert v == pytest.approx(ev.ReachSupervisor._TRK_VSTAR
                              * max(0.0, np.cos(err)))


def test_command_branch_order():
    """The geodesic branch must run BEFORE the replan trigger (no NLP
    solves during a geodesic return) and AFTER the dock stop."""
    src = inspect.getsource(ev.ReachSupervisor.command)
    i_dock = src.index('DOCK_TOL')
    i_geo = src.index('self._geo_return is not None')
    i_replan = src.index('self._replan(state6, mission_t)')
    assert i_dock < i_geo < i_replan


def test_turnback_geodesic_handover():
    """On turn-back with return_policy='geodesic': build the geodesic,
    clear the plan, and RETURN before the NLP return solve (fallback
    only when the geodesic is unavailable)."""
    src = inspect.getsource(ev.ReachSupervisor._replan)
    i_geo = src.index("_build_geodesic_return")
    i_fallback = src.index("path is not None")
    i_nlp = src.index("res2 = self._solve(state6, outbound=False")
    assert i_geo < i_fallback < i_nlp
    assert "self._plan_U = None" in src          # no plan left to execute


def test_tracker_constants_match_oracle():
    """The tracker must be the oracle's, exactly (La=0.3 with the 0.35
    routing buffer is a measured collision-margin choice, script 24)."""
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'scripts', '24_reach_oracle.py')
    spec = importlib.util.spec_from_file_location('o24', p)
    o24 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(o24)
    assert ev.ReachSupervisor._TRK_LA == o24.LA
    assert ev.ReachSupervisor._TRK_KP == o24.KP
    assert ev.ReachSupervisor._TRK_VSTAR == o24.VSTAR
    assert ev.ReachSupervisor._TRK_ROUTE_BUFFER == o24.ROUTE_BUFFER
