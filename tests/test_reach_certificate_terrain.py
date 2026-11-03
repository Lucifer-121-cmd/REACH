"""Move-2 guards: the per-map certificate refit on terrain J-fields.

The certificate is paper contribution 2, so the fit itself is pinned:
coverage construction (not least squares), terrain-aware rollout (the
plant's multipliers, not the nominal coefficients), slope 1.0 on the
J-field, and the trial record naming which intercept it used.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.terrain import TerrainField, TerrainRegion
from env.scenarios import ScenarioConfig
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "m20", os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts", "20_reach_certificate_terrain.py"))
m20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m20)


def _map_with_terrain(m1=2.0, m4=2.0):
    """A hand-authored terrain map: simpler than the generated ones so the
    expected values are derivable by hand."""
    terrain = TerrainField([TerrainRegion(6.0, 0.0, 14.0, 5.0, m1=m1, m4=m4)],
                           bounds=(0.0, 0.0, 20.0, 5.0))
    return ScenarioConfig(
        name="cert_probe",
        start=np.array([1.0, 2.5, 0.0]), goal=np.array([1.0, 2.5]),
        obstacles=[(10.0, 4.0, 0.8)], bounds=(0.0, 0.0, 20.0, 5.0),
        description="certificate fit probe",
        terrain=terrain, dock=np.array([1.0, 2.5]))


def test_terrain_rollout_pays_terrain_multipliers():
    """A rollout crossing the m-region must cost more than the nominal one,
    and by the right order: the c1/c4 terms carry m, c2/c3/c5 do not."""
    sc = _map_with_terrain(m1=3.0, m4=3.0)
    path = np.array([[1.0, 2.5], [6.0, 2.5], [12.0, 2.5]])
    hi = m20._rollout_terrain(path, (1.0, 2.5, 0.0), (12.0, 2.5),
                              sc.terrain)
    lo = m20._rollout_terrain(path, (1.0, 2.5, 0.0), (12.0, 2.5), None)
    assert hi['E'] > lo['E']
    # the c1/c4 terms scale; c5 does not
    assert hi['terms']['c1'] > lo['terms']['c1'] * 1.5
    assert hi['terms']['c4'] > lo['terms']['c4'] * 1.5
    assert hi['terms']['c5'] == pytest.approx(lo['terms']['c5'])


def test_fit_is_a_coverage_construction():
    """a_q at 0.95 must give 95% coverage BY CONSTRUCTION -- and at a
    stricter coverage, a larger intercept."""
    sc = _map_with_terrain()
    c95 = m20.sample_and_fit(sc, stride=10, coverage=0.95)
    c99 = m20.sample_and_fit(sc, stride=10, coverage=0.99)
    assert c95['coverage_achieved'] == pytest.approx(0.95, abs=0.01)
    assert c99['a_q'] >= c95['a_q']
    assert c95['denomination'] == 'J' and c95['slope'] == 1.0


def test_metric_map_is_rejected_loudly():
    """A terrain-blind map has a metric field; fitting the J-form on it
    would silently mix denominations."""
    sc = _map_with_terrain()
    sc.terrain = None
    with pytest.raises(ValueError, match="not 'J'"):
        m20.sample_and_fit(sc)


def test_trial_record_names_the_certificate_source():
    """A placeholder trial must say LEGACY_PLACEHOLDER in its record."""
    from evaluation.reach_evaluator import ReachTrialResult
    r = ReachTrialResult(map_name="m", arm="reach", B=220.0, seed=1,
                         outcome="docked", returned=True, reach_m=4.0,
                         reach_J=90.0, energy_spent=100.0, mission_time=20.0,
                         min_margin_J=10.0, margin_at_turnback_J=12.0,
                         turnback_time=5.0, turnback_reach_m=4.0,
                         collision_cause=None, replans=8,
                         solve_ms_mean=20.0, solve_ms_max=40.0,
                         final_price=None)
    d = r.to_record()
    assert d['a_q'] == 24.0
    assert d['a_q_source'] == 'LEGACY_PLACEHOLDER_24J'


def test_fitted_values_bracket_the_placeholder():
    """The real fits (REPRODUCE.md §11.9) run 28.6-54.0 J; the 24 J
    placeholder under-reserves on every generated map. Pin the range so a
    refit that collapses toward the placeholder is visible."""
    # one generated map, coarse stride: the value must be well above 24 J
    from planning.reach_maps import MapSpec, generate_map
    sc, _ = generate_map(MapSpec(seed=20260911, split="train"))
    cert = m20.sample_and_fit(sc, stride=12)
    assert cert['a_q'] > 30.0, cert['a_q']
