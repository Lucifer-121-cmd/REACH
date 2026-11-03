"""Terrain multiplier fields (PLAN.md R2) — guards for the REACH foundation.

The physics motivation is pinned by the rejection audit: λ(v) is flat to
0.35% across the operating band, so homogeneous terrain gives route choice
no authority. Spatial multipliers on c1/c4 are what make the min-energy
route differ from the shortest one; these tests pin the pieces every REACH
experiment will stand on:

* the field validates loudly (overlap, out-of-bounds, degenerate, bounds)
* all-ones terrain is EXACTLY the nominal model (bit-identical energy) --
  the property that keeps every legacy number valid
* the terrain-weighted geodesic field with all-ones multipliers is exactly
  λ_min × the metric field (units consistency of the 'J' denomination)
* the plant pays terrain-scaled power at its position; mismatch and
  actuation noise do what R5 says they do, reproducibly
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.terrain import (M_MAX, M_MIN, TerrainError, TerrainField,
                         TerrainRegion)
from env.scenarios import ScenarioConfig, load_scenario_from_yaml
from env.pioneer_env import PioneerEnv
from mpc.energy_model import load_default_coefficients
from planning.energy_dijkstra import build_field


# ---------------------------------------------------------------------------
# TerrainField itself
# ---------------------------------------------------------------------------

def test_region_degenerate_raises():
    with pytest.raises(TerrainError):
        TerrainRegion(x0=2.0, y0=0.0, x1=2.0, y1=1.0)


def test_region_multiplier_bounds_enforced():
    with pytest.raises(TerrainError):
        TerrainRegion(0, 0, 1, 1, m1=M_MIN - 0.01)
    with pytest.raises(TerrainError):
        TerrainRegion(0, 0, 1, 1, m4=M_MAX + 0.01)


def test_overlap_rejected_loudly():
    """Two overlapping regions make the multiplier at a point ambiguous --
    the plant and the planner would disagree about the same floor."""
    with pytest.raises(TerrainError, match="overlap"):
        TerrainField([TerrainRegion(0, 0, 2, 2, m1=2.0),
                      TerrainRegion(1, 1, 3, 3, m1=2.0)])


def test_out_of_bounds_region_rejected():
    with pytest.raises(TerrainError, match="outside the arena"):
        TerrainField([TerrainRegion(-1, 0, 1, 2)], bounds=(0, 0, 20, 4))


def test_empty_field_rejected():
    with pytest.raises(TerrainError):
        TerrainField([])


def test_multipliers_inside_outside_boundary():
    f = TerrainField([TerrainRegion(5, 0, 10, 2, m1=2.0, m4=3.0)],
                     bounds=(0, 0, 20, 4))
    assert f.multipliers(7.0, 1.0) == (2.0, 3.0)
    assert f.multipliers(12.0, 1.0) == (1.0, 1.0)      # beyond region
    assert f.multipliers(5.0, 1.0) == (2.0, 3.0)       # x0 inclusive
    assert f.multipliers(10.0, 1.0) == (1.0, 1.0)      # x1 exclusive


def test_lambda_grid_matches_formula():
    f = TerrainField([TerrainRegion(0, 0, 10, 4, m1=2.0, m4=3.0)],
                     bounds=(0, 0, 20, 4))
    X, Y = np.meshgrid(np.linspace(0, 20, 9), np.linspace(0, 4, 5),
                       indexing="ij")
    LAM = f.lambda_grid(X, Y, load_default_coefficients())
    c = load_default_coefficients()
    expect_left = 2.0 * np.sqrt(2.0 * c["c1"] * c["c5"]) + 3.0 * c["c4"]
    expect_right = 2.0 * np.sqrt(c["c1"] * c["c5"]) + c["c4"]
    assert LAM[0, 0] == pytest.approx(expect_left)
    assert LAM[-1, 0] == pytest.approx(expect_right)


def test_yaml_roundtrip_through_scenario(tmp_path):
    yaml_text = """
name: terrain_probe
description: test
bounds: [0.0, 0.0, 20.0, 4.0]
start: [0.5, 2.0, 0.0]
goal: [19.5, 2.0]
obstacles: [[10.0, 2.0, 0.4]]
terrain:
  - {x0: 5.0, y0: 0.0, x1: 10.0, y1: 4.0, m1: 2.0, m4: 2.0}
dock: [0.5, 2.0]
"""
    p = tmp_path / "terrain_probe.yaml"
    p.write_text(yaml_text)
    sc = load_scenario_from_yaml(str(p))
    assert sc.terrain is not None and len(sc.terrain.regions) == 1
    assert sc.terrain.regions[0].m1 == 2.0
    assert sc.dock is not None and sc.dock.tolist() == [0.5, 2.0]


def test_legacy_scenario_yaml_has_no_terrain_and_no_dock(tmp_path):
    yaml_text = """
name: plain
description: legacy
bounds: [0.0, 0.0, 20.0, 4.0]
start: [0.5, 2.0, 0.0]
goal: [19.5, 2.0]
obstacles: [[10.0, 2.0, 0.4]]
"""
    p = tmp_path / "plain.yaml"
    p.write_text(yaml_text)
    sc = load_scenario_from_yaml(str(p))
    assert sc.terrain is None and sc.dock is None


# ---------------------------------------------------------------------------
# Plant integration
# ---------------------------------------------------------------------------

def _straight_run_energy(terrain=None, mismatch=0.0, noise=None):
    env = PioneerEnv({"dt": 0.1})
    sc = ScenarioConfig(name="probe", start=np.array([0.5, 2.0, 0.0]),
                        goal=np.array([19.5, 2.0]), obstacles=[],
                        bounds=(0, 0, 20, 4))
    env.set_scenario(sc.obstacles, sc.bounds, terrain=terrain)
    env.set_power_mismatch(mismatch)
    if noise is not None:
        env.set_actuation_noise(*noise, seed=42)
    env.reset(sc.start, lidar_seed=1)
    for _ in range(100):
        env.step(1.095445, 0.0)   # v* straight line
    st = env.get_state_vector()
    return float(st[5]), float(st[0])


def test_all_ones_terrain_is_exactly_nominal():
    """A field of multiplier-1 regions must change NOTHING -- not even one
    ulp. This is the property that lets terrain maps coexist with every
    tracked legacy number."""
    e_nom, _ = _straight_run_energy(terrain=None)
    ones = TerrainField([TerrainRegion(0, 0, 20, 4, m1=1.0, m4=1.0)],
                        bounds=(0, 0, 20, 4))
    e_one, _ = _straight_run_energy(terrain=ones)
    assert e_one == e_nom


def test_terrain_scales_energy_by_lambda_ratio():
    """Driving at the NOMINAL v* through an m1=m4=2 region costs
    λ_m(v*) = m1·c1·v* + m4·c4 + c5/v* per metre there (c5 is NOT
    terrain-scaled -- idle draw is idle everywhere). The measured ratio must
    land on that fixed-speed ratio, which is *larger* than the minimised
    floor ratio 2√(m1·c1·c5)+m4·c4 / λ_min, because the robot does not
    re-optimise its speed in this open-loop test."""
    hi = TerrainField([TerrainRegion(0, 0, 20, 4, m1=2.0, m4=2.0)],
                      bounds=(0, 0, 20, 4))
    c = load_default_coefficients()
    vs = 1.095445
    lam2 = 2.0 * c["c1"] * vs + 2.0 * c["c4"] + c["c5"] / vs
    lam1 = c["c1"] * vs + c["c4"] + c["c5"] / vs
    e_hi, _ = _straight_run_energy(terrain=hi)
    e_nom, _ = _straight_run_energy(terrain=None)
    assert e_hi > e_nom
    # motor-lag transient makes it slightly under the steady-state ratio
    assert e_hi / e_nom <= lam2 / lam1 + 1e-9
    assert e_hi / e_nom >= lam2 / lam1 - 0.10


def test_power_mismatch_scales_energy():
    e_nom, _ = _straight_run_energy()
    e_p15, _ = _straight_run_energy(mismatch=0.15)
    assert e_p15 == pytest.approx(1.15 * e_nom, rel=1e-12)


def test_actuation_noise_is_reproducible_and_nonzero():
    e1, x1 = _straight_run_energy(noise=(0.05, 0.1))
    e2, x2 = _straight_run_energy(noise=(0.05, 0.1))
    assert e1 == e2 and x1 == x2          # seeded: bit-reproducible
    e0, x0 = _straight_run_energy(noise=None)
    assert abs(x1 - x0) > 1e-6            # noise actually did something


# ---------------------------------------------------------------------------
# Geodesic integration
# ---------------------------------------------------------------------------

def _probe_scenario():
    return ScenarioConfig(name="probe", start=np.array([0.5, 2.0, 0.0]),
                          goal=np.array([19.5, 2.0]),
                          obstacles=[(10.0, 2.0, 0.4)], bounds=(0, 0, 20, 4))


def test_ones_terrain_field_equals_lambda_min_times_metric():
    ones = TerrainField([TerrainRegion(0, 0, 20, 4)], bounds=(0, 0, 20, 4))
    sc = _probe_scenario()
    m = build_field(sc)
    j = build_field(sc, terrain=ones)
    assert j.denomination == "J" and m.denomination == "m"
    assert j.terrain is ones
    # constant λ trapezoid = λ × metric distance, exactly
    assert j.D[~np.isinf(j.D)] == pytest.approx(
        m.lambda_min * m.D[~np.isinf(m.D)], rel=1e-12)


def test_terrain_field_floor_at_equals_dstar():
    hi = TerrainField([TerrainRegion(5, 0, 15, 4, m1=3.0, m4=3.0)],
                      bounds=(0, 0, 20, 4))
    j = build_field(_probe_scenario(), terrain=hi)
    assert j.floor_at(0.5, 2.0) == j.dstar(0.5, 2.0)
    np.testing.assert_array_equal(j.E_floor, j.D)
    # the high-terrain route out of the start must cost more than λ_min × m
    m = build_field(_probe_scenario())
    assert j.dstar(0.5, 2.0) > m.lambda_min * m.dstar(0.5, 2.0)


def test_npz_roundtrip_preserves_denomination_and_terrain(tmp_path):
    hi = TerrainField([TerrainRegion(5, 0, 15, 4, m1=3.0, m4=3.0)],
                      bounds=(0, 0, 20, 4))
    from planning.energy_dijkstra import save_npz, load_npz
    j = build_field(_probe_scenario(), terrain=hi)
    p = str(tmp_path / "field.npz")
    save_npz(j, p)
    r = load_npz(p)
    assert r.denomination == "J"
    assert r.terrain is not None and r.terrain.regions[0].m1 == 3.0
    np.testing.assert_array_equal(r.D, j.D)


def test_interpolants_match_numpy_at_nodes_and_smooth_within_a_pitch():
    """The NLP-side interpolants must agree with the plant-side numpy
    evaluation ON the grid nodes. BETWEEN nodes the bilinear interpolant
    smooths the piecewise-constant rectangles across one grid pitch (e.g.
    0.1 m inside an edge at h=0.5 you get a blend, not the region value) --
    a documented, shared approximation. What must NOT happen is disagreement
    at nodes or error beyond the pitch-wide edge band."""
    import casadi as ca
    f = TerrainField([TerrainRegion(5, 0, 15, 4, m1=2.5, m4=1.5)],
                     bounds=(0, 0, 20, 4))
    h = 0.5
    i1, i4, _ = f.build_interpolants((0, 0, 20, 4), h=h)
    nodes = [(0.0, 0.0), (7.5, 2.0), (20.0, 4.0), (5.0, 0.5), (14.5, 3.5),
             (15.0, 0.0), (4.5, 2.0)]
    for (x, y) in nodes:
        m1n, m4n = f.multipliers(x, y)
        assert float(i1(ca.DM([x, y]))) == pytest.approx(m1n), (x, y)
        assert float(i4(ca.DM([x, y]))) == pytest.approx(m4n), (x, y)
    # 0.1 m inside the x=15 edge: a blend of 2.5 and 1.0 -- bounded error,
    # never a value outside [1.0, 2.5]
    v = float(i1(ca.DM([14.9, 0.2])))
    assert 1.0 <= v <= 2.5
    assert v == pytest.approx(2.5 - 0.8 * (2.5 - 1.0))   # exactly the blend
