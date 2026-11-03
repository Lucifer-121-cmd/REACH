"""Guards for the REACH map generator (PLAN.md R2).

Every criterion here is one the rejection sampler enforces, re-checked
independently -- a generator that silently stops enforcing its own rules
would produce drag races again, and the REACH claim would be measured on
maps that cannot show it.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.scenarios import load_scenario_from_yaml
from mpc.energy_model import load_default_coefficients
from planning.energy_dijkstra import build_field
from planning.reach_maps import (MAX_OBS, RESERVE_HEADROOM, MapSpec,
                                 _check_map, generate_map, generate_split,
                                 scenario_to_yaml_dict)

SEEDS = [20260911, 20260912, 20260913, 20260914, 20260915, 20260916]
COEFFS = load_default_coefficients()


@pytest.mark.parametrize("seed", SEEDS)
def test_every_criterion_holds_on_generated_maps(seed):
    """Re-run the checker on the accepted map -- and re-derive the two
    decisive ones (straight-line blocked, turn-back interior) from the map
    itself rather than trusting the sampler's bookkeeping."""
    sc, spec = generate_map(MapSpec(seed=seed, split="train"))
    ok, reason = _check_map(sc, spec, COEFFS)
    assert ok, reason

    assert len(sc.obstacles) <= MAX_OBS
    assert sc.dock is not None and np.allclose(sc.dock, sc.goal), \
        "goal must BE the dock (the geodesic field measures energy-to-dock)"

    # straight-line blocked, derived independently
    L = sc.bounds[2]
    dock = np.asarray(sc.dock)
    far = np.array([L - 1.0, dock[1]])
    blocked = any(
        _seg_clear(dock, far, cx, cy) <= r + 0.25 for cx, cy, r in sc.obstacles)
    assert blocked, "the +x ray from the dock must be obstructed"

    # turn-back interior, derived independently: round-trip J to far end
    f = build_field(sc, coeffs=COEFFS, terrain=sc.terrain)
    e_far = f.dstar(far[0], far[1])
    if f.denomination == "m":
        e_far *= f.lambda_min
    assert 2.0 * e_far >= RESERVE_HEADROOM * spec.b_max


def _seg_clear(p0, p1, cx, cy):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    t = 0.0 if L2 == 0 else float(np.clip((np.array([cx, cy]) - p0) @ d / L2, 0, 1))
    return float(np.hypot(*(p0 + t * d - np.array([cx, cy]))))


def test_same_seed_same_map():
    a, _ = generate_map(MapSpec(seed=20260911, split="train"))
    b, _ = generate_map(MapSpec(seed=20260911, split="train"))
    assert scenario_to_yaml_dict(a) == scenario_to_yaml_dict(b)


def test_different_seed_different_map():
    a, _ = generate_map(MapSpec(seed=20260911, split="train"))
    b, _ = generate_map(MapSpec(seed=20260912, split="train"))
    assert scenario_to_yaml_dict(a) != scenario_to_yaml_dict(b)


def test_split_seeds_are_disjoint(tmp_path):
    """Held-out maps must not be re-labelled training maps."""
    m = generate_split(str(tmp_path), n_train=2, n_heldout=2, seed0=20260911)
    train_seeds = {e["seed"] for e in m["splits"]["train"]}
    held_seeds = {e["seed"] for e in m["splits"]["heldout"]}
    assert train_seeds and held_seeds and not (train_seeds & held_seeds)
    assert (m["specs"]["heldout_seed_range"][0]
            > m["specs"]["train_seed_range"][1])


def test_yaml_roundtrip_preserves_reach_semantics(tmp_path):
    import yaml
    sc, _ = generate_map(MapSpec(seed=20260911, split="train"))
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(scenario_to_yaml_dict(sc), sort_keys=False))
    r = load_scenario_from_yaml(str(p))
    assert np.allclose(r.dock, sc.dock)
    assert np.allclose(r.goal, sc.goal)
    assert r.terrain is not None and len(r.terrain.regions) == len(sc.terrain.regions)
    assert [tuple(o) for o in r.obstacles] == [tuple(o) for o in sc.obstacles]


def test_manifest_records_rejections_and_terrain(tmp_path):
    m = generate_split(str(tmp_path), n_train=1, n_heldout=1, seed0=20260911)
    for split in ("train", "heldout"):
        e = m["splits"][split][0]
        assert "rejections" in e and e["rejections"] >= 0
        assert e["terrain"] is None or "regions" in e["terrain"] or "λ" in e["terrain"]


def test_terrain_maps_have_higher_energy_floor_than_metric(tmp_path):
    """The whole point of R2: on a terrain map the energy-geodesic to the
    far end must cost more than the homogeneous floor would claim, i.e. the
    terrain actually bites on the priced route."""
    sc, _ = generate_map(MapSpec(seed=20260913, split="train"))
    if sc.terrain is None:
        pytest.skip("this seed sampled no terrain")
    jt = build_field(sc, coeffs=COEFFS, terrain=sc.terrain)
    mt = build_field(sc, coeffs=COEFFS)
    far = (sc.bounds[2] - 1.0, float(sc.dock[1]))
    assert jt.denomination == "J"
    assert jt.dstar(*far) > mt.lambda_min * mt.dstar(*far)
