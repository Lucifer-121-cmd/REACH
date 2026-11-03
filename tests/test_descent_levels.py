"""Smoke guard: the persisted P4 level-2/3 analyses exist and run.

2026-09-21 (fresh_full_audit finding 1): truth_paper.md §3.11 quoted
steady-state and realignment-transient figures that existed in no
artefact and were computed by no repo code. scripts/25 now computes and
persists them per map. This test runs both analyses on the tracked
`corridor` scenario (portable: no runs/ dependency) and asserts the
output schema — it does not assert physics, only that the persisted
quantities exist and are well-formed, so a refactor that silently drops
level 2/3 from the artefact fails here.
"""
import importlib.util
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'scripts', '25_descent_check.py')
_spec = importlib.util.spec_from_file_location('descent_25', _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _corridor():
    from env.scenarios import get_scenario
    return get_scenario('corridor')


def test_steady_state_check_schema():
    r = mod.steady_state_check(_corridor(), a_q=23.73, stride=20)
    assert set(r) >= {'definition', 'bilinear', 'nearest'}, sorted(r)
    for tag in ('bilinear', 'nearest'):
        rec = r[tag]
        assert rec['n_states'] > 50, rec
        assert 0.0 <= rec['violation_rate'] <= 1.0, rec
        assert rec['mean_deficit_J'] >= 0.0, rec
    # intercept invariance: same deficits under any a_q (they cancel)
    r2 = mod.steady_state_check(_corridor(), a_q=100.0, stride=20)
    for tag in ('bilinear', 'nearest'):
        assert r2[tag]['n_bad'] == r[tag]['n_bad'], (tag, r[tag], r2[tag])
        assert r2[tag]['mean_deficit_J'] == r[tag]['mean_deficit_J']


def test_realignment_transient_schema():
    r = mod.realignment_transient(_corridor(), a_q=23.73, stride=20,
                                  cell_every=2)
    assert set(r) >= {'definition', 'n_cells', 'n_invalid',
                      'worst_transient_J', 'p95_transient_J',
                      'median_transient_J', 'covered_by_intercept_x',
                      'n_positive'}, sorted(r)
    assert r['n_cells'] > 10, r
    assert r['worst_transient_J'] >= 0.0
    assert r['p95_transient_J'] <= r['worst_transient_J'] + 1e-12
    if r['worst_transient_J'] > 0:
        # nonnegative leak, covered ratio finite and positive
        assert math.isfinite(r['covered_by_intercept_x'])
        assert r['covered_by_intercept_x'] > 0


def test_bilinear_matches_grid_at_nodes():
    from planning.energy_dijkstra import build_field
    jf = build_field(_corridor())
    checked = 0
    for x in jf.X[::7]:
        for y in jf.Y[::5]:
            x, y = float(x), float(y)
            if not math.isfinite(jf.dstar(x, y)):
                continue
            b = mod.bilinear_dstar(jf, x, y)
            # near the obstacle-inflation boundary a cell corner can be
            # off the reachable set: bilinear honestly returns inf there
            if not math.isfinite(b):
                continue
            # at a grid node bilinear must equal the cell value
            assert abs(b - jf.dstar(x, y)) < 1e-9, (x, y, b, jf.dstar(x, y))
            checked += 1
    assert checked > 20, checked
