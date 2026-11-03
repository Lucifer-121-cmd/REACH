"""The paired campaign tests must use all 250 pairs, and refuse when they can't.

`05_statistical_eval.py` reports `total_trials: 250` and computes every p-value,
Cliff's delta and CI from `seed_results[0]` alone -- n = 50. PLAN.md's own power
analysis says n = 50 is underpowered 4.7-5.8x for the 0.16 J target effect. The
campaign already produces 250 seed-aligned pairs per cell; this script uses them.

These tests cover the three things that could make it silently wrong: pairing
records that do not correspond, comparing energy across a crashed arm, and letting
an underflowed p-value reach a family correction as 0.0.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _mod():
    path = os.path.join(ROOT, 'scripts', '16_paired_campaign_tests.py')
    spec = importlib.util.spec_from_file_location('paired_16', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules['paired_16'] = m
    spec.loader.exec_module(m)
    return m


def _rec(master_seed, trial_index, energy, success=True, time=17.0):
    return {'master_seed': master_seed, 'trial_index': trial_index,
            'seed': master_seed + trial_index, 'energy': energy,
            'time': time, 'success': success}


def _write_jsonl(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for r in recs:
            fh.write(json.dumps(r) + '\n')


def test_misaligned_arms_refuse_rather_than_pair_by_position(tmp_path):
    """Two arms with different seeds have no trial-level correspondence."""
    m = _mod()
    _write_jsonl(str(tmp_path / 'ep2' / 'trials_corridor_ep2mpc_seed1000.jsonl'),
                 [_rec(1000, i, 128.0 + i * 0.01) for i in range(5)])
    _write_jsonl(str(tmp_path / 'baselines' / 'trials_corridor_apf_seed2000.jsonl'),
                 [_rec(2000, i, 149.0 + i * 0.01) for i in range(5)])
    with pytest.raises(SystemExit) as e:
        m.load_pairs(str(tmp_path), 'corridor', 'apf')
    assert 'NOT seed-aligned' in str(e.value)


def test_duplicate_records_refuse(tmp_path):
    """The JSONL sinks append, so a re-run under the same run-id doubles records."""
    m = _mod()
    p = str(tmp_path / 'ep2' / 'trials_corridor_ep2mpc_seed1000.jsonl')
    _write_jsonl(p, [_rec(1000, 0, 128.0), _rec(1000, 0, 128.5)])
    with pytest.raises(SystemExit) as e:
        m._read_jsonl_arm(p)
    assert 'duplicate trial' in str(e.value)


def test_energy_is_suppressed_when_one_arm_crashes(tmp_path):
    """PLAN.md §8: energy on a crashed run is a truncated partial sum."""
    m = _mod()
    keys = [(21000, i) for i in range(20)]
    ep2 = [_rec(21000, i, 134.0 + i * 0.01, success=True) for i in range(20)]
    apf = [_rec(21000, i, 49.0 + i * 0.01, success=False) for i in range(20)]
    out = m.compare_arm('chicane', 'apf', keys, ep2, apf)

    assert out['energy'] is None, 'energy compared across a crashed arm'
    assert out['time'] is None
    assert 'truncated partial sum' in out['energy_suppressed_because']
    # Solvability is still reported -- that IS the result on this map.
    mc = out['mcnemar_success']
    assert mc['b10'] == 20 and mc['b01'] == 0
    assert out['ep2_success_rate'] == 1.0 and out['base_success_rate'] == 0.0


def test_both_full_success_produces_a_real_energy_comparison(tmp_path):
    m = _mod()
    n = 30
    keys = [(1000, i) for i in range(n)]
    # EP2 uniformly 1.0 J worse: one-sign differences, the case that underflows.
    ep2 = [_rec(1000, i, 128.0 + i * 0.001) for i in range(n)]
    std = [_rec(1000, i, 127.0 + i * 0.001) for i in range(n)]
    out = m.compare_arm('corridor', 'standard_mpc_casadi', keys, ep2, std)

    e = out['energy']
    assert e is not None and out['both_arms_100pc']
    assert e['ep2_better'] is False, 'sign convention inverted'
    assert e['cliffs_delta'] > 0.9, 'EP2 draws the larger value; delta must be +'
    assert e['wilcoxon']['method'] == 'exact', "must not fall back to 'auto'"
    ci = e['paired_diff_ci95']
    assert ci['ci_low'] > 0 and ci['ci_high'] > 0, 'CI must exclude zero'
    assert abs(ci['mean_diff'] - 1.0) < 1e-6
    # TOST against the declared 0.16 J target: 1.0 J apart is not equivalent.
    assert e['tost_vs_target_effect']['equivalent'] is False


def test_underflowed_p_is_clamped_to_its_floor_before_correction():
    """scipy's exact test underflows to 0.0 at n=250; Holm must not print 0.0."""
    from evaluation.statistics import holm_bonferroni, wilcoxon_paired
    n = 250
    a = np.arange(n, dtype=float) + 1000.0
    b = a - 1.0                      # every difference the same sign
    w = wilcoxon_paired(a, b, method='exact')
    assert w['at_resolution_floor'] is True
    assert w['p_floor'] > 0.0
    clamped = max(float(w['p_value']), float(w['p_floor']))
    assert clamped > 0.0, 'the clamp is what keeps 0.0 out of the family table'
    h = holm_bonferroni({'std': clamped})
    assert h['results']['std']['p_adjusted'] > 0.0
    assert h['results']['std']['reject'] is True
