"""Tests for the F7 statistics protocol (B6).

The four things these pin down are the four I got wrong on the first pass, each
of which would have silently reproduced a defect from the rejected submission:

1. ``format_p_value`` must render the floor as ``2e-15``, not ``2e-14``. Deriving
   the exponent from ``-log10`` rounds the wrong way and claims a tighter bound
   than the test can support.
2. ``wilcoxon_paired`` must use ``method='exact'``. scipy's ``'auto'`` switches to
   the normal approximation above n=25 and returns 1.54e-12 where the exact test
   returns the 1.776e-15 floor -- so with ``'auto'`` the resolution-limit guard
   can never fire and the forbidden number slips through unflagged.
3. ``power_analysis_summary`` must use the two-sample form, because
   ``SIGMA_E_RANGE`` is the per-trial SD and a paired difference has SD
   ``sigma*sqrt(2)``. Feeding the per-trial sigma to the paired form halves the
   variance and lands on ~118 instead of ~236.
4. Vectorised Cliff's delta must be bit-identical to the double loop it replaced,
   including under heavy ties.
"""
import numpy as np
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.statistics import (
    cliffs_delta, wilcoxon_p_floor, format_p_value, wilcoxon_paired,
    bootstrap_ci_paired, mcnemar_exact, tost, holm_bonferroni, required_n,
    power_analysis_summary, SIGMA_E_RANGE, TARGET_EFFECT_J,
)

FORBIDDEN_POINT_ESTIMATE = 1.7763568394002505e-15


# --- 1. the p-value floor ---------------------------------------------------

def test_floor_at_n50_is_exactly_the_forbidden_number():
    """PLAN.md §9: 1.78e-15 is 2/2^50, not an estimate of anything."""
    assert wilcoxon_p_floor(50) == FORBIDDEN_POINT_ESTIMATE


def test_floor_renders_as_2e_minus_15_not_2e_minus_14():
    text = format_p_value(FORBIDDEN_POINT_ESTIMATE, 50)
    assert '2e-15' in text, text
    assert '2e-14' not in text
    assert 'resolution limit' in text


def test_real_small_p_is_not_mislabelled_as_the_floor():
    # 1.19239e-9 is the true maximum from the corpus; it must print as a value.
    text = format_p_value(1.19239e-9, 50)
    assert 'resolution limit' not in text
    assert '1.2e-09' in text


def test_floor_scales_with_n():
    assert wilcoxon_p_floor(10) == pytest.approx(2.0 / 2 ** 10)
    assert wilcoxon_p_floor(0) == 1.0
    assert wilcoxon_p_floor(1) == 1.0


# --- 2. exact wilcoxon ------------------------------------------------------

def test_wilcoxon_uses_exact_and_reaches_the_floor():
    a = np.arange(50, dtype=float)
    r = wilcoxon_paired(a, a + 5.0)
    assert r['method'] == 'exact'
    assert r['p_value'] == pytest.approx(FORBIDDEN_POINT_ESTIMATE, rel=1e-12)
    assert r['at_resolution_floor'] is True
    assert '2e-15' in r['p_text']


def test_wilcoxon_auto_would_miss_the_floor():
    """Documents why the default is 'exact' and not 'auto'."""
    a = np.arange(50, dtype=float)
    approx = wilcoxon_paired(a, a + 5.0, method='approx')
    assert approx['p_value'] > wilcoxon_p_floor(50) * 100
    assert approx['at_resolution_floor'] is False


def test_wilcoxon_all_zero_differences():
    a = np.ones(10)
    r = wilcoxon_paired(a, a)
    assert r['p_value'] == 1.0 and r['n_nonzero'] == 0
    assert r['at_resolution_floor'] is False


def test_wilcoxon_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        wilcoxon_paired([1, 2, 3], [1, 2])


# --- 3. power ---------------------------------------------------------------

def test_power_lands_on_the_plan_figure():
    p = power_analysis_summary()
    assert p['n_required_at_sigma_low'] == 236     # sigma = 0.62, PLAN's "~240"
    assert p['n_required_at_sigma_high'] == 292    # sigma = 0.69
    assert 4.0 <= p['underpowered_factor_range'][0] <= 6.0


def test_paired_and_two_sample_forms_agree_via_sqrt2():
    lo = SIGMA_E_RANGE[0]
    two = required_n(lo, TARGET_EFFECT_J, paired=False)['n_per_arm']
    pair = required_n(lo * np.sqrt(2), TARGET_EFFECT_J, paired=True)['n_per_arm']
    assert two == pair


def test_required_n_rejects_nonsense():
    for bad in ((0.0, 0.16), (0.65, 0.0), (-1.0, 0.16)):
        with pytest.raises(ValueError):
            required_n(*bad)


# --- 4. vectorised Cliff's delta -------------------------------------------

def _loop_cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return 0.0
    dom = sum(1 if ai > bi else (-1 if ai < bi else 0) for ai in a for bi in b)
    return float(dom / (a.size * b.size))


def test_cliffs_delta_is_bit_identical_to_the_loop_with_ties():
    rng = np.random.default_rng(11)
    for _ in range(60):
        n, m = int(rng.integers(1, 40)), int(rng.integers(1, 40))
        a = np.round(rng.normal(0, 1, n), 1)     # 1 dp forces many exact ties
        b = np.round(rng.normal(0.2, 1, m), 1)
        assert cliffs_delta(a, b) == _loop_cliffs(a, b)


def test_cliffs_delta_extremes():
    assert cliffs_delta([5, 6], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [5, 6]) == -1.0
    assert cliffs_delta([1, 1, 1], [1, 1, 1]) == 0.0
    assert cliffs_delta([], [1, 2]) == 0.0


# --- the rest of the protocol ----------------------------------------------

def test_bootstrap_ci_is_on_the_paired_difference():
    rng = np.random.default_rng(3)
    a = rng.normal(100, 0.65, 50)
    r = bootstrap_ci_paired(a, a + 2.0)
    assert r['mean_diff'] == pytest.approx(-2.0)
    # a constant offset has zero variance, so the CI must collapse onto it
    assert r['ci_low'] == pytest.approx(-2.0) and r['ci_high'] == pytest.approx(-2.0)


def test_mcnemar_exact_uses_discordant_pairs_only():
    r = mcnemar_exact([1] * 50, [1] * 47 + [0] * 3)
    assert (r['b10'], r['b01'], r['n_discordant']) == (3, 0, 3)
    assert r['p_value'] == pytest.approx(0.25)      # 2 * 0.5**3
    assert 'resolution limit' in r['p_text']         # 0.25 IS the floor at n=3


def test_mcnemar_identical_arms_is_no_evidence_not_an_error():
    r = mcnemar_exact([1, 0, 1], [1, 0, 1])
    assert r['n_discordant'] == 0 and r['p_value'] == 1.0


def test_tost_declares_equivalence_only_within_margin():
    rng = np.random.default_rng(5)
    a = rng.normal(100, 0.65, 50)
    assert tost(a, a + rng.normal(0, 0.65, 50), margin=1.0)['equivalent'] is True
    assert tost(a, a + 5.0, margin=0.5)['equivalent'] is False


def test_tost_requires_a_positive_declared_margin():
    with pytest.raises(ValueError):
        tost([1, 2, 3], [1, 2, 3], margin=0.0)


def test_holm_is_monotone_and_more_powerful_than_bonferroni():
    fam = {'a': 0.001, 'b': 0.02, 'c': 0.04, 'd': 0.30, 'e': 0.55}
    h = holm_bonferroni(fam)
    adj = [h['results'][k]['p_adjusted'] for k in sorted(fam, key=fam.get)]
    assert adj == sorted(adj), "adjusted p-values must be monotone in rank"
    # Holm's smallest adjustment is m*p, same as Bonferroni; later ones are looser
    assert h['results']['a']['p_adjusted'] == pytest.approx(5 * 0.001)
    assert h['results']['e']['p_adjusted'] <= 1.0


def test_holm_empty_family():
    assert holm_bonferroni({})['family_size'] == 0


# ---------------------------------------------------------------------------
# 2026-09-11: the n=1250 underflow. The 2026-09 campaigns run 250 trials x 5
# seed blocks, and at n >= 1024 the exact-test floor 2/2**n underflows to
# exactly 0.0 -- which un-clamps every guard built on it (max(p, 0) stays 0,
# Holm prints adjusted p = 0.0, format_p_value prints "p < 0e+00"). Found on
# run_20260910T174433Z's paired_campaign_tests.json.
# ---------------------------------------------------------------------------

def test_p_floor_does_not_underflow_to_zero_at_n1250():
    """The floor must stay positive so clamping and Holm keep working."""
    from evaluation.statistics import wilcoxon_p_floor
    for n in (1023, 1024, 1250, 5000):
        floor = wilcoxon_p_floor(n)
        assert 0.0 < floor <= 5e-324 or 0.0 < floor < 1e-300, (
            f"n={n}: floor must be positive, got {floor!r}")


def test_format_p_value_at_underflow_never_prints_zero():
    """p < 0e+00 is not a number that may appear in a paper."""
    from evaluation.statistics import format_p_value, wilcoxon_p_floor
    floor = wilcoxon_p_floor(1250)
    text = format_p_value(0.0, 1250)
    assert "0e+00" not in text, text
    assert text.startswith("p < "), text
    assert float(text.split()[2]) > 0, text


def test_holm_stays_positive_when_p_underflows():
    """Holm over underflowed p's must not produce an adjusted p of 0.0."""
    from evaluation.statistics import holm_bonferroni, wilcoxon_p_floor
    floor = wilcoxon_p_floor(1250)
    adj = holm_bonferroni({'a': floor, 'b': floor, 'c': 0.01})
    for label, entry in adj['results'].items():
        assert entry['p_adjusted'] > 0, (label, entry)
