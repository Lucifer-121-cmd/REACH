"""Statistical tests for EP²-MPC evaluation."""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats

logger = logging.getLogger(__name__)


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples.
    
    H0: a and b come from the same distribution.
    
    Args:
        a, b: paired measurement arrays
    
    Returns:
        (statistic, p_value)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length: {len(a)} vs {len(b)}")
    
    differences = a - b
    nonzero = differences[differences != 0]
    
    if len(nonzero) == 0:
        return 0.0, 1.0
    
    try:
        result = stats.wilcoxon(a, b, alternative='two-sided')
        return float(result.statistic), float(result.pvalue)
    except ValueError as e:
        logger.debug("Wilcoxon fallback to neutral result: %s", e)
        return 0.0, 1.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size for two independent samples.

    d = (P(a > b) - P(a < b)) ∈ [-1, 1]
    Positive d means a tends to be larger than b.

    Vectorised. The previous implementation was a double Python loop over
    every pair, which is O(n*m) interpreted work: 2,500 iterations at n = 50 but
    57,600 at the n ≈ 240 per arm the power analysis requires, times every
    comparison in every family. This computes the identical quantity by sorting
    b once and counting with searchsorted, which is O((n+m) log m) and exact --
    ties are handled explicitly rather than by floating-point comparison order.

    Args:
        a, b: measurement arrays (can have different lengths)

    Returns:
        d ∈ [-1, 1]
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        return 0.0

    b_sorted = np.sort(b)
    # for each a_i: how many b are strictly less, and how many are <= it
    less = np.searchsorted(b_sorted, a, side='left').sum()      # b < a_i
    less_or_eq = np.searchsorted(b_sorted, a, side='right').sum()  # b <= a_i
    greater = n_a * n_b - less_or_eq                             # b > a_i
    return float((less - greater) / (n_a * n_b))


def interpret_cliffs_delta(d: float) -> str:
    """Interpret Cliff's delta magnitude.
    
    Thresholds from Romano et al. 2006:
    |d| < 0.147: negligible
    |d| < 0.330: small
    |d| < 0.474: medium
    |d| >= 0.474: large
    """
    abs_d = abs(d)
    if abs_d < 0.147:
        return 'negligible'
    elif abs_d < 0.330:
        return 'small'
    elif abs_d < 0.474:
        return 'medium'
    else:
        return 'large'


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute parametric confidence interval using t-distribution.
    
    Args:
        data: sample data
        confidence: confidence level (default 0.95)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    mean = float(np.mean(data))
    if n == 1:
        return mean, mean, mean
    
    se = stats.sem(data)
    ci = stats.t.interval(confidence, df=n - 1, loc=mean, scale=se)
    return mean, float(ci[0]), float(ci[1])


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.
    
    Args:
        data: sample data
        n_bootstrap: number of bootstrap samples
        confidence: confidence level
        seed: random seed
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    
    rng = np.random.default_rng(seed)
    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    
    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return float(np.mean(data)), lower, upper


def compare_methods(
    results_ep2: 'EvalResult',
    results_baseline: 'EvalResult',
    metric: str = 'energy'
) -> Dict[str, Any]:
    """Full statistical comparison between EP²-MPC and a baseline.
    
    Args:
        results_ep2: EP²-MPC evaluation results
        results_baseline: baseline evaluation results
        metric: 'energy', 'time', 'solver_ms', or 'success'
    
    Returns:
        dict with statistic, p_value, cliffs_d, interpretation, CI
    """
    metric_map = {
        'energy': lambda r: [t.energy for t in r.trial_results],
        'time': lambda r: [t.time for t in r.trial_results],
        'solver_ms': lambda r: [t.solver_ms for t in r.trial_results],
        'success': lambda r: [float(t.success) for t in r.trial_results],
    }
    
    if metric not in metric_map:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(metric_map.keys())}")
    
    extract = metric_map[metric]
    a = np.array(extract(results_ep2))
    b = np.array(extract(results_baseline))
    
    # Align lengths for paired test
    n = min(len(a), len(b))
    a_paired, b_paired = a[:n], b[:n]
    
    stat, pval = wilcoxon_test(a_paired, b_paired)
    d = cliffs_delta(a, b)
    interp = interpret_cliffs_delta(d)
    mean_a, lo_a, hi_a = compute_confidence_interval(a)
    mean_b, lo_b, hi_b = compute_confidence_interval(b)
    
    return {
        'metric': metric,
        'ep2_mean': mean_a,
        'ep2_ci': (lo_a, hi_a),
        'baseline_mean': mean_b,
        'baseline_ci': (lo_b, hi_b),
        'wilcoxon_stat': stat,
        'p_value': pval,
        'significant': pval < 0.05,
        'cliffs_d': d,
        'effect_size': interp,
        'n_ep2': len(a),
        'n_baseline': len(b),
    }


# ===========================================================================
# The statistics protocol the rebuild requires
#
# Six things, each of which the rejected submission got wrong:
#   1. p-value floors. `1.78e-15` was quoted as a point estimate. It is not an
#      estimate of anything -- it is 2/2^50, the smallest two-sided p the exact
#      signed-rank test can return at n = 50. Reporting it as a number implies
#      resolution the design does not have.
#   2. Paired bootstrap CIs on the DIFFERENCE, not two independent CIs whose
#      overlap gets eyeballed.
#   3. Exact McNemar for success rates. Success is paired binary data; a chi2
#      approximation is wrong at these counts.
#   4. TOST with a pre-declared margin, so "no significant difference" can
#      become "equivalent within delta" instead of an absence of evidence.
#   5. Holm-Bonferroni within declared families.
#   6. The power result recorded as code, not prose: sigma_E = 0.62-0.69 J
#      against a 0.16 J target effect needs n ~= 240 per arm.
# ===========================================================================

#: Measured per-trial energy standard deviation, all three maps.
SIGMA_E_RANGE = (0.62, 0.69)

#: The effect size the rejected paper's headline energy claim rested on.
TARGET_EFFECT_J = 0.16


def wilcoxon_p_floor(n_nonzero: int) -> float:
    """Smallest two-sided p the exact signed-rank test can return.

    With ``n`` nonzero differences there are ``2**n`` equally likely sign
    assignments under H0; the two-sided extreme (all one sign, either way)
    carries probability ``2 / 2**n``. At n = 50 that is 1.7763568394002505e-15 --
    the number the rejected paper quoted as though it were an estimate.

    At n >= 1024, ``2.0 ** (1 - n)`` underflows to exactly 0.0. Returning 0.0
    would break every downstream guard that clamps to the floor: ``max(p, 0)``
    stays 0, Holm prints an adjusted p of exactly 0.0, and ``format_p_value``
    renders "p < 0e+00" -- the nonsense-number bug this module exists to
    prevent, resurfacing at exactly the n=1250 the 2026-09 campaigns run.
    The smallest positive denormal is returned instead: the true floor is
    far below it, so "p < 5e-324" is a true, conservative bound (run
    2026-09-10/11, n=1250 paired tests).
    """
    if n_nonzero <= 0:
        return 1.0
    floor = min(1.0, 2.0 ** (1 - n_nonzero))
    if floor <= 0.0:
        return float(np.nextafter(0.0, 1.0))   # ~5e-324, smallest positive
    return floor


def format_p_value(p: float, n_nonzero: Optional[int] = None,
                   digits: int = 3) -> str:
    """Render a p-value honestly, naming the resolution limit when it is hit.

    >>> format_p_value(1.7763568394002505e-15, 50)
    'p < 2e-15 (exact-test resolution limit, n=50)'
    """
    if p is None or not np.isfinite(p):
        return 'p = n/a'
    if n_nonzero is not None:
        floor = wilcoxon_p_floor(n_nonzero)
        if p <= floor * (1.0 + 1e-9):
            # Format the floor itself. Deriving an exponent from -log10 rounds the
            # wrong way (floor(14.75) = 14 for a value of 1.78e-15) and would
            # print a bound the test cannot actually claim.
            return (f"p < {floor:.0e} "
                    f"(exact-test resolution limit, n={n_nonzero})")
    if p < 1e-4:
        return f"p = {p:.1e}"
    return f"p = {p:.{digits}f}"


def wilcoxon_paired(a, b, method: str = 'exact') -> Dict[str, Any]:
    """Exact paired signed-rank test, reported with its resolution limit.

    ``method='exact'`` by default and deliberately. scipy's ``'auto'`` switches
    to the normal approximation above n = 25, which at n = 50 returns 1.54e-12
    for an all-one-sign difference where the exact test returns 1.776e-15 -- the
    floor. With ``'auto'`` the resolution-limit guard below can never fire, so
    an exact-test floor would get reported as an ordinary p-value again.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"paired arrays must match: {a.size} vs {b.size}")
    d = a - b
    n_nonzero = int(np.count_nonzero(d))
    floor = wilcoxon_p_floor(n_nonzero)
    if n_nonzero == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'n_pairs': int(a.size),
                'n_nonzero': 0, 'p_floor': 1.0, 'at_resolution_floor': False,
                'method': method,
                'p_text': 'p = 1.000 (all differences exactly zero)'}
    try:
        res = stats.wilcoxon(a, b, alternative='two-sided', method=method)
    except (ValueError, MemoryError) as exc:
        logger.warning("exact wilcoxon unavailable at n=%d (%s); falling back to "
                       "the normal approximation -- the p-value floor no longer "
                       "applies", n_nonzero, exc)
        res = stats.wilcoxon(a, b, alternative='two-sided', method='approx')
        method = 'approx'
    p = float(res.pvalue)
    return {'statistic': float(res.statistic), 'p_value': p,
            'n_pairs': int(a.size), 'n_nonzero': n_nonzero, 'p_floor': floor,
            'at_resolution_floor': bool(p <= floor * (1.0 + 1e-9)),
            'method': method,
            'p_text': format_p_value(p, n_nonzero)}


def bootstrap_ci_paired(a, b, n_boot: int = 10000, conf: float = 0.95,
                        seed: int = 42) -> Dict[str, Any]:
    """Percentile bootstrap CI on the MEAN PAIRED DIFFERENCE ``a - b``.

    Resampling pairs, not the two arms independently, is the point: the arms
    share matched seeds, so the pairing carries most of the precision. Two
    separate CIs and an eyeballed overlap throws that away.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"paired arrays must match: {a.size} vs {b.size}")
    d = a - b
    n = d.size
    if n < 2:
        return {'mean_diff': float(d.mean()) if n else 0.0,
                'ci_low': float('nan'), 'ci_high': float('nan'), 'n_pairs': n}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    alpha = 1.0 - conf
    return {'mean_diff': float(d.mean()),
            'ci_low': float(np.percentile(means, 100 * alpha / 2)),
            'ci_high': float(np.percentile(means, 100 * (1 - alpha / 2))),
            'n_pairs': int(n), 'conf': conf, 'n_boot': n_boot}


def mcnemar_exact(success_a, success_b) -> Dict[str, Any]:
    """Exact McNemar test for two paired binary (success) vectors.

    Success is paired binary data on matched seeds, so the only informative
    trials are the discordant ones: ``b01`` (a failed, b succeeded) and ``b10``
    (a succeeded, b failed). Under H0 each discordant pair is a fair coin, so the
    exact test is a two-sided binomial on ``b10`` out of ``b01 + b10``. The chi2
    approximation is not usable at these counts, and with 0 discordant pairs
    there is simply no evidence either way -- reported as p = 1.0 rather than
    as an error.
    """
    a = np.asarray(success_a).astype(bool).ravel()
    b = np.asarray(success_b).astype(bool).ravel()
    if a.size != b.size:
        raise ValueError(f"paired arrays must match: {a.size} vs {b.size}")
    b10 = int(np.sum(a & ~b))
    b01 = int(np.sum(~a & b))
    n_disc = b10 + b01
    if n_disc == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(b10, n_disc, 0.5, alternative='two-sided').pvalue)
    return {'b10': b10, 'b01': b01, 'n_discordant': n_disc,
            'n_pairs': int(a.size), 'p_value': p,
            'rate_a': float(a.mean()), 'rate_b': float(b.mean()),
            'p_floor': (2.0 ** (1 - n_disc)) if n_disc else 1.0,
            'p_text': format_p_value(p, n_disc if n_disc else None)}


def tost(a, b, margin: float, alpha: float = 0.05) -> Dict[str, Any]:
    """Two one-sided tests for EQUIVALENCE within a pre-declared ``margin``.

    Answers "are these two arms equivalent to within margin?" rather than
    "did we fail to detect a difference?". The margin must be declared before
    looking at the data; it is an argument here precisely so it appears in the
    caller and gets recorded.

    Equivalent iff both one-sided paired t-tests reject at ``alpha``:
        H01: mean_diff <= -margin      H02: mean_diff >= +margin
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"paired arrays must match: {a.size} vs {b.size}")
    if margin <= 0:
        raise ValueError("margin must be > 0 and declared in advance")
    d = a - b
    n = d.size
    mean_d = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float('inf')
    if se == 0.0:
        p_lower = 0.0 if mean_d > -margin else 1.0
        p_upper = 0.0 if mean_d < margin else 1.0
    else:
        t_lower = (mean_d + margin) / se
        t_upper = (mean_d - margin) / se
        p_lower = float(stats.t.sf(t_lower, df=n - 1))
        p_upper = float(stats.t.cdf(t_upper, df=n - 1))
    p_tost = max(p_lower, p_upper)
    return {'mean_diff': mean_d, 'margin': float(margin), 'alpha': alpha,
            'p_lower': p_lower, 'p_upper': p_upper, 'p_tost': p_tost,
            'equivalent': bool(p_tost < alpha), 'n_pairs': int(n),
            'se': se}


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni correction within ONE declared family.

    Holm rather than plain Bonferroni because it is uniformly more powerful at
    the same family-wise error rate. The family must be declared by the caller:
    passing every comparison in the paper as one family is as wrong as running
    none, in the opposite direction.

    Returns per-label adjusted p-values (monotonically enforced) and reject flags.
    """
    if not pvalues:
        return {'alpha': alpha, 'family_size': 0, 'results': {}, 'n_rejected': 0}
    labels = list(pvalues)
    p = np.array([float(pvalues[k]) for k in labels], dtype=float)
    m = p.size
    order = np.argsort(p, kind='stable')
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # enforce monotonicity
        adj[i] = min(1.0, running)
    reject = adj < alpha
    return {
        'alpha': alpha, 'family_size': m,
        'n_rejected': int(reject.sum()),
        'results': {labels[i]: {'p_raw': float(p[i]),
                                'p_adjusted': float(adj[i]),
                                'reject': bool(reject[i]),
                                'rank': int(np.where(order == i)[0][0]) + 1}
                    for i in range(m)},
    }


def required_n(sigma: float, effect: float, alpha: float = 0.05,
               power: float = 0.80, paired: bool = True) -> Dict[str, Any]:
    """Per-arm sample size for a two-sided test at the given power.

    With sigma_E = 0.62-0.69 J per trial against a 0.16 J
    target effect, 80% power at alpha = 0.05 needs n ~= 240 per arm. Every
    headline comparison in the rejected paper ran at n = 50, i.e. underpowered
    by roughly 5x.

    ``paired=True`` treats ``sigma`` as the SD of the paired difference. With
    ``paired=False`` the two-sample form doubles the variance requirement.
    """
    if sigma <= 0 or effect <= 0:
        raise ValueError("sigma and effect must both be > 0")
    z_a = stats.norm.isf(alpha / 2.0)
    z_b = stats.norm.isf(1.0 - power)
    factor = 1.0 if paired else 2.0
    n = factor * ((z_a + z_b) ** 2) * (sigma / effect) ** 2
    return {'n_per_arm': int(np.ceil(n)), 'sigma': sigma, 'effect': effect,
            'alpha': alpha, 'power': power, 'paired': paired,
            'n_exact': float(n)}


def power_analysis_summary() -> Dict[str, Any]:
    """The recorded power result for the energy comparisons, as data not prose.

    Note the ``paired=False`` form. SIGMA_E_RANGE is the **per-trial** energy SD,
    so the SD of a paired difference between two arms is sigma_E * sqrt(2). The
    two-sample expression carries that factor of 2 in the variance and gives
    identical numbers to the paired expression fed sigma_E*sqrt(2) -- verified.
    Feeding the per-trial sigma straight into the paired form would understate
    the requirement by a factor of 2 and land on ~118-146 instead of ~236-292.
    """
    lo, hi = SIGMA_E_RANGE
    n_lo = required_n(lo, TARGET_EFFECT_J, paired=False)['n_per_arm']
    n_hi = required_n(hi, TARGET_EFFECT_J, paired=False)['n_per_arm']
    return {
        'sigma_E_range_per_trial': SIGMA_E_RANGE,
        'target_effect_J': TARGET_EFFECT_J,
        'n_required_at_sigma_low': n_lo,      # 236 at sigma = 0.62
        'n_required_at_sigma_high': n_hi,     # 292 at sigma = 0.69
        'n_used_in_rejected_paper': 50,
        'underpowered_factor_range': (round(n_lo / 50.0, 1), round(n_hi / 50.0, 1)),
        'note': 'Every headline energy comparison in 26-3816v1 ran at n = 50 '
                'paired trials against an effect of 0.16 J with a per-trial '
                'sigma of 0.62-0.69 J: underpowered by roughly 5x. n ~= 240 '
                'is the sigma = 0.62 end of this range.',
    }
