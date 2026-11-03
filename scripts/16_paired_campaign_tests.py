#!/usr/bin/env python3
"""Paired tests over the FULL campaign, at n = 250 instead of n = 50.

Why this exists
---------------
`05_statistical_eval.py` computes its descriptive statistics over all 250 trials
(5 master seeds x 50) but runs every inferential test on `seed_results[0]` alone,
paired against baselines collected at `master_seed=0`. So every p-value, Cliff's
delta and CI in `statistical_results.json` rests on **n = 50**, while the file
also reports `total_trials: 250`. With sigma_E = 0.62-0.69 J per trial, n = 50 is
underpowered 4.7-5.8x for the 0.16 J target effect (it requires n ~ 236-292 per
arm). A paper that reports 250 trials and a
p-value computed from 50 is misreporting its own power.

The campaign already fixes this and nobody read it. `run_campaign.py` enforces
cross-arm seed alignment in `validate_plan`, and the result is that
`runs/<ID>/ep2/trials_<scen>_ep2mpc_seed*.jsonl` and
`runs/<ID>/baselines/trials_<scen>_<arm>_seed*.jsonl` carry the **same 250
(master_seed, trial_index, seed) triples** per scenario. Verified on
run_20260904T092518Z: 250 == 250, key sets identical, all three maps, both arms.

So this script reads that JSONL and runs the B6 protocol on the full 250 pairs.
No re-run, no solver, no evaluation -- pure analysis, seconds.

Protocol (evaluation/statistics.py, all of it exact where an exact form exists)
------------------------------------------------------------------------------
* `wilcoxon_paired(method='exact')` -- never 'auto', which switches to the normal
  approximation above n=25 and would report 1.54e-12 where the exact test reports
  its 1.776e-15 resolution floor.
* `format_p_value` -- a p at the floor is reported AS the floor, not as a point
  estimate. At n=250 the floor is 2/2^250, i.e. astronomically small, so the
  floor stops being the binding constraint that it is at n=50.
* `cliffs_delta` + `interpret_cliffs_delta` -- effect size, not just significance.
* `bootstrap_ci_paired` -- CI on the mean PAIRED difference, resampling pairs.
* `mcnemar_exact` -- success is paired binary data; only discordant pairs inform.
* `tost` -- equivalence against the 0.16 J target, because "not significantly
  different" is not the same claim as "equivalent".
* `holm_bonferroni` -- one declared family per scenario, per metric.
* `power_analysis_summary` -- what n this design actually needed.

Energy is compared ONLY where both arms reach 100% success: energy
accumulated on a crashed run is a truncated partial sum, not a smaller cost.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCENARIOS = ('corridor', 's_curve', 'chicane')

#: Arms the campaign runs with seed alignment against EP2-MPC. Anything else in
#: baseline_<scen>.json (energy_penalized_mpc, the N=24/N=50 variants) comes from
#: 02_run_baselines.py at a DIFFERENT master seed and is therefore not pairable
#: here -- silently pairing it would invent a correspondence that does not exist.
PAIRABLE_ARMS = ('apf', 'standard_mpc_casadi')

FULL_SUCCESS = 0.999


def _read_jsonl_arm(pattern: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Index one arm's per-trial records by (master_seed, trial_index)."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            key = (int(r['master_seed']), int(r['trial_index']))
            if key in out:
                raise SystemExit(
                    f"duplicate trial {key} in {path}. The JSONL sinks append, so "
                    f"a stage re-run under the same run-id doubles its records. "
                    f"Move the old files aside and re-run, or the pairing is wrong.")
            out[key] = r
    return out


def load_pairs(run_root: str, scenario: str, arm: str):
    """Return (keys, ep2_records, arm_records) over the intersection, or None."""
    ep2 = _read_jsonl_arm(
        os.path.join(run_root, 'ep2', f'trials_{scenario}_ep2mpc_seed*.jsonl'))
    base = _read_jsonl_arm(
        os.path.join(run_root, 'baselines', f'trials_{scenario}_{arm}_seed*.jsonl'))
    if not ep2 or not base:
        return None
    if set(ep2) != set(base):
        only_ep2 = sorted(set(ep2) - set(base))[:4]
        only_base = sorted(set(base) - set(ep2))[:4]
        raise SystemExit(
            f"{scenario}/{arm}: the two arms are NOT seed-aligned, so they cannot "
            f"be paired. ep2-only e.g. {only_ep2}; {arm}-only e.g. {only_base}. "
            f"run_campaign.py's validate_plan is supposed to guarantee alignment; "
            f"if this fires, the two arms came from different plans.")
    keys = sorted(ep2)
    return keys, [ep2[k] for k in keys], [base[k] for k in keys]


def compare_arm(scenario: str, arm: str, keys, ep2_recs, base_recs) -> Dict[str, Any]:
    """Run the full B6 protocol on one (scenario, arm) pair set."""
    from evaluation.statistics import (bootstrap_ci_paired, cliffs_delta,
                                       interpret_cliffs_delta, mcnemar_exact,
                                       tost, wilcoxon_paired, TARGET_EFFECT_J)

    ep2_succ = np.array([bool(r['success']) for r in ep2_recs])
    base_succ = np.array([bool(r['success']) for r in base_recs])
    ep2_E = np.array([float(r['energy']) for r in ep2_recs])
    base_E = np.array([float(r['energy']) for r in base_recs])
    ep2_T = np.array([float(r['time']) for r in ep2_recs])
    base_T = np.array([float(r['time']) for r in base_recs])

    out: Dict[str, Any] = {
        'scenario': scenario, 'arm': arm, 'n_pairs': len(keys),
        'master_seeds': sorted({k[0] for k in keys}),
        'ep2_success_rate': float(ep2_succ.mean()),
        'base_success_rate': float(base_succ.mean()),
        'mcnemar_success': mcnemar_exact(ep2_succ, base_succ),
    }

    both_full = (ep2_succ.mean() >= FULL_SUCCESS and base_succ.mean() >= FULL_SUCCESS)
    out['both_arms_100pc'] = bool(both_full)
    if not both_full:
        # Energy on a crashed run is a truncated partial sum. Report the
        # solvability difference and stop; do NOT emit an energy p-value that would
        # be quoted as an efficiency result.
        out['energy'] = None
        out['energy_suppressed_because'] = (
            f"not both arms at 100% success (ep2={ep2_succ.mean():.3f}, "
            f"{arm}={base_succ.mean():.3f}); energy on a crashed run is a "
            f"truncated partial sum, not a smaller cost")
        out['time'] = None
        return out

    for label, a, b in (('energy', ep2_E, base_E), ('time', ep2_T, base_T)):
        w = wilcoxon_paired(a, b, method='exact')
        d = float(cliffs_delta(a, b))
        ci = bootstrap_ci_paired(a, b)
        blk = {
            'ep2_mean': float(a.mean()), 'base_mean': float(b.mean()),
            'ep2_std': float(a.std(ddof=1)), 'base_std': float(b.std(ddof=1)),
            'wilcoxon': w,
            'cliffs_delta': d, 'effect': interpret_cliffs_delta(d),
            # cliff > 0 means EP2 draws the LARGER value, i.e. worse for both
            # energy and time. Stated explicitly because the sign convention is
            # the single easiest thing to invert when writing the paper.
            'ep2_better': bool(a.mean() < b.mean()),
            'paired_diff_ci95': ci,
            'pct_vs_base': float((b.mean() - a.mean()) / b.mean() * 100.0),
        }
        if label == 'energy':
            blk['tost_vs_target_effect'] = tost(a, b, margin=TARGET_EFFECT_J)
        out[label] = blk
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id', dest='run_id', required=True,
                    help='REQUIRED. Reads <runs>/<run-id>/{ep2,baselines}/*.jsonl.')
    ap.add_argument('--out', default=None,
                    help='Default: <runs>/<run-id>/statistics/'
                         'paired_campaign_tests.json')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    from utils.paths import run_dir
    run_root = run_dir(args.run_id)

    from evaluation.statistics import (format_p_value, holm_bonferroni,
                                       power_analysis_summary)

    per_scenario: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        arms: Dict[str, Any] = {}
        for arm in PAIRABLE_ARMS:
            loaded = load_pairs(run_root, scenario, arm)
            if loaded is None:
                logger.warning("%s/%s: no per-trial JSONL, skipping", scenario, arm)
                continue
            keys, ep2_recs, base_recs = loaded
            arms[arm] = compare_arm(scenario, arm, keys, ep2_recs, base_recs)
            logger.info("%s vs %-20s n=%d  ep2_succ=%.2f base_succ=%.2f",
                        scenario, arm, len(keys),
                        arms[arm]['ep2_success_rate'],
                        arms[arm]['base_success_rate'])
        if not arms:
            continue
        # One declared family per (scenario, metric). Passing every comparison in
        # the paper as one family is as wrong as running no correction at all.
        #
        # Clamp each p to its own resolution floor before correcting. At n = 250
        # with every difference the same sign, scipy's exact computation underflows
        # to literal 0.0 (corridor and s_curve do; chicane returns the floor,
        # 1.105e-75, from identical structure). Feeding 0.0 to Holm prints an
        # adjusted p of 0.0, which is not a number that may appear in a paper --
        # the test cannot resolve below its floor, so the floor IS the honest
        # figure. `format_p_value` already does this for the display string; the
        # family table has to agree with it.
        family = {}
        for a, b in arms.items():
            if not b.get('energy'):
                continue
            w = b['energy']['wilcoxon']
            family[a] = max(float(w['p_value']), float(w['p_floor']))
        per_scenario[scenario] = {
            'arms': arms,
            'holm_energy_family': holm_bonferroni(family) if family else None,
            'holm_note': ('p-values clamped to the exact test resolution floor '
                          '(2/2^n_nonzero) before correction; scipy underflows to '
                          '0.0 at n=250 for an all-one-sign difference'),
            'family_declaration': (
                'one family per scenario per metric: the pairable campaign arms '
                'compared against EP2-MPC on that map'),
        }

    # Actual pair count, measured from the JSONL — the Sep-2026 campaigns moved
    # from 50 to 250 trials per seed block, so per-cell n is whatever the run
    # recorded. A hardcoded 250 here would mislabel a 1250-pair run, and an
    # artefact that misstates its own n is the defect this script exists to fix.
    observed_ns = sorted({
        a['n_pairs'] for s in per_scenario.values() for a in s.get('arms', {}).values()
    })
    n_pairs_label = (observed_ns[0] if len(observed_ns) == 1
                     else f"variable {observed_ns[0]}-{observed_ns[-1]}")
    payload = {
        'run_id': args.run_id,
        'n_pairs_per_cell': n_pairs_label,
        'why': ('05_statistical_eval.py runs its inferential tests on one master '
                'seed (n=50) while reporting total_trials over all seeds. These '
                'use ALL seed-aligned pairs from the campaign JSONL (n recorded '
                'per cell; the Sep-2026 campaigns run 250 trials x 5 seed blocks '
                '= 1250 pairs per scenario per arm).'),
        'observed_n_pairs': observed_ns,
        'pairable_arms': list(PAIRABLE_ARMS),
        'unpairable_note': (
            'energy_penalized_mpc and standard_mpc_casadi[N=24|50] are NOT here: '
            'they come from 02_run_baselines.py at a different master seed, so no '
            'trial-level correspondence with the EP2 campaign exists.'),
        'power': power_analysis_summary(),
        'scenarios': per_scenario,
    }

    out = args.out or os.path.join(run_dir(args.run_id, 'statistics'),
                                   'paired_campaign_tests.json')
    from utils.safe_write import safe_write_json
    from utils.provenance import write_sidecar, sidecar_path
    safe_write_json(out, payload, force=args.force)
    inputs = sorted(glob.glob(os.path.join(run_root, 'ep2', 'trials_*.jsonl'))
                    + glob.glob(os.path.join(run_root, 'baselines', 'trials_*.jsonl')))
    write_sidecar(out, run_id=args.run_id, inputs=inputs,
                  script='scripts/16_paired_campaign_tests.py', force=args.force)

    print(f"\n{'scenario':<10} {'arm':<22} {'n':>4} {'EP2 E':>8} {'base E':>8} "
          f"{'%':>7} {'cliff':>7} {'p':>14}")
    for scenario, blk in per_scenario.items():
        for arm, r in blk['arms'].items():
            e = r.get('energy')
            if not e:
                print(f"{scenario:<10} {arm:<22} {r['n_pairs']:4d} "
                      f"{'--':>8} {'--':>8} {'--':>7} {'--':>7} "
                      f"{'solvability':>14}   (McNemar "
                      f"{r['mcnemar_success']['p_text'] if 'p_text' in r['mcnemar_success'] else r['mcnemar_success'].get('p_value')})")
                continue
            print(f"{scenario:<10} {arm:<22} {r['n_pairs']:4d} "
                  f"{e['ep2_mean']:8.2f} {e['base_mean']:8.2f} "
                  f"{e['pct_vs_base']:+7.2f} {e['cliffs_delta']:+7.3f} "
                  f"{e['wilcoxon']['p_text']:>14}")
    print(f"\nwritten : {out}")
    print(f"sidecar : {sidecar_path(out)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
