#!/usr/bin/env python3
"""Re-tune the APF baseline's gains against the CORRECTED obstacle geometry.

Why this exists
---------------
C1/C2 added ``r_body`` to the potential field's clearance, so obstacles are now
measured body-to-surface as they are in the MPC. That is geometrically right, but
``k_rep`` and ``d_influence`` were tuned against the OLD centre-to-surface
convention, and inflating every obstacle by 0.25 m without re-tuning them turned a
working s_curve slalom into a guaranteed crash:

    tracked (pre-repair) corridor 1.00   s_curve 1.00   chicane 0.00
    after the repair, untuned corridor 1.00   s_curve 0.00   chicane 0.00

Shipping that as "the corrected baseline" would repeat the exact error that killed
the original submission -- a comparator crippled by its own geometry -- just with
the sign flipped. The parity discipline exists to give every arm its best honest
shot, so the
gains have to be re-fitted to the new convention.

Selection rule, fixed in advance
--------------------------------
Maximise mean success rate across all three maps; break ties on lowest mean
energy over the maps where the arm succeeds. Energy is never used to break a tie
between different success rates, because energy on a crashed run is meaningless
(a truncated partial sum -- no energy comparison unless both arms are at 100%).

The search box is declared here and recorded in the output so the tuning-parity
table can quote it.

Single core, no solver, minutes. Writes nothing unless --out is given.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s [%(levelname)s] %(message)s')

#: Declared search box. Centred on the pre-C2 values (2.2, 2.2) and widened
#: upward, because the corrected clearance makes obstacles effectively larger and
#: the repulsion has to reach further to steer around them.
K_REP_GRID = (1.0, 1.5, 2.2, 3.0, 4.0, 5.5, 7.0)
D_INFLUENCE_GRID = (1.5, 2.2, 3.0, 4.0, 5.0)

SCENARIOS = ('corridor', 's_curve', 'chicane')


def evaluate_gains(k_rep: float, d_influence: float, cfg: dict,
                   n_trials: int, master_seed: int) -> Dict[str, Any]:
    """Run every map at one (k_rep, d_influence) and summarise."""
    from evaluation.baselines import APFBaseline, run_baseline
    from env.scenarios import get_scenario

    per_map: Dict[str, Dict[str, float]] = {}
    for scen in SCENARIOS:
        ctrl = APFBaseline(robot=cfg['robot'], dt=cfg['mpc'].get('dt', 0.1),
                           k_rep=k_rep, d_influence=d_influence)
        res = run_baseline(ctrl, get_scenario(scen), n_trials=n_trials,
                           config=cfg, master_seed=master_seed)
        per_map[scen] = {'success_rate': float(res.success_rate),
                         'energy_mean': float(res.energy_mean),
                         'time_mean': float(res.time_mean),
                         'path_distance_mean': float(res.path_distance_mean)}
    succ = float(np.mean([m['success_rate'] for m in per_map.values()]))
    solved = [m for m in per_map.values() if m['success_rate'] >= 0.999]
    energy_on_solved = (float(np.mean([m['energy_mean'] for m in solved]))
                        if solved else float('inf'))
    return {'k_rep': k_rep, 'd_influence': d_influence, 'per_map': per_map,
            'mean_success': succ, 'n_maps_solved': len(solved),
            'energy_on_solved': energy_on_solved}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-trials', type=int, default=20,
                    help='trials per map per grid point during the sweep')
    ap.add_argument('--validate-trials', type=int, default=50,
                    help='trials per map for the winner')
    ap.add_argument('--master-seed', type=int, default=1000)
    ap.add_argument('--config', default=None)
    ap.add_argument('--run-id', dest='run_id', default=None,
                    help='when given (and --out is not), writes to '
                         '<runs>/<run-id>/tuning/apf_gain_sweep.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    from utils.config_loader import load_config
    cfg = load_config(args.config)
    cfg.setdefault('evaluation', {})['progress_every_trials'] = 0
    if not cfg.get('robot'):
        raise SystemExit("config['robot'] is required")

    grid = list(itertools.product(K_REP_GRID, D_INFLUENCE_GRID))
    print(f"APF gain sweep: {len(grid)} points x {len(SCENARIOS)} maps x "
          f"{args.n_trials} trials = {len(grid)*len(SCENARIOS)*args.n_trials} trials")
    print(f"  k_rep grid       : {K_REP_GRID}")
    print(f"  d_influence grid : {D_INFLUENCE_GRID}")
    print(f"  pre-C2 defaults  : k_rep=2.2, d_influence=2.2")
    print()
    print(f"  {'k_rep':>6} {'d_infl':>7} {'corridor':>9} {'s_curve':>8} {'chicane':>8} "
          f"{'mean':>6} {'E(solved)':>10}")

    rows: List[Dict[str, Any]] = []
    for k_rep, d_inf in grid:
        r = evaluate_gains(k_rep, d_inf, cfg, args.n_trials, args.master_seed)
        rows.append(r)
        pm = r['per_map']
        e = r['energy_on_solved']
        print(f"  {k_rep:6.2f} {d_inf:7.2f} {pm['corridor']['success_rate']:9.2f} "
              f"{pm['s_curve']['success_rate']:8.2f} {pm['chicane']['success_rate']:8.2f} "
              f"{r['mean_success']:6.2f} {(f'{e:10.2f}' if np.isfinite(e) else '         -')}")

    # Selection rule, applied exactly as declared above.
    best = max(rows, key=lambda r: (r['mean_success'], -r['energy_on_solved']))
    print()
    print(f"  WINNER: k_rep={best['k_rep']}, d_influence={best['d_influence']}  "
          f"mean_success={best['mean_success']:.2f}  "
          f"maps_solved={best['n_maps_solved']}/{len(SCENARIOS)}")

    print(f"\n  validating at {args.validate_trials} trials/map ...")
    val = evaluate_gains(best['k_rep'], best['d_influence'], cfg,
                         args.validate_trials, args.master_seed)
    baseline_default = evaluate_gains(2.2, 2.2, cfg, args.validate_trials,
                                      args.master_seed)
    print(f"  {'map':<10} {'tuned succ':>11} {'tuned E':>9} {'untuned succ':>13} {'untuned E':>10}")
    for scen in SCENARIOS:
        t, u = val['per_map'][scen], baseline_default['per_map'][scen]
        print(f"  {scen:<10} {t['success_rate']:11.2f} {t['energy_mean']:9.2f} "
              f"{u['success_rate']:13.2f} {u['energy_mean']:10.2f}")

    payload = {
        'search_box': {'k_rep': list(K_REP_GRID),
                       'd_influence': list(D_INFLUENCE_GRID),
                       'n_points': len(grid)},
        'selection_rule': 'max mean success across maps; ties broken by lowest '
                          'mean energy over solved maps only',
        'sweep_n_trials': args.n_trials,
        'validate_n_trials': args.validate_trials,
        'master_seed': args.master_seed,
        'rows': rows,
        'winner': {'k_rep': best['k_rep'], 'd_influence': best['d_influence']},
        'winner_validated': val,
        'untuned_pre_c2_defaults_validated': baseline_default,
        'note': 'Re-tuned because the geometry repair added r_body to the APF '
                'clearance; the pre-repair gains (2.2, 2.2) were fitted to the old '
                'centre-to-surface convention. Recorded so the tuning-parity table '
                'can declare both search boxes.',
    }
    out = args.out
    if out is None and args.run_id:
        from utils.paths import run_dir as _run_dir
        out = os.path.join(_run_dir(args.run_id, 'tuning'), 'apf_gain_sweep.json')
    if out:
        from utils.safe_write import safe_write_json
        safe_write_json(out, payload, force=args.force)
        print(f"\n  written: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
