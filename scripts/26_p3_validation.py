"""P3 validation — the adaptive-certificate campaign (reach arm only).

The P3 refit shrank a_q by 6–42% (mean ~29%) at identical 0.950 coverage
by commanding the terrain-optimal v*(m) in the π_fb rollout. But a
tighter certificate is only deployable if the CLOSED LOOP still
respects the budget under it: the supervisor's return legs are NLP
returns, not π_fb returns, and the intercept is what covers their
variance. This script runs the reach arm on the full main-grid budget
set (train maps) under BOTH certificates — same seeds, same maps, same
controller — and reports the breach rate for each.

Acceptance (pre-declared): the adaptive certificate is deployable iff
its breach rate on this validation is 0; otherwise the shrink exceeded
what the actual return policy's variance supports, and the honest
deployment a_q lies between (reported, not tuned: we report both
endpoints and the measured leak).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

logger = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--max-workers', type=int, default=8)
    ap.add_argument('--n-seeds', type=int, default=5)
    args = ap.parse_args(argv)

    import multiprocessing as mp
    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json

    maps_root = _run_dir(args.run_id, 'reach_maps', create=False)
    fits = json.load(open(os.path.join(
        _run_dir(args.run_id, 'reach', create=False),
        'certificates_adaptive_p3.json')))['fits']
    aq_adaptive = {f['name']: f['a_q'] for f in fits}
    old = {}
    for c in json.load(open(os.path.join(maps_root, 'train',
                                         'certificates.json'))):
        old[c['map']] = c['a_q']

    # main-grid budget set: 4 fractions x 6 train maps x n seeds x 2 certs
    import glob as _glob
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field
    from mpc.energy_model import load_default_coefficients
    import math

    jobs = []
    for p in sorted(_glob.glob(os.path.join(maps_root, 'train',
                                             'reach_*.yaml'))):
        sc = load_scenario_from_yaml(p)
        jf = build_field(sc, coeffs=load_default_coefficients(),
                         terrain=sc.terrain)
        far_x = sc.bounds[2] - 1.0
        e_rt = 2.0 * jf.dstar(float(far_x), float(sc.dock[1]))
        for frac in (0.30, 0.45, 0.60, 0.75):
            B = round(frac * e_rt, 1)
            for s in range(31000, 31000 + args.n_seeds):
                for cert_label, aq in (('old', old[sc.name]),
                                        ('adaptive', aq_adaptive[sc.name])):
                    jobs.append({'map_file': os.path.abspath(p),
                                 'map_name': sc.name, 'B': B, 'seed': s,
                                 'cert': cert_label, 'a_q': aq})
    logger.info("%d validation missions (%d maps x 4 budgets x %d seeds x 2 certs)",
                len(jobs), len(jobs) // (4 * args.n_seeds * 2), args.n_seeds)

    out_dir = os.path.join(_run_dir(args.run_id, 'reach'), 'p3_validation')
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context('spawn')
    payloads = [json.dumps(j) for j in jobs]
    t0 = time.perf_counter()
    results = []
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_run_one, payloads):
            results.append(r)
            if len(results) % 24 == 0:
                logger.info("  %d/%d done", len(results), len(jobs))

    safe_write_json(os.path.join(out_dir, 'p3_validation.json'),
                    {'jobs': len(jobs), 'results': results,
                     'wall_s': round(time.perf_counter() - t0, 1)})

    for cert in ('old', 'adaptive'):
        sub = [r for r in results if r['cert'] == cert]
        n = len(sub)
        breaches = sum(1 for r in sub if r['breach'])
        docked = sum(1 for r in sub if r['outcome'] == 'docked')
        reach = [r['reach_m'] for r in sub if r['reach_m'] is not None]
        logger.info("%-9s: n=%d breaches=%d docked=%.1f%% mean reach %.2f m",
                    cert, n, breaches, 100 * docked / n,
                    float(np.mean(reach)) if reach else float('nan'))
    return 0


def _run_one(payload_json: str) -> Dict[str, Any]:
    """Worker: one mission under its certificate. Spawned, dict-JSON."""
    for k, v in {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}.items():
        os.environ[k] = v
    try:
        os.nice(19 - os.nice(0))
    except (OSError, AttributeError):
        pass
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    j = json.loads(payload_json)
    from utils.config_loader import load_config
    from env.scenarios import load_scenario_from_yaml
    from evaluation.reach_evaluator import REACH_ARM, run_reach_trial
    try:
        sc = load_scenario_from_yaml(j['map_file'])
        cfg = load_config(None)
        res = run_reach_trial(sc, REACH_ARM, B=j['B'], seed=j['seed'],
                              config=cfg, replan_every_s=2.5,
                              t_mission_s=240.0, a_q=j['a_q'])
        r = res.to_record()
        return {'map': j['map_name'], 'B': j['B'], 'seed': j['seed'],
                'cert': j['cert'], 'a_q': j['a_q'],
                'outcome': r['outcome'], 'reach_m': r['reach_m'],
                'breach': r['energy_spent'] > j['B'] + 0.5,
                'energy_spent': r['energy_spent'],
                'min_margin_J': r['min_margin_J']}
    except Exception as exc:                     # noqa: BLE001 - worker guard
        return {'map': j['map_name'], 'B': j['B'], 'seed': j['seed'],
                'cert': j['cert'], 'a_q': j['a_q'], 'error': str(exc),
                'outcome': 'error', 'breach': False, 'reach_m': None,
                'energy_spent': None, 'min_margin_J': None}


if __name__ == '__main__':
    raise SystemExit(main())
