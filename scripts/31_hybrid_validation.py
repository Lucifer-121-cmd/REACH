"""The hybrid arm validation: reach (NLP return) vs reach_hybrid
(geodesic return), same seeds, same maps, same budgets.

Motivation (measured, 2026-09-16): the threshold baseline
(scripts/29) posts 0/240 breaches with 90%
docking and 10.33 m reach — dominating the constrained arm (0/240, 10%,
9.43 m) — because its pi_fb-on-geodesic return leg docks while the NLP
return strands near-dock. The hybrid keeps the constrained NLP
outbound (the contribution: the terminal reserve row on the fitted
certificate) and hands the return leg to the same tracker the
certificate model assumes. Acceptance (pre-declared): the hybrid is
the better deployment iff it holds 0 breaches AND materially improves
docking/reach over the NLP-return arm; if docking does not improve,
the return leg was not the binding deficiency and the honest
deployment remains the NLP arm (reported, not tuned).

Stages: main grid (6 maps x 4 budgets x 10 seeds x 2 arms = 480),
mismatch +15% (6 x 1 x 10 x 2 = 120), held-out (3 x 4 x 10 x 2 = 240).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

# Single-threaded BLAS/OMP, set BEFORE the first numpy/casadi import: with
# the 'spawn' pool each worker re-imports this module, so module scope is
# the only place that lands before OpenBLAS/MUMPS read the environment.
# The first hybrid launch (2026-09-16) set these inside _run_one -- AFTER
# numpy had already been imported at module scope -- so every worker ran
# multi-threaded BLAS and the realization drifted from the pinned campaign
# lineage (its reach-arm rows matched the pinned campaign on only 48/60
# outcomes, median |dSpent| 0.15 J). Pin at import time, as scripts/21
# has always done. One regime, one lineage.
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

logger = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOISE = (0.05, 0.10)


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--max-workers', type=int, default=150)
    ap.add_argument('--stage', default='all',
                    choices=('all', 'heldout_mismatch'),
                    help=("'all' = the documented 840-mission validation "
                          "(main + mismatch + heldout); 'heldout_mismatch' "
                          "= the 2026-09-22 scope-closing addition: the "
                          "controlled pair on the 3 HELD-OUT maps at "
                          "+15% (3 maps x 0.60-E_rt x 10 seeds x 2 arms "
                          "= 60 missions). NOTE: the sink is a single "
                          "hybrid_validation.json written per run-id -- "
                          "run the new stage under a FRESH run-id, never "
                          "the deployed one."))
    args = ap.parse_args(argv)

    import multiprocessing as mp
    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json

    maps_root = _run_dir(args.run_id, 'reach_maps_adaptive', create=False)
    import glob as _glob
    import math
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field
    from mpc.energy_model import load_default_coefficients

    def map_entries(split):
        # The DEPLOYED per-map certificates (scripts/21's pattern): the
        # campaign must not run on the 24 J legacy placeholder, and
        # neither may this comparison -- the hybrid's margin IS the
        # certificate, so the runs must use the fitted values and the
        # rows must carry them.
        cert_path = os.path.join(maps_root, split, 'certificates.json')
        if not os.path.isfile(cert_path):
            raise SystemExit(
                f"missing {cert_path}: run scripts/20_reach_certificate_terrain"
                " first -- this comparison must run on the fitted adaptive"
                " certificates, not the 24 J placeholder")
        certs = {c['map']: c for c in json.load(open(cert_path))}
        entries = []
        for p in sorted(_glob.glob(os.path.join(maps_root, split,
                                                'reach_*.yaml'))):
            sc = load_scenario_from_yaml(p)
            if sc.name not in certs:
                raise SystemExit(f"{sc.name}: no certificate fit in "
                                 f"{cert_path}")
            jf = build_field(sc, coeffs=load_default_coefficients(),
                             terrain=sc.terrain)
            far_x = sc.bounds[2] - 1.0
            e_rt = 2.0 * jf.dstar(float(far_x), float(sc.dock[1]))
            entries.append((os.path.abspath(p), sc.name, e_rt,
                            float(certs[sc.name]['a_q'])))
        return entries

    jobs = []
    for p, name, e_rt, a_q in map_entries('train'):
        for frac in (0.30, 0.45, 0.60, 0.75):
            B = round(frac * e_rt, 1)
            for s in range(31000, 31010):
                for arm in ('reach', 'reach_hybrid'):
                    jobs.append({'map_file': p, 'map_name': name, 'B': B,
                                 'seed': s, 'arm': arm, 'stage': 'main',
                                 'power_mismatch': 0.0, 'a_q': a_q})
        B = round(0.60 * e_rt, 1)
        for s in range(31000, 31010):
            for arm in ('reach', 'reach_hybrid'):
                jobs.append({'map_file': p, 'map_name': name, 'B': B,
                             'seed': s, 'arm': arm, 'stage': 'mismatch',
                             'power_mismatch': 0.15, 'a_q': a_q})
    for p, name, e_rt, a_q in map_entries('heldout'):
        for frac in (0.30, 0.45, 0.60, 0.75):
            B = round(frac * e_rt, 1)
            for s in range(31000, 31010):
                for arm in ('reach', 'reach_hybrid'):
                    jobs.append({'map_file': p, 'map_name': name, 'B': B,
                                 'seed': s, 'arm': arm, 'stage': 'heldout',
                                 'power_mismatch': 0.0, 'a_q': a_q})
        if args.stage in ('all', 'heldout_mismatch'):
            B = round(0.60 * e_rt, 1)
            for s in range(31000, 31010):
                for arm in ('reach', 'reach_hybrid'):
                    jobs.append({'map_file': p, 'map_name': name, 'B': B,
                                 'seed': s, 'arm': arm,
                                 'stage': 'heldout_mismatch',
                                 'power_mismatch': 0.15, 'a_q': a_q})
    if args.stage == 'heldout_mismatch':
        jobs = [j for j in jobs if j['stage'] == 'heldout_mismatch']
    logger.info("%d missions (main 480 + mismatch 120 + heldout 240 "
                "+ heldout_mismatch 60; stage=%s)", len(jobs), args.stage)

    out_dir = os.path.join(_run_dir(args.run_id, 'reach'), 'hybrid_validation')
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context('spawn')
    payloads = [json.dumps(j) for j in jobs]
    t0 = time.perf_counter()
    results = []
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_run_one, payloads):
            results.append(r)
            if len(results) % 120 == 0:
                logger.info("  %d/%d done (%.0fs)", len(results), len(jobs),
                            time.perf_counter() - t0)

    safe_write_json(os.path.join(out_dir, 'hybrid_validation.json'),
                    {'jobs': len(jobs), 'results': results,
                     'wall_s': round(time.perf_counter() - t0, 1),
                     'blas_threads': {k: os.environ.get(k) for k in
                                      ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                       'OPENBLAS_NUM_THREADS',
                                       'NUMEXPR_NUM_THREADS')}})

    print(f"\n{'stage':<9} {'arm':<13} {'n':>4} {'breach':>8} {'dock%':>6} "
          f"{'reach':>6}")
    for stage in ('main', 'mismatch', 'heldout', 'heldout_mismatch'):
        for arm in ('reach', 'reach_hybrid'):
            sel = [r for r in results if r['stage'] == stage
                   and r['arm'] == arm and not r.get('error')]
            if not sel:
                continue
            br = sum(1 for r in sel if r['breach'])
            dk = 100 * sum(1 for r in sel if r['outcome'] == 'docked') / len(sel)
            mr = float(np.mean([r['reach_m'] for r in sel
                                if r['reach_m'] is not None]))
            print(f"{stage:<9} {arm:<13} {len(sel):>4} {br:>5}/{len(sel):<3} "
                  f"{dk:>6.1f} {mr:>6.2f}")
    return 0


def _run_one(payload_json: str) -> Dict[str, Any]:
    """Worker: one mission under its arm. Spawned, dict-JSON.

    Thread pinning happens at MODULE import (top of this file), before
    numpy loads; nothing to do here except descend to idle priority.
    """
    try:
        os.nice(19 - os.nice(0))
    except (OSError, AttributeError):
        pass
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    j = json.loads(payload_json)
    from utils.config_loader import load_config
    from env.scenarios import load_scenario_from_yaml
    from evaluation.reach_evaluator import ARMS, run_reach_trial
    try:
        sc = load_scenario_from_yaml(j['map_file'])
        cfg = load_config(None)
        res = run_reach_trial(sc, ARMS[j['arm']], B=j['B'], seed=j['seed'],
                              config=cfg, replan_every_s=2.5,
                              t_mission_s=240.0, a_q=j['a_q'],
                              power_mismatch=j['power_mismatch'],
                              actuation_noise=(NOISE if j['power_mismatch']
                                               else None))
        r = res.to_record()
        return {'stage': j['stage'], 'arm': j['arm'], 'map': j['map_name'],
                'B': j['B'], 'seed': j['seed'], 'a_q': j['a_q'],
                'outcome': r['outcome'], 'reach_m': r['reach_m'],
                'breach': r['energy_spent'] > j['B'] + 0.5,
                'energy_spent': r['energy_spent'],
                'turnback_time': r['turnback_time'],
                'min_margin_J': r['min_margin_J'],
                'replans': r['replans']}
    except Exception as exc:                     # noqa: BLE001 - worker guard
        return {'stage': j['stage'], 'arm': j['arm'], 'map': j['map_name'],
                'B': j['B'], 'seed': j['seed'], 'error': str(exc),
                'outcome': 'error', 'breach': False, 'reach_m': None,
                'energy_spent': None, 'turnback_time': None,
                'min_margin_J': None, 'replans': None}


if __name__ == '__main__':
    raise SystemExit(main())
