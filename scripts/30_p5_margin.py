"""P5 — the mismatch margin: how far can a_q shrink before breaches?

The mechanism behind the mismatch-compliance result: under +15% plant/model
mismatch, compliance held because MEASURED spent energy (closed-loop
feedback) plus the conservative intercept absorbed the ~15-20 J
return-model error. This script quantifies that margin: run the reach
arm at +15% mismatch with actuation noise (σv=0.05, σω=0.10), at the
binding 0.60 E_rt budget, with the adaptive
certificate intercept SCALED by s in [1.0 .. 0.0], and find where
breaches first appear.

Pre-registered reading (2026-09-16, before running): at s=1.0 zero
breaches (the campaign result); as s shrinks, the NLP's terminal row
tightens until the +15% return-model error is no longer covered — the
breach onset marks the certificate's true robustness margin. The
honest number is the onset point, wherever it is.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

# Single-threaded BLAS/OMP, set BEFORE the first numpy/casadi import. With
# the 'spawn' pool each worker re-imports this module, so module scope is
# the only place that lands before OpenBLAS/MUMPS read the environment.
# The first P5 launch (2026-09-16) set these inside _run_one -- AFTER
# numpy had already been imported at module scope -- so every worker ran
# multi-threaded BLAS and the realization drifted from the pinned
# campaign lineage: measured on the scale=1.0 ladder (same 60 missions as
# the +15% campaign arm), 48/60 outcomes matched with median |dSpent|
# 0.15 J and 12 knife-edge flips (spill 0.45 J -> 0.54 J on THE one
# marginal mission, flipping its breach flag). One regime, one lineage:
# pin at import time, as scripts/21 has always done.
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

logger = None
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: the shrink ladder (fractions of the adaptive a_q)
SCALES = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.0)
SEEDS = tuple(range(31000, 31010))
NOISE = (0.05, 0.10)


def main(argv=None) -> int:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    global logger
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--max-workers', type=int, default=60)
    args = ap.parse_args(argv)

    import multiprocessing as mp
    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json

    maps_root = _run_dir(args.run_id, 'reach_maps_adaptive', create=False)
    fits = json.load(open(os.path.join(
        _run_dir(args.run_id, 'reach', create=False),
        'certificates_adaptive_p3.json')))['fits']
    aq = {f['name']: f['a_q'] for f in fits}

    import glob as _glob
    import math
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field
    from mpc.energy_model import load_default_coefficients

    jobs = []
    for p in sorted(_glob.glob(os.path.join(maps_root, 'train',
                                            'reach_*.yaml'))):
        sc = load_scenario_from_yaml(p)
        jf = build_field(sc, coeffs=load_default_coefficients(),
                         terrain=sc.terrain)
        far_x = sc.bounds[2] - 1.0
        e_rt = 2.0 * jf.dstar(float(far_x), float(sc.dock[1]))
        B = round(0.60 * e_rt, 1)
        for s in SCALES:
            for seed in SEEDS:
                # FULL precision: the 2026-09-16/17 runs rounded to 6
                # decimals, and the resulting 9e-8 J intercept shift was
                # enough to flip the one knife-edge mission (spill 0.45 J
                # at full precision, 0.54 J rounded -- the breach flag's
                # 0.5 J tolerance sits inside its ulp basin). Inputs must
                # be bit-equal to the campaign lineage or the experiment
                # silently becomes a different one.
                jobs.append({'map_file': os.path.abspath(p),
                             'map_name': sc.name, 'B': B, 'seed': seed,
                             'scale': s, 'a_q': s * aq[sc.name]})
    logger.info("%d missions (%d maps x %d scales x %d seeds)",
                len(jobs), len(jobs) // (len(SCALES) * len(SEEDS)),
                len(SCALES), len(SEEDS))

    out_dir = os.path.join(_run_dir(args.run_id, 'reach'), 'p5_margin')
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context('spawn')
    payloads = [json.dumps(j) for j in jobs]
    t0 = time.perf_counter()
    results = []
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_run_one, payloads):
            results.append(r)
            if len(results) % 60 == 0:
                logger.info("  %d/%d done", len(results), len(jobs))

    safe_write_json(os.path.join(out_dir, 'p5_margin.json'),
                    {'jobs': len(jobs), 'results': results,
                     'wall_s': round(time.perf_counter() - t0, 1),
                     'blas_threads': {k: os.environ.get(k) for k in
                                      ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                       'OPENBLAS_NUM_THREADS',
                                       'NUMEXPR_NUM_THREADS')}})
    print(f"{'scale':>6} {'a_q mean':>9} {'breaches':>9} {'docked':>8} {'reach':>7}")
    for s in SCALES:
        sel = [r for r in results if r['scale'] == s and not r.get('error')]
        br = sum(1 for r in sel if r['breach'])
        dk = 100 * sum(1 for r in sel if r['outcome'] == 'docked') / len(sel)
        mr = float(np.mean([r['reach_m'] for r in sel
                            if r['reach_m'] is not None]))
        print(f"{s:>6.2f} {s*29.6:>9.1f} {br:>4}/{len(sel):<4} {dk:>7.1f}% {mr:>7.2f}")
    return 0


def _run_one(payload_json: str) -> Dict[str, Any]:
    """Worker: one mission at a scaled intercept. Spawned, dict-JSON.

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
    from evaluation.reach_evaluator import REACH_ARM, run_reach_trial
    try:
        sc = load_scenario_from_yaml(j['map_file'])
        cfg = load_config(None)
        res = run_reach_trial(sc, REACH_ARM, B=j['B'], seed=j['seed'],
                              config=cfg, replan_every_s=2.5,
                              t_mission_s=240.0, a_q=j['a_q'],
                              power_mismatch=0.15,
                              actuation_noise=NOISE)
        r = res.to_record()
        return {'map': j['map_name'], 'B': j['B'], 'seed': j['seed'],
                'scale': j['scale'], 'a_q': j['a_q'],
                'outcome': r['outcome'], 'reach_m': r['reach_m'],
                'breach': r['energy_spent'] > j['B'] + 0.5,
                'energy_spent': r['energy_spent'],
                'min_margin_J': r['min_margin_J']}
    except Exception as exc:                     # noqa: BLE001 - worker guard
        return {'map': j['map_name'], 'B': j['B'], 'seed': j['seed'],
                'scale': j['scale'], 'a_q': j['a_q'], 'error': str(exc),
                'outcome': 'error', 'breach': False, 'reach_m': None,
                'energy_spent': None, 'min_margin_J': None}


if __name__ == '__main__':
    raise SystemExit(main())
