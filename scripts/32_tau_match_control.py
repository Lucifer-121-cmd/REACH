"""Matched-τ control: the reach arm on the main grid with the planning
model's motor-lag constants MATCHED to the plant.

Pre-registered (2026-09-22) BEFORE this run: the transient optimism
(build-time τ_v=0.15/τ_ω=0.10 vs the plant's 0.3/0.2)
is absorbed by closed-loop replanning on measured state, so we expect
breaches to remain 0/240 and mean reach to move by at most a few
percent. Either outcome is reportable: negligible shift closes the
question with a number; a material shift means the declared
simplification mattered.

Design: identical to the main grid's reach-arm rows (6 train maps ×
4 budgets × 10 seeds = 240 missions, deployed adaptive certificates,
replan 2.5 s) with ONE difference — the config's mpc block carries
tau_v = 0.3 / tau_omega = 0.2 (configs/control/tau_matched.yaml), so the
NLP's lag integrator matches the plant exactly. The comparison baseline
is the main3 reach rows: 0/240 breaches, 10.0% docked, 9.43 m.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys

# Single-threaded BLAS/OMP, set BEFORE the first numpy/casadi import:
# OpenBLAS/MUMPS read the environment at import time, and results must
# land on the pinned campaign lineage -- an ulp-scale numeric change
# re-routes the non-smooth NLP into different basins. Same idiom
# as scripts/30/31.
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONTROL_CONFIG = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'configs', 'control', 'tau_matched.yaml')


def _run_one(payload_json: str) -> dict:
    """Worker: one mission. Spawned, dict-JSON; pinning happened at import."""
    try:
        os.nice(19 - os.nice(0))
    except (OSError, AttributeError):
        pass
    j = json.loads(payload_json)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.config_loader import load_config
    from env.scenarios import load_scenario_from_yaml
    from evaluation.reach_evaluator import ARMS, run_reach_trial, \
        reach_trial_sink, emit_reach_trial
    try:
        sc = load_scenario_from_yaml(j['map_file'])
        cfg = load_config(CONTROL_CONFIG)
        res = run_reach_trial(sc, ARMS['reach'], B=j['B'], seed=j['seed'],
                              config=cfg, replan_every_s=2.5,
                              t_mission_s=240.0, a_q=j['a_q'])
        sink = reach_trial_sink(j['out_dir'], 'tau_match_control')
        emit_reach_trial(sink, res)
        rec = res.to_record()
        return {'ok': True, 'map': j['map_name'], 'B': j['B'],
                'seed': j['seed'], 'outcome': rec['outcome'],
                'breach': rec['energy_spent'] > j['B'] + 0.5,
                'reach_m': rec['reach_m'], 'energy_spent': rec['energy_spent'],
                'docked': rec['outcome'] == 'docked'}
    except Exception as exc:                     # noqa: BLE001 - worker guard
        return {'ok': False, 'map': j['map_name'], 'B': j['B'],
                'seed': j['seed'], 'error': f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    import argparse
    import time
    import multiprocessing as mp
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--maps-root', required=True,
                    help='deployed adaptive maps root (input, read-only)')
    ap.add_argument('--max-workers', type=int, default=150)
    ap.add_argument('--n-seeds', type=int, default=10)
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field
    from mpc.energy_model import load_default_coefficients
    import glob as _glob

    maps_root = args.maps_root
    out_dir = os.path.join(_run_dir(args.run_id, 'reach'), 'tau_match_control')
    os.makedirs(out_dir, exist_ok=True)

    # Same E_rt definition as scripts/31 (verified shared 2026-09-22)
    jobs = []
    for p in sorted(_glob.glob(os.path.join(maps_root, 'train',
                                            'reach_*.yaml'))):
        sc = load_scenario_from_yaml(p)
        certs = {c['map']: float(c['a_q']) for c in
                 json.load(open(os.path.join(maps_root, 'train',
                                             'certificates.json')))}
        a_q = certs[sc.name]
        jf = build_field(sc, coeffs=load_default_coefficients(),
                         terrain=sc.terrain)
        far_x = sc.bounds[2] - 1.0
        e_rt = 2.0 * jf.dstar(float(far_x), float(sc.dock[1]))
        for frac in (0.30, 0.45, 0.60, 0.75):
            B = round(frac * e_rt, 1)
            for s in range(31000, 31000 + args.n_seeds):
                jobs.append({'map_file': os.path.abspath(p),
                             'map_name': sc.name, 'B': B, 'seed': s,
                             'a_q': a_q, 'out_dir': out_dir})
    logger.info("%d missions (6 train maps x 4 budgets x %d seeds, reach "
                "arm, tau matched 0.3/0.2)", len(jobs), args.n_seeds)

    ctx = mp.get_context('spawn')
    t0 = time.perf_counter()
    results = []
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_run_one, [json.dumps(j) for j in jobs]):
            results.append(r)
            if len(results) % 40 == 0:
                logger.info("  %d/%d done (%.0fs)", len(results), len(jobs),
                            time.perf_counter() - t0)

    safe_write_json(os.path.join(out_dir, 'tau_match_control.json'),
                    {'jobs': len(jobs), 'results': results,
                     'control_config': CONTROL_CONFIG,
                     'wall_s': round(time.perf_counter() - t0, 1),
                     'blas_threads': {k: os.environ.get(k) for k in
                                      ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                       'OPENBLAS_NUM_THREADS',
                                       'NUMEXPR_NUM_THREADS')}})

    ok = [r for r in results if r.get('ok')]
    n = len(ok)
    br = sum(1 for r in ok if r['breach'])
    dk = sum(1 for r in ok if r['docked'])
    mr = float(np.mean([r['reach_m'] for r in ok])) if ok else 0.0
    print(f"\nMATCHED-TAU CONTROL (reach arm, main grid):")
    print(f"  n={n}/{len(jobs)}  breaches={br}  docked={dk} ({100*dk/max(n,1):.1f}%)"
          f"  mean reach={mr:.2f} m")
    print(f"  baseline (main3 reach rows): 0/240 breaches, 10.0% docked, 9.43 m")
    print(f"written: {out_dir}/tau_match_control.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
