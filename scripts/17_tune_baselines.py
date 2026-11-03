#!/usr/bin/env python3
"""Re-tune BOTH comparator arms against the corrected geometry, in parallel.

Why this exists
---------------
The paper was rejected because its margin came from a crippled comparator. Every
geometry fix we make re-cripples the comparators unless their gains are re-fitted,
because every gain in `evaluation/baselines.py` and `mpc/standard_mpc_casadi.py`
was tuned against the geometry that preceded the fix. This has now happened twice:

  1. The geometry repair added ``r_body`` to the APF clearance. The pre-repair
     gains (2.2, 2.2) were
     fitted centre-to-surface, so inflating every obstacle by 0.25 m turned a
     working s_curve slalom into a guaranteed crash (1.00 -> 0.00).
  2. Activating the wall terms -- the APF had none at all, and the standard MPC's
     was inert because nobody passed ``wall_bounds`` -- broke both arms again
     (APF 0.00/0.00/0.60, std MPC 1.00/0.00/0.00), because ``w_obs`` and the APF
     gains were fitted with no wall force in the field.

Fix (1) was found by measuring: all 250 chicane APF failures ended at
y = 3.750-3.796 in an arena bounded at y = 4.0, attributed 'wall' by
``EP2Evaluator._collision_cause``. So "chicane 0.00 is intrinsic to reactive
control" was never true; it was our missing term. With a wall term at an untuned
0.60 m influence the APF already reaches 0.60 on chicane.

So the gains are not free parameters to be set by hand. They are fitted here, over
a declared box, by a rule fixed before the data is seen, and the whole grid is
recorded so a reviewer can see the box and the rule rather than a chosen point.

Selection rule, fixed in advance
--------------------------------
1. Maximise mean success rate across all three maps.
2. Break ties on lowest mean energy over the maps where the arm reaches 100%.
Energy never breaks a tie between different success rates: energy on a crashed run
is a truncated partial sum, not a smaller cost.

Parallel over grid points: one process per point, spawn, single-threaded BLAS,
nice 19. Energy and success are load-independent, so the fan-out does not touch
the quantities being fitted; solve times from this script are NOT timing data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCENARIOS = ('corridor', 's_curve', 'chicane')
FULL = 0.999

SINGLE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1', 'MPLBACKEND': 'Agg',
}

# ---------------------------------------------------------------------------
# Declared search boxes. Widened deliberately: a box centred tightly on the old
# value assumes the old value was nearly right, which is the assumption that just
# failed twice.
# ---------------------------------------------------------------------------

#: APF: repulsion gain x obstacle influence x WALL influence.
#: Wall influence is separate from obstacle influence and much smaller -- a wall is
#: an infinite plane, so the same radius integrates to a far larger force. The upper
#: end (1.2 m) is already most of the 1.00 m usable band on chicane.
#: k_rep runs down to 0.02 because the first sweep's winner landed on the box's
#: lower edge (0.25 of 0.25..4.0). A winner on a boundary is not a fitted optimum,
#: it is evidence the box was placed wrong, and reporting it as "tuned" would be
#: the same sin as reporting an untuned comparator. Extending downward also tests a
#: real hypothesis: that with the wall term supplying the containment, the obstacle
#: repulsion the old gains carried was mostly overkill.
APF_BOX = {
    'k_rep':            (0.02, 0.05, 0.10, 0.25, 0.5, 1.0, 1.5, 2.2, 3.0, 4.0),
    'd_influence':      (1.0, 1.5, 2.2, 3.0, 4.0),
    'd_wall_influence': (0.20, 0.30, 0.45, 0.60, 0.80, 1.00, 1.20),
}

#: Standard MPC: the obstacle/wall weight and the safety margin. w_obs is swept as
#: a MULTIPLIER on the existing per-scenario weights so the relative tuning between
#: maps (420/540/560) is preserved and only its scale is fitted. d_safe is swept
#: over EP2's own evolved range [0.26, 0.38] so neither arm is held to a stricter
#: margin than the other -- the evolved values are 0.3225/0.2922/0.3252.
#: ``w_goal_scale`` is in the box because leaving it out crippled the arm, and the
#: reason is a property of the benchmark that the rejected submission never
#: surfaced: EVERY goal on EVERY map sits INSIDE the wall margin. All three goals
#: are at x = 19.5 in a 20 m arena, i.e. 0.50 m from the right wall, while a
#: d_safe of 0.26-0.38 plus r_body = 0.25 demands 0.51-0.63 m of centre clearance.
#: So the terminal approach is a soft-constraint trade-off for every arm, EP2-MPC
#: included (its evolved wall_margin = d_safe = 0.2922-0.3252 needs 0.54-0.58 m).
#:
#: Measured: at d_safe = 0.30 the standard MPC threads the whole s_curve without a
#: single collision and then parks at (19.13, 0.73) for the full 60 s timeout --
#: 0.44 m short, held off by its own wall penalty, because w_goal = 11-14 against
#: w_obs = 420-560 means clearance outweighs the goal by ~40x. At d_safe = 0.26 it
#: reaches the 0.2 m tolerance and scores 137.4 J, indistinguishable from the
#: 137.53 J it had with the walls inert. Nothing about that is a lookahead failure
#: or a bad w_obs; it is a weight balance EP2 got from evolution and the comparator
#: was never given. Sweeping it is what "every arm gets its best honest shot" means.
STD_MPC_BOX = {
    'w_obs_scale':  (0.25, 0.5, 1.0, 2.0),
    'd_safe':       (0.26, 0.28, 0.30, 0.34, 0.38),
    'w_goal_scale': (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
}


@dataclass
class Point:
    """One grid point: the arm's parameters, plus what to measure it with."""
    arm: str
    params: Dict[str, float]
    n_trials: int
    master_seeds: Tuple[int, ...]
    config_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Worker: evaluate one grid point on all three maps
# ---------------------------------------------------------------------------

def eval_point(payload: dict) -> dict:
    """Run every map at one parameter setting. Runs in a spawned child."""
    os.environ.update(SINGLE_THREAD_ENV)
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass
    logging.basicConfig(level=logging.ERROR)

    from utils.config_loader import load_config
    from env.scenarios import get_scenario

    arm = payload['arm']
    params = payload['params']
    n_trials = payload['n_trials']
    seeds = payload['master_seeds']

    cfg = load_config(payload.get('config_path'))
    cfg.setdefault('evaluation', {})['progress_every_trials'] = 0
    if not cfg.get('robot'):
        raise SystemExit("config['robot'] is required")

    t0 = time.time()
    per_map: Dict[str, Any] = {}
    try:
        for scen in SCENARIOS:
            succ, energies = [], []
            for ms in seeds:
                if arm == 'apf':
                    from evaluation.baselines import APFBaseline, run_baseline
                    ctrl = APFBaseline(
                        robot=cfg['robot'], dt=cfg['mpc'].get('dt', 0.1),
                        k_rep=params['k_rep'],
                        d_influence=params['d_influence'],
                        d_wall_influence=params['d_wall_influence'])
                    res = run_baseline(ctrl, get_scenario(scen),
                                       n_trials=n_trials, config=cfg,
                                       master_seed=ms)
                else:
                    from evaluation.baselines import (
                        run_standard_mpc_casadi_baseline,
                        SCENARIO_STD_MPC_WEIGHTS)
                    base_w = SCENARIO_STD_MPC_WEIGHTS[scen]
                    res = run_standard_mpc_casadi_baseline(
                        scen, n_trials=n_trials, config=cfg, master_seed=ms,
                        weight_overrides={
                            'w_obs': base_w['w_obs'] * params['w_obs_scale'],
                            'd_safe': params['d_safe'],
                            'w_goal': base_w['w_goal'] * params['w_goal_scale'],
                        })
                succ.append(float(res.success_rate))
                energies.append(float(res.energy_mean))
            per_map[scen] = {'success_rate': float(np.mean(succ)),
                             'energy_mean': float(np.mean(energies))}
    except Exception as exc:            # a bad point must not kill the sweep
        return {**payload, 'ok': False, 'error': f'{type(exc).__name__}: {exc}',
                'wall_s': time.time() - t0}

    mean_succ = float(np.mean([m['success_rate'] for m in per_map.values()]))
    solved = [m for m in per_map.values() if m['success_rate'] >= FULL]
    return {
        **payload, 'ok': True, 'per_map': per_map,
        'mean_success': mean_succ,
        'n_maps_solved': len(solved),
        'energy_on_solved': (float(np.mean([m['energy_mean'] for m in solved]))
                             if solved else float('inf')),
        'wall_s': time.time() - t0,
    }


def select(rows: List[dict]) -> Optional[dict]:
    """The rule, applied exactly as declared in the module docstring."""
    good = [r for r in rows if r.get('ok')]
    if not good:
        return None
    return max(good, key=lambda r: (r['mean_success'], -r['energy_on_solved']))


def _grid(arm: str) -> List[Dict[str, float]]:
    box = APF_BOX if arm == 'apf' else STD_MPC_BOX
    keys = list(box)
    return [dict(zip(keys, combo)) for combo in itertools.product(*box.values())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', choices=('apf', 'std_mpc'), required=True)
    ap.add_argument('--n-trials', type=int, default=20,
                    help='trials per map per master seed during the sweep')
    ap.add_argument('--sweep-seeds', type=int, default=2,
                    help='master seeds in the sweep; >1 so the fit is not to one '
                         'seed realisation')
    ap.add_argument('--validate-trials', type=int, default=50)
    ap.add_argument('--validate-seeds', type=int, default=5)
    ap.add_argument('--max-workers', type=int, default=64)
    ap.add_argument('--config', default=None)
    ap.add_argument('--run-id', dest='run_id', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    import multiprocessing as mp
    ctx = mp.get_context('spawn')       # never fork: a forked child inherits RNG

    grid = _grid(args.arm)
    sweep_seeds = tuple(1000 * (i + 1) for i in range(args.sweep_seeds))
    val_seeds = tuple(1000 * (i + 1) for i in range(args.validate_seeds))
    box = APF_BOX if args.arm == 'apf' else STD_MPC_BOX

    n_trials_total = (len(grid) * len(SCENARIOS) * args.n_trials
                      * len(sweep_seeds))
    print(f"tuning {args.arm}: {len(grid)} grid points x {len(SCENARIOS)} maps x "
          f"{len(sweep_seeds)} seeds x {args.n_trials} trials "
          f"= {n_trials_total} trials")
    for k, v in box.items():
        print(f"  {k:<18}: {v}")
    print(f"  workers           : {args.max_workers}")
    print(f"  sweep seeds       : {sweep_seeds}")
    print()

    payloads = [asdict(Point(arm=args.arm, params=p, n_trials=args.n_trials,
                             master_seeds=sweep_seeds,
                             config_path=args.config)) for p in grid]

    t0 = time.time()
    rows: List[dict] = []
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(eval_point, payloads):
            rows.append(r)
            pm = r.get('per_map') or {}
            if r.get('ok'):
                e = r['energy_on_solved']
                # Built outside the f-string: nested same-type quotes inside an
                # f-string are a SyntaxError before Python 3.12 and this is 3.11.
                succ_str = '/'.join('%.2f' % pm[s]['success_rate']
                                    for s in SCENARIOS)
                par_str = ' '.join('%s=%g' % (k, v)
                                   for k, v in r['params'].items())
                e_str = ('  E=%.2f' % e) if np.isfinite(e) else '  E=-'
                print(f"  [{len(rows):3d}/{len(grid)}] {par_str}  {succ_str}"
                      f"  mean={r['mean_success']:.3f}{e_str}")
            else:
                print(f"  [{len(rows):3d}/{len(grid)}] FAILED {r.get('error')}")
    sweep_s = time.time() - t0

    best = select(rows)
    if best is None:
        print("every grid point failed")
        return 1
    print(f"\nsweep done in {sweep_s:.0f}s")
    print(f"WINNER  {best['params']}  mean_success={best['mean_success']:.3f}  "
          f"maps_solved={best['n_maps_solved']}/{len(SCENARIOS)}")

    ties = [r for r in rows if r.get('ok')
            and abs(r['mean_success'] - best['mean_success']) < 1e-9]
    print(f"  {len(ties)} of {len(rows)} points tie on success; winner has the "
          f"lowest energy over solved maps")

    print(f"\nvalidating at {args.validate_trials} trials x "
          f"{len(val_seeds)} seeds ...")
    val = eval_point({**asdict(Point(arm=args.arm, params=best['params'],
                                     n_trials=args.validate_trials,
                                     master_seeds=val_seeds,
                                     config_path=args.config))})
    for scen in SCENARIOS:
        m = val['per_map'][scen]
        print(f"  {scen:<9} succ={m['success_rate']:.2f}  E={m['energy_mean']:8.2f}")

    payload = {
        'arm': args.arm,
        'search_box': {k: list(v) for k, v in box.items()},
        'n_grid_points': len(grid),
        'selection_rule': ('1. max mean success across all three maps. '
                           '2. ties broken by lowest mean energy over the maps '
                           'reaching 100% only. Energy never breaks a tie between '
                           'different success rates.'),
        'sweep': {'n_trials': args.n_trials, 'master_seeds': list(sweep_seeds),
                  'wall_s': sweep_s, 'max_workers': args.max_workers},
        'validation': {'n_trials': args.validate_trials,
                       'master_seeds': list(val_seeds)},
        'rows': rows,
        'winner': best['params'],
        'winner_sweep': {k: best[k] for k in
                         ('per_map', 'mean_success', 'n_maps_solved',
                          'energy_on_solved')},
        'winner_validated': val,
        'n_tied_on_success': len(ties),
        'why': ('gains fitted with the wall terms ACTIVE. Every previous value was '
                'fitted with no wall force in the field: the APF had no wall term '
                'at all and the standard MPC never received wall_bounds, while '
                'EP2-MPC has carried four live wall terms since C1.'),
    }
    out = args.out
    if out is None and args.run_id:
        from utils.paths import run_dir
        out = os.path.join(run_dir(args.run_id, 'tuning'),
                           f'{args.arm}_gain_sweep.json')
    if out:
        from utils.safe_write import safe_write_json
        safe_write_json(out, payload, force=args.force)
        print(f"\nwritten: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
