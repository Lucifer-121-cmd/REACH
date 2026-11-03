"""Step-change mechanism experiment: const_weight at price 50 (2026-10-04).

Parallel re-run of the step-change experiment with const_weight at the
sweep-selected price 50 (matching the campaign grid after Defect #7).

Usage:
    python scripts/35_stepchange_const_price50.py --run-id <id> --budget 280
    python scripts/35_stepchange_const_price50.py --run-id <id> --budget 320
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Step-change map parameters (matching script 22)
STEP_X = 15.0
A_Q = 53.543
BUDGET = 280.0
SEED0 = 41000


def build_step_map():
    """Same map as script 22's build_step_map."""
    from env.scenarios import ScenarioConfig
    from env.terrain import TerrainField, TerrainRegion
    terrain = TerrainField(
        [TerrainRegion(STEP_X, 0.0, 30.0, 6.0, m1=3.0, m4=3.0)],
        bounds=(0.0, 0.0, 30.0, 6.0))
    return ScenarioConfig(
        name='reach_stepchange',
        start=np.array([1.0, 3.0, 0.0]),
        goal=np.array([1.0, 3.0]),
        obstacles=[(8.0, 3.0, 0.8), (13.0, 4.9, 0.8)],
        bounds=(0.0, 0.0, 30.0, 6.0),
        terrain=terrain,
        description='step-change map: terrain m=3 beyond x=15',
    )


def _run_one(task):
    """Module-level worker: one step-change mission (picklable)."""
    from evaluation.reach_evaluator import ReachArm, run_reach_trial
    from utils.config_loader import load_config
    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer
    from mpc.reach_mpc import ReachSpec
    from evaluation.reach_evaluator import ReachSupervisor
    from mpc.energy_model import load_default_coefficients

    sc = build_step_map()
    cfg = load_config(None)

    if task['arm'] == 'const_weight':
        arm = ReachArm('const_weight', use_constraint=False,
                       price0=50.0, turnback_rule='energy')
    elif task['arm'] == 'dual':
        arm = ReachArm('dual', use_constraint=False, dual=True,
                       price0=0.0, turnback_rule='energy')
    else:
        arm = ReachArm('reach')

    # Trial result (outcome + summary stats)
    res = run_reach_trial(sc, arm, B=task['B'], seed=task['seed'],
                           config=cfg, replan_every_s=2.5,
                           a_q=task['a_q'])
    rec = res.to_record()

    # Trajectory profile for the speed figure (deterministic re-run
    # with a recording env — same seeds, same result)
    env_cfg = dict(cfg.get('robot', {}))
    env_cfg['dt'] = float(cfg.get('dt', 0.1))
    env_cfg.setdefault('energy_coeffs', load_default_coefficients())
    env = PioneerEnv(env_cfg)
    randomizer = TrialRandomizer()
    perturbed, v0, w0 = randomizer.randomize_trial(sc, seed=task['seed'])
    env.set_scenario(perturbed.obstacles, perturbed.bounds, terrain=sc.terrain)
    env.reset(perturbed.start, lidar_seed=task['seed'] + 1_000_003)
    env._v, env._omega = v0, w0

    spec = ReachSpec(B=task['B'], certificate='affine',
                     a_q=task['a_q'], lam=1.0)
    sup = ReachSupervisor(sc, spec, arm, cfg, N=60, dt_plan=0.5,
                          replan_every_s=2.5)
    dt = float(cfg.get('dt', 0.1))
    xs, vs, margins = [], [], []
    for _ in range(int(240.0 / dt)):
        st = env.get_state_vector()
        x, y, v, E = float(st[0]), float(st[1]), float(st[3]), float(st[5])
        if env.check_collision():
            break
        dj = sup._field.dstar(x, y)
        margin = (task['B'] - E - (task['a_q'] + dj)
                  if dj is not None and math.isfinite(dj) else None)
        margins.append(margin)
        xs.append(x)
        vs.append(v)
        u = sup.command(st, 0.0)
        if u is None:
            break
        env.step(u[0], u[1])

    rec['x_profile'] = xs
    rec['v_profile'] = vs
    rec['margin_profile'] = margins
    rec['max_x'] = max(xs) if xs else 0.0
    rec['min_margin'] = (min(m for m in margins if m is not None)
                         if any(m is not None for m in margins) else None)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--budget', type=float, default=BUDGET)
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--max-workers', type=int, default=10)
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json

    out_dir = os.path.join(_run_dir(args.run_id, 'reach'),
                           f'stepchange_const50_B{int(args.budget)}')
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for arm in ('reach', 'dual', 'const_weight'):
        for s in range(args.n_seeds):
            tasks.append({'arm': arm, 'seed': SEED0 + s,
                          'B': args.budget, 'a_q': A_Q})
    print(f'[setup] {len(tasks)} missions, budget={args.budget}')

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done += 1
            logger.info("  [%d/%d] %s seed %d -> %s, max_x=%.1f",
                        done, len(tasks), rec['arm'], rec['seed'],
                        rec['outcome'], rec['max_x'])

    results.sort(key=lambda r: (r['arm'], r['seed']))
    out = os.path.join(out_dir, 'stepchange_results.json')
    safe_write_json(out, {'map': 'reach_stepchange', 'B': args.budget,
                          'step_x': STEP_X, 'a_q': A_Q, 'trials': results})
    print(f'\n[done] {len(results)} missions -> {out} '
          f'({time.perf_counter()-t0:.0f}s)')

    for t in results:
        xs, vs = t['x_profile'], t['v_profile']
        v_step = [v for x, v in zip(xs, vs) if 15 <= x < 17]
        print(f"  {t['arm']:14s} seed {t['seed']}: {t['outcome']:9s} "
              f"max_x={t['max_x']:5.1f} "
              f"v@15-17={np.mean(v_step) if v_step else float('nan'):5.2f} "
              f"min_marg={t['min_margin'] if t['min_margin'] is not None else float('nan'):7.1f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
