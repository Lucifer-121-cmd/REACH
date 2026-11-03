"""Script 22 — the step-change experiment (anticipation vs reaction).

The mechanism demo: a hand-authored map with a terrain step (m1=m4=3
beyond x=15), budget B=240 J tuned so reaching past the step is possible
ONLY by banking energy before it. The J-field prices the step into
Ebar(x) for every x before it — the constrained arm's reserve check sees
the step coming and its outbound plan should slow BEFORE x=15. The dual
arm's price reacts only after the reserve error appears, i.e. only after
the robot is already paying the step's cost.

Pre-registered predictions:
1. The reach arm's outbound speed profile dips before x=15.
2. The dual arm maintains speed into the step; its price rises only after
   the error; it turns back later and deeper (smaller margin at
   turn-back, or strands in the step).
3. Margin-at-max-x is larger for the reach arm at equal reach.

Design note (post Move-3 fix): both arms share the energy turn-back
rule, so turn-back TIMING is matched; the isolated difference is the
in-loop price ascent (dual) vs the in-plan certificate pricing (reach).
The experiment demonstrates anticipation inside the plan vs reaction on
top of the plan.

Outputs per arm: speed-vs-x profile, margin-vs-x profile, turn-back
point, outcome. Written to <runs>/<id>/reach/stepchange/ with the map.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List

# Single-threaded BLAS/OMP, set BEFORE the first numpy/casadi import:
# OpenBLAS/MUMPS read the environment at import time, and this driver's
# results must land on the pinned campaign lineage -- an ulp-scale
# thread-environment change is a different experiment (measured: a
# 6-decimal intercept rounding departed from the campaign lineage in
# 12/60 outcomes).
# Same idiom as scripts/30. Added 2026-09-20; the 2026-09-12/15
# artefacts produced by this script predate the pin and remain the runs
# of record (their lineage is recorded per artefact).
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Budget. First draft 240 J turned the constrained arm back at 6.6 m --
#: with the fitted a_q=51 J on this map, the certified return cost from
#: the step boundary leaves no room to reach it. 280 J: round-trip
#: floor to x=16 is 214 J + certificate ~51 J = 265 J, so crossing the
#: step boundary is certified-feasible ONLY at an efficient approach;
#: a wasteful one (full speed, sharp turns) strands inside the step.
#: --B (added 2026-09-22, default 280 = the run of record) exists for
#: the robustness replication at a second budget; anything other
#: than 280 writes under the GIVEN run-id only (never re-run the
#: deployed stepchange dir with a different B).
BUDGET = 280.0
STEP_X = 15.0
A_Q = 44.71            # fallback; refit on THIS map by default
N_SEEDS = 5
SEED0 = 41_000


def build_step_map():
    """The hand-authored step-change map (see module docstring).

    Obstacle placement v3 (2026-09-12): v1 put (8, 4.5) and (12, 1.5)
    opposite each other, leaving a 0.2 m squeeze at x~11 -- every arm
    collided there, measuring the map, not the controllers. v2 staggered
    them off the line and accidentally OPENED the straight line (both
    cleared y=3 by 0.85 m) -- a drag race, no detour. v3: obstacle 1 ON
    the dock line (8, 3.0) forces the weave; obstacle 2 high (13, 4.9)
    keeps the low route wide (free band y in [0, 1.6] at x=8 and [0, 3.5]
    at x=13 -- contiguous, >= 1.35 m, the generator's OBSTACLE_GAP
    discipline applied by hand). Straight line blocked by construction.
    """
    from env.scenarios import ScenarioConfig
    from env.terrain import TerrainField, TerrainRegion
    terrain = TerrainField([TerrainRegion(STEP_X, 0.0, 30.0, 6.0,
                                          m1=3.0, m4=3.0)],
                           bounds=(0.0, 0.0, 30.0, 6.0))
    return ScenarioConfig(
        name="reach_stepchange",
        start=np.array([1.0, 3.0, 0.0]), goal=np.array([1.0, 3.0]),
        obstacles=[(8.0, 3.0, 0.8), (13.0, 4.9, 0.8)],
        bounds=(0.0, 0.0, 30.0, 6.0),
        description=("step-change map: terrain m=3 beyond x=15; B=280 J "
                     "allows crossing only with banking; v2 staggered "
                     "obstacles (the v1 squeeze measured the map, not the "
                     "controllers)"),
        terrain=terrain, dock=np.array([1.0, 3.0]))


def run_profile(scenario, arm, B: float, seed: int, a_q: float,
                config: dict) -> Dict[str, Any]:
    """One mission, recording the per-step (x, v, margin) profile."""
    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer
    from mpc.reach_mpc import ReachSpec
    from mpc.energy_model import load_default_coefficients
    from evaluation.reach_evaluator import ReachSupervisor

    env_cfg = dict(config.get('robot', {}))
    env_cfg['dt'] = float(config.get('dt', 0.1))
    env_cfg.setdefault('energy_coeffs', load_default_coefficients())
    env = PioneerEnv(env_cfg)
    randomizer = TrialRandomizer()
    perturbed, v0, w0 = randomizer.randomize_trial(scenario, seed=seed)
    env.set_scenario(perturbed.obstacles, perturbed.bounds,
                     terrain=scenario.terrain)
    env.reset(perturbed.start, lidar_seed=seed + 1_000_003)
    env._v, env._omega = v0, w0

    spec = ReachSpec(B=B, certificate='affine', a_q=a_q, lam=1.0)
    sup = ReachSupervisor(scenario, spec, arm, config, N=60, dt_plan=0.5,
                          replan_every_s=2.5)
    dock = np.asarray(scenario.dock, dtype=float)
    dt = float(config.get('dt', 0.1))
    xs, vs, margins = [], [], []
    outcome = 'timeout'
    for step in range(int(240.0 / dt)):
        st = env.get_state_vector()
        if env.check_collision():
            outcome = 'collision'; break
        if np.hypot(*(st[:2] - dock)) <= 0.30 and len(xs) > 30:
            outcome = 'docked'; break
        if st[5] >= B:
            outcome = 'stranded'; break
        xs.append(float(st[0])); vs.append(float(st[3]))
        margins.append(sup.margin(st[:2], st[5]))
        v, w = sup.command(st, step * dt)
        env.step(v, w)
    return {
        'arm': arm.name, 'seed': seed, 'outcome': outcome,
        'B': B, 'a_q': a_q,
        'x_profile': xs, 'v_profile': vs, 'margin_profile': margins,
        'max_x': max(xs) if xs else None,
        'turnback_time': sup.turnback_t,
        'min_margin': min(margins) if margins else None,
        'final_price': (sup.price if arm.dual else None),
        'energy_spent': float(env.get_state_vector()[5]),
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--n-seeds', type=int, default=N_SEEDS)
    ap.add_argument('--refit-aq', action='store_true', default=True,
                    help='fit a_q on THIS map before running (script 20)')
    ap.add_argument('--const-price', type=float, default=None,
                    help='override const_weight price0 (Defect #7 fix: 50.0)')
    ap.add_argument('--budget', type=float, default=None,
                    help='override B (default: BUDGET=%.0f = the run of '
                         'record). For the robustness replication at a second '
                         'budget; run under a FRESH run-id.' % BUDGET)
    args = ap.parse_args(argv)
    budget = float(args.budget if args.budget is not None else BUDGET)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    from evaluation.reach_evaluator import (REACH_ARM, DUAL_ARM,
                                              CONST_WEIGHT_ARM, ReachArm)
    if args.const_price is not None:
        CONST_WEIGHT_ARM = ReachArm('const_weight', use_constraint=False,
                                    price0=float(args.const_price),
                                    turnback_rule='energy')
        logger.info("const_weight price0 overridden to %.1f", args.const_price)

    _suffix = f'_const{int(args.const_price)}' if args.const_price else ''
    out_dir = os.path.join(_run_dir(args.run_id, 'reach'),
                           f'stepchange{_suffix}')
    os.makedirs(out_dir, exist_ok=True)

    sc = build_step_map()
    # persist the map itself (the experiment is meaningless without it)
    import yaml
    from planning.reach_maps import scenario_to_yaml_dict
    with open(os.path.join(out_dir, 'map.yaml'), 'w') as fh:
        yaml.safe_dump(scenario_to_yaml_dict(sc), fh, sort_keys=False)

    a_q = A_Q
    if args.refit_aq:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '20_reach_certificate_terrain.py')
        spec20 = importlib.util.spec_from_file_location('m20', p)
        m20 = importlib.util.module_from_spec(spec20)
        spec20.loader.exec_module(m20)
        cert = m20.sample_and_fit(sc, stride=6)
        a_q = float(cert['a_q'])
        safe_write_json(os.path.join(out_dir, 'certificate.json'), cert)
        logger.info("a_q fitted on the step map: %.2f J (coverage %.3f)",
                    a_q, cert['coverage_achieved'])

    from utils.config_loader import load_config
    cfg = load_config(None)

    results = []
    for arm in (REACH_ARM, DUAL_ARM, CONST_WEIGHT_ARM):
        for s in range(args.n_seeds):
            logger.info("%s seed %d ...", arm.name, SEED0 + s)
            r = run_profile(sc, arm, budget, SEED0 + s, a_q, cfg)
            results.append(r)
            logger.info("  -> %s, max_x=%.1f, min_margin=%.1f, price=%s",
                        r['outcome'], r['max_x'] or 0,
                        r['min_margin'] if r['min_margin'] is not None else float('nan'),
                        r['final_price'])

    safe_write_json(os.path.join(out_dir, 'stepchange_results.json'),
                    {'map': 'reach_stepchange', 'B': budget,
                     'step_x': STEP_X, 'a_q': a_q, 'trials': results})

    # the verdict table: speed at x in bands, per arm
    print(f"\n{'arm':<14}{'outcome':>10}{'max_x':>7}{'min_marg':>9}"
          f"{'v@13-15':>8}{'v@15-17':>8}{'turn_x':>7}")
    for r in results:
        xs, vs = r['x_profile'], r['v_profile']
        def vband(a, b):
            sel = [v for x, v in zip(xs, vs) if a <= x < b]
            return sum(sel) / len(sel) if sel else float('nan')
        tb_idx = next((i for i, x in enumerate(xs)
                       if r['turnback_time'] is not None
                       and i * 0.1 >= r['turnback_time']), None)
        print(f"{r['arm']:<14}{r['outcome']:>10}{(r['max_x'] or 0):>7.1f}"
              f"{(r['min_margin'] if r['min_margin'] is not None else float('nan')):>9.1f}"
              f"{vband(13, 15):>8.2f}{vband(15, 17):>8.2f}"
              f"{(xs[tb_idx] if tb_idx is not None else float('nan')):>7.1f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
