"""P1 — the oracle gap: plan-then-track upper bound vs the closed loop.

The question (P1): "is 8.42 m good?" — i.e. how much
reach does the ReachMPC closed loop give up against a controller that
KNOWS the map and executes the certificate's own model of the controller
perfectly?

THE ORACLE ('certified plan-then-track'):
1. TARGET: the certified-ceiling cell — the free cell with MAXIMUM METRIC
   distance subject to 2*D_J(cell) + a_q <= B. This is the optimal
   certified-feasible target: the farthest point (in metres) whose
   round trip at the energy floor still fits the budget with the
   certificate's reserve.
2. ROUTE: the J-geodesic dock->target (the cheapest-energy route to the
   best target — steepest descent from the target, reversed).
3. TRACKER: pi_fb gains (pure pursuit at v*, kp=2.0, La=0.5 lookahead,
   slow-while-realigning) — the certificate's own controller model,
   executed on the perturbed plant.
4. TURN: the certified energy rule, checked every plant step:
   B - E_spent <= a_q + D_J(x); the return follows the J-geodesic from
   the turn position (planned once, then tracked — plan-then-track).
5. Same budgets, seeds (exact per-seed pairing with the campaign's
   reach-arm trials), perturbed plants, certificates, and mission rules
   as the campaign.

The first draft of this oracle tracked the ASCENDING GRADIENT of the
J-field — which points toward where energy-to-dock increases fastest per
METRE, i.e. into expensive terrain (it dove into the bottom wall on the
smoke test). The fix is the target+route design above. The same
observation exposes a property of the closed loop itself: the ReachMPC
objective maximises D_J (ENERGY distance), which is metric-blind at the
constraint boundary — all boundary points with equal D_J score equally,
whether far-and-cheap or near-and-expensive. The oracle's
reach_J-vs-reach_m comparison with the campaign records quantifies this
denomination bias.

CEILINGS (arithmetic, from the fields):
- certified ceiling: max metric distance of any free cell with
  2*D_J(cell) + a_q <= B  (the oracle's target, floor-executed);
- physical ceiling: max metric distance of any free cell with
  2*D_J(cell) <= B  (no certificate — the absolute bound).

Outputs runs/<id>/reach/oracle/oracle_results.json. Feeds the paper's
verified-results register.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Single-threaded BLAS/OMP, set BEFORE the first numpy/casadi import:
# OpenBLAS/MUMPS read the environment at import time, and results must
# land on the pinned campaign lineage -- an ulp-scale thread-environment
# change is a different experiment (measured: a 6-decimal intercept
# rounding departed from the campaign lineage in 12/60 outcomes).
# Same idiom as scripts/30. Added 2026-09-20 during the full audit
# sweep; the 2026-09-14/15 oracle artefacts predate the pin and remain
# the runs of record (their lineage is recorded per artefact).
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pi_fb gains -- IDENTICAL to the certificate rollout (script 20)
VSTAR = math.sqrt(3.0 / 2.5)
VMAX, WMAX = 1.2, 3.14159
KP = 2.0
#: Lookahead reduced from pi_fb's 0.5 to 0.3: pure pursuit cuts corners by
#: up to La/2, and with La=0.5 the cut (0.25 m) plus the +-5 cm obstacle
#: perturbation consumed the routing buffer and the first draft collided
#: on the RETURN leg (measured, map 0911 seed 31000: collision at t=13.2 s
#: on the way home). La=0.3 with the 0.35 m routing buffer below leaves
#: ~0.15 m of margin; the gains stay pi_fb's.
LA = 0.3
#: Routing-only inflation beyond r_body: the same execution buffer the
#: closed loop's hard clearance uses (ReachWeights.hard_clearance). The
#: RESERVE arithmetic (turn-back, ceilings, D_J lookups) stays on the
#: standard r_body-inflated field -- the certificate is what it is.
ROUTE_BUFFER = 0.35
GTOL = 0.30           # dock tolerance (evaluator value)
MIN_OUTBOUND = 1.0    # m before "docked" counts (evaluator rule)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _ceiling_cells(jfield, routefield, mfield, a_q: float, B: float
                   ) -> Tuple[Optional[Tuple[float, float]], Optional[float],
                              Optional[float]]:
    """(certified-target cell, certified ceiling m, physical ceiling m).

    The certified target must be free on BOTH the standard field (reserve
    arithmetic) and the routing field (buffered) -- otherwise the tracker
    cannot reach it without entering its own safety buffer."""
    free, D_J, D_m = jfield.free, jfield.D, mfield.D
    rfree = routefield.free
    best_cert, best_phys = None, None
    cert_cell = None
    for i in range(D_J.shape[0]):
        for j in range(D_J.shape[1]):
            if not free[i, j] or not rfree[i, j] or not math.isfinite(D_J[i, j]):
                continue
            dm = D_m[i, j]
            if not math.isfinite(dm):
                continue
            if 2 * D_J[i, j] + a_q <= B and (best_cert is None or dm > best_cert):
                best_cert = float(dm)
                cert_cell = (float(jfield.X[i]), float(jfield.Y[j]))
            if 2 * D_J[i, j] <= B and (best_phys is None or dm > best_phys):
                best_phys = float(dm)
    return cert_cell, best_cert, best_phys


def _track_step(path: np.ndarray, idx: int, x: float, y: float, th: float
                ) -> Tuple[int, float, float]:
    """One pi_fb pure-pursuit step toward the polyline. Returns (idx, v, w)."""
    while idx < len(path) - 1 and math.hypot(path[idx][0] - x,
                                             path[idx][1] - y) < LA:
        idx += 1
    tx, ty = path[idx]
    err = _wrap(math.atan2(ty - y, tx - x) - th)
    w_cmd = max(-WMAX, min(WMAX, KP * err))
    v_cmd = VSTAR * max(0.0, math.cos(err))
    return idx, v_cmd, w_cmd


def oracle_rollout(scenario, jfield, routefield, mfield, B: float, a_q: float,
                   seed: int, config: dict,
                   t_mission_s: float = 240.0) -> Dict[str, Any]:
    """One oracle mission (see module docstring for the definition)."""
    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer
    from planning.energy_dijkstra import geodesic_path, R_BODY

    target, cert_m, phys_m = _ceiling_cells(jfield, routefield, mfield,
                                             a_q, B)
    if target is None:
        return {'outcome': 'no_certified_target', 'reach_m': 0.0,
                'reach_J': 0.0, 'energy_spent': 0.0, 'min_margin_J': None,
                'turnback_time': None, 'turnback_reach_m': None,
                'mission_time': 0.0, 'certified_ceiling_m': cert_m,
                'physical_ceiling_m': phys_m}

    env_cfg = dict(config.get('robot', {}))
    env_cfg['dt'] = float(config.get('dt', 0.1))
    from mpc.energy_model import load_default_coefficients
    env_cfg.setdefault('energy_coeffs', load_default_coefficients())
    env = PioneerEnv(env_cfg)
    randomizer = TrialRandomizer()
    perturbed, v0, w0 = randomizer.randomize_trial(scenario, seed=seed)
    env.set_scenario(perturbed.obstacles, perturbed.bounds,
                     terrain=scenario.terrain)
    env.reset(perturbed.start, lidar_seed=seed + 1_000_003)
    env._v, env._omega = v0, w0

    dock = np.asarray(scenario.dock, dtype=float)
    dt = float(config.get('dt', 0.1))

    # plan: outbound = reversed descent polyline target->dock on the ROUTE
    # field (buffered inflation; reserve arithmetic stays on jfield)
    descent = geodesic_path(routefield, target)
    if descent is None or len(descent) < 2:
        return {'outcome': 'no_geodesic', 'reach_m': 0.0, 'reach_J': 0.0,
                'energy_spent': 0.0, 'min_margin_J': None,
                'turnback_time': None, 'turnback_reach_m': None,
                'mission_time': 0.0, 'certified_ceiling_m': cert_m,
                'physical_ceiling_m': phys_m}
    outbound = np.asarray(descent[::-1])

    phase = 'out'
    idx = 0
    reach_m = reach_J = 0.0
    outcome = 'timeout'
    min_margin = math.inf
    turnback_t = turnback_reach = None
    t = 0.0
    for _ in range(int(t_mission_s / dt)):
        st = env.get_state_vector()
        x, y, E = float(st[0]), float(st[1]), float(st[5])
        if env.check_collision():
            outcome = 'collision'
            break
        dm = mfield.dstar(x, y)
        if math.isfinite(dm):
            reach_m = max(reach_m, dm)
        if (math.hypot(x - dock[0], y - dock[1]) <= GTOL
                and reach_m > MIN_OUTBOUND):
            outcome = 'docked'
            break
        if E >= B:
            outcome = 'stranded'
            break
        dj = jfield.dstar(x, y)
        if math.isfinite(dj):
            reach_J = max(reach_J, dj)
            min_margin = min(min_margin, B - E - (a_q + dj))
        if phase == 'out':
            # certified turn-back, checked every step
            if math.isfinite(dj) and (B - E) <= a_q + dj:
                phase = 'ret'
                turnback_t, turnback_reach = t, reach_m
                ret = geodesic_path(routefield, (x, y))
                if ret is None or len(ret) < 2:
                    outcome = 'no_return_geodesic'
                    break
                outbound = np.asarray(ret)   # now tracks dock-ward
                idx = 0
            idx, v_cmd, w_cmd = _track_step(outbound, idx, x, y, float(st[2]))
        else:
            idx, v_cmd, w_cmd = _track_step(outbound, idx, x, y, float(st[2]))
            if idx >= len(outbound) - 1 and math.hypot(x - dock[0], y - dock[1]) < 1.0:
                # polyline exhausted near the dock: head straight for it
                err = _wrap(math.atan2(dock[1] - y, dock[0] - x) - float(st[2]))
                w_cmd = max(-WMAX, min(WMAX, KP * err))
                v_cmd = VSTAR * max(0.0, math.cos(err))
        env.step(v_cmd, w_cmd)
        t += dt
    return {
        'outcome': outcome, 'reach_m': reach_m, 'reach_J': reach_J,
        'energy_spent': float(env.get_state_vector()[5]),
        'min_margin_J': (None if min_margin == math.inf else min_margin),
        'turnback_time': turnback_t, 'turnback_reach_m': turnback_reach,
        'mission_time': t,
        'certified_ceiling_m': cert_m, 'physical_ceiling_m': phys_m,
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--maps-root', default=None,
                    help='maps root containing {train,heldout}/ with certificates.json '
                         '(default: this run-id\'s reach_maps)')
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    from utils.config_loader import load_config
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field
    import glob as _glob

    cfg = load_config(None)
    reach_root = _run_dir(args.run_id, 'reach', create=False)
    maps_root = (args.maps_root if args.maps_root
                 else _run_dir(args.run_id, 'reach_maps', create=False))

    sinks = {
        'train': os.path.join(reach_root,
                              'main3_adaptive_20260915/reach_trials_train.jsonl'),
        'heldout': os.path.join(reach_root,
                                'heldout3_adaptive_20260915/reach_trials_heldout.jsonl'),
    }
    reach_rows: Dict[Tuple[str, str, float], List[dict]] = {}
    for split, sink in sinks.items():
        for line in open(sink):
            r = json.loads(line)
            if r['arm'] != 'reach':
                continue
            reach_rows.setdefault((split, r['map_name'], r['B']), []).append(r)

    certs: Dict[str, float] = {}
    for split in ('train', 'heldout'):
        for c in json.load(open(os.path.join(maps_root, split,
                                             'certificates.json'))):
            certs[c['map']] = float(c['a_q'])
    map_files: Dict[str, str] = {}
    for split in ('train', 'heldout'):
        for p in sorted(_glob.glob(os.path.join(maps_root, split,
                                                'reach_*.yaml'))):
            sc = load_scenario_from_yaml(p)
            map_files[sc.name] = p

    t0 = time.perf_counter()
    results: List[dict] = []
    for (split, map_name, B), rows in sorted(reach_rows.items()):
        sc = load_scenario_from_yaml(map_files[map_name])
        jfield = build_field(sc, terrain=sc.terrain)
        routefield = build_field(sc, terrain=sc.terrain,
                                 r_body=0.25 + ROUTE_BUFFER)
        mfield = build_field(sc)
        a_q = certs[map_name]
        for r in rows:
            o = oracle_rollout(sc, jfield, routefield, mfield, B, a_q,
                               r['seed'], cfg)
            o.update({'map': map_name, 'split': split, 'B': B,
                      'seed': r['seed'], 'a_q': a_q,
                      'reach_arm_reach_m': r['reach_m'],
                      'reach_arm_reach_J': r.get('reach_J'),
                      'reach_arm_outcome': r['outcome']})
            results.append(o)
        sel = [o for o in results if o['map'] == map_name and o['B'] == B]
        mean_o = np.mean([o['reach_m'] for o in sel])
        mean_r = np.mean([r['reach_m'] for r in rows])
        mean_orJ = np.mean([r.get('reach_J') or 0 for r in rows])
        logger.info("%s B=%.0f: oracle %.2f m (cert ceil %.2f) vs arm %.2f m "
                    "(arm reach_J %.1f)", map_name, B, mean_o,
                    sel[0]['certified_ceiling_m'] or float('nan'),
                    mean_r, mean_orJ)

    oracle_reach = [o['reach_m'] for o in results]
    arm_reach = [o['reach_arm_reach_m'] for o in results]
    oracle_rJ = [o['reach_J'] for o in results]
    arm_rJ = [o['reach_arm_reach_J'] or 0 for o in results]
    out = {
        'definition': ('oracle = certified plan-then-track: target = max-'
                       'metric cell with 2*D_J + a_q <= B; route = J-geodesic; '
                       'tracker = pi_fb gains; certified turn-back every step; '
                       'same B/seeds/perturbations/certificates as the reach '
                       'arm (exact pairing). Upper bound at equal information.'),
        'n_missions': len(results), 'trials': results,
        'wall_s': round(time.perf_counter() - t0, 1),
        'summary': {
            'oracle_mean_reach_m': float(np.mean(oracle_reach)),
            'reach_arm_mean_reach_m': float(np.mean(arm_reach)),
            'gap_m': float(np.mean(oracle_reach) - np.mean(arm_reach)),
            'gap_pct_of_oracle': float(100 * (np.mean(oracle_reach)
                                              - np.mean(arm_reach))
                                       / np.mean(oracle_reach)),
            'oracle_mean_reach_J': float(np.mean(oracle_rJ)),
            'reach_arm_mean_reach_J': float(np.mean(arm_rJ)),
            'oracle_outcomes': {
                k: sum(1 for o in results if o['outcome'] == k)
                for k in ('docked', 'stranded', 'collision', 'timeout')},
        },
    }
    out_dir = os.path.join(reach_root, 'oracle')
    os.makedirs(out_dir, exist_ok=True)
    safe_write_json(os.path.join(out_dir, 'oracle_results.json'), out)
    s = out['summary']
    print(f"\noracle reach {s['oracle_mean_reach_m']:.2f} m vs arm "
          f"{s['reach_arm_mean_reach_m']:.2f} m  (gap {s['gap_m']:.2f} m = "
          f"{s['gap_pct_of_oracle']:.1f}% of oracle)")
    print(f"energy-reach: oracle {s['oracle_mean_reach_J']:.1f} J vs arm "
          f"{s['reach_arm_mean_reach_J']:.1f} J")
    print(f"oracle outcomes: {s['oracle_outcomes']}")
    print(f"written: {out_dir}/oracle_results.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
