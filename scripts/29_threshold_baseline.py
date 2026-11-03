"""C — the threshold-return baseline: the published practice, swept.

The persistent-autonomy literature's standard energy policy is a
THRESHOLD RETURN: drive the mission; turn back when the remaining
budget no longer covers a safety-factored estimate of the return cost,
``B - E <= gamma * D_estimate(x)``, with gamma hand-tuned per
deployment. This script instantiates that family faithfully on our
mission and sweeps gamma:

- target: the farthest METRIC free cell (the mission objective — no
  budget certification, that is the planner's job and the point);
- route: the J-geodesic to it (same routing field + buffer as the
  oracle, scripts/24_reach_oracle.py — the machinery this derives from);
- tracker: pi_fb gains, identical to the certificate rollout;
- turn-back: the gamma rule checked EVERY plant step on the true
  J-field (the estimate the practice would use);
- same budgets, seeds, perturbed plants, and mission rules as the
  campaign (exact pairing with the reach-arm rows).

The comparison isolates what the certificate adds over the standard
practice: gamma must be hand-tuned and is terrain-dependent, while
a_q = p95(V_fb - D_J) automates the margin per map. Expected result
(pre-registered 2026-09-16 before running): gamma=1 breaches (the
return costs exceed the floor by the realignment + traversal excess);
large gamma over-reserves and loses reach; NO single gamma achieves
zero breaches across all maps at good reach — the per-map spread of
the right margin is exactly what the certificate encodes.

Outputs runs/<id>/reach/threshold_baseline/threshold_trials.jsonl +
a summary JSON. Feeds the threshold-baseline results section and
fig_enforce/fig_gamma.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List

# Single-threaded BLAS/OMP, set BEFORE the first numpy import so spawn
# workers load OpenBLAS pinned (module scope runs before numpy in the
# child). The threshold arm is NLP-free -- its results were already
# deterministic -- but every fan-out campaign lives in one regime.
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pi_fb / routing constants — IDENTICAL to the oracle (script 24); loaded
# from it so the two can never drift
_spec = importlib.util.spec_from_file_location(
    "reach_oracle_mod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "24_reach_oracle.py"))
_o24 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_o24)
_track_step, _wrap = _o24._track_step, _o24._wrap
GTOL, MIN_OUTBOUND = _o24.GTOL, _o24.MIN_OUTBOUND

#: the sweep (pre-registered): 1.0 = no margin; 1.1-1.5 = the hand-tuned
#: range; 2.0 = heavy over-reserving
GAMMAS = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)
SEEDS = tuple(range(31000, 31010))          # the campaign's 10 seeds
NOISE = (0.05, 0.10)                        # the R5 mismatch spec


def _farthest_metric_cell(jfield, routefield, mfield):
    """The mission target: farthest METRIC free cell (no budget logic)."""
    free, D_J, D_m = jfield.free, jfield.D, mfield.D
    rfree = routefield.free
    best, cell = None, None
    for i in range(D_J.shape[0]):
        for j in range(D_J.shape[1]):
            if not free[i, j] or not rfree[i, j] or not math.isfinite(D_J[i, j]):
                continue
            dm = D_m[i, j]
            if math.isfinite(dm) and (best is None or dm > best):
                best = float(dm)
                cell = (float(jfield.X[i]), float(jfield.Y[j]))
    return cell, best


def threshold_rollout(scenario, jfield, routefield, mfield, B: float,
                      gamma: float, seed: int, config: dict,
                      power_mismatch: float = 0.0,
                      t_mission_s: float = 240.0) -> Dict[str, Any]:
    """One threshold-return mission (see module docstring)."""
    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer
    from planning.energy_dijkstra import geodesic_path

    target, _ = _farthest_metric_cell(jfield, routefield, mfield)
    if target is None:
        return {'outcome': 'no_target', 'reach_m': 0.0, 'reach_J': 0.0,
                'energy_spent': 0.0, 'turnback_time': None,
                'turnback_reach_m': None, 'mission_time': 0.0}

    env_cfg = dict(config.get('robot', {}))
    env_cfg['dt'] = float(config.get('dt', 0.1))
    from mpc.energy_model import load_default_coefficients
    env_cfg.setdefault('energy_coeffs', load_default_coefficients())
    env = PioneerEnv(env_cfg)
    randomizer = TrialRandomizer()
    perturbed, v0, w0 = randomizer.randomize_trial(scenario, seed=seed)
    env.set_scenario(perturbed.obstacles, perturbed.bounds,
                     terrain=scenario.terrain)
    if power_mismatch:
        env.set_power_mismatch(power_mismatch)
        # the mismatch stage's R5 spec: actuation noise rides with the
        # power mismatch (campaign script 21, mismatch stage)
        env.set_actuation_noise(NOISE[0], NOISE[1], seed=seed + 2_000_003)
    # smoke-test bug (caught before any data was written): the first
    # draft never reset the env -- the robot sat at the origin and
    # collided with the wall on step 1 at E=0
    env.reset(perturbed.start, lidar_seed=seed + 1_000_003)
    env._v, env._omega = v0, w0
    dock = np.asarray(scenario.dock, dtype=float)
    dt = float(config.get('dt', 0.1))

    descent = geodesic_path(routefield, target)
    if descent is None or len(descent) < 2:
        return {'outcome': 'no_geodesic', 'reach_m': 0.0, 'reach_J': 0.0,
                'energy_spent': 0.0, 'turnback_time': None,
                'turnback_reach_m': None, 'mission_time': 0.0}
    outbound = np.asarray(descent[::-1])

    phase = 'out'
    idx = 0
    reach_m = reach_J = 0.0
    outcome = 'timeout'
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
        if phase == 'out':
            # THE rule under test: threshold return with safety factor
            # gamma on the geodesic return estimate
            if math.isfinite(dj) and (B - E) <= gamma * dj:
                phase = 'ret'
                turnback_t, turnback_reach = t, reach_m
                ret = geodesic_path(routefield, (x, y))
                if ret is None or len(ret) < 2:
                    outcome = 'no_return_geodesic'
                    break
                outbound = np.asarray(ret)
                idx = 0
            idx, v_cmd, w_cmd = _track_step(outbound, idx, x, y, float(st[2]))
        else:
            idx, v_cmd, w_cmd = _track_step(outbound, idx, x, y, float(st[2]))
            if idx >= len(outbound) - 1 and math.hypot(x - dock[0], y - dock[1]) < 1.0:
                err = _wrap(math.atan2(dock[1] - y, dock[0] - x) - float(st[2]))
                w_cmd = max(-_o24.WMAX, min(_o24.WMAX, _o24.KP * err))
                v_cmd = _o24.VSTAR * max(0.0, math.cos(err))
        env.step(v_cmd, w_cmd)
        t += dt
    return {'outcome': outcome, 'reach_m': reach_m, 'reach_J': reach_J,
            'energy_spent': float(env.get_state_vector()[5]),
            'turnback_time': turnback_t, 'turnback_reach_m': turnback_reach,
            'mission_time': t}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--maps-root', default=None,
                    help='default: reach_maps_adaptive (the deployed certs; '
                         'the baseline itself uses no certificate — a_q is '
                         'recorded only for the implied-margin comparison)')
    ap.add_argument('--gammas', default=None,
                    help='comma list; default: ' + ','.join(map(str, GAMMAS)))
    ap.add_argument('--mismatch', action='store_true',
                    help='ALSO run the +15%% power-mismatch stage '
                         '(0.60 E_rt, R5 noise) for every gamma')
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    from utils.config_loader import load_config
    from env.scenarios import load_scenario_from_yaml
    from planning.energy_dijkstra import build_field

    gammas = ([float(g) for g in args.gammas.split(',')]
              if args.gammas else list(GAMMAS))
    maps_root = (args.maps_root or
                 _run_dir(args.run_id, 'reach_maps_adaptive', create=False))
    cfg = load_config(None)

    # maps + E_rt + a_q exactly as the campaign loads them
    certs = {c['map']: c for c in json.load(
        open(os.path.join(maps_root, 'train', 'certificates.json')))}
    import glob as _glob
    maps = []
    for path in sorted(_glob.glob(os.path.join(maps_root, 'train',
                                               'reach_*.yaml'))):
        sc = load_scenario_from_yaml(path)
        if sc.name not in certs:
            continue
        jf = build_field(sc, terrain=sc.terrain)
        far_x = sc.bounds[2] - 1.0
        e_far = jf.dstar(float(far_x), float(sc.dock[1]))
        maps.append({'path': path, 'name': sc.name, 'sc': sc, 'jf': jf,
                     'E_rt': 2.0 * float(e_far),
                     'a_q': float(certs[sc.name]['a_q'])})

    out_dir = os.path.join(_run_dir(args.run_id, 'reach'), 'threshold_baseline')
    os.makedirs(out_dir, exist_ok=True)
    sink = os.path.join(out_dir, 'threshold_trials.jsonl')
    fh = open(sink, 'w')

    jobs = []   # (map, B, seed, gamma, mismatch)
    for m in maps:
        for frac in (0.30, 0.45, 0.60, 0.75):
            B = round(frac * m['E_rt'], 1)
            for s in SEEDS:
                for g in gammas:
                    jobs.append((m, B, s, g, 0.0))
        if args.mismatch:
            B = round(0.60 * m['E_rt'], 1)
            for s in SEEDS:
                for g in gammas:
                    jobs.append((m, B, s, g, 0.15))

    t0 = time.perf_counter()
    n = 0
    fields = {}
    for m, B, s, g, mm in jobs:
        if (m['name'], mm) not in fields:
            sc, jf = m['sc'], m['jf']
            rf = build_field(sc, terrain=sc.terrain,
                             r_body=0.25 + _o24.ROUTE_BUFFER)
            mf = build_field(sc)
            fields[(m['name'], mm)] = (sc, jf, rf, mf)
        sc, jf, rf, mf = fields[(m['name'], mm)]
        r = threshold_rollout(sc, jf, rf, mf, B, g, s, cfg,
                              power_mismatch=mm)
        r.update({'map': m['name'], 'B': B, 'seed': s, 'gamma': g,
                  'power_mismatch': mm, 'a_q_ref': m['a_q'],
                  'breach': r['energy_spent'] > B + 0.5})
        fh.write(json.dumps(r) + '\n')
        n += 1
        if n % 240 == 0:
            logger.info("  %d/%d missions (%.0fs)", n, len(jobs),
                        time.perf_counter() - t0)
    fh.close()

    # summary: per gamma, nominal and mismatch
    rows = [json.loads(l) for l in open(sink)]
    summary = {}
    for mm_label, mm in (('nominal', 0.0), ('mismatch+15', 0.15)):
        for g in gammas:
            sel = [r for r in rows if r['gamma'] == g
                   and r['power_mismatch'] == mm]
            if not sel:
                continue
            summary.setdefault(mm_label, {})[str(g)] = {
                'n': len(sel),
                'breaches': sum(1 for r in sel if r['breach']),
                'breach_rate': sum(1 for r in sel if r['breach']) / len(sel),
                'docked_rate': sum(1 for r in sel if r['outcome'] == 'docked')
                               / len(sel),
                'mean_reach_m': float(np.mean(
                    [r['reach_m'] for r in sel])),
                # per-map breach counts: the certificate's point is that
                # the right margin is map-dependent
                'per_map_breaches': {
                    mn: sum(1 for r in sel if r['map'] == mn and r['breach'])
                    for mn in sorted({r['map'] for r in sel})},
            }
    safe_write_json(os.path.join(out_dir, 'threshold_summary.json'),
                    {'definition': __doc__.split('\n')[0],
                     'gammas': gammas, 'n_missions': len(rows),
                     'wall_s': round(time.perf_counter() - t0, 1),
                     'summary': summary})
    print(f"{len(rows)} missions in {time.perf_counter()-t0:.0f}s -> {sink}")
    for mm_label in summary:
        print(f"  {mm_label}:")
        for g, d in sorted(summary[mm_label].items(), key=lambda kv: float(kv[0])):
            print(f"    gamma={g:>4}: breaches {d['breaches']}/{d['n']} "
                  f"({100*d['breach_rate']:.1f}%), docked "
                  f"{100*d['docked_rate']:.1f}%, reach {d['mean_reach_m']:.2f} m")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
