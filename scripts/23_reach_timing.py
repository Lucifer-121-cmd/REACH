"""Script 23 — quiet-box timing for the ReachMPC planning solve.

The campaign's solve times (mean ~22 s) were measured under 150-worker
fan-out, which is invalid for any compute claim -- cross-process solve
times differ 4-6x for identical genomes on a loaded box.
This script measures the SAME quantity on RESERVED, verified-quiet cores:
one full mission's replan sequence (the step-change map is the natural
subject — same map, same seed as the Move-4 runs), each solve timed
individually, cold and warm separated.

The supervisory framing (GATE-E: fixed horizons solve the tracking
problem) means this is a PLANNER's cost, not a per-step deadline — the
paper reports the distribution and the replan cadence it must support,
not a 100 ms real-time claim.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--cpus', type=int, nargs='+', required=True,
                    help='reserved CPU set; the script pins itself there')
    ap.add_argument('--seed', type=int, default=41000)
    ap.add_argument('--B', type=float, default=280.0)
    ap.add_argument('--a-q', type=float, default=53.54)
    args = ap.parse_args(argv)

    # pin BEFORE any measurement; verify the affinity actually took
    try:
        os.sched_setaffinity(0, set(args.cpus))
        actual = sorted(os.sched_getaffinity(0))
        assert set(args.cpus) <= set(actual), (args.cpus, actual)
    except (AttributeError, AssertionError) as exc:
        raise SystemExit(f"could not pin to {args.cpus}: {exc}")

    from utils import preflight
    pf = preflight.check(mode='timing', cpus=args.cpus)
    if not pf['ok']:
        for r in pf['reasons']:
            print(f"REFUSE: {r}")
        return 2

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m22", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "22_reach_stepchange.py"))
    m22 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m22)

    from utils.config_loader import load_config
    from mpc.reach_mpc import ReachMPC, ReachSpec

    sc = m22.build_step_map()
    cfg = load_config(None)
    spec_r = ReachSpec(B=args.B, certificate='affine', a_q=args.a_q, lam=1.0)
    mpc = ReachMPC(sc, spec_r, N=60, dt=0.5, config=cfg)

    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer

    env_cfg = dict(cfg.get('robot', {}))
    env_cfg['dt'] = 0.1
    env = PioneerEnv(env_cfg)
    rand = TrialRandomizer()
    perturbed, v0, w0 = rand.randomize_trial(sc, seed=args.seed)
    env.set_scenario(perturbed.obstacles, perturbed.bounds,
                     terrain=sc.terrain)
    env.reset(perturbed.start, lidar_seed=args.seed + 1_000_003)
    env._v, env._omega = v0, w0

    # The replan sequence the supervisor would actually run: solve, hold
    # the plan for the cadence, replan from the drifted state (warm where
    # the phase allows). We time solves only; the plant stepping between
    # them is not part of the measurement.
    records: List[Dict[str, Any]] = []
    prev_X = prev_U = None
    executed = 0
    for step in range(int(60.0 / 0.1)):          # 60 s of mission
        st = env.get_state_vector()
        if env.check_collision():
            break
        mission_t = step * 0.1
        if step % 25 == 0:                        # every 2.5 s: a replan
            warm = None
            if prev_X is not None:
                warm = (prev_X, prev_U, executed)
            t0 = time.perf_counter()
            res = mpc.solve(st, B=args.B, warm=warm)
            dt_ms = (time.perf_counter() - t0) * 1e3
            iters = mpc._solver.stats().get('iter_count')
            records.append({'t': round(mission_t, 1),
                            'warm': warm is not None,
                            'ms': round(dt_ms, 1),
                            'iters': iters,
                            'status': res['return_status'],
                            'slack_J': round(res['slack_J'], 2)})
            logger.info("t=%5.1fs %s solve %8.1f ms (%d iters, %s)",
                        mission_t, 'warm' if warm is not None else 'cold',
                        dt_ms, iters or -1, res['return_status'][:24])
            if np.all(np.isfinite(res['X'])):
                prev_X, prev_U = res['X'], res['U']
                executed = 0
        # execute the current plan node for this plant step
        if prev_U is not None:
            u = prev_U[:, min(executed, prev_U.shape[1] - 1)]
            env.step(float(u[0]), float(u[1]))
            if step % 5 == 4:
                executed += 1

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    out_dir = os.path.join(_run_dir(args.run_id, 'reach'), 'timing')
    os.makedirs(out_dir, exist_ok=True)
    cold = [r['ms'] for r in records if not r['warm']]
    warm = [r['ms'] for r in records if r['warm']]
    summary = {
        'map': sc.name, 'seed': args.seed, 'B': args.B, 'a_q': args.a_q,
        'cpus': args.cpus, 'cpu_affinity_at_run': sorted(os.sched_getaffinity(0)),
        'n_solves': len(records), 'records': records,
        'cold_ms': {'mean': float(np.mean(cold)) if cold else None,
                    'p50': float(np.percentile(cold, 50)) if cold else None,
                    'max': float(np.max(cold)) if cold else None},
        'warm_ms': {'mean': float(np.mean(warm)) if warm else None,
                    'p50': float(np.percentile(warm, 50)) if warm else None,
                    'max': float(np.max(warm)) if warm else None},
        'note': ('quiet-box timing on reserved cores; the supervisory '
                 'replan cadence is 2.5 s, so the planner cost criterion '
                 'is ms-per-solve << 2500 ms, not a 100 ms control period'),
    }
    safe_write_json(os.path.join(out_dir, 'reach_planning_timing.json'),
                    summary)
    print(f"\ncold: mean {summary['cold_ms']['mean']:.0f} ms  "
          f"p50 {summary['cold_ms']['p50']:.0f}  max {summary['cold_ms']['max']:.0f}")
    if warm:
        print(f"warm: mean {summary['warm_ms']['mean']:.0f} ms  "
              f"p50 {summary['warm_ms']['p50']:.0f}  max {summary['warm_ms']['max']:.0f}")
    print(f"written: {out_dir}/reach_planning_timing.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
