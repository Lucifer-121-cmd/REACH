#!/usr/bin/env python3
"""Script 13 — GATE-E: isolate the horizon law from everything else.

Written to run as-is once the machine is free; the `timing` subcommand refuses
to start unless the box is quiet.

What it settles (the horizon gate, and Reviewer 13's central request). Table I
concluded that fixed N = 12/24/50 fail while the adaptive law succeeds, but the
Table I baseline was a *different controller* with `d_safe` below the collision
radius. Reviewer 13 asked for the matched experiment: hold every parameter,
constraint, initialisation, warm-start and solver setting fixed and replace only
eq. (12) with a fixed horizon.

That is what the `fixed:N` arms do, and they do it by the smallest possible
intervention: set `N_min = N_max = N_default = N` so `AdaptiveHorizon.compute`
clips to a constant. No other line changes, so any difference is attributable to
the horizon law alone.

Pre-registered outcomes -- **there is no result that saves both claims**:
  * s_curve reaches 100% at some N <= 24  =>  a deadline-feasible fixed horizon
    solves everything; horizon adaptation is withdrawn from the contribution
    list in one sentence.
  * s_curve needs N >= 36  =>  no configuration is real-time feasible on this
    formulation and the real-time framing must become supervisory (2-3 Hz above
    a reactive tracker).

Two subcommands, because the two halves have opposite machine requirements:

  solvability   success rates and exact McNemar. Load-INSENSITIVE: success is a
                logical outcome, unaffected by how busy the box is.
  timing        per-step solve-time distributions and the a + b*N cost model.
                Requires an EXCLUSIVE machine and refuses otherwise.

Usage:
    python scripts/13_horizon_isolation.py solvability --run-id run_x --scenario s_curve
    python scripts/13_horizon_isolation.py timing      --run-id run_x --scenario s_curve
    python scripts/13_horizon_isolation.py solvability --run-id run_x --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

#: The pre-registered horizon grid.
DEFAULT_FIXED_N = (12, 16, 20, 24)

#: Extended grid, used only if the pre-registered one fails to reach 100%.
ESCALATION_N = (28, 32, 36, 40, 50)


def parse_arms(spec: str, escalate: bool = False) -> List[str]:
    """``'adaptive,fixed:12,fixed:16'`` or ``'default'`` -> a list of arm names."""
    if spec in ('default', 'gate-e'):
        grid = DEFAULT_FIXED_N + (ESCALATION_N if escalate else ())
        return ['adaptive'] + [f'fixed:{n}' for n in grid]
    arms = [a.strip() for a in spec.split(',') if a.strip()]
    for a in arms:
        if a != 'adaptive' and not (a.startswith('fixed:') and a[6:].isdigit()):
            raise ValueError(f"bad arm {a!r}; use 'adaptive' or 'fixed:<N>'")
    return arms


def arm_config(base: Dict[str, Any], arm: str, *, prebuild: bool,
               deadline_s: Optional[float]) -> Dict[str, Any]:
    """Config for one arm. Only the horizon law differs between arms.

    For ``fixed:N`` this pins N_min = N_max = N_default = N, which makes
    ``AdaptiveHorizon.compute`` clip every value to N. Everything else -- weights,
    d_safe, dt, motor lag, IPOPT settings, warm-start behaviour -- is untouched,
    which is precisely the matched comparison Reviewer 13 asked for and Table I
    did not provide.
    """
    cfg = copy.deepcopy(base)
    mpc = cfg.setdefault('mpc', {})
    mpc['prebuild_solver_bank'] = bool(prebuild)
    if deadline_s is not None:
        mpc['deadline_s'] = float(deadline_s)
    if arm == 'adaptive':
        return cfg
    n = int(arm.split(':', 1)[1])
    # Set all three together. Setting N_min/N_max without N_default leaves the
    # constructor building a solver for a horizon no step will ever request.
    mpc['N_min'] = n
    mpc['N_max'] = n
    mpc['N_default'] = n
    return cfg


def _apply_lidar_seed_mode(ev, mode: str, arm: str) -> None:
    """Set the evaluator's LiDAR seeding policy for one arm.

    'matched'  every arm derives its seed from the trial seed alone, so trial k
               sees the SAME scan noise in every arm. That is what makes the arms
               genuinely paired and is the default.
    'per-arm'  the arm name is folded into the offset, so arms see different
               noise. Use only when independence between arms is wanted.
    'off'      unseeded. Trials are NOT reproducible; diagnostics only.
    """
    if mode == 'off':
        ev.lidar_seed_offset = None
    elif mode == 'per-arm':
        # zlib.crc32, not hash(): PYTHONHASHSEED is randomised per process, so
        # hash(arm) would give a different offset on every run and this mode
        # would not be reproducible -- the opposite of the point.
        import zlib
        ev.lidar_seed_offset = 1_000_003 + (zlib.crc32(arm.encode()) % 9973) * 1_000
    else:                                  # 'matched'
        ev.lidar_seed_offset = 1_000_003


def run_arm(arm: str, scenario: str, base_cfg: Dict[str, Any], params,
            *, n_trials: int, master_seed: int, run_dir: str,
            prebuild: bool, deadline_s: Optional[float],
            lidar_seed_mode: str) -> Dict[str, Any]:
    """Evaluate one arm and return its per-trial detail."""
    from evaluation.ep2_evaluator import EP2Evaluator

    cfg = arm_config(base_cfg, arm, prebuild=prebuild, deadline_s=deadline_s)
    ev = EP2Evaluator(cfg)
    _apply_lidar_seed_mode(ev, lidar_seed_mode, arm)
    ev.set_trial_sink(run_dir, f"{scenario}_{arm.replace(':', '')}")
    t0 = time.perf_counter()
    res = ev.evaluate(params, scenario, n_trials=n_trials, master_seed=master_seed)
    trials = res.trial_results
    return {
        'arm': arm, 'scenario': scenario, 'n_trials': len(trials),
        'wall_s': time.perf_counter() - t0,
        'success': [bool(t.success) for t in trials],
        'success_rate': float(res.success_rate),
        'energy': [float(t.energy) for t in trials],
        'time_s': [float(t.time) for t in trials],
        'collision_cause': [t.collision_cause for t in trials],
        'n_steps': [int(t.n_steps) for t in trials],
        # per-step, flattened across trials -- the survival-curve input
        'solver_ms_all': [float(x) for t in trials for x in t.solver_ms_all],
        'N_all': [int(x) for t in trials for x in t.N_all],
        'iter_count_all': [int(x) for t in trials for x in t.iter_count_all],
        'built_all': [bool(x) for t in trials for x in t.built_all],
        'deadline_misses': int(sum(t.deadline_misses for t in trials)),
    }


def run_interleaved(arms: List[str], scenario: str, base_cfg: Dict[str, Any],
                    params, *, n_trials: int, master_seed: int, run_dir: str,
                    prebuild: bool, deadline_s: Optional[float],
                    lidar_seed_mode: str) -> Dict[str, Dict[str, Any]]:
    """Trial-by-trial interleaving: arm A trial k, arm B trial k, arm C trial k...

    Required for any timing comparison because machine
    state drifts: thermal throttling, cache occupancy and any background load all
    change over the minutes an arm takes. Running arm A to completion and then
    arm B attributes that drift to the arm. Interleaving spreads it evenly.

    It also gives matched pairs for free: trial k of every arm uses the same seed,
    so the same start-pose perturbation, obstacle jitter and (with
    ``lidar_seed_mode='matched'``) the same LiDAR noise stream.
    """
    from evaluation.ep2_evaluator import EP2Evaluator
    from mpc.mpc_params import MPCParams

    evs: Dict[str, Any] = {}
    for arm in arms:
        cfg = arm_config(base_cfg, arm, prebuild=prebuild, deadline_s=deadline_s)
        ev = EP2Evaluator(cfg)
        _apply_lidar_seed_mode(ev, lidar_seed_mode, arm)
        ev.set_trial_sink(run_dir, f"{scenario}_{arm.replace(':', '')}")
        evs[arm] = ev

    from env.scenarios import get_scenario
    sc = get_scenario(scenario)
    out: Dict[str, Dict[str, Any]] = {a: {'arm': a, 'scenario': scenario,
                                          'trials': []} for a in arms}
    t0 = time.perf_counter()
    for k in range(n_trials):
        seed = master_seed + k
        for arm in arms:                       # same k, same seed, every arm
            ev = evs[arm]
            r = ev._run_trial(params, sc, seed=seed)
            ev._emit_trial(r, scenario=scenario, trial_index=k,
                           master_seed=master_seed, arm=arm,
                           params=params.to_dict())
            out[arm]['trials'].append(r)
        if (k + 1) % 10 == 0:
            logger.info("interleaved progress: %d/%d trials x %d arms (%.1fs)",
                        k + 1, n_trials, len(arms), time.perf_counter() - t0)

    summary: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        trials = out[arm]['trials']
        summary[arm] = {
            'arm': arm, 'scenario': scenario, 'n_trials': len(trials),
            'wall_s': time.perf_counter() - t0,
            'success': [bool(t.success) for t in trials],
            'success_rate': float(np.mean([t.success for t in trials])),
            'energy': [float(t.energy) for t in trials],
            'time_s': [float(t.time) for t in trials],
            'collision_cause': [t.collision_cause for t in trials],
            'n_steps': [int(t.n_steps) for t in trials],
            'solver_ms_all': [float(x) for t in trials for x in t.solver_ms_all],
            'N_all': [int(x) for t in trials for x in t.N_all],
            'iter_count_all': [int(x) for t in trials for x in t.iter_count_all],
            'built_all': [bool(x) for t in trials for x in t.built_all],
            'deadline_misses': int(sum(t.deadline_misses for t in trials)),
        }
    return summary


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_solvability(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Success rates, exact McNemar against the adaptive arm, Holm over the family."""
    from evaluation.statistics import mcnemar_exact, holm_bonferroni

    ref = summary.get('adaptive')
    rows: Dict[str, Any] = {}
    pvals: Dict[str, float] = {}
    for arm, s in summary.items():
        row = {'success_rate': s['success_rate'], 'n_trials': s['n_trials'],
               'deadline_misses': s['deadline_misses'],
               'collision_causes': {c: s['collision_cause'].count(c)
                                    for c in set(s['collision_cause'])}}
        if ref is not None and arm != 'adaptive':
            mc = mcnemar_exact(ref['success'], s['success'])
            row['mcnemar_vs_adaptive'] = mc
            pvals[arm] = mc['p_value']
        rows[arm] = row
    # ONE declared family: the fixed-N grid against the adaptive law.
    holm = holm_bonferroni(pvals) if pvals else {'family_size': 0, 'results': {}}
    for arm, res in holm.get('results', {}).items():
        rows[arm]['holm'] = res

    fixed = {int(a.split(':')[1]): s['success_rate']
             for a, s in summary.items() if a.startswith('fixed:')}
    full = sorted(n for n, r in fixed.items() if r >= 0.999)
    verdict = _gate_e_verdict(full, fixed)
    return {'arms': rows, 'holm_family': holm, 'fixed_success': fixed,
            'smallest_full_success_N': full[0] if full else None,
            'verdict': verdict}


def _gate_e_verdict(full: List[int], fixed: Dict[int, float]) -> Dict[str, str]:
    """Apply the pre-registered decision rule. No room for interpretation."""
    if full and full[0] <= 24:
        return {
            'outcome': 'FIXED_HORIZON_SUFFICIENT',
            'text': f"A fixed N = {full[0]} <= 24 reaches 100% success. A "
                    f"deadline-feasible fixed horizon solves the scenario, so "
                    f"horizon adaptation is withdrawn from the contribution "
                    f"list.",
        }
    if full and full[0] >= 36:
        return {
            'outcome': 'NO_REALTIME_FEASIBLE_CONFIG',
            'text': f"100% success first appears at N = {full[0]} >= 36. No "
                    f"configuration is real-time feasible on this formulation; the "
                    f"real-time framing must become supervisory (2-3 Hz above a "
                    f"reactive tracker).",
        }
    if full:
        return {
            'outcome': 'BETWEEN_25_AND_35',
            'text': f"100% success first appears at N = {full[0]}, between the two "
                    f"pre-registered thresholds. Record it and decide explicitly; "
                    f"do not pick the reading that suits the claim.",
        }
    return {
        'outcome': 'NO_FIXED_N_SUCCEEDS',
        'text': f"No fixed N in {sorted(fixed)} reaches 100%. Escalate with "
                f"--escalate before drawing any conclusion; if the extended grid "
                f"also fails, the adaptive law is doing something a fixed horizon "
                f"cannot, and THAT is the result.",
    }


def analyse_timing(summary: Dict[str, Dict[str, Any]],
                   deadline_ms: float) -> Dict[str, Any]:
    """Per-step solve-time distributions, survival curve, and the a + b*N model."""
    rows: Dict[str, Any] = {}
    for arm, s in summary.items():
        ms = np.asarray(s['solver_ms_all'], dtype=float)
        if ms.size == 0:
            rows[arm] = {'n_solves': 0}
            continue
        built = np.asarray(s['built_all'], dtype=bool)
        # Report per-STEP quantiles, not statistics of per-trial means. Per-trial
        # means must never be labelled per-step solve time.
        rows[arm] = {
            'n_solves': int(ms.size),
            'mean_ms': float(ms.mean()),
            'median_ms': float(np.median(ms)),
            'p95_ms': float(np.percentile(ms, 95)),
            'p99_ms': float(np.percentile(ms, 99)),
            'max_ms': float(ms.max()),
            'frac_over_deadline': float(np.mean(ms > deadline_ms)),
            'deadline_misses': s['deadline_misses'],
            'n_construction_steps': int(built.sum()),
            'median_ms_excluding_construction': (
                float(np.median(ms[~built])) if (~built).any() else None),
            'median_ms_construction_only': (
                float(np.median(ms[built])) if built.any() else None),
            'survival_curve': _survival(ms),
        }
    return {'arms': rows, 'deadline_ms': deadline_ms,
            'cost_model': fit_cost_model(summary)}


def _survival(ms: np.ndarray, n_points: int = 200) -> Dict[str, List[float]]:
    """P(solve time > t) sampled on a log grid -- the headline real-time artefact.

    The per-step survival curve over ~26,000 steps is the single most
    convincing real-time artefact this pipeline produces, and it is only
    possible because per-step solve times are persisted (solver_ms_all).
    """
    lo = max(float(ms.min()), 1e-3)
    grid = np.geomspace(lo, float(ms.max()) * 1.01, n_points)
    surv = [float(np.mean(ms > t)) for t in grid]
    return {'t_ms': grid.tolist(), 'p_exceed': surv}


def fit_cost_model(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Least-squares fit of solve time ~ a + b*N on UNCHANGED-N, warm steps.

    This is the input to the deadline-feasible scheduler
    ``N_k = max{N : a + b*N <= T_deadline - eps}``. Construction steps are
    excluded: including them fits the
    amortisation artefact instead of the horizon cost, which is the whole error
    the REALTIME null is about.
    """
    Ns: List[float] = []
    ts: List[float] = []
    for s in summary.values():
        ms = np.asarray(s['solver_ms_all'], dtype=float)
        nn = np.asarray(s['N_all'], dtype=float)
        bb = np.asarray(s['built_all'], dtype=bool)
        if ms.size == 0:
            continue
        keep = ~bb
        Ns.extend(nn[keep].tolist())
        ts.extend(ms[keep].tolist())
    if len(Ns) < 3 or len(set(Ns)) < 2:
        return {'ok': False, 'reason': 'need >=2 distinct N on non-construction steps',
                'n_points': len(Ns)}
    N = np.asarray(Ns); T = np.asarray(ts)
    A = np.vstack([np.ones_like(N), N]).T
    (a, b), *_ = np.linalg.lstsq(A, T, rcond=None)
    resid = T - (a + b * N)
    ss_tot = float(((T - T.mean()) ** 2).sum())
    return {'ok': True, 'a_ms': float(a), 'b_ms_per_N': float(b),
            'r2': float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else None,
            'n_points': int(N.size), 'N_range': [float(N.min()), float(N.max())],
            'note': 'fitted on unchanged-N warm steps only; construction steps '
                    'excluded so the fit measures horizon cost, not amortisation'}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)

    for name, helptext in [('solvability', 'success rates + exact McNemar (load-insensitive)'),
                           ('timing', 'per-step solve times + cost model (EXCLUSIVE machine)')]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument('--run-id', required=True)
        p.add_argument('--scenario', default='s_curve',
                       choices=['corridor', 's_curve', 'chicane'])
        p.add_argument('--results-dir', default='results')
        p.add_argument('--evolution-root', default=None,
                       help='Directory containing evolution/<scenario>/ from '
                            'which the genome loads (e.g. runs/<id>/reevolution). '
                            'Default: config results_dir = the TRACKED '
                            'pre-reevolution archive.')
        p.add_argument('--config', default=None)
        p.add_argument('--arms', default='default',
                       help="'default' for adaptive + fixed 12/16/20/24, or a "
                            "comma list like 'adaptive,fixed:12,fixed:36'")
        p.add_argument('--escalate', action='store_true',
                       help='add N in 28/32/36/40/50 to the default grid')
        p.add_argument('--n-trials', type=int, default=50)
        p.add_argument('--master-seed', type=int, default=7000)
        p.add_argument('--interleave', dest='interleave', action='store_true',
                       default=(name == 'timing'),
                       help='trial-by-trial interleaving across arms '
                            '(default ON for timing, OFF for solvability)')
        p.add_argument('--no-interleave', dest='interleave', action='store_false')
        p.add_argument('--prebuild', action='store_true',
                       default=(name == 'timing'),
                       help='pre-build the solver bank so no solve pays NLP '
                            'construction (default ON for timing)')
        p.add_argument('--lidar-seed-mode', default='matched',
                       choices=['matched', 'per-arm', 'off'],
                       help="'matched' (default): trial k uses the same LiDAR "
                            "seed in every arm, so arms are genuinely paired. "
                            "'off': unseeded, not reproducible -- diagnostics only.")
        p.add_argument('--deadline-ms', type=float, default=100.0)
        p.add_argument('--cpus', type=int, nargs='*', default=None,
                       help='timing only: reserved logical CPUs. The process is '
                            'PINNED to them (sched_setaffinity) and the preflight '
                            'gates on their busy fraction instead of on global '
                            'load average. Required on any box that also runs an '
                            'interactive desktop, where global load never drops '
                            'below ~1.5. Get a suggestion with: python -m '
                            'utils.preflight --mode timing --suggest-cpus 4')
        p.add_argument('--calibrate', action='store_true',
                       help='fit and report the a + b*N solver cost model')
        p.add_argument('--dry-run', action='store_true',
                       help='validate and print the plan; execute nothing')
        p.add_argument('--force', action='store_true')

    args = ap.parse_args(argv)

    from utils.config_loader import load_config
    from utils.checkpoint_utils import load_best_evolved_params
    from utils.safe_write import safe_write_json, run_id_dir
    from utils.provenance import git_commit_info
    from utils import preflight

    arms = parse_arms(args.arms, escalate=args.escalate)
    base_cfg = load_config(args.config)
    base_cfg.setdefault('evaluation', {})['progress_every_trials'] = 0
    results_dir = base_cfg.get('results_dir', args.results_dir)
    from utils.paths import run_dir as _run_dir
    run_dir = _run_dir(args.run_id, 'gate_e', create=not args.dry_run)

    print(f"GATE-E / {args.mode}")
    print(f"  scenario     : {args.scenario}")
    print(f"  arms         : {arms}")
    print(f"  trials/arm   : {args.n_trials}   seeds [{args.master_seed}, "
          f"{args.master_seed + args.n_trials - 1}]")
    print(f"  interleave   : {args.interleave}")
    print(f"  prebuild     : {args.prebuild}")
    print(f"  lidar seeds  : {args.lidar_seed_mode}")
    print(f"  run dir      : {run_dir}")
    print(f"  total trials : {len(arms) * args.n_trials}")

    if args.mode == 'timing':
        if args.cpus:
            # Pin BEFORE measuring anything. Without this the gate would check
            # cores the process then wanders off.
            try:
                os.sched_setaffinity(0, set(args.cpus))
                print(f"  pinned to CPUs   : {sorted(os.sched_getaffinity(0))}")
            except (AttributeError, OSError) as exc:
                print(f"\nRefusing: could not pin to CPUs {args.cpus}: {exc}")
                return 2
        pf = preflight.check('timing', cpus=args.cpus)
        print()
        print(preflight.format_report(pf))
        if not pf['ok']:
            print("\nRefusing to run the timing half on a loaded machine. There is "
                  "no override flag: cross-process solve times differ 4-6x for "
                  "identical genomes and the resulting numbers look publishable.")
            if not args.cpus:
                print("If the residual load is an interactive desktop rather than "
                      "another campaign, reserve cores instead of waiting:\n"
                      "  CPUS=$(python -m utils.preflight --mode timing "
                      "--suggest-cpus 4)\n"
                      "  ... 13_horizon_isolation.py timing --cpus $CPUS ...")
            return 2
        if args.interleave is False:
            print("\nRefusing: timing comparisons require trial-by-trial "
                  "interleaving. Drop --no-interleave.")
            return 2

    if args.dry_run:
        print("\n--dry-run: plan valid, nothing executed.")
        return 0

    # --evolution-root (added 2026-09-11, same defect as 04/05/
    # collect_trajectories): the genome source must be sayable, or a gate
    # measures the tracked pre-reevolution genome while the run directory
    # looks like it describes the new one. The first 2026-09-11 gate_e
    # re-run hit exactly this (quarantined); its verdict happened to match,
    # but provenance must be correct, not accidentally so.
    genome_dir = getattr(args, 'evolution_root', None) or results_dir
    params = load_best_evolved_params(args.scenario, genome_dir)
    if getattr(args, 'evolution_root', None) is None:
        logger.info("genome source: %s (config fallback -- the TRACKED "
                    "pre-reevolution archive)", genome_dir)
        logger.info("genome: LLM-guided provenance -- stated on any interim "
                    "result until the plain-NSGA-II re-tune")
        genome_provenance = ('LLM-guided search (tracked archive); no LLM '
                             'claim is made')
    else:
        logger.info("genome source: %s", genome_dir)
        genome_provenance = f'plain NSGA-II, LLM disabled ({genome_dir})'

    if args.interleave:
        summary = run_interleaved(
            arms, args.scenario, base_cfg, params, n_trials=args.n_trials,
            master_seed=args.master_seed, run_dir=run_dir,
            prebuild=args.prebuild, deadline_s=args.deadline_ms / 1000.0,
            lidar_seed_mode=args.lidar_seed_mode)
    else:
        summary = {a: run_arm(a, args.scenario, base_cfg, params,
                              n_trials=args.n_trials, master_seed=args.master_seed,
                              run_dir=run_dir, prebuild=args.prebuild,
                              deadline_s=args.deadline_ms / 1000.0,
                              lidar_seed_mode=args.lidar_seed_mode)
                   for a in arms}

    payload: Dict[str, Any] = {
        'mode': args.mode, 'run_id': args.run_id, 'scenario': args.scenario,
        'arms': arms, 'n_trials': args.n_trials, 'master_seed': args.master_seed,
        'interleaved': args.interleave, 'prebuild': args.prebuild,
        'lidar_seed_mode': args.lidar_seed_mode,
        'genome_provenance': genome_provenance,
        'reserved_cpus': sorted(args.cpus) if args.cpus else None,
        'cpu_affinity_at_run': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'preflight': {k: v for k, v in (pf.items() if args.mode == 'timing' else []) if k != 'foreign_python'},
        'git': git_commit_info(),
    }
    if args.mode == 'solvability':
        payload['solvability'] = analyse_solvability(summary)
        v = payload['solvability']['verdict']
        print(f"\nsuccess rates: "
              f"{ {a: round(s['success_rate'], 3) for a, s in summary.items()} }")
        print(f"\nGATE-E VERDICT [{v['outcome']}]\n  {v['text']}")
    else:
        payload['timing'] = analyse_timing(summary, args.deadline_ms)
        for a, r in payload['timing']['arms'].items():
            if r.get('n_solves'):
                print(f"  {a:<14} n={r['n_solves']:6d}  median={r['median_ms']:7.1f} "
                      f"p95={r['p95_ms']:7.1f}  over-deadline={100*r['frac_over_deadline']:5.1f}%")
        cm = payload['timing']['cost_model']
        if cm.get('ok'):
            print(f"  cost model: {cm['a_ms']:.2f} + {cm['b_ms_per_N']:.3f}*N ms "
                  f"(R2={cm['r2']:.3f}, n={cm['n_points']})")

    out = os.path.join(run_dir, f"gate_e_{args.mode}_{args.scenario}.json")
    safe_write_json(out, payload, force=args.force)
    print(f"\nwritten: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
