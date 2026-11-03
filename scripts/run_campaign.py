#!/usr/bin/env python3
"""Fan-out campaign runner. One process per physical core, explicit seeding.

No parallel harness existed before this file: `grep -rn
"multiprocessing|ProcessPool|concurrent.futures|joblib"` over the project
returned nothing, and `evaluation/ep2_evaluator.py` is a plain for-loop. The old
box was one serial lane at 30/72/64 s per trial. So this is built from scratch rather
than ported, and it commits to four non-negotiable rules:

1. **One process per physical core, single-threaded BLAS.** Each worker gets
   OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_MAXIMUM_THREADS=1 in its own environment. 192
   processes each spawning 192 BLAS threads runs slower than a laptop.
2. **Explicit per-worker seeding, never fork inheritance.** The start method is
   forced to 'spawn'. Under 'fork' a worker inherits the parent's RNG state, so
   N workers can silently produce identical "random" trials -- and the tracked
   corpus was generated on Windows, where 'spawn' is the only option, so a
   fork-based harness would not even reproduce the recorded protocol.
3. **`nice -n 19`.** This box is shared. Workers must yield to anything else.
4. **A hard --max-workers cap, and a --dry-run that validates the whole plan
   without executing.** The default cap is deliberately conservative; the real
   number comes from the preflight's recommendation or from the operator.

Every task writes one JSONL record per trial into the run-id directory, so an
interrupted campaign keeps everything up to the interruption.

Usage:
    python scripts/run_campaign.py --dry-run --run-id run_x --stage ep2
    python scripts/run_campaign.py --run-id run_x --stage ep2      --max-workers 100
    python scripts/run_campaign.py --run-id run_x --stage baselines --max-workers 100
    python scripts/run_campaign.py --run-id run_x --stage gate_r2   --max-workers 10
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCENARIOS = ('corridor', 's_curve', 'chicane')

#: Conservative default. The operator or the preflight sets the real number.
DEFAULT_MAX_WORKERS = 8

#: Every worker exports these before importing numpy/casadi.
SINGLE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'MPLBACKEND': 'Agg',
}


@dataclass
class Task:
    """One unit of parallel work: a scenario x arm x seed-block.

    ``master_seed`` is assigned by the planner, not derived from the worker's
    identity or from any inherited RNG state, so the plan is reproducible
    independently of how many workers happen to run it.
    """
    stage: str
    scenario: str
    arm: str
    master_seed: int
    n_trials: int
    run_dir: str
    config_path: Optional[str] = None
    param_overrides: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    #: Root CONTAINING evolution/<scenario>/. Explicit, and recorded in the
    #: manifest, because the default is the TRACKED pre-C1 archive: without it a
    #: campaign meant to validate a re-evolved genome silently measures the old one
    #: and reports the result as new. That is the provenance failure this rebuild
    #: exists to eliminate, so it must appear in the command line and the manifest.
    evolution_root: Optional[str] = None

    @property
    def tag(self) -> str:
        return f"{self.scenario}_{self.arm}_seed{self.master_seed}"


def _apply_single_thread_env() -> None:
    """Set BLAS thread limits. Must run BEFORE numpy/casadi are imported."""
    for k, v in SINGLE_THREAD_ENV.items():
        os.environ[k] = v


def _renice() -> None:
    try:
        os.nice(19 - os.nice(0))
    except (OSError, AttributeError):
        pass


def run_task(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry point. Module-level and dict-argumented so it pickles.

    Runs in a *spawned* interpreter: nothing is inherited from the parent, which
    is the point. Seeds come from the task, thread limits from the environment we
    set here before the first numpy import.
    """
    _apply_single_thread_env()
    _renice()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    task = Task(**task_dict)
    t0 = time.perf_counter()
    out: Dict[str, Any] = {'tag': task.tag, 'pid': os.getpid(),
                           'scenario': task.scenario, 'arm': task.arm,
                           'master_seed': task.master_seed}
    try:
        import numpy as np
        from utils.config_loader import load_config
        from utils.checkpoint_utils import load_best_evolved_params
        from evaluation.ep2_evaluator import EP2Evaluator

        # Belt and braces: prove the limits actually took effect in this process.
        out['blas_threads'] = {k: os.environ.get(k) for k in SINGLE_THREAD_ENV}

        config = load_config(task.config_path)
        config.setdefault('evaluation', {})['progress_every_trials'] = 0
        # Build the solver bank once per worker, outside any timed window.
        config.setdefault('mpc', {})['prebuild_solver_bank'] = True

        # ---- arm dispatch -------------------------------------------------
        # This MUST branch on the arm. An earlier version ran EP2Evaluator for
        # every task regardless, so `--stage baselines` would have produced 30
        # EP2-MPC runs labelled as baselines -- worse than having no stage at all,
        # because the mislabelling is invisible in the aggregates.
        if task.arm in ('apf', 'standard_mpc_casadi'):
            from env.scenarios import get_scenario
            from evaluation.baselines import (
                APFBaseline, run_baseline, run_standard_mpc_casadi_baseline)
            scenario = get_scenario(task.scenario)
            robot = config.get('robot')
            if not robot:
                raise ValueError("config['robot'] is required for baseline arms")
            if task.arm == 'apf':
                ctrl = APFBaseline(robot=robot, dt=config['mpc'].get('dt', 0.1))
                res = run_baseline(ctrl, scenario, n_trials=task.n_trials,
                                   config=config, master_seed=task.master_seed,
                                   run_dir=task.run_dir, tag=task.tag)
            else:
                # scenario NAME, not the ScenarioConfig: this wrapper resolves the
                # scenario itself. Passing the object gave
                # "AttributeError: 'ScenarioConfig' object has no attribute 'lower'"
                # in all 15 standard_mpc tasks of the first campaign run.
                res = run_standard_mpc_casadi_baseline(
                    task.scenario, n_trials=task.n_trials, config=config,
                    master_seed=task.master_seed,
                    run_dir=task.run_dir, tag=task.tag)
            out.update({
                'ok': True, 'n_trials': res.n_trials,
                'success_rate': res.success_rate,
                'energy_mean': res.energy_mean, 'energy_std': res.energy_std,
                'time_mean': res.time_mean,
                'solver_ms_mean_of_trial_means': res.solver_ms_mean,
                'controller': task.arm,
                'note': 'baseline arm; solver_ms_mean is a mean OF PER-TRIAL MEANS',
            })
            out['wall_s'] = time.perf_counter() - t0
            return out

        evo_root = task.evolution_root or config.get('results_dir', 'results')
        out['evolution_root'] = evo_root
        params = load_best_evolved_params(task.scenario, evo_root)
        # Record the genome actually used, so a manifest can be checked against the
        # numbers it claims rather than trusted.
        out['genome'] = params.to_dict()
        if task.param_overrides:
            # GATE-R2 and any other parameter sweep: start from the deployed
            # genome and override named weights, so the sweep varies exactly one
            # thing and everything else stays matched.
            from dataclasses import replace as _replace
            ov = dict(task.param_overrides)
            mult = ov.pop('_w_p_multiplier', None)
            if mult is not None:
                ov['w_p'] = float(params.w_p) * float(mult)
            params = _replace(params, **ov)
            out['param_overrides'] = dict(task.param_overrides)
            out['resolved_w_p'] = float(params.w_p)
            out['resolved_w_E'] = float(params.w_E)
        out['params'] = params.to_dict()
        out['controller'] = 'ep2mpc'
        ev = EP2Evaluator(config)
        ev.set_trial_sink(task.run_dir, task.tag)
        res = ev.evaluate(params, task.scenario,
                          n_trials=task.n_trials, master_seed=task.master_seed)
        out.update({
            'ok': True,
            'n_trials': res.n_trials,
            'success_rate': res.success_rate,
            'energy_mean': res.energy_mean,
            'energy_std': res.energy_std,
            'time_mean': res.time_mean,
            'solver_ms_mean_of_trial_means': res.solver_ms_mean,
            'note': 'solver_ms_mean is a mean OF PER-TRIAL MEANS, not a '
                    'per-step statistic; '
                    'per-step samples are in the JSONL, field solver_ms_all',
        })
    except BaseException as exc:                     # noqa: BLE001 - report, never crash the pool
        out.update({'ok': False, 'error': f"{type(exc).__name__}: {exc}"})
    out['wall_s'] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

#: GATE-R2: sweep w_p over these multiples of nominal with w_E = 0.
#: The point is to BOUND what any energy-side claim can be worth, not to find a
#: better genome. Predicted span is <=3% and non-monotone (U-shaped about v*,
#: because slowing below v* raises J/m -- 8.45 J/m at v = 0.5 -- and lengthens
#: idle accrual against c5 = 3 W).
GATE_R2_W_P_MULTIPLIERS = (0.01, 0.1, 1.0, 10.0, 100.0)

#: Scenarios GATE-R2 runs on (corridor and s_curve).
GATE_R2_SCENARIOS = ('corridor', 's_curve')


def build_plan(stage: str, run_dir: str, *, scenarios=SCENARIOS, n_seeds: int = 5,
               n_trials: int = 50, seed_base: int = 1000,
               config_path: Optional[str] = None,
               evolution_root: Optional[str] = None) -> List[Task]:
    """Enumerate every task up front, with its seed fixed by position.

    Seeds are ``seed_base + 10_000*scenario_index + 1_000*seed_index``, spaced so
    that no two blocks can overlap even if n_trials is later increased -- a
    silent seed collision would make two "independent" arms share trials.
    """
    tasks: List[Task] = []

    if stage == 'gate_r2':
        # One arm per w_p multiplier, w_E pinned to 0.
        #
        # Every arm gets the SAME seed block per scenario. That is not a detail:
        # the predicted attainable energy span is <=3%, which is smaller
        # than the per-trial sigma_E of 0.62-0.69 J. With unpaired seeds a large
        # part of the observed span would be seed noise rather than the w_p
        # effect, and the gate would "measure" a range it cannot resolve. The
        # plan validator enforces this.
        maps = [s for s in scenarios if s in GATE_R2_SCENARIOS]
        if not maps:
            raise ValueError(f"GATE-R2 runs on {GATE_R2_SCENARIOS}; got {scenarios}")
        for si, scenario in enumerate(maps):
            seed = seed_base + 10_000 * si
            for mult in GATE_R2_W_P_MULTIPLIERS:
                tasks.append(Task(
                    stage=stage, scenario=scenario, arm=f'wp{mult:g}x_wE0',
                    master_seed=seed,
                    n_trials=n_trials, run_dir=run_dir, config_path=config_path,
                    param_overrides={'w_E': 0.0, '_w_p_multiplier': mult},
                    evolution_root=evolution_root,
                ))
        return tasks

    arms = {'ep2': ['ep2mpc'], 'baselines': ['apf', 'standard_mpc_casadi']}.get(stage)
    if arms is None:
        raise ValueError(f"unknown stage {stage!r}; "
                         f"known: ep2, baselines, gate_r2")
    for si, scenario in enumerate(scenarios):
        for arm in arms:
            for k in range(n_seeds):
                tasks.append(Task(
                    stage=stage, scenario=scenario, arm=arm,
                    master_seed=seed_base + 10_000 * si + 1_000 * k,
                    n_trials=n_trials, run_dir=run_dir, config_path=config_path,
                    evolution_root=evolution_root,
                ))
    return tasks


def validate_plan(tasks: List[Task], max_workers: int) -> List[str]:
    """Everything checkable without executing. Returns a list of problems."""
    problems: List[str] = []
    if not tasks:
        problems.append("plan is empty")

    # Seed blocks must not overlap WITHIN one (scenario, arm) -- that would mean
    # the same trial counted twice. ACROSS arms the same seeds are mandatory, not
    # a collision: matched seeds are what makes the comparison paired, and the
    # whole statistics protocol (paired Wilcoxon, exact McNemar, bootstrap over
    # matched pairs) depends on arm A trial k and arm B trial k being the same
    # perturbation. An earlier version of this check flagged that as an error and
    # refused every multi-arm stage.
    by_arm: Dict[tuple, List[tuple]] = {}
    for t in tasks:
        by_arm.setdefault((t.scenario, t.arm), []).append(
            (t.master_seed, t.master_seed + t.n_trials - 1, t.tag))
    for (scenario, arm), blocks in by_arm.items():
        blocks.sort()
        for (lo, hi, tag), (olo, ohi, otag) in zip(blocks, blocks[1:]):
            if ohi >= lo and olo <= hi:
                problems.append(
                    f"seed range collision within {scenario}/{arm}: "
                    f"{tag} [{lo},{hi}] overlaps {otag} [{olo},{ohi}]")

    # Cross-arm seed alignment is a REQUIREMENT, so check it holds.
    per_scenario: Dict[str, Dict[str, set]] = {}
    for t in tasks:
        per_scenario.setdefault(t.scenario, {}).setdefault(t.arm, set()).add(t.master_seed)
    for scenario, arms in per_scenario.items():
        if len(arms) > 1:
            seed_sets = list(arms.values())
            if any(s != seed_sets[0] for s in seed_sets[1:]):
                problems.append(
                    f"{scenario}: arms do not share the same seed set, so the "
                    f"comparison would not be paired: "
                    f"{ {a: sorted(s) for a, s in arms.items()} }")

    try:
        from env.scenarios import get_scenario
        for scenario in {t.scenario for t in tasks}:
            get_scenario(scenario)
    except Exception as exc:                          # noqa: BLE001
        problems.append(f"scenario will not load: {exc}")

    if max_workers < 1:
        problems.append("--max-workers must be >= 1")
    out_dirs = {t.run_dir for t in tasks}
    for d in out_dirs:
        # The run dir is created at execution time, so validate the nearest
        # EXISTING ancestor rather than requiring the immediate parent to exist.
        probe = os.path.abspath(d)
        while not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if not os.access(probe, os.W_OK):
            problems.append(f"cannot create run dir under {probe} (not writable)")
    return problems


def print_plan(tasks: List[Task], max_workers: int, per_trial_s: float) -> None:
    total_trials = sum(t.n_trials for t in tasks)
    waves = (len(tasks) + max_workers - 1) // max_workers
    serial_h = total_trials * per_trial_s / 3600.0
    print(f"  tasks              : {len(tasks)}")
    print(f"  trials             : {total_trials}")
    print(f"  max workers        : {max_workers}")
    print(f"  waves              : {waves}")
    print(f"  serial equivalent  : {serial_h:.2f} h at {per_trial_s:.0f} s/trial")
    print(f"  wall-clock estimate: {serial_h / min(max_workers, len(tasks)):.2f} h "
          f"(perfect scaling; add ~10 s/worker for spawn + solver bank)")
    print(f"  {'tag':<40} {'seeds':<18} trials")
    for t in tasks:
        print(f"  {t.tag:<40} [{t.master_seed},{t.master_seed + t.n_trials - 1}]"
              f"{'':<3} {t.n_trials}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-id', required=True,
                    help='REQUIRED. One id threaded through every stage of a '
                         'campaign, so "which run produced this" has one answer.')
    ap.add_argument('--stage', required=True,
                    choices=['ep2', 'baselines', 'gate_r2'],
                    help="'ep2' the main EP2-MPC campaign; 'baselines' the "
                         "comparator arms; 'gate_r2' the w_p sweep with w_E = 0 "
                         "that bounds any energy-side claim.")
    ap.add_argument('--results-dir', default='results',
                    help='Root under which <results-dir>/campaigns/<run-id>/ is created.')
    ap.add_argument('--config', default=None)
    ap.add_argument('--evolution-root', dest='evolution_root', default=None,
                    help='Root CONTAINING evolution/<scenario>/ from which the '
                         'EP2-MPC genome is loaded. Unset falls back to the '
                         "config's results_dir, which is the TRACKED PRE-C1 "
                         'archive -- so to validate a re-evolved genome you MUST '
                         'pass this, e.g. runs/<RUN_ID>/reevolution. Without it a '
                         'campaign meant to measure the new genome silently '
                         'measures the old one and reports it as new. The value and '
                         'the genome actually loaded are both recorded per task in '
                         'the campaign manifest.')
    ap.add_argument('--scenarios', nargs='*', default=list(SCENARIOS), choices=list(SCENARIOS))
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS,
                    help=f'HARD cap on concurrent processes (default '
                         f'{DEFAULT_MAX_WORKERS}, deliberately conservative). '
                         f'Never exceeds the physical core count.')
    ap.add_argument('--per-trial-s', type=float, default=45.0,
                    help='Estimate used only for the wall-clock projection.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Validate and print the plan; execute nothing.')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='Bulk stages only. Timing stages never skip.')
    args = ap.parse_args(argv)

    from utils.preflight import check, format_report, physical_cores
    from utils.paths import run_dir as _run_dir, check_space, runs_target

    cores = physical_cores()
    max_workers = max(1, min(args.max_workers, cores))
    if max_workers != args.max_workers:
        print(f"NOTE: --max-workers {args.max_workers} clamped to {max_workers} "
              f"(physical cores = {cores})")

    # Output goes to the runs root on /data, reached through the repo symlink.
    # --results-dir remains the READ root for the tracked record (genomes,
    # scenario configs); it is not where output lands.
    run_dir = _run_dir(args.run_id, args.stage, create=not args.dry_run)
    space = check_space()
    tasks = build_plan(args.stage, run_dir, scenarios=tuple(args.scenarios),
                       n_seeds=args.n_seeds, n_trials=args.n_trials,
                       config_path=args.config,
                       evolution_root=args.evolution_root)

    print(f"campaign run-id : {args.run_id}")
    print(f"stage           : {args.stage}   (bulk / load-insensitive)")
    _evo_note = (args.evolution_root if args.evolution_root
                 else '(unset -> config results_dir, the TRACKED pre-C1 archive)')
    print(f"evolution root  : {_evo_note}")
    print(f"run dir         : {run_dir}")
    print(f"                  -> {os.path.realpath(run_dir)}")
    print(f"output device    : {space['free_gb']:.1f} GB free on "
          f"{space['target']}  (separate device: {not space['same_device_as_repo']})")
    problems = validate_plan(tasks, max_workers)
    if not space['ok']:
        problems.append(
            f"only {space['free_gb']:.1f} GB free on the output device "
            f"{space['target']} (need >= {space['min_free_gb']} GB)")
    print_plan(tasks, max_workers, args.per_trial_s)

    pf = check('bulk', max_workers=max_workers)
    print()
    print(format_report(pf))
    if problems:
        print("\nPLAN INVALID:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if not pf['ok'] and not args.skip_preflight:
        print("\nRefusing to start: preflight failed. Re-run with --skip-preflight "
              "only if you accept the throughput hit (bulk stages tolerate load; "
              "timing stages never do).")
        return 2
    if args.dry_run:
        print("\n--dry-run: plan is valid and nothing was executed.")
        return 0

    os.makedirs(run_dir, exist_ok=True)
    from utils.paths import write_run_metadata
    write_run_metadata(args.run_id, {'stages_started': [args.stage]})
    # 'spawn', not 'fork': a forked worker inherits the parent's RNG state, so N
    # workers can produce identical "random" trials without any error.
    ctx = mp.get_context('spawn')
    payloads = [asdict(t) for t in tasks]
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ctx.Pool(processes=max_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_task, payloads):
            results.append(r)
            state = 'ok' if r.get('ok') else f"FAILED {r.get('error')}"
            print(f"  [{len(results)}/{len(tasks)}] {r['tag']:<40} "
                  f"{r['wall_s']:7.1f}s  {state}")

    manifest = os.path.join(run_dir, 'campaign_manifest.json')
    from utils.safe_write import safe_write_json
    from utils.provenance import git_commit_info
    safe_write_json(manifest, {
        'run_id': args.run_id, 'stage': args.stage,
        'max_workers': max_workers, 'physical_cores': cores,
        'wall_s': time.perf_counter() - t0,
        'git': git_commit_info(),
        'preflight': {k: v for k, v in pf.items() if k != 'foreign_python'},
        'single_thread_env': SINGLE_THREAD_ENV,
        'tasks': payloads, 'results': results,
    })
    n_bad = sum(1 for r in results if not r.get('ok'))
    print(f"\n{len(results) - n_bad}/{len(results)} tasks ok in "
          f"{time.perf_counter() - t0:.1f}s   manifest: {manifest}")
    return 1 if n_bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
