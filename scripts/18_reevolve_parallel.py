#!/usr/bin/env python3
"""Re-evolve the EP2-MPC genome in the CORRECTED landscape, plain NSGA-II, in parallel.

Why this exists
---------------
The three deployed genomes in ``results/evolution/*/pareto_gen019.json`` were
searched under a cost landscape that no longer exists:

  * pre-C1 geometry -- no ``r_body`` in ``d_surface``, no C2 hinge, so the obstacle
    penalty activated at a different point on every step of every trial;
  * a 35 s trial timeout and ``N_max = 24``, neither of which is in any config today;
  * comparators that were crippled in ways since fixed, which does not
    change the genome's own fitness but does change what "good" was calibrated
    against.

So EP2-MPC is currently being scored in a landscape it was never fitted to, and it
loses to a corrected standard MPC on corridor and s_curve by 1.25 J and 0.48 J
(the paired n = 250 tests, CIs excluding zero). Re-evolving is the only honest route to
an energy claim: either the genome recovers the margin in the corrected landscape,
or it does not and the paper says so.

This runs the LLM **disabled** -- plain NSGA-II. The
reported genome must not have LLM provenance.

Parallelism
-----------
``LLMEvolution.run`` already exposes an ``eval_hook`` (added for the abandoned
distributed path) that receives the whole population and returns one dict per
candidate. That is the correct injection point: the genetic operators, archive,
hypervolume and checkpointing stay exactly as they are, and only the evaluation
fans out. One spawned process per candidate, single-threaded BLAS, nice 19.

Serial, one generation of 16 candidates x 50 trials is ~40 min; the original search
was 20 generations of that per map, i.e. ~12 h each. Fanned out, a generation costs
one candidate's time, which makes a much larger search affordable -- and a
better-converged front is a better genome is a better energy number.

Output safety
-------------
``LLMEvolution`` writes to ``<results_dir>/evolution/<scenario>/``, and with the
default ``results_dir`` that is the tracked, IRREPLACEABLE
``results/evolution/<scenario>/pareto_gen019.json``: corridor and s_curve were
evolved at ``N_max = 24`` under a 35 s timeout and cannot be re-derived. This script
therefore overrides ``results_dir`` to a run-scoped directory and REFUSES to run if
the resolved path lands inside the repo's ``results/``.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SINGLE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1', 'MPLBACKEND': 'Agg',
}

#: Objective vector on an evaluation that raised. Mirrors LLMEvolution's own
#: fallback exactly so a crashed candidate is ranked identically either way.
FALLBACK_OBJ = [5000.0, 30.0, 800.0, 1.0]
FALLBACK_EVAL = {
    'energy_mean': 5000.0, 'energy_std': 0.0, 'time_mean': 30.0, 'time_std': 0.0,
    'solver_ms_mean': 800.0, 'solver_ms_std': 0.0, 'success_rate': 0.0,
    'path_distance_mean': 0.0, 'path_distance_std': 0.0, 'n_trials': 0,
    'trial_results': [], 'solver_status_counts': {},
}


def eval_one(payload: dict) -> dict:
    """Evaluate one genome. Runs in a spawned child."""
    os.environ.update(SINGLE_THREAD_ENV)
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass
    logging.basicConfig(level=logging.ERROR)

    from dataclasses import asdict
    from mpc.mpc_params import MPCParams
    from evaluation.ep2_evaluator import EP2Evaluator

    idx = payload['index']
    t0 = time.time()
    try:
        cfg = payload['config']
        params = MPCParams.from_dict(payload['params'])
        ev = EP2Evaluator(cfg)
        res = ev.evaluate(params, payload['scenario'],
                          n_trials=payload['n_trials'],
                          master_seed=payload['master_seed'])
        try:
            d = asdict(res)
        except TypeError:
            d = {k: getattr(res, k) for k in
                 ('energy_mean', 'time_mean', 'solver_ms_mean', 'success_rate')}
        d['objectives'] = list(np.asarray(res.objectives, dtype=float))
        d['index'] = idx
        d['elapsed_sec'] = time.time() - t0
        # trial_results are TrialResult dataclasses; keep them but make them JSON
        # safe, because _save_candidate_result serialises whatever it is handed.
        return d
    except Exception as exc:
        out = dict(FALLBACK_EVAL)
        out['objectives'] = list(FALLBACK_OBJ)
        out['index'] = idx
        out['elapsed_sec'] = time.time() - t0
        out['error'] = f'{type(exc).__name__}: {exc}'
        return out


def build_hook(pool, config: dict, scenario: str, n_trials: int,
               master_seed: int, log_every: int = 1):
    """Return an eval_hook(population, gen) that fans the population out."""
    state = {'gen_t0': None}

    def hook(population, gen) -> List[Dict[str, Any]]:
        t0 = time.time()
        payloads = [{'index': i, 'params': p.to_dict(), 'config': config,
                     'scenario': scenario, 'n_trials': n_trials,
                     'master_seed': master_seed}
                    for i, p in enumerate(population)]
        # imap_unordered then reorder: LLMEvolution indexes the returned list
        # positionally against `population`, so returning them out of order would
        # silently attach every candidate's objectives to the wrong genome.
        got = []
        n = len(payloads)
        for d in pool.imap_unordered(eval_one, payloads):
            got.append(d)
            # Without this the log is silent for the entire generation, because
            # LLMEvolution only persists candidates after the hook returns. 33
            # minutes of no output is indistinguishable from a hang.
            if len(got) % max(1, n // 6) == 0 or len(got) == n:
                logger.info("  gen %d: %d/%d candidates (%.0fs elapsed)",
                            gen, len(got), n, time.time() - t0)
        got.sort(key=lambda d: d['index'])
        assert [d['index'] for d in got] == list(range(len(population))), \
            'candidate/result misalignment'
        n_err = sum(1 for d in got if d.get('error'))
        e = [d['objectives'][0] for d in got]
        f = [d['objectives'][3] for d in got]
        ok = [ei for ei, fi in zip(e, f) if fi < 0.1]
        logger.info("gen %d: %d candidates in %.0fs  errors=%d  "
                    "best_E(success)=%s  n_success=%d",
                    gen, len(got), time.time() - t0, n_err,
                    f'{min(ok):.2f}' if ok else '-', len(ok))
        state['gen_t0'] = t0
        return got

    return hook


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', required=True,
                    choices=('corridor', 's_curve', 'chicane'))
    ap.add_argument('--run-id', dest='run_id', required=True,
                    help='REQUIRED. Output goes to '
                         '<runs>/<run-id>/reevolution/evolution/<scenario>/.')
    ap.add_argument('--pop-size', type=int, default=64,
                    help='4x the original 16. The fan-out makes a generation cost '
                         'one candidate, so population is nearly free and a wider '
                         'front is a better genome.')
    ap.add_argument('--generations', type=int, default=40,
                    help='2x the original 20.')
    ap.add_argument('--n-trials', type=int, default=20,
                    help='trials per candidate evaluation DURING SEARCH. 20, not '
                         '50: sigma_E is 0.62-0.69 J/trial so 20 trials give a SEM '
                         'of ~0.15 J, ample to RANK candidates whose energies '
                         'differ by 1-10 J. The winning genome is validated at '
                         '50 trials x 5 seeds afterwards, which is where the '
                         'reported number comes from.')
    ap.add_argument('--timeout-per-trial', type=float, default=30.0,
                    help='seconds per trial during search. The config default is '
                         '60 s, a safety margin, and it sets the WORST CASE for a '
                         'whole generation: the pool barrier means a generation '
                         'takes as long as its slowest candidate, so one genome '
                         'that never reaches the goal costs 50 x 60 s = 50 min. '
                         'Measured nominal traversal is 17.0/19.6/18.0 s, so 30 s '
                         'is still ~1.5x nominal. A genome needing more than that '
                         'is spending far too much energy to ever be selected, so '
                         'cutting it early is conservative, not lossy.')
    ap.add_argument('--master-seed', type=int, default=0)
    ap.add_argument('--init-seed', dest='init_seed', type=int, default=None,
                    help='Seed for the STRATIFIED INITIAL POPULATION, enabling a '
                         'true independent replicate. LLMEvolution.run hardcodes '
                         'StratifiedPopulation(seed=42), so two runs differing only '
                         'in --master-seed start from the SAME 48 genomes and are '
                         'not independent trajectories -- they differ only through '
                         'evaluation noise propagating into selection. A negative '
                         'result from one stochastic search is weak evidence; this '
                         'is what makes replication possible.')
    ap.add_argument('--max-workers', type=int, default=64)
    ap.add_argument('--config', default=None)
    ap.add_argument('--tag', default=None,
                    help='extra subdirectory, e.g. to keep two search sizes apart')
    args = ap.parse_args(argv)

    from utils.config_loader import load_config
    from utils.paths import run_dir

    cfg = copy.deepcopy(load_config(args.config))

    # ---- Output safety: never the tracked results/evolution tree ----------
    root = run_dir(args.run_id, 'reevolution')
    if args.tag:
        root = os.path.join(root, args.tag)
    os.makedirs(root, exist_ok=True)
    repo_results = os.path.realpath(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'results'))
    if os.path.realpath(root).startswith(repo_results):
        raise SystemExit(
            f"refusing to run: resolved output {root} is inside {repo_results}. "
            f"LLMEvolution writes <results_dir>/evolution/<scenario>/pareto_gen*.json "
            f"and would overwrite the only surviving record of the deployed genomes, "
            f"which were evolved at N_max=24 under a 35 s timeout and cannot be "
            f"re-derived from any current config.")
    cfg['results_dir'] = root

    # ---- clean provenance: the reported genome must not have LLM provenance --
    cfg.setdefault('evolution', {})
    cfg['evolution']['use_llm_params'] = False
    cfg['evolution']['llm_params_mode'] = 'random'
    cfg['evolution']['pop_size'] = args.pop_size
    cfg['evolution']['generations'] = args.generations
    cfg['evolution']['use_tqdm'] = False
    cfg.setdefault('evaluation', {})['n_trials'] = args.n_trials
    cfg['evaluation']['progress_every_trials'] = 0
    cfg['evaluation']['timeout_per_trial'] = args.timeout_per_trial

    import multiprocessing as mp
    ctx = mp.get_context('spawn')       # fork would share the parent's RNG state

    from evolution.llm_evolution import LLMEvolution

    print(f"re-evolution: {args.scenario}")
    print(f"  results_dir  : {root}")
    print(f"  pop x gens   : {args.pop_size} x {args.generations} "
          f"= {args.pop_size * args.generations} candidate evaluations")
    print(f"  n_trials     : {args.n_trials}  -> "
          f"{args.pop_size * args.generations * args.n_trials} trials")
    print(f"  trial timeout: {args.timeout_per_trial} s  (worst-case generation "
          f"= {args.n_trials * args.timeout_per_trial / 60:.0f} min)")
    print(f"  LLM          : DISABLED (use_llm_params=False, mode=random)")
    print(f"  workers      : {args.max_workers}")
    print()

    t0 = time.time()
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        evo = LLMEvolution(config=cfg, scenario_name=args.scenario)
        hook = build_hook(pool, cfg, args.scenario, args.n_trials, args.master_seed)
        init_pop = None
        if args.init_seed is not None:
            from evolution.population import StratifiedPopulation
            sp = StratifiedPopulation(seed=args.init_seed)
            sp.initialize(args.pop_size)
            init_pop = sp.get_population()
            logger.info("independent replicate: stratified initial population "
                        "seeded %d (default is a hardcoded 42)", args.init_seed)
        archive = evo.run(n_generations=args.generations,
                          pop_size=args.pop_size,
                          initial_population=init_pop,
                          eval_hook=hook)
    wall = time.time() - t0

    entries = archive.get_pareto_front()
    full = [e for e in entries
            if len(e.objectives) >= 4 and float(e.objectives[3]) < 0.01]
    print(f"\ndone in {wall/60:.1f} min")
    print(f"  archive        : {len(entries)} entries, {len(full)} at 100% success")
    if full:
        best = min(full, key=lambda e: float(e.objectives[0]))
        print(f"  best at 100%   : E={float(best.objectives[0]):.2f} J  "
              f"T={float(best.objectives[1]):.2f} s  "
              f"solver={float(best.objectives[2]):.1f} ms")
        summary_best = {'objectives': [float(v) for v in best.objectives],
                        'params': best.params.to_dict()}
    else:
        summary_best = None
    print(f"  hv history     : {evo.hv_history[:1]} -> {evo.hv_history[-1:]}")
    print(f"  written under  : {root}/evolution/{args.scenario}/")

    summary = {
        'scenario': args.scenario, 'run_id': args.run_id,
        'pop_size': args.pop_size, 'generations': args.generations,
        'n_trials': args.n_trials, 'master_seed': args.master_seed,
        'timeout_per_trial': args.timeout_per_trial,
        'init_seed': args.init_seed,
        'init_population': ('stratified(seed=%s)' % args.init_seed
                            if args.init_seed is not None
                            else 'stratified(seed=42), LLMEvolution default'),
        'llm_disabled': True, 'wall_s': wall,
        'hv_history': [float(h) for h in evo.hv_history],
        'results_dir': root,
        'archive_size': len(entries),
        'archive_full_success': len(full),
        'best_at_full_success': summary_best,
        'why': ('re-evolved in the post-C1/C2 landscape; the tracked genomes were '
                'searched under pre-C1 geometry, a 35 s timeout and N_max=24'),
    }
    from utils.safe_write import safe_write_json
    safe_write_json(os.path.join(root, f'reevolution_{args.scenario}.json'), summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
