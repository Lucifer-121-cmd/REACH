#!/usr/bin/env python3
"""Script 05: Final statistical evaluation of EP²-MPC vs baselines.

Runs 5 seeds × 50 trials per method per scenario.
Each seed uses a distinct master_seed offset (seed * 1000) so the 50 trial
perturbations differ across seeds, giving genuinely independent evaluations.

Wilcoxon signed-rank tests are performed on per-trial paired data (n=50),
not on per-seed means. Cliff's delta and 95% bootstrap CI are also reported.

Fixes applied:
    F1 — Uses best evolved Pareto params (not DEFAULT_PARAMS).
    F2 — Pareto file selected by modification time (not alphabetical sort).
    F3 — Saves adaptive horizon per-step history for Script 07 Fig 3.

Usage:
    python scripts/05_statistical_eval.py [--n_seeds 5] [--n_trials 50]
    python scripts/05_statistical_eval.py --scenario corridor
"""
import sys
import os
import argparse
import json
import numpy as np
import logging
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from utils.config_loader import load_config
from utils.safe_write import safe_write_json, run_id_dir


# ---------------------------------------------------------------------------
# Load best evolved Pareto params (sort by mtime, not name)
# ---------------------------------------------------------------------------

def load_best_evolved_params(scenario_name: str, results_dir: str):
    """Load the best energy-minimising Pareto solution from evolution results.

    Uses shared checkpoint_utils which picks the highest-generation pareto file
    via recursive glob (not mtime sort).
    """
    from utils.checkpoint_utils import load_best_evolved_params as _load
    return _load(scenario_name, results_dir)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    scenario_name: str,
    n_seeds: int,
    n_trials: int,
    config: dict,
    ep2_params_override=None,
    evolution_root: str = None,
) -> dict:
    """Run full 5-seed × 50-trial evaluation for a scenario.

    ep2_params uses the best evolved Pareto candidate, not DEFAULT_PARAMS.
    AdaptiveHorizon per-step history is collected for Fig 3.

    ``evolution_root`` (added 2026-09-11) is the directory *containing*
    ``evolution/<scenario>/`` from which the genome loads. When omitted it
    falls back to ``config['results_dir']`` -- the tracked archive, i.e. the
    PRE-reevolution genome. This is exactly how the 2026-09-10
    statistical_results.json came to describe the old genome while the
    campaign beside it had measured the re-evolved one: this script had no
    way to say otherwise. The resolved source is recorded in the output
    JSON as ``ep2_evolution_root`` so a mislabelled artefact is visible in
    the artefact itself.
    """
    from evaluation.ep2_evaluator import EP2Evaluator
    from evaluation.baselines import collect_all_baselines
    from evaluation.statistics import (
        wilcoxon_test, cliffs_delta, interpret_cliffs_delta, bootstrap_ci
    )
    from mpc.mpc_params import MPCParams

    results_dir = (evolution_root if evolution_root is not None
                   else config.get('results_dir', 'results'))
    evaluator   = EP2Evaluator(config)

    # use evolved params, not DEFAULT_PARAMS
    if ep2_params_override is not None:
        ep2_params = ep2_params_override
        logger.info("Using manually supplied EP²-MPC params")
    else:
        if evolution_root is None:
            logger.warning(
                "Genome source: %s (config fallback -- the TRACKED pre-reevolution "
                "archive; pass --evolution-root to measure a re-evolved genome)",
                results_dir)
        else:
            logger.info("Genome source: %s", results_dir)
        ep2_params = load_best_evolved_params(scenario_name, results_dir)

    # -----------------------------------------------------------------------
    # Per-seed per-trial data collection
    # -----------------------------------------------------------------------
    all_ep2_energies:   List[float] = []
    all_ep2_times:      List[float] = []
    all_ep2_success:    List[float] = []
    all_ep2_solver_ms:  List[float] = []
    ep2_solver_status_counts: Dict[str, int] = {}

    seed_results = []
    horizon_histories = []   # collect for Fig 3

    for seed in range(n_seeds):
        master_seed = seed * 1000
        logger.info("EP²-MPC seed %d/%d on %s (master_seed=%d)",
                    seed + 1, n_seeds, scenario_name, master_seed)
        try:
            result = evaluator.evaluate(
                ep2_params, scenario_name,
                n_trials=n_trials,
                master_seed=master_seed,
            )
            for r in result.trial_results:
                all_ep2_energies.append(r.energy)
                all_ep2_times.append(r.time)
                all_ep2_success.append(float(r.success))
                all_ep2_solver_ms.append(r.solver_ms)
            for k, v in result.solver_status_counts.items():
                ep2_solver_status_counts[k] = ep2_solver_status_counts.get(k, 0) + v
            seed_results.append(result)

            # extract horizon history from first trial's TrialResult
            if result.trial_results:
                h = getattr(result.trial_results[0], 'adaptive_horizon_history', [])
                if h:
                    horizon_histories.append(h)
        except Exception as e:
            logger.warning("EP²-MPC failed for seed %d: %s", seed, e)

    # -----------------------------------------------------------------------
    # Paired baseline collection (seed 0 for Wilcoxon pairing)
    # -----------------------------------------------------------------------
    PAIRED_MASTER_SEED = 0
    baselines_raw = collect_all_baselines(
        scenario_name, n_trials=n_trials, config=config,
        master_seed=PAIRED_MASTER_SEED,
    )

    # -----------------------------------------------------------------------
    # Wilcoxon + Cliff's delta (per-trial paired, n=n_trials)
    # -----------------------------------------------------------------------
    wilcoxon_results: Dict[str, dict] = {}
    paired_ep2_energy: List[float] = []
    paired_ep2_time:   List[float] = []
    if seed_results:
        first = seed_results[0]
        paired_ep2_energy = [r.energy for r in first.trial_results]
        paired_ep2_time   = [r.time   for r in first.trial_results]

    for bname, bresult in baselines_raw.items():
        base_energies = [r.energy for r in bresult.trial_results]
        base_times    = [r.time   for r in bresult.trial_results]
        n_paired = min(len(paired_ep2_energy), len(base_energies))
        if n_paired < 5:
            continue

        e_stat, e_p = wilcoxon_test(
            np.array(paired_ep2_energy[:n_paired]),
            np.array(base_energies[:n_paired]),
        )
        t_stat, t_p = wilcoxon_test(
            np.array(paired_ep2_time[:n_paired]),
            np.array(base_times[:n_paired]),
        )
        e_d = cliffs_delta(
            np.array(paired_ep2_energy[:n_paired]),
            np.array(base_energies[:n_paired]),
        )
        t_d = cliffs_delta(
            np.array(paired_ep2_time[:n_paired]),
            np.array(base_times[:n_paired]),
        )

        # ep2_better: True when EP²-MPC uses LESS energy (cliff < 0) or
        # has substantially higher success rate (solvability win)
        ep2_success_mean  = float(np.mean(all_ep2_success)) if all_ep2_success else 0.0
        base_success_mean = float(bresult.success_rate)
        solvability_win   = (ep2_success_mean - base_success_mean) > 0.3

        wilcoxon_results[bname] = {
            'energy': {
                'W': float(e_stat), 'p': float(e_p),
                'cliffs_d': float(e_d),
                'effect': interpret_cliffs_delta(e_d),
                'significant': bool(e_p < 0.05),
                'ep2_better': solvability_win or (float(e_d) < 0),
            },
            'time': {
                'W': float(t_stat), 'p': float(t_p),
                'cliffs_d': float(t_d),
                'effect': interpret_cliffs_delta(t_d),
                'significant': bool(t_p < 0.05),
                'ep2_better': solvability_win or (float(t_d) < 0),
            },
            'n_paired':       n_paired,
            'solvability_win': solvability_win,
            'ep2_success':    ep2_success_mean,
            'base_success':   base_success_mean,
        }

    # -----------------------------------------------------------------------
    # Bootstrap 95% CI on flattened data (all seeds × n_trials)
    # -----------------------------------------------------------------------
    ci_energy = bootstrap_ci(np.array(all_ep2_energies)) if all_ep2_energies else (0, 0, 0)
    ci_time   = bootstrap_ci(np.array(all_ep2_times))    if all_ep2_times   else (0, 0, 0)

    return {
        'scenario':            scenario_name,
        'n_seeds':             n_seeds,
        'n_trials_per_seed':   n_trials,
        'total_trials':        len(all_ep2_energies),
        'ep2_evolution_root':  str(results_dir),
        'ep2_energy_mean':     float(np.mean(all_ep2_energies)) if all_ep2_energies else None,
        'ep2_energy_std':      float(np.std(all_ep2_energies))  if all_ep2_energies else None,
        'ep2_energy_ci95':     (float(ci_energy[1]), float(ci_energy[2])),
        'ep2_time_mean':       float(np.mean(all_ep2_times))    if all_ep2_times    else None,
        'ep2_time_std':        float(np.std(all_ep2_times))     if all_ep2_times    else None,
        'ep2_time_ci95':       (float(ci_time[1]),  float(ci_time[2])),
        'ep2_success_rate':    float(np.mean(all_ep2_success))  if all_ep2_success  else None,
        'ep2_solver_ms_mean':   float(np.mean(all_ep2_solver_ms))           if all_ep2_solver_ms else None,
        'ep2_solver_ms_median': float(np.median(all_ep2_solver_ms))         if all_ep2_solver_ms else None,
        'ep2_solver_ms_p95':    float(np.percentile(all_ep2_solver_ms, 95)) if all_ep2_solver_ms else None,
        'ep2_solver_ms_all':    [float(x) for x in all_ep2_solver_ms],
        'ep2_solver_status_counts': ep2_solver_status_counts,
        'baselines': {
            name: {
                'energy_mean':    res.energy_mean,
                'energy_std':     res.energy_std,
                'time_mean':      res.time_mean,
                'time_std':       res.time_std,
                'success_rate':   res.success_rate,
                'solver_ms_mean': res.solver_ms_mean,
            }
            for name, res in baselines_raw.items()
        },
        'wilcoxon': wilcoxon_results,
        'horizon_histories': horizon_histories,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(all_stats: dict) -> None:
    """Log formatted results table to console."""
    logger.info("=" * 100)
    logger.info("FINAL STATISTICAL EVALUATION RESULTS")
    logger.info("=" * 100)
    for scenario, stats in all_stats.items():
        logger.info("Scenario: %s  (%d total trials)",
                    scenario, stats.get('total_trials', 0))
        logger.info("  %-30s  %22s  %10s  %10s  %10s",
                    'Method', 'Energy (J)', 'Time (s)', 'Success', 'Solver ms')
        logger.info("  " + "-" * 85)

        ep2_e  = stats.get('ep2_energy_mean')
        ep2_t  = stats.get('ep2_time_mean')
        ep2_s  = stats.get('ep2_success_rate')
        ep2_ms = stats.get('ep2_solver_ms_mean')
        if ep2_e is not None:
            ci_e   = stats.get('ep2_energy_ci95', (ep2_e, ep2_e))
            ep2_med = stats.get('ep2_solver_ms_median', ep2_ms)
            ep2_p95 = stats.get('ep2_solver_ms_p95', ep2_ms)
            logger.info("  %-30s  %6.1f [%5.1f, %5.1f]  %7.1f  %9.1f%%  %8.1f",
                        'EP2-MPC (ours)',
                        ep2_e, ci_e[0], ci_e[1],
                        ep2_t or 0,
                        (ep2_s or 0) * 100,
                        ep2_ms or 0)
            logger.info("  %-30s  solver: median=%.1f ms  p95=%.1f ms",
                        '', ep2_med or 0, ep2_p95 or 0)

        for bname, bvals in stats.get('baselines', {}).items():
            logger.info("  %-30s  %6.1f +/- %-8.1f    %7.1f  %9.1f%%  %8.1f",
                        bname,
                        bvals['energy_mean'], bvals['energy_std'],
                        bvals['time_mean'],
                        bvals['success_rate'] * 100,
                        bvals['solver_ms_mean'])

        if stats.get('wilcoxon'):
            n_paired = next(iter(stats['wilcoxon'].values()), {}).get('n_paired', 0)
            logger.info("  Wilcoxon signed-rank tests (paired, n=%d):", n_paired)
            for bname, wres in stats['wilcoxon'].items():
                e = wres['energy']
                t = wres['time']
                solv = '[SOLV]' if wres.get('solvability_win') else ''
                logger.info(
                    "    vs %-22s  E: W=%6.1f p=%.4f %s cliff=%+.3f (%s) | "
                    "T: p=%.4f %s  %s",
                    bname,
                    e['W'], e['p'],
                    'sig' if e['significant'] else 'n.s.',
                    e['cliffs_d'], e['effect'],
                    t['p'],
                    'sig' if t['significant'] else 'n.s.',
                    solv,
                )
        logger.info("")


# ---------------------------------------------------------------------------
# Save horizon history for Fig 3
# ---------------------------------------------------------------------------

def merge_scenario_json(path: str, new_blocks: dict) -> dict:
    """Merge ``new_blocks`` over whatever scenario blocks already exist at ``path``.

    A single-scenario invocation used to rewrite this file from scratch, deleting
    the other two scenarios. Merging makes a partial run additive:
    scenarios just evaluated are replaced, scenarios not evaluated are preserved
    exactly as they were.
    """
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                existing = json.load(fh)
            if not isinstance(existing, dict):
                raise ValueError(
                    f"expected a dict of scenario blocks, got {type(existing).__name__}")
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Refusing to write {path}: it exists but could not be parsed "
                f"for merging ({exc}). Move it aside or pass --out-dir."
            ) from exc
        kept = sorted(set(existing) - set(new_blocks))
        if kept:
            logger.info("Preserving existing scenario blocks: %s", kept)
    merged = dict(existing)
    merged.update(new_blocks)
    return merged


def save_horizon_history(all_stats: dict, save_dir: str,
                         force: bool = False) -> None:
    """Save adaptive horizon histories so Script 07 Fig 3 uses real data."""
    histories = []
    for scenario, stats in all_stats.items():
        for h in stats.get('horizon_histories', []):
            if h:
                histories.append(h)
                break   # one representative trial per scenario is sufficient

    if not histories:
        logger.info("No horizon history to save (evaluator may not expose it).")
        return

    path = os.path.join(save_dir, 'adaptive_horizon_history.json')
    safe_write_json(path, {'histories': histories}, force=force)
    logger.info("Horizon history saved to %s  (%d traces)", path, len(histories))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='EP2-MPC Statistical Evaluation')
    parser.add_argument('--n_seeds',    type=int, default=5)
    parser.add_argument('--n_trials',   type=int, default=50)
    parser.add_argument('--scenario',   type=str, default='all',
                        choices=['all', 'corridor', 's_curve', 'chicane'])
    parser.add_argument('--config',     type=str, default=None)
    parser.add_argument('--ep2_params', type=str, default=None,
                        help='Optional path to evolved params JSON (auto-loaded if omitted)')
    parser.add_argument('--evolution-root', type=str, default=None,
                        help='Directory containing evolution/<scenario>/ from which '
                             'the EP2 genome loads (e.g. runs/<id>/reevolution). '
                             'Default: config results_dir, which is the TRACKED '
                             'pre-reevolution archive.')
    parser.add_argument('--run-id', dest='run_id', type=str, default=None,
                        help='Output subdirectory under results/statistics/ '
                             '(default: run_<UTC timestamp>).')
    parser.add_argument('--out-dir', dest='out_dir', type=str, default=None,
                        help='Explicit output directory, bypassing --run-id.')
    parser.add_argument('--force', action='store_true',
                        help='Permit overwriting git-tracked output files. Even '
                             'with --force the write MERGES per-scenario blocks '
                             'rather than replacing the file.')
    args = parser.parse_args()

    config      = load_config(args.config)
    results_dir = config.get('results_dir', 'results')

    # Optional manual param override (auto-loads evolved params if not supplied)
    ep2_params_override = None
    if args.ep2_params:
        from mpc.mpc_params import MPCParams
        with open(args.ep2_params, encoding='utf-8') as f:
            ep2_params_override = MPCParams.from_dict(json.load(f))
        logger.info("Manual params loaded from %s", args.ep2_params)

    from env.scenarios import list_scenarios
    scenarios = list_scenarios() if args.scenario == 'all' else [args.scenario]

    stats_root = os.path.join(results_dir, 'statistics')
    if args.out_dir:
        stats_dir = args.out_dir
        os.makedirs(stats_dir, exist_ok=True)
    else:
        from utils.paths import run_dir as _run_dir
        stats_dir = _run_dir(args.run_id, 'statistics')
    logger.info("Statistics output directory: %s", stats_dir)
    if len(scenarios) < 3:
        logger.warning(
            "Partial run: only %s. The output file will contain only these "
            "scenario blocks; any pre-existing blocks at the target path are "
            "MERGED, never dropped.", scenarios,
        )

    all_results: dict = {}
    for scenario in scenarios:
        logger.info("\nEvaluating scenario: %s", scenario)
        results = run_full_evaluation(
            scenario, args.n_seeds, args.n_trials, config, ep2_params_override,
            evolution_root=args.evolution_root,
        )
        all_results[scenario] = results

    print_results_table(all_results)

    # Save main results JSON (horizon_histories excluded — saved separately).
    # MERGE semantics: a single-scenario invocation used to rewrite this file
    # from scratch, deleting the other two scenarios' blocks.
    out_path = os.path.join(stats_dir, 'statistical_results.json')
    EXCLUDE_FROM_JSON = {'horizon_histories', 'ep2_solver_ms_all'}
    results_to_save = {
        sc: {k: v for k, v in data.items() if k not in EXCLUDE_FROM_JSON}
        for sc, data in all_results.items()
    }
    merged = merge_scenario_json(out_path, results_to_save)
    safe_write_json(out_path, merged, force=args.force)
    logger.info("Results saved to %s (scenario blocks: %s)",
                out_path, sorted(merged))

    # save horizon history separately for Fig 3
    save_horizon_history(all_results, stats_dir, force=args.force)

    return 0


if __name__ == '__main__':
    sys.exit(main())