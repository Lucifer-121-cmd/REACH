#!/usr/bin/env python3
"""Script 02: Run all baseline controllers on EP²-MPC scenarios.

Steps:
1. Load config.yaml
2. For each of 3 scenarios, run all baselines (50 trials each)
3. Save results to results/baselines/{scenario}/
4. Print summary table
5. Check GATE 2 (std/mean < 15%)

Usage:
    python scripts/02_run_baselines.py [--scenario all] [--n_trials 50]
"""
import sys
import os
import argparse
import json
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


from utils.config_loader import load_config


def check_gate2(results: dict, scenario_name: str) -> bool:
    """GATE 2: Coefficient of variation (std/mean) < 15% for energy and time."""
    logger.info("=== GATE 2 Check: %s ===", scenario_name)
    gate2_pass = True

    for name, result in results.items():
        cv_energy = result.energy_std / (result.energy_mean + 1e-6)
        cv_time = result.time_std / (result.time_mean + 1e-6)

        energy_ok = cv_energy < 0.15
        time_ok = cv_time < 0.15

        status = "PASS" if (energy_ok and time_ok) else "FAIL"
        logger.info("  %-20s: CV_energy=%.3f %s, CV_time=%.3f %s [%s]",
                    name, cv_energy, '✓' if energy_ok else '✗',
                    cv_time, '✓' if time_ok else '✗', status)

        if not (energy_ok and time_ok):
            gate2_pass = False

    return gate2_pass


def print_summary_table(all_results: dict):
    """Log summary table of baseline results."""
    logger.info("=" * 100)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 100)
    logger.info("%-12s %-22s %-14s %-12s %-14s %-10s",
                'Scenario', 'Controller', 'Energy(J)', 'Time(s)', 'Solver(ms)', 'Success%')
    logger.info("-" * 100)

    for scenario, results in all_results.items():
        for name, result in results.items():
            logger.info("%-12s %-22s %8.1f±%5.1f  %6.1f±%4.1f  %8.2f±%5.2f  %8.1f%%",
                        scenario, name,
                        result.energy_mean, result.energy_std,
                        result.time_mean, result.time_std,
                        result.solver_ms_mean, result.solver_ms_std,
                        result.success_rate * 100)
    logger.info("=" * 100)


def main():
    """Parse CLI arguments, run baseline controllers on selected scenarios, and save results."""
    parser = argparse.ArgumentParser(description='Run EP²-MPC Baseline Controllers')
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['all', 'corridor', 's_curve', 'chicane'],
                        help='Which scenario to run')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of trials per baseline')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file path (default: configs/global_config.yaml)')
    parser.add_argument('--run-id', dest='run_id', required=True,
                        help='REQUIRED. Output goes to '
                             '<results_dir>/baselines/<run-id>/. The flat tracked '
                             'path is never written.')
    parser.add_argument('--results-dir', dest='results_dir', default=None,
                        help="Root for output; defaults to config['results_dir'].")
    parser.add_argument('--force', action='store_true',
                        help='Permit overwriting git-tracked output.')
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.get('robot'):
        raise SystemExit(
            "config['robot'] is required: every baseline arm must run at "
            "the robot's real limits, not a hard-coded literal.")

    from evaluation.baselines import collect_all_baselines
    from env.scenarios import list_scenarios
    from utils.safe_write import safe_write_json, run_id_dir

    scenarios = [args.scenario] if args.scenario != 'all' else list_scenarios()
    results_root = args.results_dir or config.get('results_dir', 'results')

    all_results = {}
    gate2_results = {}
    
    for scenario_name in scenarios:
        logger.info("=" * 60)
        logger.info("Running baselines on scenario: %s", scenario_name)
        logger.info("=" * 60)

        # One campaign directory per run-id, MIRRORING the results-root layout
        # (<root>/campaigns/<run-id>/baselines/, .../statistics/, .../ablation/).
        # That way a reader which legitimately wants a results-root -- e.g.
        # summarize_paper_tables --results_dir -- can be pointed at the campaign
        # directory and find what it expects, without any fallback and without any
        # reader having to guess whether it was handed a root or a leaf.
        from utils.paths import run_dir as _run_dir
        save_dir = _run_dir(args.run_id, 'baselines')

        results = collect_all_baselines(
            scenario_name,
            n_trials=args.n_trials,
            config=config
        )
        all_results[scenario_name] = results

        # Save results
        results_data = {}
        for name, result in results.items():
            results_data[name] = {
                'energy_mean': result.energy_mean,
                'energy_std': result.energy_std,
                'time_mean': result.time_mean,
                'time_std': result.time_std,
                'solver_ms_mean': result.solver_ms_mean,
                'solver_ms_std': result.solver_ms_std,
                'success_rate': result.success_rate,
                'n_trials': result.n_trials,
            }

        # baseline_<scenario>.json, NOT baseline_results.json. Every reader --
        # summarize_paper_tables.py:142 and summarize_evolution.py:183 -- loads
        # baseline_<scenario>.json, so the old name produced a file nothing read:
        # the run reported success while the tables kept serving stale values.
        results_path = os.path.join(save_dir, f'baseline_{scenario_name}.json')
        safe_write_json(results_path, {scenario_name: results_data},
                        force=args.force)
        logger.info("Results saved to %s", results_path)

        gate2_results[scenario_name] = check_gate2(results, scenario_name)

    print_summary_table(all_results)

    logger.info("=" * 60)
    logger.info("GATE 2 Summary:")
    all_pass = True
    for scenario, passed in gate2_results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        logger.info("  %s: %s", scenario, status)
        if not passed:
            all_pass = False
    logger.info("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
