#!/usr/bin/env python3
"""Script 04: Ablation study for EP²-MPC components.

Studies (all run with 5 seeds × 50 trials, paired Wilcoxon on per-trial data):

1. EP²-MPC vs Standard MPC (5-state, no energy state)            → validates Claim 1 overall
2. EP²-MPC vs Bayesian Optimization (same 10D space, no LLM)     → validates Claim 3 LLM value
3. EP²-MPC vs NSGA-II (no LLM, pure evolutionary)                → validates Claim 3 LLM value
4. EP²-MPC vs EP²-MPC fixed horizon (N_min=N_max=12)             → validates Claim 2 horizon adaptation
5. EP²-MPC vs EP²-MPC without energy state (5-state model)       → validates Claim 1 energy state
6. Full dual-signal vs proximity-only adaptive horizon            → validates Claim 2 energy signal
7. Full dual-signal vs energy-only adaptive horizon               → validates Claim 2 proximity signal

Fixes applied:
    C5 — Ablation 4 uses N_min=N_max to truly fix the horizon (not just N_base).
    H4 — Ablations 6/7 use best evolved Pareto params when available.
    H5 — Ablations 6/7 use explicit disable flags in AdaptiveHorizon config,
         not fragile threshold tricks.
    F1 — EP²-MPC evaluated with best evolved Pareto params (not DEFAULT_PARAMS).
    F2 — Pareto file selection by modification time (not alphabetical sort).
    F3 — NSGA-II results cached to avoid redundant 5-day optimization.
    F4 — BO results cached if prior run exists.
    F5 — NSGA-II evolution output redirected to ablation subdir (prevents
         overwriting main evolution pareto files).
    F6 — Direction-aware significance: validates claim only when EP²-MPC is
         actually better (lower energy OR higher success), not just different.

Usage:
    python scripts/04_ablation.py --study all [--n_seeds 5] [--scenario corridor]
    python scripts/04_ablation.py --study 4 --scenario s_curve
"""
import sys
import os
import copy
import argparse
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from utils.config_loader import load_config
from utils.safe_write import safe_write_json, run_id_dir

BO_N_CALLS       = 50
BO_TRIALS_PER_CALL = 20
N_TRIALS         = 50


ABLATION_NAMES = {
    1: 'ep2mpc_vs_standard_mpc',
    2: 'ep2mpc_vs_bayesian_opt',
    3: 'ep2mpc_vs_nsga2_no_llm',
    4: 'ep2mpc_vs_fixed_horizon',
    5: 'ep2mpc_vs_no_energy_state',
    6: 'proximity_only_vs_dual_signal',
    7: 'energy_only_vs_dual_signal',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_evolved_params(scenario_name: str, results_dir: str):
    """Load the best energy-minimising Pareto solution from evolution results.

    Uses shared checkpoint_utils which picks the highest-generation pareto file
    via recursive glob (not mtime sort).
    """
    from utils.checkpoint_utils import load_best_evolved_params as _load
    return _load(scenario_name, results_dir)


def make_fixed_horizon_evaluator(config: dict, fixed_n: int):
    """Return an EP2Evaluator whose AdaptiveHorizon always returns fixed_n.

    Sets N_min = N_max = fixed_n in the mpc config so phi has no effect on N.
    This is the correct way to ablate horizon adaptation.
    """
    from evaluation.ep2_evaluator import EP2Evaluator
    fixed_cfg = copy.deepcopy(config)
    fixed_cfg.setdefault('mpc', {})
    fixed_cfg['mpc']['N_min'] = fixed_n
    fixed_cfg['mpc']['N_max'] = fixed_n   # forces N = fixed_n always
    return EP2Evaluator(fixed_cfg)


def make_single_signal_evaluator(config: dict, disable_energy: bool, disable_prox: bool):
    """Return an EP2Evaluator with one adaptive signal disabled.

    Sets the disable flag in the mpc config; AdaptiveHorizon reads it and
    zeroes the corresponding phi component.
    """
    from evaluation.ep2_evaluator import EP2Evaluator
    sig_cfg = copy.deepcopy(config)
    sig_cfg.setdefault('mpc', {})
    sig_cfg['mpc']['disable_energy_signal'] = disable_energy
    sig_cfg['mpc']['disable_prox_signal']   = disable_prox
    return EP2Evaluator(sig_cfg)


# ---------------------------------------------------------------------------
# Main per-study runner
# ---------------------------------------------------------------------------

def _emit_ablation_trials(sink_dir, study_id, scenario, arm, by_seed):
    """Persist per-trial records so a power question never needs a 7 h re-run.

    04_ablation.py wrote only aggregates, so when the n=50-vs-250 problem surfaced
    there was nothing on disk to recompute from -- the only options were re-running
    every study or leaving the p-values wrong. B1 solved exactly this for the main
    campaign (TrialResult + set_trial_sink); the ablation never got it.
    """
    if not sink_dir:
        return
    os.makedirs(sink_dir, exist_ok=True)
    path = os.path.join(sink_dir,
                        f'trials_{scenario}_study{study_id}_{arm}.jsonl')
    n = 0
    # 'w', not 'a': re-running one study under the same run-id must replace its
    # records, not double them. The campaign sinks append and that is why
    # 16_paired_campaign_tests.py has to refuse duplicate keys.
    with open(path, 'w', encoding='utf-8') as fh:
        for ms in sorted(by_seed):
            for i, r in enumerate(by_seed[ms].trial_results):
                rec = {'master_seed': ms, 'trial_index': i,
                       'study_id': study_id, 'scenario': scenario, 'arm': arm}
                for k in ('energy', 'time', 'success', 'collision', 'n_steps',
                          'path_distance', 'solver_ms', 'seed'):
                    if hasattr(r, k):
                        v = getattr(r, k)
                        rec[k] = bool(v) if isinstance(v, (bool,)) else v
                fh.write(json.dumps(rec, default=float) + '\n')
                n += 1
    logger.info("  wrote %d per-trial records: %s", n, path)


def run_ablation_study(
    study_id: int,
    n_seeds: int,
    scenario_name: str,
    config: dict,
    trial_sink_dir: Optional[str] = None,
    evolution_root: str = None,
) -> dict:
    """Run ablation study *study_id* with *n_seeds* seeds.

    Returns a dict containing ep2_results and ablation_results lists, plus
    per-trial paired data for Wilcoxon tests.

    ``evolution_root`` (added 2026-09-11) overrides ONLY the genome source
    (the directory containing ``evolution/<scenario>/``). The BO and NSGA-II
    caches deliberately stay on ``config['results_dir']`` -- they stand for
    ~13.4 days of tracked, unrecoverable compute
    and must not be re-searched or written into a run tree. The resolved
    genome source is recorded in the output as ``ep2_genome_source``: the
    2026-09-06 ablation pre-dates re-evolution and measured the old genome
    with nothing in its output saying so.
    """
    from mpc.mpc_params import MPCParams, DEFAULT_PARAMS
    from evaluation.ep2_evaluator import EP2Evaluator
    from evaluation.statistics import (wilcoxon_test, cliffs_delta,
                                       interpret_cliffs_delta,
                                       wilcoxon_paired, bootstrap_ci_paired)

    results_dir = config.get('results_dir', 'results')   # caches (never moved)
    genome_dir  = (evolution_root if evolution_root is not None
                   else results_dir)
    if evolution_root is None:
        logger.warning(
            "Genome source: %s (config fallback -- the TRACKED pre-reevolution "
            "archive; pass --evolution-root to ablate the re-evolved genome)",
            genome_dir)
    else:
        logger.info("Genome source: %s", genome_dir)
    evaluator   = EP2Evaluator(config)

    # --- Use best evolved params, NOT DEFAULT_PARAMS ---
    ep2_params  = load_best_evolved_params(scenario_name, genome_dir)

    study_name  = ABLATION_NAMES[study_id]
    logger.info("Ablation study %d: %s | scenario=%s | seeds=%d",
                study_id, study_name, scenario_name, n_seeds)

    # -------------------------------------------------------------------
    # One-time setup for studies 2 and 3 (optimization runs before seed loop)
    # -------------------------------------------------------------------
    bo_best_params      = None
    nsga2_best_params   = None

    # --- Skip BO optimization if a cached result exists ---
    if study_id == 2:
        bo_path = os.path.join(
            results_dir, 'bo_comparison',
            f'bo_{scenario_name}_n{BO_N_CALLS}.json'
        )
        # Only run BO subprocess if no cached result
        if not os.path.exists(bo_path):
            import subprocess
            bo_script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'scripts', '06_bo_comparison.py',
            )
            cmd = [
                sys.executable, bo_script,
                '--scenario', scenario_name,
                '--n_calls', str(BO_N_CALLS),
                '--n_trials', str(BO_TRIALS_PER_CALL),
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception as exc:
                logger.warning("BO optimization failed: %s", exc)
        else:
            logger.info("Using cached BO result from %s", bo_path)

        # Load BO result (whether just computed or cached)
        if os.path.exists(bo_path):
            try:
                with open(bo_path, encoding='utf-8') as f:
                    bo_data = json.load(f)
                if bo_data.get('best_params'):
                    bo_best_params = MPCParams.from_dict(bo_data['best_params'])
                    logger.info("BO best params loaded (energy=%.1fJ)",
                                bo_data.get('best_energy', 0))
                else:
                    logger.warning("BO result missing 'best_params' — "
                                   "Study 2 will use evolved EP²-MPC params as ablation")
            except Exception as exc:
                logger.warning("Failed to load BO results: %s", exc)

    # --- Cache NSGA-II results; redirect the output dir ---
    elif study_id == 3:
        nsga2_cache = os.path.join(results_dir, 'ablation',
                                   f'nsga2_cache_{scenario_name}.json')
        if os.path.exists(nsga2_cache):
            # Load cached NSGA-II best params — skip optimization entirely
            try:
                with open(nsga2_cache, encoding='utf-8') as f:
                    cached = json.load(f)
                nsga2_best_params = MPCParams.from_dict(cached['best_params'])
                logger.info("Loaded cached NSGA-II params from %s", nsga2_cache)
            except Exception as exc:
                logger.warning("Failed to load NSGA-II cache: %s", exc)
        else:
            # Run NSGA-II optimization with output redirected
            nsga2_cfg = copy.deepcopy(config)
            nsga2_cfg['evolution']['llm_params_mode'] = 'random'
            nsga2_cfg['evolution']['use_llm_params']  = False
            nsga2_cfg['evolution']['pop_size']        = 16
            nsga2_cfg['evolution']['generations']     = 10
            # Redirect output to prevent overwriting the main evolution record
            nsga2_results_dir = os.path.join(results_dir, 'ablation', 'nsga2_runs')
            nsga2_cfg['results_dir'] = nsga2_results_dir
            os.makedirs(nsga2_results_dir, exist_ok=True)
            logger.info("NSGA-II output redirected to %s", nsga2_results_dir)

            try:
                from evolution.llm_evolution import LLMEvolution
                evo     = LLMEvolution(config=nsga2_cfg, scenario_name=scenario_name)
                archive = evo.run(n_generations=10, pop_size=16)
                best    = None
                for entry in archive.get_pareto_front():
                    if entry.objectives[3] < 0.01:
                        if best is None or entry.objectives[0] < best.objectives[0]:
                            best = entry
                if best is None and archive.get_pareto_front():
                    best = min(archive.get_pareto_front(),
                               key=lambda e: e.objectives[0])
                if best is not None:
                    nsga2_best_params = best.params
                    # Cache for future runs
                    os.makedirs(os.path.dirname(nsga2_cache), exist_ok=True)
                    with open(nsga2_cache, 'w', encoding='utf-8') as f:
                        json.dump({
                            'best_params': best.params.to_dict(),
                            'objectives': list(best.objectives),
                            'scenario': scenario_name,
                        }, f, indent=2)
                    logger.info("NSGA-II cache saved to %s (energy=%.1fJ)",
                                nsga2_cache, best.objectives[0])
            except Exception as exc:
                logger.warning("NSGA-II optimization failed: %s", exc)

    # For ablations 6/7: load the best evolved params
    evolved_best = None
    if study_id in (6, 7):
        evolved_best = load_best_evolved_params(scenario_name, genome_dir)

    # -------------------------------------------------------------------
    # Seed loop — each seed uses master_seed = seed * 1000
    # -------------------------------------------------------------------
    ep2_results:      List = []
    ablation_results: List = []
    # Keyed by master_seed, not list position: if one arm raises for a seed, the
    # two lists desynchronise and pairing by index silently compares seed 3 of one
    # arm against seed 4 of the other.
    ep2_by_seed:      Dict[int, Any] = {}
    abl_by_seed:      Dict[int, Any] = {}

    for seed in range(n_seeds):
        master_seed = seed * 1000
        logger.info("  Seed %d/%d (master_seed=%d)", seed + 1, n_seeds, master_seed)

        # EP²-MPC baseline for this study (evolved params, never defaults)
        try:
            ep2_result = evaluator.evaluate(
                ep2_params, scenario_name,
                n_trials=N_TRIALS, master_seed=master_seed,
            )
            ep2_results.append(ep2_result)
            ep2_by_seed[master_seed] = ep2_result
        except Exception as exc:
            logger.warning("EP²-MPC evaluation failed for seed %d: %s", seed, exc)

        # Ablation variant
        result = None
        try:
            if study_id == 1:
                # ONE implementation of the standard-MPC comparator, not two.
                #
                # This used to be a 40-line inline trial loop building
                # `StandardMPCCasadi(N=12, config=...)` with no weights and calling
                # `.solve()` with no `wall_bounds` -- so it ran the CLASS DEFAULTS
                # (w_obs=360, w_goal=10, d_safe=0.30) with the wall term inert,
                # while `run_standard_mpc_casadi_baseline` ran the fitted
                # per-scenario weights with walls active. Same label, different
                # controller, in two tables of the same paper:
                #
                #            ablation study 1      baseline table
                #   corridor    127.09 J             127.08 J     <- agreed, hid it
                #   s_curve     138.56 J             137.20 J     +1.36 J
                #   chicane     135.90 J             133.55 J     +2.35 J
                #
                # The 2.35 J gap is larger than every EP2-vs-standard-MPC margin in
                # the corpus. Copying the fitted weights into the inline loop would
                # have created a fourth definition free to drift again, so the loop
                # is deleted and the shared wrapper is called instead -- the same
                # consolidation A3k did for solver options.
                from evaluation.baselines import run_standard_mpc_casadi_baseline
                result = run_standard_mpc_casadi_baseline(
                    scenario_name=scenario_name,
                    n_trials=N_TRIALS,
                    config=config,
                    master_seed=master_seed,
                )
            elif study_id == 2:
                # Bayesian Optimization best params
                params_to_eval = bo_best_params or ep2_params
                result = evaluator.evaluate(
                    params_to_eval, scenario_name,
                    n_trials=N_TRIALS, master_seed=master_seed,
                )

            elif study_id == 3:
                # NSGA-II best params (no LLM)
                params_to_eval = nsga2_best_params or ep2_params
                result = evaluator.evaluate(
                    params_to_eval, scenario_name,
                    n_trials=N_TRIALS, master_seed=master_seed,
                )

            elif study_id == 4:
                # FIX C5: truly fixed horizon — N_min=N_max=12 in config.
                # This disables adaptation entirely regardless of phi.
                fixed_n = 12
                fixed_eval = make_fixed_horizon_evaluator(config, fixed_n)
                result = fixed_eval.evaluate(
                    ep2_params, scenario_name,
                    n_trials=N_TRIALS, master_seed=master_seed,
                )

            elif study_id == 5:
                # 5-state ablation: no cumulative energy state or terminal energy cost
                from mpc.ep2_mpc_5state import EP2MPC5State
                eval_5state = EP2Evaluator(config, mpc_class=EP2MPC5State)
                result = eval_5state.evaluate(
                    ep2_params, scenario_name,
                    n_trials=N_TRIALS, master_seed=master_seed,
                )

            elif study_id == 6:
                # FIX H4 + H5: proximity-only — use best evolved params,
                # disable energy signal via explicit flag (not threshold hack).
                prox_only_eval = make_single_signal_evaluator(
                    config, disable_energy=True, disable_prox=False
                )
                result = prox_only_eval.evaluate(
                    evolved_best or ep2_params, scenario_name,
                    n_trials=N_TRIALS, master_seed=master_seed,
                )

            elif study_id == 7:
                # FIX H4 + H5: energy-only — use best evolved params,
                # disable proximity signal via explicit flag.
                energy_only_eval = make_single_signal_evaluator(
                    config, disable_energy=False, disable_prox=True
                )
                result = energy_only_eval.evaluate(
                    evolved_best or ep2_params, scenario_name,
                    n_trials=N_TRIALS, master_seed=master_seed,
                )

        except Exception as exc:
            logger.warning("Ablation %d seed %d failed: %s", study_id, seed, exc)

        if result is not None:
            ablation_results.append(result)
            abl_by_seed[master_seed] = result

    # -------------------------------------------------------------------
    # Compute paired Wilcoxon on per-trial arrays (seed 0 pairing)
    # -------------------------------------------------------------------
    wilcoxon_energy = wilcoxon_time = cliffs_energy = cliffs_time = None
    sig_energy = sig_time = False

    # --- Direction-aware significance ---
    # ep2_energy_lower = True when Cliff's delta < 0 (EP²-MPC uses less energy)
    # ep2_better = True when EP²-MPC is genuinely better (lower energy AND
    #              comparable/higher success, OR higher success when both differ)
    ep2_energy_lower = None
    ep2_better       = None

    # Pair over EVERY seed, not seed 0 alone.
    #
    # This used to read `ep2_results[0].trial_results` and
    # `ablation_results[0].trial_results`, so every p-value, Cliff's delta and
    # significance flag came from n = 50 while the means printed beside them were
    # over all 5 x 50 = 250 trials -- the same split as 05_statistical_eval.py.
    # With sigma_E = 0.62-0.69 J/trial against the 0.16 J target effect, n = 50
    # is underpowered 4.7-5.8x (n ~ 236-292 needed),
    # and it showed: No-LLM/chicane reported "significant" at p = 3.8e-4 on
    # |dE| = 0.084 J, i.e. an effect BELOW the target effect size, with Cliff's
    # delta = 0.061.
    #
    # `wilcoxon_paired(method='exact')`, not `wilcoxon_test`, which leaves scipy's
    # `method='auto'` in place. 'auto' happens to pick the exact test at n = 50 but
    # switches to the normal approximation above it, so raising n to 250 with the
    # old helper would have silently DEGRADED the test while appearing to
    # strengthen it.
    paired_seeds = sorted(set(ep2_by_seed) & set(abl_by_seed))
    n_paired = 0
    wilcoxon_energy_full = wilcoxon_time_full = None
    paired_ci_energy = None
    if paired_seeds:
        ep2_e, ep2_t, abl_e, abl_t = [], [], [], []
        for ms in paired_seeds:
            a, b = ep2_by_seed[ms].trial_results, abl_by_seed[ms].trial_results
            k = min(len(a), len(b))
            if k != len(a) or k != len(b):
                logger.warning("seed %d: trial counts differ (%d vs %d); pairing "
                               "the first %d", ms, len(a), len(b), k)
            ep2_e += [r.energy for r in a[:k]]
            ep2_t += [r.time   for r in a[:k]]
            abl_e += [r.energy for r in b[:k]]
            abl_t += [r.time   for r in b[:k]]
        n_paired = len(ep2_e)
        if n_paired >= 5:
            we_d = wilcoxon_paired(np.array(ep2_e), np.array(abl_e), method='exact')
            wt_d = wilcoxon_paired(np.array(ep2_t), np.array(abl_t), method='exact')
            wilcoxon_energy_full, wilcoxon_time_full = we_d, wt_d
            # Clamp to the resolution floor: at n = 250 with all-one-sign
            # differences scipy underflows to literal 0.0, and a p-value of exactly
            # zero must never reach a table.
            pe = max(float(we_d['p_value']), float(we_d['p_floor']))
            pt = max(float(wt_d['p_value']), float(wt_d['p_floor']))
            # Tuple kept for existing readers (summarize_paper_tables reads [1]).
            wilcoxon_energy = (float(we_d['statistic']), pe)
            wilcoxon_time   = (float(wt_d['statistic']), pt)
            cliffs_energy = cliffs_delta(np.array(ep2_e), np.array(abl_e))
            cliffs_time   = cliffs_delta(np.array(ep2_t), np.array(abl_t))
            sig_energy    = pe < 0.05
            sig_time      = pt < 0.05
            paired_ci_energy = bootstrap_ci_paired(np.array(ep2_e), np.array(abl_e))
            logger.info("  paired over %d seeds, n=%d pairs  p_E=%s  cliff_E=%+.4f",
                        len(paired_seeds), n_paired, we_d['p_text'], cliffs_energy)

            # direction check
            # Cliff's delta < 0 means group a (EP²-MPC) tends to be smaller
            ep2_energy_lower = (cliffs_energy is not None and cliffs_energy < 0)

    _emit_ablation_trials(trial_sink_dir, study_id, scenario_name, 'ep2', ep2_by_seed)
    _emit_ablation_trials(trial_sink_dir, study_id, scenario_name, 'abl', abl_by_seed)

    # Compute mean success rates for direction-aware comparison
    ep2_success_mean = float(np.mean([r.success_rate for r in ep2_results])) if ep2_results else None
    abl_success_mean = float(np.mean([r.success_rate for r in ablation_results])) if ablation_results else None

    # EP²-MPC is "better" when:
    #   (a) Both succeed similarly AND EP²-MPC uses less energy, OR
    #   (b) EP²-MPC has substantially higher success rate (solvability win)
    if ep2_success_mean is not None and abl_success_mean is not None:
        success_gap = ep2_success_mean - abl_success_mean
        if success_gap > 0.3:
            # Ablation fails significantly more — EP²-MPC wins on solvability
            ep2_better = True
        elif abs(success_gap) <= 0.3:
            # Similar success rates — compare energy direction
            ep2_better = bool(ep2_energy_lower) if ep2_energy_lower is not None else None
        else:
            # EP²-MPC fails more — EP²-MPC is worse
            ep2_better = False

    return {
        'study_id':           study_id,
        'study_name':         study_name,
        'scenario':           scenario_name,
        'n_seeds':            n_seeds,
        'ep2_genome_source':  str(genome_dir),
        'ep2_energy_mean':    float(np.mean([r.energy_mean for r in ep2_results])) if ep2_results else None,
        'ep2_time_mean':      float(np.mean([r.time_mean   for r in ep2_results])) if ep2_results else None,
        'ep2_success':        ep2_success_mean,
        'abl_energy_mean':    float(np.mean([r.energy_mean for r in ablation_results])) if ablation_results else None,
        'abl_time_mean':      float(np.mean([r.time_mean   for r in ablation_results])) if ablation_results else None,
        'abl_success':        abl_success_mean,
        # n_paired is the number the p-value actually rests on. It is emitted
        # NEXT TO the means so a reader cannot mistake one for the other again.
        'n_paired':           n_paired,
        'paired_master_seeds': paired_seeds,
        'wilcoxon_energy':    wilcoxon_energy,
        'wilcoxon_time':      wilcoxon_time,
        'wilcoxon_energy_full': wilcoxon_energy_full,
        'wilcoxon_time_full':   wilcoxon_time_full,
        'paired_energy_ci95': paired_ci_energy,
        'cliffs_energy':      cliffs_energy,
        'cliffs_time':        cliffs_time,
        'significant_energy': sig_energy,
        'significant_time':   sig_time,
        'ep2_energy_lower':   ep2_energy_lower,
        'ep2_better':         ep2_better,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_ablation_table(results: dict) -> None:
    """Print a compact ablation summary table."""
    logger.info("=" * 110)
    logger.info("ABLATION RESULTS")
    logger.info("=" * 110)
    logger.info("%-40s  %10s  %10s  %8s  %8s  %8s  %8s  %8s",
                'Study', 'EP2 E(J)', 'Abl E(J)', 'ΔE(%)', 'p(E)', 'Cliffs', 'Sig?', 'Better?')
    logger.info("-" * 110)
    for study_name, res in results.items():
        ep2_e = res.get('ep2_energy_mean')
        abl_e = res.get('abl_energy_mean')
        if ep2_e is None or abl_e is None:
            continue
        delta_pct = (abl_e - ep2_e) / abl_e * 100 if abl_e else 0
        wres = res.get('wilcoxon_energy')
        p_val = wres[1] if wres else float('nan')
        cliff = res.get('cliffs_energy', float('nan'))
        sig   = '✓' if res.get('significant_energy') else '✗'
        better = '✓' if res.get('ep2_better') else ('✗' if res.get('ep2_better') is False else '?')
        logger.info("%-40s  %10.1f  %10.1f  %+7.1f%%  %8.4f  %+7.3f  %s  %s",
                    study_name[:40], ep2_e, abl_e, delta_pct, p_val, cliff or 0.0, sig, better)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description='EP²-MPC Ablation Studies')
    parser.add_argument('--study', type=str, default='all',
                        help='Study ID (1-7) or "all"')
    parser.add_argument('--n_seeds',  type=int, default=5)
    parser.add_argument('--scenario', type=str, default='corridor',
                        choices=['corridor', 's_curve', 'chicane', 'all'],
                        help="one map, or 'all' for every map in turn. 'all' is "
                             "what launch_campaign.sh passes, and "
                             "what 05_statistical_eval.py already accepted; this "
                             "script used to reject it with 'invalid choice', "
                             "which is why stage 10 failed and took stage 11 "
                             "down with it (no --ablation-dir to plot from).")
    parser.add_argument('--config',   type=str, default=None)
    parser.add_argument('--evolution-root', type=str, default=None,
                        help='Directory containing evolution/<scenario>/ from which '
                             'the EP2 genome loads (e.g. runs/<id>/reevolution). '
                             'Caches (BO, NSGA-II) always come from config '
                             'results_dir. Default: config results_dir, which is '
                             'the TRACKED pre-reevolution archive.')
    parser.add_argument('--run-id', dest='run_id', type=str, default=None,
                        help='Output goes to <runs>/<run-id>/ablation/ on /data '
                             'via utils.paths.run_dir. Aggregates are NEVER '
                             'written to the legacy flat path, which holds the 21 '
                             'tracked study files (their per-trial data is '
                             'already unrecoverable).')
    parser.add_argument('--out-dir', dest='out_dir', type=str, default=None,
                        help='Explicit output directory, bypassing --run-id. '
                             'Still guarded against overwriting tracked files.')
    parser.add_argument('--force', action='store_true',
                        help='Permit overwriting git-tracked output files.')
    args = parser.parse_args()

    config = load_config(args.config)
    ablation_root = os.path.join(config.get('results_dir', 'results'), 'ablation')
    if args.out_dir:
        save_dir = args.out_dir
        os.makedirs(save_dir, exist_ok=True)
    else:
        from utils.paths import run_dir as _run_dir
        save_dir = _run_dir(args.run_id, 'ablation')
    logger.info("Ablation output directory: %s", save_dir)

    studies = list(ABLATION_NAMES.keys()) if args.study == 'all' else [int(args.study)]
    scenarios = (['corridor', 's_curve', 'chicane'] if args.scenario == 'all'
                 else [args.scenario])

    # Per scenario, not one merged dict: the aggregate keys are
    # "<study_id>_<study_name>" and would collide across maps, silently keeping
    # only the last map's numbers in the printed table.
    for scenario in scenarios:
        all_results: dict = {}
        for study_id in studies:
            res = run_ablation_study(study_id, args.n_seeds, scenario, config,
                                     trial_sink_dir=os.path.join(save_dir, 'trials'),
                                     evolution_root=args.evolution_root)
            key = f"{study_id}_{res['study_name']}"
            all_results[key] = res

            out_path = os.path.join(
                save_dir, f'ablation_{scenario}_study{study_id}.json'
            )
            safe_write_json(out_path, res, force=args.force)
            logger.info("Study %d results saved to %s", study_id, out_path)

        if len(scenarios) > 1:
            logger.info("=" * 70)
            logger.info("Ablation table for scenario: %s", scenario)
        print_ablation_table(all_results)
    return 0


if __name__ == '__main__':
    sys.exit(main())