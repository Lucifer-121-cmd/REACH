"""Shared utilities for checkpoint and Pareto file discovery.

Centralises the logic for finding the best checkpoint/pareto files
across the gen*/ subdirectory structure, replacing fragile os.listdir
and latest_checkpoint.json patterns with generation-number-based selection.
"""

import glob
import json
import logging
import os
import re

logger = logging.getLogger(__name__)


def find_best_checkpoint(results_dir: str, scenario: str) -> str:
    """Return path to the highest-generation checkpoint for a scenario.

    Searches recursively through gen*/ subdirectories.
    Falls back to latest_checkpoint.json only if no numbered checkpoints exist.
    """
    base = os.path.join(results_dir, 'evolution', scenario)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Evolution directory not found: {base}")
    files = glob.glob(os.path.join(base, '**', 'checkpoint_gen*.json'), recursive=True)
    if not files:
        latest = os.path.join(base, 'latest_checkpoint.json')
        if os.path.isfile(latest):
            logger.warning(
                "No numbered checkpoints for %s; falling back to latest_checkpoint.json",
                scenario,
            )
            return latest
        raise FileNotFoundError(f"No checkpoints found for {scenario} in {base}")

    def _gen_num(p):
        m = re.search(r'checkpoint_gen(\d+)', p)
        return int(m.group(1)) if m else -1

    return max(files, key=_gen_num)


def find_best_pareto(results_dir: str, scenario: str) -> str:
    """Return path to the highest-generation pareto file for a scenario."""
    base = os.path.join(results_dir, 'evolution', scenario)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Evolution directory not found: {base}")
    files = glob.glob(os.path.join(base, '**', 'pareto_gen*.json'), recursive=True)
    if not files:
        raise FileNotFoundError(f"No pareto files found for {scenario} in {base}")

    def _gen_num(p):
        m = re.search(r'pareto_gen(\d+)', p)
        return int(m.group(1)) if m else -1

    return max(files, key=_gen_num)


def load_hv_history(results_dir: str, scenario: str) -> list:
    """Load full HV history from the best checkpoint for a scenario."""
    path = find_best_checkpoint(results_dir, scenario)
    with open(path, encoding='utf-8') as f:
        ckpt = json.load(f)
    hv_hist = ckpt.get('hv_history', [])
    if hv_hist:
        logger.info(
            "Loaded %d HV entries from %s (gen %s)",
            len(hv_hist),
            path,
            ckpt.get('generation', '?'),
        )
    return hv_hist


def load_best_evolved_params(scenario_name: str, results_dir: str):
    """Load the best (lowest-energy, success=100%) params from best pareto file.

    Note the argument order: ``(scenario_name, results_dir)``. A reversed call
    used to be masked by the DEFAULT_PARAMS fallback removed below.

    Returns:
        MPCParams for the deployed genome.

    Raises:
        FileNotFoundError: if no Pareto archive exists for the scenario, or the
            archive is empty. It never silently substitutes DEFAULT_PARAMS --
            that fallback meant a mis-resolved path ran default weights and
            reported them as evolved.
        ValueError: if the archive holds no 100%-success candidate. The old
            behaviour served the lowest-energy entry behind a warning; that
            entry is the MOST-broken candidate (crash-truncated energies read
            cheap), so it is refused.
    """
    from mpc.mpc_params import DEFAULT_PARAMS, MPCParams

    try:
        path = find_best_pareto(results_dir, scenario_name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No Pareto archive for scenario {scenario_name!r} under "
            f"{results_dir!r}: {exc}\n"
            f"  Refusing to fall back to DEFAULT_PARAMS -- that would run "
            f"default weights and report them as evolved.\n"
            f"  Check the argument order: load_best_evolved_params"
            f"(scenario_name, results_dir)."
        ) from exc

    with open(path, encoding='utf-8') as f:
        entries = json.load(f)

    if not entries:
        raise FileNotFoundError(
            f"Pareto archive {path} is empty; refusing to fall back to "
            f"DEFAULT_PARAMS."
        )

    best_entry = None
    for e in entries:
        objs = e.get('objectives', [5000, 30, 600, 1.0])
        failure_rate = objs[3] if len(objs) > 3 else 1.0
        if failure_rate < 0.01:
            if best_entry is None or objs[0] < best_entry['objectives'][0]:
                best_entry = e

    if best_entry is None:
        # A crashed trial's energy is a truncated
        # partial sum, so the lowest-energy entry is the MOST-broken
        # candidate -- it both fails and reads cheap. Serving it as the
        # deployment genome behind a warning is exactly the fail-open
        # this module exists to prevent; refuse instead.
        # (Fixed 2026-09-20; the warning-only fallback had been flagged
        # as "should raise" since 2026-09-05 and left in place.)
        raise ValueError(
            f"No 100%-success candidate in {path}; refusing to serve the "
            f"lowest-energy fallback. A crashed trial's energy is a "
            f"truncated partial sum, so the lowest-energy entry is the "
            f"most-broken candidate, not a deployment one. Re-run the "
            f"search, or select an entry explicitly at the call site."
        )

    params = MPCParams.from_dict(best_entry['params'])
    if params.to_dict() == DEFAULT_PARAMS:
        raise ValueError(
            f"Pareto archive {path} yielded params bit-identical to "
            f"DEFAULT_PARAMS. That is almost certainly a resolution bug, not a "
            f"genuine evolution result."
        )
    logger.info(
        "Loaded evolved params from %s (gen %s, archive-objective energy=%.1fJ, "
        "success=%.0f%%)",
        path,
        best_entry.get('generation', '?'),
        best_entry['objectives'][0],
        (1.0 - best_entry['objectives'][3]) * 100,
    )
    return params