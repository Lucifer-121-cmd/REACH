"""Guard: the downstream measurement scripts must let the caller say WHICH
genome is measured, and must record the resolved source in their output.

Why this test exists (2026-09-11): ``05_statistical_eval.py``,
``04_ablation.py`` and ``collect_trajectories.py`` all loaded the genome
from ``config['results_dir']`` -- the tracked archive, i.e. the
PRE-reevolution genome -- with no way to override it. The 2026-09-10
``statistical_results.json`` therefore described the old genome while the
campaign directory beside it held re-evolved-genome trials, and nothing
anywhere recorded the mismatch. This is the same defect class as the
126.7-vs-135.0 figure/table split the submission was rejected for: an
artefact that does not say what produced it.
"""
import ast
import os
import re
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = {
    'scripts/05_statistical_eval.py': 'ep2_evolution_root',
    'scripts/04_ablation.py':         'ep2_genome_source',
    'scripts/collect_trajectories.py': None,  # logs only; no JSON output dict
    'scripts/13_horizon_isolation.py': None,  # genome_provenance in output JSON
}


def _src(name: str) -> str:
    with open(os.path.join(REPO, name), encoding='utf-8') as f:
        return f.read()


@pytest.mark.parametrize('script', sorted(SCRIPTS))
def test_evolution_root_cli_flag_exists(script):
    """Every downstream script accepts --evolution-root (genome source)."""
    src = _src(script)
    assert "--evolution-root" in src, (
        f"{script} lost its --evolution-root flag; without it the script "
        "measures the tracked pre-reevolution genome with no way to say "
        "otherwise")
    # argparse registration, not just a mention in a comment
    assert "add_argument('--evolution-root'" in src


@pytest.mark.parametrize('script,output_key', [
    (k, v) for k, v in SCRIPTS.items() if v is not None
])
def test_output_records_genome_source(script, output_key):
    """The artefact names the genome it measured, so mislabelling is visible
    in the artefact itself (the quarantine doc for the 2026-09-10
    statistical_results.json)."""
    src = _src(script)
    assert f"'{output_key}'" in src or f'"{output_key}"' in src, (
        f"{script} does not record {output_key} in its output; a stale "
        "genome cannot be detected from the artefact alone")


def test_04_ablation_caches_stay_on_tracked_results():
    """--evolution-root must move ONLY the genome source. The BO and
    NSGA-II caches stand for ~13.4 days of unrecoverable tracked compute
    (PLAN.md §8 "never delete") and must keep resolving to
    config['results_dir'] even when a run-tree evolution root is given."""
    src = _src('scripts/04_ablation.py')
    assert "genome_dir  = (evolution_root" in src
    # every cache path must still be built from results_dir, not genome_dir
    for cache_marker in ("bo_comparison", "nsga2_cache_", "nsga2_runs"):
        for m in re.finditer(re.escape(cache_marker), src):
            line = src[:m.start()].splitlines()[-1]
            assert "genome_dir" not in line, (
                f"ablation cache path {cache_marker!r} uses genome_dir; "
                "caches must stay on the tracked results tree")


def test_05_genome_fallback_warns():
    """Omitting --evolution-root must log a warning that the fallback is the
    tracked pre-reevolution archive -- silent fallbacks are how the
    2026-09-10 file got mislabelled."""
    src = _src('scripts/05_statistical_eval.py')
    assert "TRACKED pre-reevolution" in src


def test_run_campaign_evolution_root_required_for_reevolved_genomes():
    """run_campaign.py records the evolution root per task; the new scripts
    must use the same directory convention (root containing
    evolution/<scenario>/)."""
    src = _src('scripts/run_campaign.py')
    assert "evolution_root" in src


def test_13_genome_provenance_follows_the_root():
    """Script 13's output must name the actual provenance: the hardcoded
    'LLM-guided search' string described the tracked archive genome, so the
    first 2026-09-11 gate_e re-run (reevolution genome, --evolution-root
    missing from the script) got labelled with the wrong provenance."""
    src = _src('scripts/13_horizon_isolation.py')
    assert "'--evolution-root'" in src or 'add_argument(\'--evolution-root' in src
    assert "genome_provenance = f'plain NSGA-II" in src
    # the hardcoded string must now only describe the fallback path
    assert "'genome_provenance': 'LLM-guided" not in src
