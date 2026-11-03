"""Regression guard: every genome recorded under results/genome_reevolved/ must
lie in PARAM_BOUNDS.

Why this exists (A4): the three tests rewritten alongside this file all turned on
``PARAM_BOUNDS['d_safe']``. If someone widens that box to make a stale assertion
pass -- or narrows it during F8 -- the tracked genomes silently fall outside the
declared search space, and every number derived from them becomes unreproducible
from the stated bounds. This test fails loudly in that case.

It reads only the tracked JSON under ``results/genome_reevolved/`` (the EP²
comparator arm's genome of record: gen059 NSGA-II winners, LLM disabled) and
asserts nothing about performance, so it is load-insensitive and safe to run at
any time.
"""
import glob
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.mpc_params import PARAM_BOUNDS, DEFAULT_PARAMS
from utils.checkpoint_utils import load_best_evolved_params

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOME_ROOT = os.path.join(REPO_ROOT, 'results', 'genome_reevolved')
_GENOME_PRESENT = os.path.isdir(os.path.join(GENOME_ROOT, 'evolution'))

pytestmark = pytest.mark.skipif(
    not _GENOME_PRESENT,
    reason='results/genome_reevolved/ not present in this checkout')
SCENARIOS = ('corridor', 's_curve', 'chicane')
COLLISION_RADIUS = 0.25

#: Files whose params predate the current box are not expected here -- if one
#: appears, that is the finding, not an exemption to add.
_TOL = 1e-9


def _params_dicts(obj, where, out):
    """Recursively collect every ``params`` dict, tagged with its location."""
    if isinstance(obj, dict):
        p = obj.get('params')
        if isinstance(p, dict) and any(k in p for k in PARAM_BOUNDS):
            out.append((where, p))
        for k, v in obj.items():
            if k != 'params':
                _params_dicts(v, f"{where}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _params_dicts(v, f"{where}[{i}]", out)


def _tracked_result_jsons():
    return sorted(glob.glob(os.path.join(GENOME_ROOT, '**', '*.json'),
                            recursive=True))


def test_deployed_genomes_are_in_bounds():
    """The three deployed genomes must validate against the declared box."""
    for scenario in SCENARIOS:
        params = load_best_evolved_params(scenario, GENOME_ROOT)
        d = params.to_dict()
        assert d != DEFAULT_PARAMS, f"{scenario}: deployed genome equals DEFAULT_PARAMS"
        for name, (lo, hi) in PARAM_BOUNDS.items():
            if name not in d:
                continue
            assert lo - _TOL <= d[name] <= hi + _TOL, (
                f"{scenario}: deployed {name}={d[name]!r} is outside "
                f"PARAM_BOUNDS[{name}]=({lo}, {hi})"
            )


def test_deployed_d_safe_clears_the_body_radius():
    for scenario in SCENARIOS:
        d_safe = load_best_evolved_params(scenario, GENOME_ROOT).d_safe
        assert d_safe > COLLISION_RADIUS, (
            f"{scenario}: deployed d_safe={d_safe} does not clear r_body="
            f"{COLLISION_RADIUS}, so its obstacle penalty is dead at every "
            f"collision-free state"
        )


def test_loader_refuses_archives_with_no_100pct_success(tmp_path):
    """An archive with no 100%-success entry must REFUSE, not serve the
    lowest-energy candidate behind a warning.

    REPRODUCE.md §7.1a: a crashed trial's energy is a truncated partial
    sum, so the lowest-energy entry is the MOST-broken candidate (it
    both fails and reads cheap) -- the 93.00 J / 45%-failure chicane
    archive point is the exact historical shape. The fail-open was
    flagged 2026-09-05, lived as a logger.warning, and was made to raise
    on 2026-09-20."""
    root = tmp_path / 'evolution' / 'chicane'
    root.mkdir(parents=True)
    entries = [
        {'objectives': [93.00, 1.0, 1.0, 0.45], 'params': {}},
        {'objectives': [117.03, 1.0, 1.0, 0.20], 'params': {}},
        {'objectives': [126.47, 1.0, 1.0, 0.10], 'params': {}},
    ]
    (root / 'pareto_gen059.json').write_text(json.dumps(entries))
    with pytest.raises(ValueError, match='No 100%-success candidate'):
        load_best_evolved_params('chicane', str(tmp_path))


def test_every_tracked_genome_is_in_bounds():
    """Sweep every params dict in every tracked evolution/ablation JSON."""
    files = _tracked_result_jsons()
    if not files:
        pytest.skip("no tracked result JSONs present in this checkout")

    found = 0
    offenders = []
    for path in files:
        try:
            with open(path, encoding='utf-8') as fh:
                blob = json.load(fh)
        except (OSError, ValueError) as exc:
            offenders.append(f"{os.path.relpath(path, REPO_ROOT)}: unreadable ({exc})")
            continue
        collected = []
        _params_dicts(blob, os.path.relpath(path, REPO_ROOT), collected)
        for where, p in collected:
            found += 1
            for name, (lo, hi) in PARAM_BOUNDS.items():
                if name not in p:
                    continue
                val = p[name]
                if not (lo - _TOL <= val <= hi + _TOL):
                    offenders.append(f"{where}: {name}={val!r} outside ({lo}, {hi})")

    assert found > 0, "found no genomes to check -- the sweep is not doing anything"
    assert not offenders, (
        f"{len(offenders)} tracked genome value(s) outside PARAM_BOUNDS "
        f"(checked {found} genomes across {len(files)} files):\n  "
        + "\n  ".join(offenders[:25])
    )
