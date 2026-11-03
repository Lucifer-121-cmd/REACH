"""Tests for the fan-out harness (B2) and the quietness preflight.

The harness is new code with no historical behaviour to preserve, so what these
tests pin down is the two properties that would silently corrupt a campaign:

* **Seed independence.** Blocks must not overlap. Two "independent" arms sharing
  trials looks like agreement and is undetectable in the aggregates.
* **Spawn, not fork.** A forked worker inherits the parent's RNG state, so N
  workers can emit identical "random" trials with no error anywhere. The
  simulator is not exercised here -- only the process-creation path and the
  environment each worker actually ends up with.
"""
import multiprocessing as mp
import os
import sys
import inspect

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    'run_campaign',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'scripts', 'run_campaign.py'))
run_campaign = importlib.util.module_from_spec(_SPEC)
sys.modules['run_campaign'] = _SPEC.name and run_campaign
_SPEC.loader.exec_module(run_campaign)

from utils import preflight


# ---------------------------------------------------------------------------
# Planning and seeding
# ---------------------------------------------------------------------------

def test_plan_is_deterministic():
    a = run_campaign.build_plan('ep2', '/tmp/x')
    b = run_campaign.build_plan('ep2', '/tmp/x')
    assert [t.tag for t in a] == [t.tag for t in b]
    assert [t.master_seed for t in a] == [t.master_seed for t in b]


def test_plan_covers_every_scenario_and_seed():
    tasks = run_campaign.build_plan('ep2', '/tmp/x', n_seeds=5)
    assert len(tasks) == 3 * 5
    assert {t.scenario for t in tasks} == set(run_campaign.SCENARIOS)
    for scenario in run_campaign.SCENARIOS:
        seeds = [t.master_seed for t in tasks if t.scenario == scenario]
        assert len(set(seeds)) == 5, "seeds within a scenario must be distinct"


def test_seed_blocks_do_not_overlap_at_default_size():
    tasks = run_campaign.build_plan('ep2', '/tmp/x', n_trials=50)
    problems = [p for p in run_campaign.validate_plan(tasks, 4) if 'collision' in p]
    assert problems == []


def test_seed_collision_is_detected_when_blocks_would_overlap():
    """n_trials wider than the 1000-seed spacing must be caught, not run."""
    tasks = run_campaign.build_plan('ep2', '/tmp/x', n_trials=2000)
    problems = [p for p in run_campaign.validate_plan(tasks, 4) if 'collision' in p]
    assert problems, "overlapping seed blocks were not detected"


def test_every_scenario_seed_range_is_disjoint_across_scenarios_too():
    tasks = run_campaign.build_plan('ep2', '/tmp/x', n_trials=50)
    ranges = [(t.scenario, t.master_seed, t.master_seed + t.n_trials - 1) for t in tasks]
    for i, (s1, a1, b1) in enumerate(ranges):
        for s2, a2, b2 in ranges[i + 1:]:
            assert not (a1 <= b2 and a2 <= b1), f"overlap {(s1,a1,b1)} vs {(s2,a2,b2)}"


def test_unknown_stage_rejected():
    with pytest.raises(ValueError):
        run_campaign.build_plan('does_not_exist', '/tmp/x')


def test_max_workers_must_be_positive():
    tasks = run_campaign.build_plan('ep2', '/tmp/x')
    assert any('max-workers' in p for p in run_campaign.validate_plan(tasks, 0))


# ---------------------------------------------------------------------------
# Process creation: spawn, and single-threaded BLAS per worker
# ---------------------------------------------------------------------------

def _child_env(_):
    """Report the env a worker sees after _apply_single_thread_env()."""
    run_campaign._apply_single_thread_env()
    return {k: os.environ.get(k) for k in run_campaign.SINGLE_THREAD_ENV}


def test_worker_gets_single_threaded_blas():
    ctx = mp.get_context('spawn')
    with ctx.Pool(1) as pool:
        env = pool.map(_child_env, [None])[0]
    for key in run_campaign.SINGLE_THREAD_ENV:
        assert env[key] == run_campaign.SINGLE_THREAD_ENV[key], (
            f"{key} is {env[key]!r} in the worker; 192 processes each spawning "
            f"192 BLAS threads runs slower than a laptop (PLAN.md §7 Rule 1)")


def _child_rng(_):
    import numpy as np
    return int(np.random.default_rng().integers(0, 2 ** 62))


def test_spawned_workers_do_not_share_rng_state():
    """Under fork these can collide; under spawn they must not."""
    ctx = mp.get_context('spawn')
    with ctx.Pool(4) as pool:
        vals = pool.map(_child_rng, range(4))
    assert len(set(vals)) == len(vals), (
        "workers produced identical draws -- RNG state is being inherited")


def test_harness_declares_spawn_not_fork():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'scripts', 'run_campaign.py'), encoding='utf-8').read()
    assert "get_context('spawn')" in src
    assert "get_context('fork')" not in src


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_preflight_reports_physical_not_logical_cores():
    cores = preflight.physical_cores()
    assert cores >= 1
    assert cores <= (os.cpu_count() or 1)


def test_preflight_timing_thresholds_are_strict():
    assert preflight.TIMING_MAX_LOAD <= 2.0
    assert preflight.TIMING_MAX_FOREIGN_PYTHON == 0


def test_preflight_rejects_unknown_mode():
    with pytest.raises(ValueError):
        preflight.check('whenever')


def test_preflight_returns_a_report_for_both_modes():
    for mode in ('timing', 'bulk'):
        res = preflight.check(mode)
        assert set(res) >= {'ok', 'mode', 'reasons', 'physical_cores', 'loadavg'}
        assert isinstance(preflight.format_report(res), str)


def test_evolution_root_is_threaded_to_every_task():
    """The genome source must be explicit, per task, and recorded.

    `run_task` loaded the genome from `config['results_dir']` -- the TRACKED pre-C1
    archive -- with no way to say otherwise. A campaign launched to validate a
    re-evolved genome would have measured the OLD one and written the numbers under
    the new run-id, and nothing in the output would have looked wrong. That is the
    same class of defect as the archive-vs-measured table row (REPRODUCE.md §5.2),
    one layer earlier and harder to see.
    """
    root = 'runs/RID/reevolution'
    for stage in ('ep2', 'baselines', 'gate_r2'):
        tasks = run_campaign.build_plan(stage, '/tmp/x', evolution_root=root)
        assert tasks, stage
        assert all(t.evolution_root == root for t in tasks), stage
    # Default stays None so the fallback is a deliberate choice, not a hidden one.
    assert all(t.evolution_root is None
               for t in run_campaign.build_plan('ep2', '/tmp/x'))


def test_evolution_root_default_is_none_not_a_guessed_path():
    """A defaulted path would reintroduce the silent-stale-read failure."""
    import dataclasses
    f = {x.name: x for x in dataclasses.fields(run_campaign.Task)}
    assert 'evolution_root' in f, 'Task lost its explicit genome source'
    assert f['evolution_root'].default is None


def test_campaign_cli_exposes_evolution_root():
    """It must be settable from the command line, or the plan cannot be honest."""
    import argparse
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        old = sys.argv
        try:
            sys.argv = ['run_campaign.py', '--help']
            run_campaign.main()
        finally:
            sys.argv = old
    help_text = buf.getvalue()
    assert '--evolution-root' in help_text
    # The help must say what the fallback is; a silent default is the whole risk.
    assert 'PRE-C1' in help_text or 'pre-C1' in help_text


def test_preflight_reserved_core_mode_ignores_foreign_load():
    """2026-09-14: with --cpus, the gate must decide on the RESERVED cores'
    busy fraction alone. The foreign-python check ran unconditionally,
    making the reserved-core gate impossible to pass while any co-tenant
    process existed -- the exact 'gate that gets overridden' failure the
    docstring warns about. PLAN.md §7 Rule 2: reserve a core set with
    nothing else running; load ELSEWHERE is irrelevant by construction.
    """
    src = inspect.getsource(preflight.check)
    # the foreign-python refusal must live inside the no-cpus branch
    timing_block = src.split("if mode == 'timing':", 1)[1].split("elif mode == 'bulk':", 1)[0]
    assert "if cpus:" in timing_block
    foreign_pos = timing_block.find("TIMING_MAX_FOREIGN_PYTHON")
    cpus_pos = timing_block.find("if cpus:")
    else_pos = timing_block.find("else:")
    assert foreign_pos > else_pos > cpus_pos >= 0, (
        "the foreign-python check must be inside the no-cpus (else) branch "
        "of the timing gate; reserved-core mode gates on per-core busy only")
