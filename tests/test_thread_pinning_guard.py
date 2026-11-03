"""Guard: parallel campaign scripts must pin BLAS/OMP threads BEFORE numpy.

2026-09-17 discovery (the P5/hybrid realization drift): scripts 30 and 31
set OMP_NUM_THREADS etc. INSIDE the worker function while importing numpy
at MODULE scope -- with the 'spawn' pool each worker re-imports the
module first, so OpenBLAS/MUMPS always initialized multi-threaded and
the env assignment was a no-op. Measured consequence vs the pinned
campaign lineage on the same 60 missions: 48/60 outcome agreement,
median |spent| drift 0.15 J, 12 knife-edge flips (one of which moved a
mission's breach flag). scripts/21 has always pinned correctly (module-scope
env application before the first numpy/casadi import inside the worker).

The rule, enforced here: a script that imports numpy at module scope
must set the thread env at module scope, ABOVE that import. A script
that imports numpy lazily must apply its env before that import (the
script-21 pattern). This is a source-order test, not a runtime one:
the failure mode is silent, so the guard must fire before any campaign
launches, not after.

2026-09-20: scripts/22, 23 and 25 (the serial REACH drivers) were added
under 'module_scope_env' -- they imported numpy at module scope with NO
thread pins at all, so any re-run would drift from the pinned campaign
lineage for the §11.15 reason. Their existing artefacts (stepchange,
planning timing, descent check) predate the fix and stay the runs of
record. The same sweep found scripts/14 (GATE-R1) and 24 (oracle)
unpinned too; they are added under 'module_scope_env' as well.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = {
    '14_gate_r1_reach.py': 'module_scope_env',
    '21_reach_campaign.py': 'env_applied_before_lazy_numpy',
    '22_reach_stepchange.py': 'module_scope_env',
    '23_reach_timing.py': 'module_scope_env',
    '24_reach_oracle.py': 'module_scope_env',
    '25_descent_check.py': 'module_scope_env',
    '29_threshold_baseline.py': 'module_scope_env',
    '30_p5_margin.py': 'module_scope_env',
    '31_hybrid_validation.py': 'module_scope_env',
}
_THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _lines(name):
    with open(os.path.join(_ROOT, 'scripts', name)) as fh:
        return fh.read().splitlines()


def _first_module_numpy(lines):
    """Index of the first top-level numpy import, or None."""
    for i, ln in enumerate(lines):
        if re.match(r'import numpy', ln):
            return i
    return None


def _first_module_env_set(lines):
    """Index of the first top-level env assignment of a thread var."""
    for i, ln in enumerate(lines):
        for var in _THREAD_VARS:
            if re.match(rf"for _k in \(?.*'{var}'", ln) or \
                    (f"os.environ['{var}']" in ln):
                return i
    return None


@pytest.mark.parametrize('name', sorted(_SCRIPTS))
def test_threads_pinned_before_numpy(name):
    lines = _lines(name)
    pattern = _SCRIPTS[name]
    if pattern == 'module_scope_env':
        i_env = _first_module_env_set(lines)
        i_np = _first_module_numpy(lines)
        assert i_env is not None, (
            f"{name}: no module-scope thread-env assignment found -- "
            "without it spawn workers run multi-threaded BLAS and the "
            "realization drifts from the campaign lineage")
        assert i_np is not None, (
            f"{name}: expected a module-scope numpy import (pattern "
            "drifted; update this guard)")
        assert i_env < i_np, (
            f"{name}: numpy imports at line {i_np + 1} but threads are "
            f"pinned at line {i_env + 1} -- the pin happens AFTER OpenBLAS "
            "reads the environment and is a no-op")
    else:
        # script-21 pattern: no module-scope numpy at all; the WORKER
        # applies the env before its (lazy) numpy/casadi-touching imports.
        assert _first_module_numpy(lines) is None, (
            f"{name}: gained a module-scope numpy import -- it must "
            "pin at module scope too (see scripts/30's header comment)")
        src = '\n'.join(lines)
        assert 'SINGLE_THREAD_ENV' in src and \
            '_apply_single_thread_env()' in src, (
            f"{name}: the lazy-import env pattern was refactored away -- "
            "restore it or switch to the module-scope pin, then update "
            "this guard")


def test_worker_fn_no_too_late_env_set():
    """No campaign script may keep a thread-env assignment inside the
    worker body: it is dead code after numpy has loaded, and it masks
    the defect (the 2026-09-16 launches looked pinned but were not)."""
    for name in _SCRIPTS:
        src = '\n'.join(_lines(name))
        parts = re.split(r'def _run_one|def run_reach_task|def threshold_rollout',
                         src)
        if len(parts) < 2:
            # serial driver (22/23/25): no worker body exists; the
            # module-scope pin is the whole discipline for these.
            continue
        body = parts[-1]
        for var in _THREAD_VARS:
            assert f"'{var}'" not in body or name == '21_reach_campaign.py', (
                f"{name}: thread var {var} is set inside a worker body "
                "(too late to matter) -- pin at module scope instead")


def _src(name):
    return open(os.path.join(_ROOT, 'scripts', name)).read()


def test_hybrid_jobs_carry_fitted_certificates():
    """The hybrid comparison (scripts/31) must run on the DEPLOYED
    per-map certificates and its rows must carry them: the certificate
    IS the hybrid's return margin, and the first two launches (2026-09-16
    and the aborted 2026-09-17 pin rerun) silently fell back to the 24 J
    legacy placeholder because no a_q was passed -- internally consistent,
    but not the configuration the paper reports. A campaign that runs
    run_reach_trial under an unverified intercept is a placeholder run
    by definition (see reach_evaluator.py's a_q docstring)."""
    src = _src('31_hybrid_validation.py')
    assert "certificates.json" in src, (
        "scripts/31 must load the per-split fitted certificates (the "
        "reach_evaluator's 24 J fallback under-reserves by up to 2.25x)")
    assert "'a_q': a_q" in src and "'a_q': j['a_q']" in src, (
        "script 31 must attach the fitted a_q to every job and echo it "
        "into every record")
    assert "a_q=j['a_q']" in src, (
        "script 31 must pass a_q to run_reach_trial (without it the "
        "mission silently runs the 24 J placeholder)")


def test_p5_scales_the_fitted_certificate():
    src = _src('30_p5_margin.py')
    assert "'a_q': j['a_q']" in src, (
        "script 30 must record the intercept each mission ran at")
    assert "a_q=j['a_q']" in src, (
        "script 30 must pass the (scaled) fitted a_q to run_reach_trial")
    assert 'round(s * aq' not in src and "round(s*aq" not in src, (
        "script 30 must NOT round the scaled intercept: a 9e-8 J shift "
        "was measured to flip the knife-edge mission (spill 0.45 -> "
        "0.54 J across the 0.5 J breach tolerance) — inputs must stay "
        "bit-equal to the campaign lineage")
