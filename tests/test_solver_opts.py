"""Tests for mpc/solver_opts.py — the IPOPT option guard.

Two invariants worth locking down:

1. Consolidating three duplicated option dicts must not have changed solver
   behaviour. The pre-consolidation values are pinned here verbatim from
   ``git show c15d84c:mpc/*.py``, so any future edit to ``build_ipopt_opts`` that
   silently changes a solver setting fails a test instead of quietly moving every
   downstream number.
2. HSL linear solvers must be unreachable by every route, not just through the
   helper's own keyword — including a YAML config and the ``extra=`` escape
   hatch. On this wheel ma27/ma57/ma86/pardiso return ``Invalid_Option`` without
   raising and ma97 segfaults the process, so neither failure mode looks like a
   configuration error in a fan-out log.
"""
import glob
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.solver_opts import (
    build_ipopt_opts, validate_linear_solver, UnusableLinearSolver,
    USABLE_LINEAR_SOLVERS, FORBIDDEN_LINEAR_SOLVERS, DEFAULT_LINEAR_SOLVER,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Verbatim from git show c15d84c:mpc/ep2_mpc.py (and _5state, identical).
PRE_CONSOLIDATION_BASE = {
    'ipopt.max_iter': 200,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-3,
    'ipopt.acceptable_iter': 5,
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.mu_init': 1e-3,
}

#: Only standard_mpc_casadi had these.
PRE_CONSOLIDATION_WARM = {
    'ipopt.warm_start_bound_push': 1e-6,
    'ipopt.warm_start_bound_frac': 1e-6,
    'ipopt.warm_start_slack_bound_push': 1e-6,
    'ipopt.warm_start_slack_bound_frac': 1e-6,
    'ipopt.warm_start_mult_bound_push': 1e-6,
}


def test_consolidated_opts_preserve_every_original_value():
    """Every key the original dicts set must keep its original value."""
    opts = build_ipopt_opts({'ipopt_max_iter': 200, 'ipopt_tol': 1e-4})
    for key, expected in PRE_CONSOLIDATION_BASE.items():
        assert key in opts, f"{key} was dropped by consolidation"
        assert opts[key] == expected, (
            f"{key} changed from {expected!r} to {opts[key]!r}; every arm "
            f"comparison against the frozen numbers would be affected"
        )


def test_consolidation_adds_only_the_linear_solver():
    """The single intended addition, so an accidental extra key is caught."""
    opts = build_ipopt_opts({'ipopt_max_iter': 200, 'ipopt_tol': 1e-4})
    added = set(opts) - set(PRE_CONSOLIDATION_BASE)
    assert added == {'ipopt.linear_solver'}, f"unexpected new options: {added}"
    # mumps is IPOPT's compiled default on this wheel, so naming it explicitly is
    # a no-op -- verified bit-exactly over 150 solves spanning N=10..50.
    assert opts['ipopt.linear_solver'] == DEFAULT_LINEAR_SOLVER == 'mumps'


def test_standard_mpc_keeps_its_five_extra_warm_start_keys():
    opts = build_ipopt_opts({}, name='standard_mpc',
                            extra=dict(PRE_CONSOLIDATION_WARM))
    for key, expected in PRE_CONSOLIDATION_WARM.items():
        assert opts[key] == expected


@pytest.mark.parametrize('bad', sorted(FORBIDDEN_LINEAR_SOLVERS))
def test_hsl_refused_via_helper(bad):
    with pytest.raises(UnusableLinearSolver):
        validate_linear_solver(bad)


@pytest.mark.parametrize('bad', sorted(FORBIDDEN_LINEAR_SOLVERS))
def test_hsl_refused_via_config_dict(bad):
    """A YAML config that sets it must not slip past."""
    with pytest.raises(UnusableLinearSolver):
        build_ipopt_opts({'ipopt_linear_solver': bad, 'dt': 0.1})


@pytest.mark.parametrize('key', ['ipopt.linear_solver', 'linear_solver'])
def test_hsl_refused_via_extra_escape_hatch(key):
    with pytest.raises(UnusableLinearSolver):
        build_ipopt_opts({}, extra={key: 'ma97'})


def test_ma97_is_documented_as_the_segfaulting_one():
    """The message must say *how* it fails, or nobody will believe the guard."""
    assert 'SEGFAULT' in FORBIDDEN_LINEAR_SOLVERS['ma97'].upper()
    with pytest.raises(UnusableLinearSolver, match='SEGFAULT'):
        validate_linear_solver('ma97')


def test_usable_solvers_are_accepted():
    for good in USABLE_LINEAR_SOLVERS:
        validate_linear_solver(good)
    assert USABLE_LINEAR_SOLVERS == {'mumps', 'spral'}


def test_unset_linear_solver_is_allowed():
    validate_linear_solver(None)


def test_deadline_translates_to_max_cpu_time():
    assert 'ipopt.max_cpu_time' not in build_ipopt_opts({})
    assert build_ipopt_opts({'ipopt_max_cpu_time': 0.25})['ipopt.max_cpu_time'] == 0.25
    o = build_ipopt_opts({'deadline_s': 0.1, 'deadline_fraction': 0.9})
    assert o['ipopt.max_cpu_time'] == pytest.approx(0.09)


def test_every_nlpsol_site_uses_the_guard():
    """Structural guard: no controller may hand-roll its own option dict."""
    offenders = []
    for path in glob.glob(os.path.join(REPO_ROOT, 'mpc', '*.py')):
        src = open(path, encoding='utf-8').read()
        if 'nlpsol(' not in src:
            continue
        if 'build_ipopt_opts' not in src:
            offenders.append(os.path.relpath(path, REPO_ROOT))
    assert not offenders, (
        f"these files call nlpsol without importing build_ipopt_opts, so an HSL "
        f"linear_solver could reach IPOPT unvalidated: {offenders}"
    )
