"""Single place where IPOPT options are constructed and validated.

Two jobs:

1. **Refuse unusable linear solvers.** The casadi 3.7.2 wheel links
   MUMPS 5.4.1 and SPRAL. It does *not* link HSL. Asking for an HSL solver does
   not raise a Python exception -- it fails in two much worse ways, both measured
   on this wheel:

   =============  ==================================================
   ``ma27``       solve returns status ``Invalid_Option``, no raise
   ``ma57``       solve returns status ``Invalid_Option``, no raise
   ``ma86``       solve returns status ``Invalid_Option``, no raise
   ``pardiso``    solve returns status ``Invalid_Option``, no raise
   ``ma97``       **SIGSEGV -- the process dies with a core dump**
   =============  ==================================================

   In a fan-out, an ``Invalid_Option`` worker looks like a failed trial and a
   segfaulting worker looks like a lost one. Neither looks like a config error.
   So the option is validated here, once, before it can reach IPOPT.

2. **Carry the deadline.** ``ipopt.max_cpu_time`` is the only mechanism that
   bounds a single solve. It is expressed as a fraction of the control period so
   that a deadline change is one number in one place.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Linear solvers actually present in the casadi 3.7.2 manylinux wheel.
USABLE_LINEAR_SOLVERS = frozenset({'mumps', 'spral'})

#: Absent, with the failure mode each one produces if requested anyway.
FORBIDDEN_LINEAR_SOLVERS = {
    'ma27': "not linked into the casadi wheel; returns Invalid_Option without raising",
    'ma57': "not linked into the casadi wheel; returns Invalid_Option without raising",
    'ma86': "not linked into the casadi wheel; returns Invalid_Option without raising",
    'ma97': "not linked into the casadi wheel; SEGFAULTS the process (SIGSEGV)",
    'pardiso': "not linked into the casadi wheel; returns Invalid_Option without raising",
    'wsmp': "not linked into the casadi wheel",
}

DEFAULT_LINEAR_SOLVER = 'mumps'


class UnusableLinearSolver(ValueError):
    """Raised when a linear solver absent from this wheel is requested."""


def validate_linear_solver(name: Optional[str]) -> None:
    """Raise :class:`UnusableLinearSolver` unless ``name`` is usable here."""
    if name is None:
        return
    key = str(name).strip().lower()
    if key in USABLE_LINEAR_SOLVERS:
        return
    reason = FORBIDDEN_LINEAR_SOLVERS.get(key, "not a known IPOPT linear solver")
    raise UnusableLinearSolver(
        f"ipopt.linear_solver={name!r} is unusable on this build: {reason}.\n"
        f"  Usable: {sorted(USABLE_LINEAR_SOLVERS)} (default {DEFAULT_LINEAR_SOLVER!r}).\n"
        f"  Verified empirically against casadi 3.7.2 / IPOPT 3.14.11 on 2026-08-31."
    )


#: Cold-start planning profile. The defaults below are TRACKING-MPC defaults and
#: they are wrong for a cold nonconvex planning solve -- reusing them silently is a
#: real defect this project already committed once.
#:
#: Measured on mpc/reach_mpc.py (N=60, 366 variables, an interpolated D* in the
#: objective): with the tracking defaults every solve returned in a suspiciously
#: uniform 1.5-1.8 s with `success=False` and the terminal reserve constraint
#: violated by 1.8-90 J. That is not a hard problem, it is `max_iter=200` plus
#: `acceptable_iter=5` at `acceptable_tol=1e-3` letting IPOPT bail to a loose point
#: after a handful of iterations, and `mu_init=1e-3` with
#: `warm_start_init_point='yes'` telling it to trust multipliers that do not exist.
#:
#: A tracking MPC re-solves a nearly identical problem every 100 ms from a warm
#: start, so those values are right there and wrong here. Nothing about the two
#: problems shares a time budget.
#: A second measured correction: `tol = 1e-6` was also wrong, and for a reason
#: specific to this objective. D* enters the cost through
#: `ca.interpolant(..., 'linear', ...)`, whose derivative is DISCONTINUOUS at every
#: grid line. Linear was chosen over a spline deliberately -- D* has genuine creases
#: along the medial axis and a spline oscillates across them
#: (planning/energy_dijkstra.py:build_interpolant) -- so the non-smoothness is a
#: design trade, not an accident. Demanding a 1e-6 KKT residual from a cost with a
#: piecewise-constant gradient asks for something the problem does not possess:
#: measured, IPOPT ran the full 3000 iterations to `Maximum_Iterations_Exceeded`
#: while the terminal reserve slack was already 0.000000 and the reach monotone in
#: budget. It was grinding on optimality it could never certify, having long since
#: found the answer.
#:
#: So the acceptance test for a REACH solve is CONSTRAINT SATISFACTION (slack >= 0),
#: not IPOPT's success flag. `tol` is set to a value the objective can actually
#: support and `acceptable_*` is what normally terminates.
PLANNING_PROFILE: Dict[str, Any] = {
    'ipopt.tol': 1e-4,
    'ipopt.mu_init': 0.1,
    'ipopt.mu_strategy': 'adaptive',
    'ipopt.warm_start_init_point': 'no',
    # Feasibility first: a reserve certificate that is violated is worthless, so
    # never accept a point that has not restored the constraints.
    'ipopt.required_infeasibility_reduction': 0.9,
    'ipopt.honor_original_bounds': 'yes',
    # 2026-09-12: every REACH solve -- cold, warm, outbound, return -- was
    # measured running the FULL 3000 iterations (40-60 s each) while the
    # reserve slack was already at 0.00 and the reach monotone: grinding on
    # a KKT residual the piecewise-linear interpolant objective cannot
    # certify (see the tol note above). The escape hatch IPOPT actually
    # offers here is the acceptable-point track: accept a point once it has
    # held acceptable_tol for acceptable_iter iterations. Loosening
    # acceptable_tol to 1e-2 and letting acceptable_iter carry the
    # diligence lets the same test terminate in a fraction of the
    # iterations -- acceptance is slack >= 0 either way, so nothing about
    # the RESULT's trustworthiness changes; only the iteration bill does.
    'ipopt.acceptable_tol': 1e-2,
    'ipopt.acceptable_iter': 15,
    # 2026-09-12, second measurement: even with the loose acceptable track,
    # every solve still ran to the full iteration cap (3000 iters, 40-60 s)
    # because the non-smooth interpolant keeps the iterates oscillating in
    # and out of the acceptable band, resetting the 15-iteration counter.
    # The acceptance criterion for a REACH solve is constraint satisfaction
    # (slack >= 0) -- the supervisor REJECTS violating plans regardless of
    # what IPOPT says -- so iterations beyond the point where the objective
    # has effectively converged buy nothing the protocol uses. 800 (the
    # first try) was TOO tight: solves came back violating, the supervisor
    # turned back at t = 0.1 s and idled at the dock until stranding -- a
    # new failure mode, not a faster mission. 1500 keeps most of the speed
    # gain while leaving the solver room to find feasible points.
    'ipopt.max_iter': 1500,
}


def build_ipopt_opts(config: Dict[str, Any], *, name: str = 'mpc',
                     profile: str = 'tracking',
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Construct the ``nlpsol`` options dict for one solver.

    ``profile`` selects the iteration/barrier regime: ``'tracking'`` for the
    warm-started per-step controllers, ``'planning'`` for a cold mission-length
    solve. See ``PLANNING_PROFILE`` for why they cannot share values.

    Recognised config keys:

    ``ipopt_max_iter``       iteration cap (default 200)
    ``ipopt_tol``            convergence tolerance (default 1e-4)
    ``ipopt_linear_solver``  validated against this wheel (default ``mumps``)
    ``ipopt_max_cpu_time``   absolute per-solve CPU budget in seconds
    ``deadline_s``           control period; used only if ``ipopt_max_cpu_time``
                             is absent, as ``deadline_fraction * deadline_s``
    ``deadline_fraction``    default 0.9
    """
    linear_solver = config.get('ipopt_linear_solver', DEFAULT_LINEAR_SOLVER)
    validate_linear_solver(linear_solver)

    opts: Dict[str, Any] = {
        'ipopt.max_iter': config.get('ipopt_max_iter', 200),
        'ipopt.tol': config.get('ipopt_tol', 1e-4),
        'ipopt.acceptable_tol': 1e-3,
        'ipopt.acceptable_iter': 5,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_init': 1e-3,
        'ipopt.linear_solver': linear_solver,
    }

    if profile == 'planning':
        opts.update(PLANNING_PROFILE)
    elif profile != 'tracking':
        raise ValueError(f"unknown solver profile {profile!r}; "
                         f"known: tracking, planning")

    max_cpu = config.get('ipopt_max_cpu_time')
    if max_cpu is None:
        deadline = config.get('deadline_s')
        if deadline is not None:
            max_cpu = float(config.get('deadline_fraction', 0.9)) * float(deadline)
    if max_cpu is not None:
        # IPOPT treats max_cpu_time as advisory: it is checked between
        # iterations, so a single long factorisation can still overrun it. The
        # deadline-miss counter in the controller is what actually measures
        # compliance; this only stops a runaway solve.
        opts['ipopt.max_cpu_time'] = float(max_cpu)

    if extra:
        for key, value in extra.items():
            if key in ('ipopt.linear_solver', 'linear_solver'):
                validate_linear_solver(value)
            opts[key] = value
    return opts
