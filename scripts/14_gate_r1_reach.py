#!/usr/bin/env python3
"""GATE-R1 — does a max-reach objective put the optimum ON the reserve boundary?

**This is the gate that decides the project's direction, and it costs no compute.**
Using the weighted geodesic field and the 6.6772 J/m floor, verify
that for a max-reach objective the optimum lies on the reserve boundary across the
plausible reserve range: that achievable outbound progress is strictly increasing
in available energy over r in [0.2, 0.8], with a turn-back point interior to the
map.

Why it matters. The earlier min-energy-to-goal formulation died because objective
and constraint pointed the same way
-- both wanted the short cheap path -- so the constraint selected nothing, and the
entire usable window was the 0.97-3.30% excess over the floor. REACH claims to
escape that because the objective wants to spend MORE than the reserve allows, so
the optimum sits on the boundary by construction. That claim is arithmetic, not
simulation, and it can be checked now:

    maximise  s_out                                   (outbound arclength)
    s.t.      E_out(s_out) + E_back(s_out) <= r * B    (reserve)

with E_out and E_back priced off the geodesic field. If the achievable s_out is
strictly increasing in r over [0.2, 0.8] and the turn-back point is interior to
the route (neither 0 nor the far end), the constraint is doing the selecting and
REACH is not the dead formulation in a new costume.

PASS  -> proceed with REACH.
FAIL  -> the reserve constraint still selects nothing. Switch to REALTIME immediately.

Pure NumPy, no solver, single core, seconds. Writes nothing unless --out is given.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

# Single-threaded BLAS/OMP, set BEFORE the first numpy/casadi import:
# OpenBLAS/MUMPS read the environment at import time, and results must
# land on the pinned campaign lineage -- ulp-scale numeric changes re-route
# the non-smooth NLP into different basins (6-decimal rounding of one
# certificate intercept shifted 12/60 outcomes once). Same
# idiom as scripts/30. Added 2026-09-20 during the full audit sweep.
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Reserve fractions to sweep -- r in [0.2, 0.8].
DEFAULT_R_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def route_profile(field, start, n: int = 400, *,
                  return_rate: str = 'lambda_min',
                  cert_a: float = 0.0, cert_b: Optional[float] = None
                  ) -> Dict[str, np.ndarray]:
    """Sample the geodesic route and price outbound / return energy along it.

    ``return_rate`` decides how the RETURN leg is priced, and it matters:

    ``'lambda_min'``  both legs at the thermodynamic floor. This is the baseline
        version, and it makes the answer trivially linear: reach = r*B / (2*lambda_min) exactly, so
        "strictly increasing in r" holds by construction and the gate tests almost
        nothing. Reported anyway, because it is the specified check.
    ``'certificate'``  the return leg is priced by the affine certificate
        ``Ebar(D*) = a + b*D*`` measured in prototypes/audit_vfb_decomposition.py
        (a = 21-26 J of fixed heading-realignment cost, b = lambda_min). This is
        what the controller would actually have to reserve, so the reach curve
        becomes affine-with-an-offset rather than linear, and the gate measures
        something real: how much reach the certificate itself costs.
    """
    from planning.energy_dijkstra import geodesic_path

    path = geodesic_path(field, start)
    if path is None or len(path) < 3:
        raise ValueError("no geodesic route from the start cell")
    seg = np.hypot(*np.diff(path, axis=0).T)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])       # arclength from start
    total = float(s_cum[-1])

    s_out = np.linspace(0.0, total, n)
    lam = field.lambda_min
    E_out = lam * s_out
    if return_rate == 'certificate':
        b = lam if cert_b is None else cert_b
        E_back = cert_a + b * s_out
    elif return_rate == 'lambda_min':
        E_back = lam * s_out
    else:
        raise ValueError(f"return_rate must be 'lambda_min' or 'certificate', "
                         f"got {return_rate!r}")
    return {'s_out': s_out, 'E_out': E_out, 'E_back': E_back,
            'E_total': E_out + E_back, 'route_length': total,
            'path': path, 's_cum': s_cum, 'return_rate': return_rate}


def max_reach(profile: Dict[str, np.ndarray], budget: float) -> Dict[str, Any]:
    """Largest outbound arclength whose out+back cost fits inside ``budget``."""
    feasible = profile['E_total'] <= budget
    if not feasible.any():
        return {'reach_m': 0.0, 'binding': True, 'interior': False,
                'slack_J': float(budget - profile['E_total'][0])}
    idx = int(np.max(np.nonzero(feasible)[0]))
    reach = float(profile['s_out'][idx])
    total_at_reach = float(profile['E_total'][idx])
    # 'binding' = the reserve constraint is what stopped us, not the map's end.
    at_route_end = idx == profile['s_out'].size - 1
    return {
        'reach_m': reach,
        'reach_fraction_of_route': reach / profile['route_length'],
        'energy_used_J': total_at_reach,
        'slack_J': float(budget - total_at_reach),
        'binding': not at_route_end,
        'interior': (0.0 < reach < profile['route_length'] * (1 - 1e-9)),
    }


def assess(name: str, field, start, r_grid, *, pack_J: Optional[float] = None,
           n: int = 400, return_rate: str = 'lambda_min',
           cert_a: float = 0.0, cert_b: Optional[float] = None) -> Dict[str, Any]:
    """Sweep the reserve fraction and apply the PASS/FAIL rule."""
    prof = route_profile(field, start, n=n, return_rate=return_rate,
                         cert_a=cert_a, cert_b=cert_b)
    # B defaults to the out-and-back cost of the WHOLE route, so r is literally
    # "what fraction of a full round trip the pack can afford". That makes r
    # comparable across maps of different length, which a raw joule budget is not.
    B = float(pack_J if pack_J is not None else prof['E_total'][-1])

    rows = []
    for r in r_grid:
        m = max_reach(prof, r * B)
        m['r'] = float(r)
        m['budget_J'] = r * B
        rows.append(m)

    reaches = np.array([x['reach_m'] for x in rows])
    strictly_increasing = bool(np.all(np.diff(reaches) > 1e-9))
    all_interior = all(x['interior'] for x in rows)
    all_binding = all(x['binding'] for x in rows)
    passed = strictly_increasing and all_interior and all_binding

    return {
        'scenario': name,
        'route_length_m': prof['route_length'],
        'lambda_min': field.lambda_min,
        'round_trip_J': B,
        'r_grid': list(map(float, r_grid)),
        'return_rate': return_rate,
        'cert_a_J': cert_a, 'cert_b_J_per_m': cert_b,
        'rows': rows,
        'reach_strictly_increasing_in_r': strictly_increasing,
        'turn_back_interior_for_all_r': all_interior,
        'reserve_binding_for_all_r': all_binding,
        'pass': passed,
    }


def verdict(per_map: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    failing = [k for k, v in per_map.items() if not v['pass']]
    if not failing:
        return {
            'outcome': 'PASS_PROCEED_WITH_REACH',
            'text': "Achievable outbound progress is strictly increasing in the "
                    "reserve fraction on every map, the turn-back point is interior "
                    "to the route throughout, and the reserve constraint is what "
                    "stops the robot rather than the end of the map. The defect that "
                    "killed the earlier formulation -- a constraint that selects "
                    "nothing -- is absent. Proceed with REACH.",
        }
    return {
        'outcome': 'FAIL_SWITCH_TO_REALTIME',
        'text': f"Failed on {failing}. The reserve constraint does not select the "
                f"turn-back point on those maps, which is the dead formulation's "
                f"defect in a new costume. Switch to REALTIME immediately rather "
                f"than engineering around it.",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenarios', nargs='*',
                    default=['corridor', 's_curve', 'chicane'])
    ap.add_argument('--r-grid', type=float, nargs='*', default=list(DEFAULT_R_GRID))
    ap.add_argument('--n-points', type=int, default=400,
                    help='turn-back candidates sampled along the route')
    ap.add_argument('--pack-J', type=float, default=None,
                    help='absolute pack energy. Default: the full round-trip cost '
                         'of the route, so r is "fraction of a round trip".')
    ap.add_argument('--return-rate', default='both',
                    choices=['lambda_min', 'certificate', 'both'],
                    help="How to price the RETURN leg. 'lambda_min' is the "
                         "specified baseline and is trivially linear; "
                         "'certificate' uses the affine Ebar = a + b*D* measured "
                         "in audit_vfb_decomposition and is the informative one. "
                         "Default runs both so the difference is visible.")
    ap.add_argument('--cert-a', type=float, default=24.0,
                    help='fixed certificate cost in J (measured 20.96/24.09/26.20 '
                         'on corridor/s_curve/chicane)')
    ap.add_argument('--cert-b', type=float, default=None,
                    help='per-metre certificate rate; defaults to lambda_min, '
                         'which is what the affine fit actually returned')
    ap.add_argument('--run-id', dest='run_id', default=None,
                    help='when given (and --out is not), output goes to '
                         '<runs>/<run-id>/gate_r1/gate_r1.json on /data')
    ap.add_argument('--out', default=None, help='explicit JSON output path')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    from env.scenarios import get_scenario
    from planning.energy_dijkstra import build_field

    per_map: Dict[str, Dict[str, Any]] = {}
    for name in args.scenarios:
        sc = get_scenario(name)
        field = build_field(sc)
        modes = (['lambda_min', 'certificate'] if args.return_rate == 'both'
                 else [args.return_rate])
        variants = {}
        for mode in modes:
            variants[mode] = assess(name, field, sc.start[:2], args.r_grid,
                                    pack_J=args.pack_J, n=args.n_points,
                                    return_rate=mode, cert_a=args.cert_a,
                                    cert_b=args.cert_b)
        res = variants[modes[0]]
        # NOT res['variants'] = variants -- variants[modes[0]] IS res, so that
        # aliases the dict into itself and json.dumps raises "Circular reference
        # detected" only at write time, i.e. after the whole gate has run.
        per_map[name] = {**res, 'variants': {k: dict(v) for k, v in variants.items()}}
        print(f"=== {name}   route {res['route_length_m']:.3f} m   "
              f"round trip {res['round_trip_J']:.2f} J   "
              f"lambda_min {res['lambda_min']:.4f} J/m ===")
        print(f"  {'r':>5} {'budget J':>10} {'reach m':>9} {'% route':>8} "
              f"{'used J':>9} {'slack J':>8}  binding interior")
        for row in res['rows']:
            print(f"  {row['r']:5.2f} {row['budget_J']:10.2f} {row['reach_m']:9.3f} "
                  f"{100*row.get('reach_fraction_of_route', 0):7.1f}% "
                  f"{row.get('energy_used_J', 0):9.2f} {row['slack_J']:8.3f}  "
                  f"{str(row['binding']):>7} {str(row['interior']):>8}")
        if len(variants) > 1:
            cv = variants['certificate']
            print(f"  {'--- with the certificate-priced return leg ---':<60}")
            for row in cv['rows']:
                print(f"  {row['r']:5.2f} {row['budget_J']:10.2f} {row['reach_m']:9.3f} "
                      f"{100*row.get('reach_fraction_of_route', 0):7.1f}% "
                      f"{row.get('energy_used_J', 0):9.2f} {row['slack_J']:8.3f}  "
                      f"{str(row['binding']):>7} {str(row['interior']):>8}")
            lo = res['rows'][0]['reach_m']; lo_c = cv['rows'][0]['reach_m']
            hi = res['rows'][-1]['reach_m']; hi_c = cv['rows'][-1]['reach_m']
            if lo > 0 and hi > 0:
                print(f"  certificate costs {100*(1-lo_c/lo):.1f}% of reach at r=0.2 "
                      f"and {100*(1-hi_c/hi):.1f}% at r=0.8")
        print(f"  strictly increasing in r : {res['reach_strictly_increasing_in_r']}")
        print(f"  turn-back interior       : {res['turn_back_interior_for_all_r']}")
        print(f"  reserve binding          : {res['reserve_binding_for_all_r']}")
        print(f"  -> {'PASS' if res['pass'] else 'FAIL'}")
        print()

    v = verdict(per_map)
    print("=" * 78)
    print(f"GATE-R1 VERDICT [{v['outcome']}]")
    print(f"  {v['text']}")
    print("=" * 78)
    print("NOTE: this is the arithmetic half of GATE-R1. It shows the reserve "
          "constraint SELECTS a turn-back point.\n"
          "It does NOT show that a receding-horizon controller can exploit that, "
          "which is what R1/R5 are for.")

    out_path = args.out
    if out_path is None and args.run_id:
        from utils.paths import run_dir as _run_dir
        out_path = os.path.join(_run_dir(args.run_id, 'gate_r1'), 'gate_r1.json')
    if out_path:
        from utils.safe_write import safe_write_json
        payload = {'per_map': per_map, 'verdict': v,
                   'r_grid': args.r_grid, 'n_points': args.n_points}
        safe_write_json(out_path, payload, force=args.force)
        print(f"\nwritten: {out_path}")
    return 0 if v['outcome'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
