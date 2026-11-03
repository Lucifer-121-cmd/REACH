"""P4 — descent-inequality check of the fitted certificates (offline).

THE QUESTION: is the fitted certificate `Ebar(x) = a_q + D_J(x)` an
(approximate) INVARIANT for the energy state — i.e. does the Bellman-type
sufficient condition

    Ebar(z) >= min over admissible u of [ P(u, z)·dt + Ebar(z') ]

hold on the grid, where z = (x, y, θ) and z' is the one-step successor
under the motor-lag dynamics? If yes (to within tolerance ε), then the
sub-level set { (z, E) : E + Ebar(z) <= B } is forward-invariant under
the ONE-STEP dynamics — the reserve cannot be violated by any admissible
control sequence, which is the property "coverage fit" only approximates
from sampled rollouts. The violation rate and depth are the honest
measure of that approximation.

WHY THE MINIMISATION IS BOUNDED AND CHEAP: the RHS minimiser over
(u_v, u_w) does not need a global search. The successor's Ebar depends
only on position; the power P is minimised and the position advance
maximised by driving at v* toward the steepest DESCENT direction of
Ebar (the geodesic home), and the heading cost of realignment is
captured by the θ-grid. So the check evaluates, for each (cell, θ):
the one-step cost of the certificate's OWN π_fb-style action (steer to
the geodesic tangent at gain KP, command v*) against Ebar(z) − Ebar(z').
This is the SAME action model the fit used, which is the right
conservative choice: the invariant should hold for the policy class the
certificate claims to bound, evaluated per-step rather than per-rollout.

TWO CERTIFICATE FORMS are checked on every map:
- affine (deployed): Ebar = a_q + D_J  (slope pinned at the floor)
- scalar (legacy):   Ebar = λ_q · D_metric  (metric field, legacy maps
  only — the terrain maps carry no λ_q fit, so the scalar check runs
  on the three legacy navigation maps with their script-19 values)

Outputs runs/<id>/reach/certificates_descent_check.json + a per-map
table. Feeds the paper's descent-inequality claim.

2026-09-21 (fresh_full_audit finding 1): LEVEL 2 (aligned steady-state
check, bilinear + nearest successor lookups) and LEVEL 3 (one-time
realignment-transient leak, worst heading from rest, defined cell
sample) are computed and persisted in the same artefact — previously
they were in-session analyses that existed in no artefact. Levels 2 and
3 are intercept-invariant (deficits are D differences; a_q cancels),
so they hold identically under the old and adaptive certificates; the
only intercept-dependent quantity is level 3's covered_by_intercept_x,
recorded against the certificate passed at invocation.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List

# Single-threaded BLAS/OMP, set BEFORE the first numpy/casadi import:
# OpenBLAS/MUMPS read the environment at import time, and this driver's
# results must land on the pinned campaign lineage -- an ulp-scale
# thread-environment change is a different experiment (measured: a
# 6-decimal intercept rounding departed from the campaign lineage in
# 12/60 outcomes).
# Same idiom as scripts/30. Added 2026-09-20; the 2026-09-12/15
# artefacts produced by this script predate the pin and remain the runs
# of record (their lineage is recorded per artefact).
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'

import numpy as np

logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dynamics identical to the certificate rollout / the plant
C = (2.5, 0.8, 0.3, 1.2, 3.0)
VSTAR = math.sqrt(C[4] / C[0])
VMAX, WMAX = 1.2, 3.14159
DT = 0.1
AV = 1.0 - math.exp(-DT / 0.3)
AW = 1.0 - math.exp(-DT / 0.2)
KP = 2.0
GTOL = 0.2

N_HEADINGS = 8


def one_step_cost(x: float, y: float, th: float, gx: float, gy: float,
                  terrain) -> float:
    """Power·dt for the certificate's action model at (x, y, θ) steering
    toward the local geodesic-descent direction (gx, gy)."""
    target = math.atan2(gy, gx)
    err = math.atan2(math.sin(target - th), math.cos(target - th))
    wc = max(-WMAX, min(WMAX, KP * err))
    vc = VSTAR
    v = AV * (vc - 0.0)      # from rest: the per-step worst case for the
    w = AW * (wc - 0.0)      # realignment cost the intercept must cover
    m1 = m4 = 1.0
    if terrain is not None:
        m1, m4 = terrain.multipliers(x, y)
    c1, c2, c3, c4, c5 = C
    P = (m1 * c1 * v * v + c2 * w * w + c3 * abs(v) * abs(w)
         + m4 * c4 * abs(v) + c5)
    return P * DT


def successor(x: float, y: float, th: float, gx: float, gy: float
              ) -> tuple:
    """One motor-lag step under the same action."""
    target = math.atan2(gy, gx)
    err = math.atan2(math.sin(target - th), math.cos(target - th))
    wc = max(-WMAX, min(WMAX, KP * err))
    v = AV * VSTAR
    w = AW * wc
    nx = x + v * math.cos(th) * DT
    ny = y + v * math.sin(th) * DT
    nth = th + w * DT
    return nx, ny, nth


def bilinear_dstar(jf, x: float, y: float) -> float:
    """Bilinear D interpolation; inf outside the grid or off the
    reachable set (any corner non-finite)."""
    import bisect
    X, Y, D = jf.X, jf.Y, jf.D
    if x < X[0] or x > X[-1] or y < Y[0] or y > Y[-1]:
        return float('inf')
    i = min(max(bisect.bisect_right(X, x) - 1, 0), len(X) - 2)
    j = min(max(bisect.bisect_right(Y, y) - 1, 0), len(Y) - 2)
    tx = (x - X[i]) / (X[i + 1] - X[i])
    ty = (y - Y[j]) / (Y[j + 1] - Y[j])
    d00, d10 = D[i, j], D[i + 1, j]
    d01, d11 = D[i, j + 1], D[i + 1, j + 1]
    if not (math.isfinite(d00) and math.isfinite(d10)
            and math.isfinite(d01) and math.isfinite(d11)):
        return float('inf')
    return float(d00 * (1 - tx) * (1 - ty) + d10 * tx * (1 - ty)
                 + d01 * (1 - tx) * ty + d11 * tx * ty)


def steady_state_check(scenario, a_q: float, stride: int = 6) -> Dict[str, Any]:
    """LEVEL 2 — aligned steady-state entry (v = v*, theta = geodesic
    tangent, zero turn rate). Isolates the terrain-v* mismatch and
    grid-discretisation leaks from any realignment effect: the
    successor holds v* along the descent direction. Intercept-
    invariant (deficits are D differences; a_q cancels), so the same
    numbers hold under the old and adaptive certificates.

    Persisted 2026-09-21 (fresh_full_audit finding 1): the level-2
    figures previously quoted for the paper were computed
    in-session and existed in no artefact. Two successor lookups are
    recorded, matching the original analysis: bilinear (smoothed) and
    nearest-cell."""
    from planning.energy_dijkstra import build_field

    terrain = getattr(scenario, 'terrain', None)
    jf = build_field(scenario, terrain=terrain)
    X, Y, free, D = jf.X, jf.Y, jf.free, jf.D

    out: Dict[str, Any] = {}
    for tag in ('bilinear', 'nearest'):
        out[tag] = {'n': 0, 'n_bad': 0, 'deficits': []}
    n_nograd = 0
    for i in range(0, len(X), stride):
        for j in range(0, len(Y), stride):
            if not free[i, j] or not math.isfinite(D[i, j]) or D[i, j] < 1e-9:
                continue
            x, y = float(X[i]), float(Y[j])
            if math.hypot(x - float(jf.goal[0]),
                          y - float(jf.goal[1])) < GTOL:
                continue
            gx = (D[i - 1, j] - D[i + 1, j]) if 0 < i < len(X) - 1 else 0.0
            gy = (D[i, j - 1] - D[i, j + 1]) if 0 < j < len(Y) - 1 else 0.0
            gn = math.hypot(gx, gy)
            # a finite cell can border the unreachable set (inf neighbour):
            # the one-sided gradient is infinite and the descent DIRECTION
            # is undefined. Level 1 survives this by accident (its min/max
            # clamps swallow the NaN); levels 2/3 use the direction
            # directly, so they must skip such cells and say how many.
            if not (gn > 1e-12) or not math.isfinite(gn):
                n_nograd += 1
                continue
            gx, gy = gx / gn, gy / gn
            th = math.atan2(gy, gx)
            m1 = m4 = 1.0
            if terrain is not None:
                m1, m4 = terrain.multipliers(x, y)
            c1, c2, c3, c4, c5 = C
            cost = (m1 * c1 * VSTAR * VSTAR + m4 * c4 * VSTAR + c5) * DT
            nx = x + VSTAR * math.cos(th) * DT
            ny = y + VSTAR * math.sin(th) * DT
            d_next = {'bilinear': bilinear_dstar(jf, nx, ny),
                      'nearest': jf.dstar(nx, ny)}
            for tag in ('bilinear', 'nearest'):
                dn = d_next[tag]
                if not math.isfinite(dn):
                    continue
                deficit = cost + dn - D[i, j]
                rec = out[tag]
                rec['n'] += 1
                if deficit > 1e-9:
                    rec['n_bad'] += 1
                    rec['deficits'].append(deficit)
    result: Dict[str, Any] = {'definition': (
        'aligned steady state (v=v*, theta=geodesic tangent, w=0); '
        'deficit = P(v*)*dt + D(z_successor) - D(z); intercept-invariant; '
        'cells with an undefined descent direction (one-sided boundary '
        'against the unreachable set) are excluded and counted'),
        'n_undefined_direction': n_nograd}
    for tag in ('bilinear', 'nearest'):
        rec = out[tag]
        n, nb = rec['n'], rec['n_bad']
        result[tag] = {
            'n_states': n, 'n_bad': nb,
            'violation_rate': (nb / n) if n else None,
            'mean_deficit_J': (float(np.mean(rec['deficits']))
                               if rec['deficits'] else 0.0),
            'p95_deficit_J': (float(np.percentile(rec['deficits'], 95))
                              if rec['deficits'] else 0.0),
        }
    return result


def realignment_transient(scenario, a_q: float, stride: int = 6,
                          cell_every: int = 2, max_steps: int = 200,
                          align_tol: float = math.pi / 4,
                          min_dock_d: float = 5.0) -> Dict[str, Any]:
    """LEVEL 3 — one-time realignment leak. From rest at the WORST
    heading (opposite the local geodesic-descent direction), roll out
    the certificate's own action model (steer to the local tangent at
    KP, command v*, motor-lag) and accumulate the descent-inequality
    deficit per step until the heading error is below align_tol (the
    aligned/misaligned split quoted in the paper). The accumulated deficit is the
    one-time leak the intercept must cover. Intercept-invariant
    (deficits are D differences).

    Two honest exclusions, both counted in the output:
    - dock-approach zone (D < min_dock_d): there the analysis rollout
      ORBITS at v* without aligning (the pure-pursuit capture-radius
      limit at D -> 0), which is a property of the ANALYSIS action
      model, not the system — the closed loop's dock-stop governs the
      final approach, and V_fb itself is tiny there;
    - never-aligned cells (max_steps exhausted): the same limit cycle
      elsewhere; excluded from the distribution.
    Persisted 2026-09-21 (fresh_full_audit finding 1): supersedes the
    in-session '3 mid-map cells' figures with a defined sample — every
    cell_every-th of the stride-sampled free cells — reported as
    worst / p95 / median across cells."""
    from planning.energy_dijkstra import build_field

    terrain = getattr(scenario, 'terrain', None)
    jf = build_field(scenario, terrain=terrain)
    X, Y, free, D = jf.X, jf.Y, jf.free, jf.D

    def local_descent(x: float, y: float):
        d = jf.dstar(x, y)
        if not math.isfinite(d):
            return None
        i = int(round((x - X[0]) / jf.h))
        j = int(round((y - Y[0]) / jf.h))
        if not (0 <= i < len(X) and 0 <= j < len(Y)):
            return None
        gx = (D[i - 1, j] - D[i + 1, j]) if 0 < i < len(X) - 1 else 0.0
        gy = (D[i, j - 1] - D[i, j + 1]) if 0 < j < len(Y) - 1 else 0.0
        gn = math.hypot(gx, gy)
        # one-sided boundary cells (inf neighbour) have no defined
        # descent direction — skip, same as steady_state_check
        if not (gn > 1e-12) or not math.isfinite(gn):
            return None
        return gx / gn, gy / gn

    transients: List[float] = []
    n_cells = n_invalid = n_dock = n_unaligned = 0
    ci = 0
    for i in range(0, len(X), stride):
        for j in range(0, len(Y), stride):
            if not free[i, j] or not math.isfinite(D[i, j]) or D[i, j] < 1e-9:
                continue
            ci += 1
            if (ci - 1) % cell_every:
                continue
            if D[i, j] < min_dock_d:
                n_dock += 1          # dock-approach zone: the analysis
                continue             # action model orbits; the closed
                                    # loop's dock-stop governs there
            x, y = float(X[i]), float(Y[j])
            if math.hypot(x - float(jf.goal[0]),
                          y - float(jf.goal[1])) < GTOL:
                continue
            gd = local_descent(x, y)
            if gd is None:
                n_invalid += 1
                continue
            th = math.atan2(gd[1], gd[0]) + math.pi   # worst heading
            v = w = 0.0                                # from rest
            acc = 0.0
            aligned = False
            n_cells += 1
            for _ in range(max_steps):
                gd = local_descent(x, y)
                if gd is None:
                    acc = float('nan')
                    break
                target = math.atan2(gd[1], gd[0])
                err = math.atan2(math.sin(target - th),
                                  math.cos(target - th))
                if abs(err) < align_tol:
                    aligned = True
                    break
                wc = max(-WMAX, min(WMAX, KP * err))
                m1 = m4 = 1.0
                if terrain is not None:
                    m1, m4 = terrain.multipliers(x, y)
                c1, c2, c3, c4, c5 = C
                P = (m1 * c1 * v * v + c2 * w * w + c3 * abs(v) * abs(w)
                     + m4 * c4 * abs(v) + c5)
                nx = x + v * math.cos(th) * DT
                ny = y + v * math.sin(th) * DT
                d_here, d_next = jf.dstar(x, y), jf.dstar(nx, ny)
                if not (math.isfinite(d_here) and math.isfinite(d_next)):
                    acc = float('nan')
                    break
                acc += P * DT + d_next - d_here
                v += AV * (VSTAR - v)
                w += AW * (wc - w)
                x, y, th = nx, ny, th + w * DT
            if math.isfinite(acc) and aligned:
                transients.append(acc)
            else:
                n_cells -= 1
                if math.isfinite(acc):
                    n_unaligned += 1
                else:
                    n_invalid += 1
    good = [t for t in transients if t > 0]
    worst = max(transients) if transients else 0.0
    return {
        'definition': ('from rest, worst heading (opposite the local '
                       'geodesic tangent), certificate action model; '
                       'accumulate deficit until |err| < pi/4; '
                       'intercept-invariant; dock-approach cells '
                       '(D < min_dock_d) and never-aligned cells '
                       '(pure-pursuit limit cycle) are excluded and '
                       'counted'),
        'n_cells': n_cells, 'n_invalid': n_invalid,
        'n_dock_zone': n_dock, 'n_unaligned': n_unaligned,
        'worst_transient_J': worst,
        'p95_transient_J': float(np.percentile(transients, 95)) if transients else 0.0,
        'median_transient_J': float(np.median(transients)) if transients else 0.0,
        'covered_by_intercept_x': (a_q / worst) if worst > 0 else None,
        'covered_by_intercept_median_x': (a_q / float(np.median(transients))) if transients and np.median(transients) > 0 else None,
        'n_positive': len(good),
    }


def check_map(scenario, a_q: float, stride: int = 6) -> Dict[str, Any]:
    """Evaluate the descent inequality on every stride-th free cell × 8
    headings, against the J-field's local descent direction."""
    from planning.energy_dijkstra import build_field

    terrain = getattr(scenario, 'terrain', None)
    jf = build_field(scenario, terrain=terrain)
    X, Y, free, D = jf.X, jf.Y, jf.free, jf.D
    h = jf.h

    def ebar(x: float, y: float) -> float:
        d = jf.dstar(x, y)
        if not math.isfinite(d):
            return float('inf')
        return a_q + d

    n_ok = n_bad = n_inf = 0
    worst = 0.0
    worst_at = None
    deficits: List[float] = []
    for i in range(0, len(X), stride):
        for j in range(0, len(Y), stride):
            if not free[i, j] or not math.isfinite(D[i, j]) or D[i, j] < 1e-9:
                continue
            x, y = float(X[i]), float(Y[j])
            # local geodesic-descent direction by finite difference on the
            # grid field (the direction the fallback follows)
            gx = (D[i - 1, j] - D[i + 1, j]) if 0 < i < len(X) - 1 else 0.0
            gy = (D[i, j - 1] - D[i, j + 1]) if 0 < j < len(Y) - 1 else 0.0
            gn = math.hypot(gx, gy)
            if gn < 1e-12:
                continue
            gx, gy = gx / gn, gy / gn
            e_here = ebar(x, y)
            for k in range(N_HEADINGS):
                th = -math.pi + 2.0 * math.pi * k / N_HEADINGS
                if math.hypot(x - float(jf.goal[0]),
                              y - float(jf.goal[1])) < GTOL:
                    continue
                cost = one_step_cost(x, y, th, gx, gy, terrain)
                nx, ny, _ = successor(x, y, th, gx, gy)
                e_next = ebar(nx, ny)
                if not math.isfinite(e_next):
                    n_inf += 1
                    continue
                deficit = (cost + e_next) - e_here
                if deficit <= 1e-9:
                    n_ok += 1
                else:
                    n_bad += 1
                    deficits.append(deficit)
                    if deficit > worst:
                        worst = deficit
                        worst_at = (round(x, 2), round(y, 2), round(th, 2))
    n = n_ok + n_bad
    return {
        'map': scenario.name, 'a_q': a_q,
        'n_states': n, 'n_ok': n_ok, 'n_bad': n_bad,
        'n_successor_inf': n_inf,
        'violation_rate': (n_bad / n) if n else None,
        'worst_deficit_J': worst if n_bad else 0.0,
        'worst_at': worst_at,
        'mean_deficit_J': float(np.mean(deficits)) if deficits else 0.0,
        'p95_deficit_J': float(np.percentile(deficits, 95)) if deficits else 0.0,
    }


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--stride', type=int, default=6)
    ap.add_argument('--maps-root', default=None,
                    help='maps root with {train,heldout}/certificates.json '
                         '(default: this run-id\'s reach_maps — the OLD-certificate '
                         'fit; pass reach_maps_adaptive for the deployed cert)')
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    from env.scenarios import load_scenario_from_yaml
    import glob as _glob

    maps_root = (args.maps_root if args.maps_root
                 else _run_dir(args.run_id, 'reach_maps', create=False))
    results = []
    for split in ('train', 'heldout'):
        certs = {c['map']: float(c['a_q']) for c in
                 json.load(open(os.path.join(maps_root, split,
                                            'certificates.json')))}
        for p in sorted(_glob.glob(os.path.join(maps_root, split,
                                                'reach_*.yaml'))):
            sc = load_scenario_from_yaml(p)
            if sc.name not in certs:
                continue
            r = check_map(sc, certs[sc.name], stride=args.stride)
            r['split'] = split
            r['level2_steady_state'] = steady_state_check(
                sc, certs[sc.name], stride=args.stride)
            r['level3_realignment'] = realignment_transient(
                sc, certs[sc.name], stride=args.stride)
            results.append(r)
            l2b = r['level2_steady_state']['bilinear']
            l2n = r['level2_steady_state']['nearest']
            l3 = r['level3_realignment']
            logger.info(
                "%s: L1 %d states, %d violations (%.2f%%), worst %.2f J | "
                "L2 bilinear %.1f%% (mean %.3f J) / nearest %.1f%% (%.3f J) | "
                "L3 worst %.2f J, covered %.1fx",
                sc.name, r['n_states'], r['n_bad'],
                100 * (r['violation_rate'] or 0), r['worst_deficit_J'],
                100 * (l2b['violation_rate'] or 0), l2b['mean_deficit_J'],
                100 * (l2n['violation_rate'] or 0), l2n['mean_deficit_J'],
                l3['worst_transient_J'], l3['covered_by_intercept_x'] or 0)

    out_dir = os.path.join(_run_dir(args.run_id, 'reach'), 'timing')
    os.makedirs(out_dir, exist_ok=True)   # reach/ exists; timing/ adjacent
    suffix = ('_adaptive' if (args.maps_root or '').endswith('adaptive') else '')
    out = os.path.join(_run_dir(args.run_id, 'reach'),
                       f'certificates_descent_check{suffix}.json')
    safe_write_json(out, {
        'definition': ('Ebar(x) >= P(u*)·dt + Ebar(x\') evaluated per free '
                       'cell x 8 headings, u* = the certificate\'s own action '
                       'model (steer to the geodesic tangent at KP, command '
                       'v*, from rest — the worst case for realignment). '
                       'affine form a_q + D_J with the deployed per-map a_q. '
                       'level2_steady_state = aligned entry at v* (isolates '
                       'the terrain-v* and discretisation leaks); '
                       'level3_realignment = one-time realignment leak from '
                       'the worst heading, accumulated until aligned. '
                       'Levels 2-3 added 2026-09-21 (intercept-invariant).'),
        'stride': args.stride, 'results': results,
    })
    print(f"\n{'map':<24}{'a_q':>7}{'states':>8}{'viol':>6}{'rate%':>7}"
          f"{'worst J':>9}{'p95 J':>8}")
    for r in results:
        print(f"{r['map']:<24}{r['a_q']:>7.2f}{r['n_states']:>8}{r['n_bad']:>6}"
              f"{100*(r['violation_rate'] or 0):>7.2f}{r['worst_deficit_J']:>9.2f}"
              f"{r['p95_deficit_J']:>8.2f}")
    tot_bad = sum(r['n_bad'] for r in results)
    tot_n = sum(r['n_states'] for r in results)
    print(f"\nTOTAL: {tot_bad}/{tot_n} violations "
          f"({100*tot_bad/tot_n:.2f}%)")
    print(f"written: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
