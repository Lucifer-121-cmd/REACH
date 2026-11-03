"""Per-map reserve certificate refit on the terrain J-fields.

Script 19 fit the certificate on the LEGACY metric fields: affine
``Ebar = a_q + λ_min·D*`` with a_q ≈ 23.7–28.0 J over the three navigation
maps. The generated REACH maps are different in kind — they carry terrain,
so their geodesic fields are J-denominated (D_J already integrates the
local λ = 2√(m1·c1·c5) + m4·c4), and the affine slope pins at 1.0: the
floor along the route IS the field value. What remains to fit per map is
the INTERCEPT — the worst-case excess of the fallback policy over that
floor:

    a_q := quantile_q( V_fb(z) − D_J(z) )     over the sampled state set

This is the same coverage construction as script 19 (a certificate has to
BOUND, so least squares bounds nothing), applied to the terrain field. V_fb
is measured by rolling out π_fb — pure pursuit on the geodesic at v*, the
same policy the legacy fit used — with TERRAIN-AWARE energy accounting:
the rollout pays m1·c1·v² + m4·c4·|v| at its current position, exactly as
the plant does. A terrain-blind rollout would understate V_fb inside the
expensive regions and the certificate would under-cover exactly where the
mission spends its reserve.

The fitted (a_q, coverage) per map is what the REACH campaign's supervisor
uses (evaluation/reach_evaluator.py currently carries a_q = 24 J, the
corridor legacy placeholder — recorded in every trial result as such).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Same declared coverage as the legacy fit. Fixed before the fit.
COVERAGE = 0.95
N_HEADINGS = 8

# π_fb and plant constants (prototypes/audit_lambda_fb_descent.py, kept
# identical so the legacy and terrain fits stay reconcilable).
C = (2.5, 0.8, 0.3, 1.2, 3.0)
VSTAR = math.sqrt(C[4] / C[0])
LAM_MIN = 2 * math.sqrt(C[0] * C[4]) + C[3]
VMAX, WMAX = 1.2, 3.14159
DT = 0.1
AV = 1.0 - math.exp(-DT / 0.3)
AW = 1.0 - math.exp(-DT / 0.2)
GTOL = 0.2
LA = 0.5          # pure-pursuit lookahead
KP = 2.0
MAX_STEPS = 3000

SINGLE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
}


def _rollout_terrain(path: np.ndarray, z0: Tuple[float, float, float],
                     goal: Tuple[float, float], terrain,
                     terrain_adaptive: bool = False) -> Dict[str, float]:
    """π_fb rollout with terrain-aware per-term energy accounting.

    Same policy as prototypes/audit_vfb_decomposition.rollout_decomposed
    (pure pursuit at v* on the geodesic path, motor-lag model, same gains
    and tolerations) — the ONLY change is that the c1/c4 terms carry the
    terrain multipliers at the robot's CURRENT position, matching
    PioneerEnv._plant_coeffs. Anything else would desynchronise this fit
    from both the plant and the legacy numbers.

    ``terrain_adaptive`` (P3, 2026-09-15): command the TERRAIN-OPTIMAL
    speed v*(m) = sqrt(c5/(m1·c1)) inside terrain regions instead of the
    nominal v*. The P4 descent check measured why: cruising at the
    nominal v* inside m=3 terrain costs 14.55 J/m while the field credits
    λ(3) = 13.09 J/m — a +1.47 J/m structural deficit that is the
    dominant leak in the certificate. The adaptive speed makes the
    steady-state descent inequality tight everywhere and shrinks V_fb
    inside terrain, hence a_q. The rollout still pays the plant's true
    power at whatever speed it actually reaches (the lag model is
    unchanged) — only the COMMAND adapts.
    """
    x, y, th = z0
    v = w = 0.0
    c1, c2, c3, c4, c5 = C
    terms = dict(c1=0.0, c2=0.0, c3=0.0, c4=0.0, c5=0.0)
    s = 0.0
    idx = 0
    for steps in range(1, MAX_STEPS + 1):
        if math.hypot(x - goal[0], y - goal[1]) < GTOL:
            break
        while idx < len(path) - 1 and math.hypot(path[idx][0] - x,
                                                 path[idx][1] - y) < LA:
            idx += 1
        tx, ty = path[idx]
        bearing = math.atan2(ty - y, tx - x)
        err = math.atan2(math.sin(bearing - th), math.cos(bearing - th))
        wc = max(-WMAX, min(WMAX, KP * err))
        vc = VSTAR
        m1 = m4 = 1.0
        if terrain is not None:
            m1, m4 = terrain.multipliers(x, y)
            if terrain_adaptive and m1 > 1.0:
                vc = math.sqrt(c5 / (m1 * c1))
        v = v + AV * (vc - v)
        w = w + AW * (wc - w)
        v = max(-VMAX, min(VMAX, v))
        w = max(-WMAX, min(WMAX, w))
        terms['c1'] += m1 * c1 * v * v * DT
        terms['c2'] += c2 * w * w * DT
        terms['c3'] += c3 * abs(v) * abs(w) * DT
        terms['c4'] += m4 * c4 * abs(v) * DT
        terms['c5'] += c5 * DT
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        th += w * DT
        s += abs(v) * DT
    else:
        return {'E': float('inf'), 'arclen': s, 'steps': MAX_STEPS,
                'reached': False, 'terms': terms}
    E = sum(terms.values())
    return {'E': E, 'arclen': s, 'steps': steps, 'reached': True,
            'terms': terms}


def sample_and_fit(scenario, *, stride: int = 6,
                   coverage: float = COVERAGE,
                   terrain_adaptive: bool = False) -> Dict[str, Any]:
    """Sample states on one map and fit a_q at the declared coverage.

    Returns the certificate dict the supervisor consumes:
    ``{'a_q': ..., 'coverage': ..., 'n_states': ..., 'denomination': 'J'}``.
    """
    from planning.energy_dijkstra import build_field, geodesic_path
    terrain = getattr(scenario, 'terrain', None)
    field = build_field(scenario, terrain=terrain)
    if field.denomination != 'J':
        raise ValueError(
            f"{scenario.name}: field is '{field.denomination}', not 'J' -- "
            "the terrain refit only applies to terrain maps. Legacy metric "
            "maps use script 19's λ_min-slope affine form.")
    # metric field for the DETOUR stat only: arclen (m) over D (m) is a
    # ratio; arclen over D_J would divide metres by joules -- a units error
    # the legacy script could not make because its field was already metric.
    metric_field = build_field(scenario)
    goal = np.asarray(scenario.dock, dtype=float)

    X, Y, free, D = field.X, field.Y, field.free, field.D
    q = 100.0 * coverage
    rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    n_states = 0
    for i in range(0, len(X), stride):
        for j in range(0, len(Y), stride):
            if not free[i, j] or not np.isfinite(D[i, j]) or D[i, j] < 1e-9:
                continue
            p = (float(X[i]), float(Y[j]))
            path = geodesic_path(field, p)
            if path is None or len(path) < 2:
                continue
            d_J = float(D[i, j])
            d_m = metric_field.dstar(p[0], p[1])
            for k in range(N_HEADINGS):
                th = -math.pi + 2.0 * math.pi * k / N_HEADINGS
                r = _rollout_terrain(np.asarray(path),
                                     (p[0], p[1], th),
                                     (float(goal[0]), float(goal[1])), terrain,
                                     terrain_adaptive=terrain_adaptive)
                n_states += 1
                if r['reached']:
                    rows.append({'x': p[0], 'y': p[1], 'theta': th,
                                 'd_J': d_J, 'V_fb': r['E'],
                                 'excess': r['E'] - d_J,
                                 'detour': (r['arclen'] / d_m
                                            if math.isfinite(d_m) and d_m > 1e-9
                                            else float('nan'))})
    if len(rows) < 50:
        raise RuntimeError(
            f"{scenario.name}: only {len(rows)} reachable rollouts from "
            f"{n_states} states -- the map is too hostile for a fit; widen "
            "the sampler (smaller stride) or regenerate the map")
    V = np.array([r['V_fb'] for r in rows])
    Dv = np.array([r['d_J'] for r in rows])
    excess = np.array([r['excess'] for r in rows])
    a_q = float(np.percentile(excess, q))
    covered = float(np.mean(V <= a_q + Dv))
    detour_p95 = float(np.percentile([r['detour'] for r in rows], 95))
    return {
        'map': scenario.name,
        'a_q': a_q,
        'slope': 1.0,
        'denomination': 'J',
        'coverage_target': coverage,
        'coverage_achieved': covered,
        'n_states': n_states,
        'n_reached': len(rows),
        'excess_mean': float(excess.mean()),
        'excess_p50': float(np.percentile(excess, 50)),
        'detour_p95': detour_p95,
        'rollout_s': round(time.perf_counter() - t0, 1),
        'terrain_adaptive': terrain_adaptive,
        'note': ('a_q = quantile_q(V_fb - D_J) over sampled states; V_fb '
                 'from a terrain-aware pi_fb rollout (pure pursuit at v* on '
                 'the J-geodesic, same gains as the legacy fit); slope 1.0 '
                 'because D_J already integrates the local floor'),
    }


def fit_maps_from_dir(maps_dir: str, stride: int = 6,
                     terrain_adaptive: bool = False) -> List[Dict[str, Any]]:
    """Fit every reach_*.yaml under maps_dir (non-recursive)."""
    import glob
    from env.scenarios import load_scenario_from_yaml
    out = []
    for path in sorted(glob.glob(os.path.join(maps_dir, '*.yaml'))):
        sc = load_scenario_from_yaml(path)
        logger.info("fitting %s ...", sc.name)
        cert = sample_and_fit(sc, stride=stride,
                               terrain_adaptive=terrain_adaptive)
        logger.info("  a_q=%.2f J (coverage %.3f, %d states, %.1fs)",
                    cert['a_q'], cert['coverage_achieved'],
                    cert['n_states'], cert['rollout_s'])
        out.append(cert)
    return out


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', default=None,
                    help='read maps from <runs>/<run-id>/reach_maps/{train,heldout}/'
                         ' and write the certificates beside them')
    ap.add_argument('--maps-dir', default=None,
                    help='explicit maps directory (alternative to --run-id)')
    ap.add_argument('--stride', type=int, default=6)
    ap.add_argument('--terrain-adaptive', action='store_true',
                    help='P3: command the terrain-optimal v*(m) in the pi_fb '
                         'rollout inside terrain regions (fixes the +1.47 J/m '
                         'descent-inequality leak found by P4)')
    ap.add_argument('--coverage', type=float, default=COVERAGE)
    args = ap.parse_args(argv)

    from utils.safe_write import safe_write_json
    from utils.paths import run_dir as _run_dir

    if args.maps_dir:
        dirs = [args.maps_dir]
    elif args.run_id:
        base = _run_dir(args.run_id, 'reach_maps', create=False)
        dirs = [os.path.join(base, s) for s in ('train', 'heldout')]
    else:
        raise SystemExit("give --run-id or --maps-dir (there is no default)")

    all_certs = {}
    for d in dirs:
        if not os.path.isdir(d):
            logger.warning("no such directory: %s", d)
            continue
        certs = fit_maps_from_dir(d, stride=args.stride,
                                 terrain_adaptive=args.terrain_adaptive)
        for c in certs:
            c['split'] = os.path.basename(os.path.normpath(d))
            all_certs[c['map']] = c
        out_path = os.path.join(d, 'certificates.json')
        safe_write_json(out_path, certs)
        logger.info("wrote %s", out_path)
    print(json.dumps({k: round(v['a_q'], 2) for k, v in all_certs.items()},
                     indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
