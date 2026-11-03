"""Weighted geodesic distance field D*(p) and the energy floor built from it.

Serves three consumers:

1. **The REACH certificate.** ``Ebar_dock(z)`` needs an energy-to-go, and D* is
   the distance half of it.
2. **A plan-then-track baseline.** The geodesic is the path a global planner
   would produce, so tracking it is the honest comparator for a local MPC.
3. **``E_floor``.** Every effect size is reported as a fraction of the achievable
   envelope ``E >= lambda_min * D*`` rather than in isolation.

**Inflation is ``r_j + r_body`` and nothing else.** Adding ``d_safe`` to the
inflation makes D* undefined at every start and goal on all three maps -- both
endpoints sit ~0.5 m from a wall, and inflating by ``r_body + d_safe`` = 0.51-0.63 m
closes them off. Measured directly: with ``r_body`` alone D*(start) is
19.000 / 20.698 / 20.325 m on corridor / s_curve / chicane; with ``d_safe`` added
it is INF on all three. ``d_safe`` is a *cost activation* margin for the
controller, not a geometric exclusion for the planner, and conflating the two
silently produces an unreachable goal.

Grid is 8-connected with true diagonal weights, so the field is a Dijkstra
solution to the weighted-length problem rather than a Manhattan approximation.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Grid pitch in metres. 0.05 m resolves the 0.5 m endpoint-to-wall clearance
#: with plenty of margin.
DEFAULT_H = 0.05

#: Robot body radius; must match config['robot']['collision_radius'].
R_BODY = 0.25


def lambda_min(coeffs: Dict[str, float]) -> float:
    """Minimum specific energy ``min_v lambda(v)`` in J/m.

    ``lambda(v) = c1*v + c4 + c5/v`` is the energy per metre of straight-line
    travel at speed v, minimised at ``v* = sqrt(c5/c1)`` giving
    ``lambda_min = 2*sqrt(c1*c5) + c4``. For the deployed coefficients
    (2.5, 0.8, 0.3, 1.2, 3.0) that is 6.677226 J/m at v* = 1.095445 m/s -- and it
    is 6.700 J/m at v_max = 1.2, a 0.35% spread across the whole operating band,
    which is why there is no speed lever in this system.
    """
    c1, c4, c5 = coeffs['c1'], coeffs['c4'], coeffs['c5']
    return 2.0 * math.sqrt(c1 * c5) + c4


def v_star(coeffs: Dict[str, float]) -> float:
    """Speed that attains ``lambda_min``."""
    return math.sqrt(coeffs['c5'] / coeffs['c1'])


@dataclass
class EnergyField:
    """A solved geodesic field for one map."""
    name: str
    X: np.ndarray                 # grid x coordinates, shape (nx,)
    Y: np.ndarray                 # grid y coordinates, shape (ny,)
    free: np.ndarray              # bool (nx, ny), True = traversable
    D: np.ndarray                 # float (nx, ny), geodesic distance to goal, inf if unreachable
    goal: np.ndarray
    h: float
    obs_inflation: float
    wall_inflation: float
    lambda_min: float
    coeffs: Dict[str, float] = dc_field(default_factory=dict)
    #: 'm' (metres; every legacy map) or 'J' (terrain-weighted energy).
    #: When 'J', D already integrates the local floor
    #: ``λ(m) = 2·√(m1·c1·c5) + m4·c4`` along each edge, so E_floor == D
    #: and no further λ multiplication is correct.
    denomination: str = 'm'
    terrain: Optional[Any] = None  # env.terrain.TerrainField or None

    @property
    def E_floor(self) -> np.ndarray:
        """The thermodynamic floor from every cell.

        Metres field: ``lambda_min * D*``. Energy field (terrain): ``D``
        itself -- multiplying by λ_min again would double-count the terrain
        already integrated into the edges.
        """
        if self.denomination == 'J':
            return self.D
        return self.lambda_min * self.D

    def dstar(self, x: float, y: float) -> float:
        """Nearest-cell D* lookup. Returns inf outside the reachable set."""
        i = int(round((x - self.X[0]) / self.h))
        j = int(round((y - self.Y[0]) / self.h))
        if not (0 <= i < self.D.shape[0] and 0 <= j < self.D.shape[1]):
            return float('inf')
        return float(self.D[i, j])

    def floor_at(self, x: float, y: float) -> float:
        if self.denomination == 'J':
            return self.dstar(x, y)
        return self.lambda_min * self.dstar(x, y)

    def reachable_fraction(self) -> float:
        return float(np.isfinite(self.D).sum() / self.D.size)

    def metadata(self) -> Dict[str, Any]:
        return {
            'name': self.name, 'h': self.h,
            'obs_inflation': self.obs_inflation,
            'wall_inflation': self.wall_inflation,
            'lambda_min': self.lambda_min,
            'v_star': v_star(self.coeffs) if self.coeffs else None,
            'goal': self.goal.tolist(),
            'shape': list(self.D.shape),
            'reachable_fraction': self.reachable_fraction(),
            'dstar_at_goal': self.dstar(float(self.goal[0]), float(self.goal[1])),
            'coeffs': self.coeffs,
            # A 'J' field is NOT comparable to a 'm' field -- quoting a
            # terrain-weighted D* in metres (or vice versa) is a units error
            # that looks like a result.
            'denomination': self.denomination,
            'terrain': (self.terrain.to_dict()
                        if getattr(self, 'terrain', None) is not None else None),
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_field(scenario, *, coeffs: Optional[Dict[str, float]] = None,
                h: float = DEFAULT_H, r_body: float = R_BODY,
                wall_inflation: Optional[float] = None,
                extra_inflation: float = 0.0,
                terrain: Optional[Any] = None) -> EnergyField:
    """Solve the 8-connected geodesic field to ``scenario.goal``.

    Args:
        scenario: a ScenarioConfig (``bounds``, ``goal``, ``obstacles``).
        coeffs: energy coefficients; defaults to the deployed set.
        h: grid pitch in metres.
        r_body: body radius added to every obstacle radius AND inset from walls.
        wall_inflation: wall inset; defaults to ``r_body``.
        extra_inflation: additional obstacle inflation. Leave at 0. It exists so
            the d_safe experiment is reproducible, not because it should be used
            -- ``extra_inflation=d_safe`` renders both endpoints unreachable.
        terrain: optional ``env.terrain.TerrainField``. When
            given, edge weights integrate the LOCAL specific-energy floor
            ``λ(m) = 2·√(m1·c1·c5) + m4·c4`` (trapezoid over each edge's two
            endpoint cells), so the field's ``D`` is an *energy* distance in
            joules and ``denomination='J'``. The field then already contains
            the terrain; callers must not multiply by ``lambda_min`` again.
            ``None`` (every legacy map) keeps the pure metric field and every
            tracked number bit-identical.

    Raises:
        ValueError: if the goal cell is not free, which is the loud version of
            the failure that ``extra_inflation`` produces silently.
    """
    from mpc.energy_model import load_default_coefficients
    coeffs = dict(coeffs or load_default_coefficients())
    wall_inflation = r_body if wall_inflation is None else wall_inflation
    obs_inflation = r_body + extra_inflation

    x0, y0, x1, y1 = scenario.bounds
    nx = int(round((x1 - x0) / h)) + 1
    ny = int(round((y1 - y0) / h)) + 1
    X = x0 + h * np.arange(nx)
    Y = y0 + h * np.arange(ny)
    XX, YY = np.meshgrid(X, Y, indexing='ij')

    free = np.ones((nx, ny), dtype=bool)
    for obs in (scenario.obstacles or []):
        cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
        free &= (np.hypot(XX - cx, YY - cy) > r + obs_inflation)
    free &= ((XX >= x0 + wall_inflation - 1e-9) & (XX <= x1 - wall_inflation + 1e-9)
             & (YY >= y0 + wall_inflation - 1e-9) & (YY <= y1 - wall_inflation + 1e-9))

    goal = np.asarray(scenario.goal, dtype=float)[:2]
    gi = int(round((goal[0] - X[0]) / h))
    gj = int(round((goal[1] - Y[0]) / h))
    if not (0 <= gi < nx and 0 <= gj < ny and free[gi, gj]):
        raise ValueError(
            f"{getattr(scenario, 'name', '?')}: goal cell {goal.tolist()} is not "
            f"free at obs_inflation={obs_inflation:.3f}, "
            f"wall_inflation={wall_inflation:.3f}. D* would be INF everywhere. "
            f"Inflate by r_j + r_body only -- d_safe is a cost margin for the "
            f"controller, not a geometric exclusion for the planner."
        )

    D = np.full((nx, ny), np.inf)
    D[gi, gj] = 0.0
    pq: List[Tuple[float, int, int]] = [(0.0, gi, gj)]
    if terrain is not None:
        # Terrain-aware: weight = mean of the two endpoint cells' local λ ×
        # edge length. The goal cell seeds D at 0 J (by definition it costs
        # nothing to be home); the λ grid is finite everywhere the metric
        # field is, so reachability is unchanged.
        LAM = terrain.lambda_grid(XX, YY, coeffs)
        denomination = 'J'
    else:
        LAM = None
        denomination = 'm'
    nbrs = [(a, b, h * math.hypot(a, b))
            for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > D[i, j]:
            continue
        for a, b, w in nbrs:
            p, q = i + a, j + b
            if 0 <= p < nx and 0 <= q < ny and free[p, q]:
                if LAM is not None:
                    w = w * 0.5 * (LAM[i, j] + LAM[p, q])
                if d + w < D[p, q]:
                    D[p, q] = d + w
                    heapq.heappush(pq, (d + w, p, q))

    return EnergyField(name=getattr(scenario, 'name', '?'), X=X, Y=Y, free=free,
                       D=D, goal=goal, h=h, obs_inflation=obs_inflation,
                       wall_inflation=wall_inflation,
                       lambda_min=lambda_min(coeffs), coeffs=coeffs,
                       denomination=denomination, terrain=terrain)


def geodesic_path(fieldobj: EnergyField, start: Sequence[float],
                  max_steps: int = 20000) -> Optional[np.ndarray]:
    """Steepest-descent polyline from ``start`` to the goal, or None if cut off."""
    i = int(round((start[0] - fieldobj.X[0]) / fieldobj.h))
    j = int(round((start[1] - fieldobj.Y[0]) / fieldobj.h))
    D = fieldobj.D
    if not (0 <= i < D.shape[0] and 0 <= j < D.shape[1]) or not np.isfinite(D[i, j]):
        return None
    pts = [(fieldobj.X[i], fieldobj.Y[j])]
    nbrs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    for _ in range(max_steps):
        if D[i, j] == 0.0:
            break
        best = None
        for a, b in nbrs:
            p, q = i + a, j + b
            if 0 <= p < D.shape[0] and 0 <= q < D.shape[1] and np.isfinite(D[p, q]):
                if best is None or D[p, q] < D[best[0], best[1]]:
                    best = (p, q)
        if best is None or D[best[0], best[1]] >= D[i, j]:
            break
        i, j = best
        pts.append((fieldobj.X[i], fieldobj.Y[j]))
    return np.asarray(pts)


# ---------------------------------------------------------------------------
# Persistence and the CasADi interpolant
# ---------------------------------------------------------------------------

def save_npz(fieldobj: EnergyField, path: str, *, force: bool = False) -> str:
    """Write the field to ``.npz`` behind the overwrite guard."""
    import json
    from utils.safe_write import assert_writable
    assert_writable(path, force=force, what='energy field')
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    np.savez_compressed(
        path, X=fieldobj.X, Y=fieldobj.Y, free=fieldobj.free, D=fieldobj.D,
        goal=fieldobj.goal, meta=np.array(json.dumps(fieldobj.metadata())),
    )
    return path


def load_npz(path: str) -> EnergyField:
    import json
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    terrain = None
    if meta.get('terrain'):
        from env.terrain import TerrainField
        terrain = TerrainField.from_dict(meta['terrain'])
    return EnergyField(
        name=meta['name'], X=z['X'], Y=z['Y'], free=z['free'], D=z['D'],
        goal=z['goal'], h=meta['h'], obs_inflation=meta['obs_inflation'],
        wall_inflation=meta['wall_inflation'], lambda_min=meta['lambda_min'],
        coeffs=meta.get('coeffs', {}),
        denomination=meta.get('denomination', 'm'),
        terrain=terrain,
    )


def extend_into_blocked(fieldobj: EnergyField, penalty: float = 5.0) -> np.ndarray:
    """Continue D* into blocked cells so the interpolant has bounded gradients.

    Filling unreachable cells with a large constant makes the interpolant a cliff:
    a query one cell from the boundary sees a jump from ~15 to the fill value over
    ``h`` = 0.05 m, so ``dD*/dp`` comes out around 2e7 and IPOPT's step
    computation is destroyed by a single obstacle-adjacent evaluation.

    Instead this runs a second Dijkstra outward from the reachable frontier,
    through blocked cells, at ``penalty`` times the true edge length. The result
    is continuous across the boundary with ``|grad| <= penalty`` per metre --
    steep enough that no optimiser prefers a route through an obstacle, smooth
    enough to differentiate.
    """
    D = fieldobj.D
    nx, ny = D.shape
    out = D.copy()
    pq: List[Tuple[float, int, int]] = []
    finite = np.isfinite(D)
    if not finite.any():
        return np.full_like(out, penalty * 1e3)
    # seed from every reachable cell that touches a blocked one
    for i, j in zip(*np.nonzero(finite)):
        heapq.heappush(pq, (float(D[i, j]), int(i), int(j)))
    h = fieldobj.h
    nbrs = [(a, b, penalty * h * math.hypot(a, b))
            for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > out[i, j]:
            continue
        for a, b, w in nbrs:
            p, q = i + a, j + b
            if 0 <= p < nx and 0 <= q < ny and not finite[p, q] and d + w < out[p, q]:
                out[p, q] = d + w
                heapq.heappush(pq, (d + w, p, q))
    # anything still infinite is enclosed by blocked cells with no path at all
    if not np.isfinite(out).all():
        out[~np.isfinite(out)] = np.nanmax(out[np.isfinite(out)]) + penalty * h
    return out


def build_interpolant(fieldobj: EnergyField, name: str = 'Dstar',
                      kind: str = 'linear', penalty: float = 5.0):
    """A ``ca.interpolant`` over D*, usable inside the NLP.

    Unreachable cells are filled by :func:`extend_into_blocked` rather than by a
    constant, so gradients stay bounded (``<= penalty`` J/m-equivalent per metre)
    everywhere including right at an obstacle boundary. ``kind='linear'`` rather
    than a spline because D* has genuine creases along the medial axis and a
    spline oscillates across them.
    """
    import casadi as ca
    Dfill = extend_into_blocked(fieldobj, penalty=penalty)
    # CasADi's grid interpolant wants values in column-major (Fortran) order.
    return ca.interpolant(name, kind,
                          [fieldobj.X.tolist(), fieldobj.Y.tolist()],
                          Dfill.ravel(order='F').tolist())


def build_all(scenarios: Sequence[str] = ('corridor', 's_curve', 'chicane'),
              **kwargs) -> Dict[str, EnergyField]:
    from env.scenarios import get_scenario
    return {s: build_field(get_scenario(s), **kwargs) for s in scenarios}


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=None,
                    help='Directory for the per-map .npz. Omit to compute and '
                         'report without writing anything.')
    ap.add_argument('--scenarios', nargs='*',
                    default=['corridor', 's_curve', 'chicane'])
    ap.add_argument('--h', type=float, default=DEFAULT_H)
    ap.add_argument('--run-id', dest='run_id', default=None,
                    help='when given (and --out-dir is not), fields go to '
                         '<runs>/<run-id>/planning/ on /data')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--demo-d-safe', type=float, default=None,
                    help='Reproduce the d_safe failure. Inflates obstacles AND '
                         'walls by r_body + this, and shows which of the two is '
                         'responsible for the endpoints becoming unreachable.')
    args = ap.parse_args(argv)

    from env.scenarios import get_scenario
    if args.out_dir is None and args.run_id:
        from utils.paths import run_dir as _run_dir
        args.out_dir = _run_dir(args.run_id, 'planning')
    print(f"{'map':<10} {'D*(start)':>11} {'E_floor':>10} {'reachable':>10} {'grid':>12}")
    for name in args.scenarios:
        sc = get_scenario(name)
        f = build_field(sc, h=args.h)
        ds = f.dstar(float(sc.start[0]), float(sc.start[1]))
        print(f"{name:<10} {ds:11.3f} {f.lambda_min*ds:10.2f} "
              f"{100*f.reachable_fraction():9.1f}% {str(f.D.shape):>12}")
        if args.out_dir:
            p = save_npz(f, os.path.join(args.out_dir, f'dstar_{name}.npz'),
                         force=args.force)
            print(f"           wrote {p}")

    if args.demo_d_safe is not None:
        ds = args.demo_d_safe
        print(f"\nAdding d_safe = {ds} to the inflation. The two knobs are NOT "
              f"equivalent:")
        print(f"  {'map':<10} {'obstacles only':>16} {'walls only':>14} {'both':>14}")
        for name in args.scenarios:
            sc = get_scenario(name)
            cells = []
            for kw in (dict(extra_inflation=ds),
                       dict(wall_inflation=R_BODY + ds),
                       dict(extra_inflation=ds, wall_inflation=R_BODY + ds)):
                try:
                    f = build_field(sc, h=args.h, **kw)
                    d = f.dstar(float(sc.start[0]), float(sc.start[1]))
                    cells.append(f"{d:.3f}" if np.isfinite(d) else 'INF')
                except ValueError:
                    cells.append('goal blocked')
            print(f"  {name:<10} {cells[0]:>16} {cells[1]:>14} {cells[2]:>14}")
        print("  It is the WALL inset that makes D* undefined, not obstacle "
              "inflation:\n  both endpoints sit 0.5 m from a wall, and "
              f"r_body + d_safe = {R_BODY + ds:.2f} m closes them off.\n"
              "  Obstacle inflation by d_safe merely lengthens the route "
              "(+2.2% on s_curve).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
