"""Terrain multiplier fields for the REACH maps.

The rejected paper's physics audit found there is *no speed lever*:
``λ(v) = c1·v + c4 + c5/v`` is flat to 0.35% across the operating band, so
no choice of cruise speed changes the energy per metre. Terrain
heterogeneity is the fix: spatial
multipliers on the ``c1`` (traction, v²) and ``c4`` (rolling, |v|) terms
give route choice 2–3× authority instead of 0.35%, make the min-energy path
genuinely different from the shortest path, and are the setting where a
node-wise reserve constraint beats a scalar dual variable.

Design, kept deliberately small:

* A ``TerrainField`` is a list of **axis-aligned rectangular regions**, each
  carrying ``(m1, m4)`` multipliers. Outside every region both multipliers
  are 1.0. Rectangles are chosen because they are piecewise constant -- the
  plant (``PioneerEnv``), the geodesic planner (``planning/energy_dijkstra``)
  and the NLP (``mpc/reach_mpc.py``, via bilinear ``ca.interpolant`` grids)
  can all evaluate the *same* field with no discretisation disagreement
  beyond the interpolant's own grid.
* Regions must not overlap and must lie inside the arena -- validated at
  construction, loudly, because a silent overlap would make the plant's
  energy and the planner's disagree about the same patch of floor.
* The local specific-energy floor with multipliers is
  ``λ(m) = 2·√(m1·c1·c5) + m4·c4``, which at ``m1 = m4 = 3`` is ~1.96× the
  nominal 6.677 J/m: the intended 2× authority.

Everything here is a leaf module: no imports from ``mpc`` or ``planning``,
so both can use it without cycles. ``casadi`` is imported lazily because
the plant only ever needs the numpy half.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Multiplier bounds. Route choice wants 2–3× authority; the bounds
#: admit that while refusing degenerate fields (m → 0 would make terrain
#: *free*, m → ∞ would make the certificate vacuous). The generator works
#: inside [1.0, 3.0]; the wider validation range only admits hand-authored
#: variants without silently lying about them.
M_MIN, M_MAX = 0.5, 4.0


class TerrainError(ValueError):
    """Raised when a terrain field is malformed. Loud, not silent."""


@dataclass(frozen=True)
class TerrainRegion:
    """One rectangular patch of floor with scaled c1/c4 coefficients."""
    x0: float
    y0: float
    x1: float
    y1: float
    m1: float = 1.0     # multiplier on c1·v²
    m4: float = 1.0     # multiplier on c4·|v|

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise TerrainError(
                f"region {self}: degenerate (x0,y0)=({self.x0},{self.y0}) "
                f">= (x1,y1)=({self.x1},{self.y1})")
        for name, m in (("m1", self.m1), ("m4", self.m4)):
            if not (M_MIN <= m <= M_MAX):
                raise TerrainError(
                    f"region {self}: {name}={m} outside [{M_MIN}, {M_MAX}]")

    def contains(self, x: float, y: float) -> bool:
        return (self.x0 <= x < self.x1) and (self.y0 <= y < self.y1)

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1,
                "m1": self.m1, "m4": self.m4}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TerrainRegion":
        return TerrainRegion(x0=float(d["x0"]), y0=float(d["y0"]),
                             x1=float(d["x1"]), y1=float(d["y1"]),
                             m1=float(d.get("m1", 1.0)),
                             m4=float(d.get("m4", 1.0)))


class TerrainField:
    """A collection of non-overlapping regions; multiplier 1 outside them."""

    def __init__(self, regions: Sequence[TerrainRegion],
                 bounds: Optional[Tuple[float, float, float, float]] = None):
        self.regions: Tuple[TerrainRegion, ...] = tuple(regions)
        self.bounds = bounds
        self._validate()

    # -- validation ---------------------------------------------------------

    def _validate(self) -> None:
        if not self.regions:
            raise TerrainError("terrain field with zero regions: either "
                               "omit terrain entirely or provide at least one")
        if self.bounds is not None:
            bx0, by0, bx1, by1 = self.bounds
            for r in self.regions:
                if not (bx0 <= r.x0 and r.x1 <= bx1 and by0 <= r.y0 and r.y1 <= by1):
                    raise TerrainError(
                        f"region {r} extends outside the arena bounds "
                        f"{self.bounds}; the plant and the planner would "
                        "disagree about off-map floor")
        for i in range(len(self.regions)):
            for j in range(i + 1, len(self.regions)):
                a, b = self.regions[i], self.regions[j]
                if (a.x0 < b.x1 and b.x0 < a.x1
                        and a.y0 < b.y1 and b.y0 < a.y1):
                    raise TerrainError(
                        f"regions overlap: {a} and {b}. Overlapping "
                        "rectangles make the multiplier at a point ambiguous "
                        "and are a generator bug, not a modelling choice")

    # -- numpy evaluation (the plant) --------------------------------------

    def multipliers(self, x: float, y: float) -> Tuple[float, float]:
        """(m1, m4) at a point. First matching region wins; none matches →
        (1, 1). Non-overlap is validated, so 'first' is 'only'."""
        for r in self.regions:
            if r.contains(x, y):
                return r.m1, r.m4
        return 1.0, 1.0

    def multiplier_grids(self, X: np.ndarray, Y: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Meshgrids of (m1, m4) over ``np.meshgrid(X, Y, indexing='ij')``."""
        M1 = np.ones_like(X, dtype=float)
        M4 = np.ones_like(Y, dtype=float)
        for r in self.regions:
            mask = ((X >= r.x0) & (X < r.x1) & (Y >= r.y0) & (Y < r.y1))
            M1 = np.where(mask, r.m1, M1)
            M4 = np.where(mask, r.m4, M4)
        return M1, M4

    def lambda_grid(self, X: np.ndarray, Y: np.ndarray,
                    coeffs: Dict[str, float]) -> np.ndarray:
        """Local specific-energy floor ``λ(m) = 2·√(m1·c1·c5) + m4·c4``
        over the grid, in J/m. This is the edge weight the terrain-aware
        geodesic integrates."""
        M1, M4 = self.multiplier_grids(X, Y)
        c1, c4, c5 = coeffs["c1"], coeffs["c4"], coeffs["c5"]
        return 2.0 * np.sqrt(M1 * c1 * c5) + M4 * c4

    # -- CasADi evaluation (the NLP) ----------------------------------------

    def build_interpolants(self, bounds: Tuple[float, float, float, float],
                           h: float = 0.25, name: str = "terrain"):
        """Bilinear ``ca.interpolant`` pair (m1, m4) over a grid of pitch *h*.

        The CasADi interpolant *extrapolates* outside its grid -- the same
        defect that let the REACH objective report D* = −7.70 m by leaving
        the arena. Callers must clamp the argument into
        the grid exactly as ``ReachMPC._D`` does for the geodesic field.
        """
        import casadi as ca
        x0, y0, x1, y1 = bounds
        nx = int(round((x1 - x0) / h)) + 1
        ny = int(round((y1 - y0) / h)) + 1
        X = x0 + h * np.arange(nx)
        Y = y0 + h * np.arange(ny)
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        M1, M4 = self.multiplier_grids(XX, YY)
        # column-major ravel, matching planning.energy_dijkstra.build_interpolant
        i1 = ca.interpolant(f"{name}_m1", "linear", [X.tolist(), Y.tolist()],
                            M1.ravel(order="F").tolist())
        i4 = ca.interpolant(f"{name}_m4", "linear", [X.tolist(), Y.tolist()],
                            M4.ravel(order="F").tolist())
        return i1, i4, (float(X[0]), float(Y[0]), float(X[-1]), float(Y[-1]))

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> List[Dict[str, float]]:
        return [r.to_dict() for r in self.regions]

    @staticmethod
    def from_dict(data: Sequence[Dict[str, Any]],
                  bounds: Optional[Tuple[float, float, float, float]] = None
                  ) -> "TerrainField":
        return TerrainField([TerrainRegion.from_dict(d) for d in data],
                            bounds=bounds)

    def describe(self) -> str:
        lam_max = max(2.0 * math.sqrt(r.m1 * 2.5 * 3.0) + r.m4 * 1.2
                      for r in self.regions)
        return (f"{len(self.regions)} regions, local λ up to "
                f"{lam_max:.2f} J/m ({lam_max / 6.677226:.2f}× nominal)")
