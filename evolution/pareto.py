"""4-objective Pareto front management for EP²-MPC.

Objectives (all minimized):
1. Energy (J)
2. Time (s)
3. Solver time (ms)
4. 1 - success_rate (inverted for HV maximization)

Reference point: [270, 52, 600, 1.0]  (calibrated from baseline data)
"""
import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, field


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Returns True if solution a Pareto-dominates b (minimize all objectives).
    
    a dominates b iff:
    - a is no worse than b in all objectives
    - a is strictly better in at least one objective
    """
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """NSGA-II fast non-dominated sort.
    
    Args:
        objectives: (n_solutions, n_obj) array
    
    Returns:
        fronts: list of lists, fronts[0] = indices of Pareto-optimal solutions
    """
    n = len(objectives)
    dominated_count = np.zeros(n, dtype=int)
    dominates_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominates_set[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    # Remove empty last front
    while fronts and not fronts[-1]:
        fronts.pop()

    return fronts


def crowding_distance(objectives: np.ndarray, front_indices: List[int]) -> np.ndarray:
    """Compute crowding distance for solutions in a front.
    
    Args:
        objectives: (n_solutions, n_obj) array
        front_indices: indices of solutions in this front
    
    Returns:
        distances: array of crowding distances for front members
    """
    n = len(front_indices)
    if n <= 2:
        return np.full(n, np.inf)

    front_obj = objectives[front_indices]
    distances = np.zeros(n)

    n_obj = front_obj.shape[1]
    for m in range(n_obj):
        sorted_idx = np.argsort(front_obj[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        obj_range = front_obj[sorted_idx[-1], m] - front_obj[sorted_idx[0], m]
        if obj_range < 1e-12:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                front_obj[sorted_idx[k + 1], m] - front_obj[sorted_idx[k - 1], m]
            ) / obj_range

    return distances


def hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """Compute hypervolume indicator using WFG sweep (exact for small n_obj).
    
    For 4 objectives, uses a Monte Carlo approximation with many samples
    to avoid WFG complexity. Falls back to exact 2D/3D if fewer objectives.
    
    Args:
        pareto_front: (n_solutions, n_obj) array of non-dominated solutions
        reference_point: (n_obj,) reference point (all solutions must dominate it)
    
    Returns:
        HV scalar value
    """
    if len(pareto_front) == 0:
        return 0.0

    front = np.array(pareto_front)
    ref = np.array(reference_point)

    # Filter solutions that dominate the reference point
    valid = np.all(front <= ref[np.newaxis, :], axis=1)
    front = front[valid]
    if len(front) == 0:
        return 0.0

    n_obj = front.shape[1]
    if n_obj == 2:
        return _hv_2d(front, ref)
    elif n_obj == 3:
        return _hv_3d_sweep(front, ref)
    else:
        # 200k samples reduce MC noise to ~0.03%, well below the
        # 0.1% stagnation threshold. Fixed seed ensures reproducible HV.
        return _hv_monte_carlo(front, ref, n_samples=200_000)


def _hv_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2D hypervolume."""
    sorted_idx = np.argsort(front[:, 0])
    front_sorted = front[sorted_idx]
    hv = 0.0
    prev_y = ref[1]
    for i in range(len(front_sorted)):
        if front_sorted[i, 1] < prev_y:
            hv += (ref[0] - front_sorted[i, 0]) * (prev_y - front_sorted[i, 1])
            prev_y = front_sorted[i, 1]
    return hv


def _hv_3d_sweep(front: np.ndarray, ref: np.ndarray) -> float:
    """3D hypervolume via slice integration."""
    sorted_idx = np.argsort(front[:, 0])
    front_sorted = front[sorted_idx]
    hv = 0.0
    prev_x = ref[0]
    for i in range(len(front_sorted) - 1, -1, -1):
        slice_front = front_sorted[i:, 1:]
        slice_ref = ref[1:]
        hv_2d = _hv_2d(slice_front, slice_ref)
        hv += hv_2d * (prev_x - front_sorted[i, 0])
        prev_x = front_sorted[i, 0]
    return hv


def _hv_monte_carlo(front: np.ndarray, ref: np.ndarray, n_samples: int = 200_000) -> float:
    """Monte Carlo hypervolume approximation for high-dimensional fronts."""
    n_obj = front.shape[1]
    # Lower bounds: minimum per objective in front
    lb = np.min(front, axis=0)
    volume_box = np.prod(ref - lb)
    if volume_box <= 0:
        return 0.0

    rng = np.random.default_rng(0)
    samples = rng.uniform(lb, ref, size=(n_samples, n_obj))
    
    # Count samples dominated by at least one solution in front
    dominated = np.zeros(n_samples, dtype=bool)
    for sol in front:
        # Sample is dominated by sol if sol[m] <= sample[m] for all m
        dominated |= np.all(samples >= sol[np.newaxis, :], axis=1)

    return float(np.sum(dominated) / n_samples * volume_box)


@dataclass
class ArchiveEntry:
    """Entry in the Pareto archive."""
    params: Any
    objectives: np.ndarray  # 4D: [energy, time, solver_ms, 1-success_rate]
    generation: int = 0
    metadata: dict = field(default_factory=dict)


class ParetoArchive:
    """Maintains a Pareto archive of non-dominated solutions.
    
    Reference point for HV: [270, 52, 600, 1.0]
    (energy=270J, time=52s, solver_ms=600ms, 1-success_rate=1.0)
    """

    # Calibrated from baseline data: max per dimension × 1.2
    # [energy_J, time_s, solver_ms, failure_rate]
    DEFAULT_REFERENCE = np.array([270.0, 52.0, 600.0, 1.0])

    def __init__(self, reference_point: Optional[np.ndarray] = None):
        self.reference_point = reference_point if reference_point is not None else self.DEFAULT_REFERENCE.copy()
        self._archive: List[ArchiveEntry] = []

    def update(self, params: Any, objectives: np.ndarray, generation: int = 0,
               metadata: Optional[dict] = None) -> bool:
        """Add candidate if non-dominated.
        
        Returns True if candidate was added.
        """
        obj = np.array(objectives, dtype=float)
        
        # Reject failed candidates. Crashed episodes produce artificially
        # low energy (short trajectory) that poisons the archive and
        # misleads the LLM strategy prompt with false "best energy" values.
        
        if obj[3] >= 0.5:   # failure_rate > 50%
            return False
        # Check if dominated by any existing solution
        for entry in self._archive:
            if dominates(entry.objectives, obj):
                return False  # new solution is dominated

        # Remove existing solutions dominated by new candidate
        self._archive = [e for e in self._archive if not dominates(obj, e.objectives)]
        
        # Add new solution
        entry = ArchiveEntry(
            params=params,
            objectives=obj,
            generation=generation,
            metadata=metadata or {}
        )
        self._archive.append(entry)
        return True

    def get_pareto_front(self) -> List[ArchiveEntry]:
        """Return current Pareto front entries."""
        return list(self._archive)

    def get_objectives_array(self) -> np.ndarray:
        """Return (n_solutions, 4) array of objectives."""
        if not self._archive:
            return np.empty((0, 4))
        return np.array([e.objectives for e in self._archive])

    def get_hypervolume(self) -> float:
        """Return current HV indicator."""
        objs = self.get_objectives_array()
        return hypervolume(objs, self.reference_point)

    def get_summary(self) -> dict:
        """Return summary statistics of current archive."""
        if not self._archive:
            return {'size': 0, 'hv': 0.0}
        objs = self.get_objectives_array()
        return {
            'size': len(self._archive),
            'hv': self.get_hypervolume(),
            'energy_min': float(np.min(objs[:, 0])),
            'energy_mean': float(np.mean(objs[:, 0])),
            'time_min': float(np.min(objs[:, 1])),
            'success_rate_max': float(1.0 - np.min(objs[:, 3])),
            'solver_ms_mean': float(np.mean(objs[:, 2])),
        }

    def size(self) -> int:
        """Number of solutions in archive."""
        return len(self._archive)

    def clear(self):
        """Clear the archive."""
        self._archive.clear()
