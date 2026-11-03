"""Stagnation detection and injection strategy for EP²-MPC evolution."""
import numpy as np
from typing import List, Optional, Deque
from collections import deque
import sys, os

from mpc.mpc_params import MPCParams, PARAM_NAMES, PARAM_BOUNDS, clamp_params, get_bounds_array, array_to_params
from evolution.pareto import ParetoArchive, dominates


class StagnationDetector:
    """Detects when evolutionary search has stagnated based on HV history.
    
    Stagnation is defined as: relative HV improvement < threshold over a window.
    """

    def __init__(self, window: int = 3, threshold: float = 0.001):
        self.window = window
        self.threshold = threshold
        self._hv_history: Deque[float] = deque(maxlen=window + 1)

    def update(self, hv: float):
        """Add new HV value to history."""
        self._hv_history.append(hv)

    def is_stagnating(self, window: Optional[int] = None, threshold: Optional[float] = None) -> bool:
        """Check if HV has been flat over the window.
        
        Returns True if max relative improvement over window < threshold.
        """
        w = window or self.window
        t = threshold or self.threshold
        
        history = list(self._hv_history)
        if len(history) < w + 1:
            return False
        
        recent = history[-(w + 1):]
        hv_base = recent[0]
        if hv_base < 1e-10:
            return False
        
        max_improvement = max(abs(h - hv_base) / hv_base for h in recent[1:])
        return max_improvement < t

    def reset(self):
        """Clear history."""
        self._hv_history.clear()

    def get_history(self) -> List[float]:
        """Return list of HV history."""
        return list(self._hv_history)

    def trend(self) -> str:
        """Return 'improving', 'stagnating', or 'insufficient_data'."""
        history = list(self._hv_history)
        if len(history) < 2:
            return 'insufficient_data'
        if self.is_stagnating():
            return 'stagnating'
        if history[-1] > history[-2]:
            return 'improving'
        return 'declining'


class InjectionStrategy:
    """Generates diverse new candidates when stagnation is detected.
    
    Uses the Pareto archive to guide injection toward underexplored regions.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        archive: ParetoArchive,
        pop_size: int,
        n_inject: int = 6,
        rng: Optional[np.random.Generator] = None
    ) -> List[MPCParams]:
        """Generate diverse new candidates for injection.
        
        Strategy:
        1. Perturb existing archive members toward underexplored objective regions
        2. Generate pure random candidates in underexplored parameter bands
        3. Add extreme-point candidates (min energy, min time, max success)
        
        Args:
            archive: current Pareto archive
            pop_size: current population size
            n_inject: number of candidates to inject
            rng: optional RNG override
        
        Returns:
            list of MPCParams to inject
        """
        _rng = rng or self.rng
        injected = []
        lbs, ubs = get_bounds_array()

        archive_entries = archive.get_pareto_front()
        
        if not archive_entries:
            # Fallback: pure random injection
            for _ in range(n_inject):
                arr = _rng.uniform(lbs, ubs)
                p = clamp_params(array_to_params(arr))
                injected.append(p)
            return injected

        # Strategy 1: Perturb archive members (n_inject // 2)
        n_perturb = n_inject // 2
        for i in range(n_perturb):
            entry = archive_entries[i % len(archive_entries)]
            from mpc.mpc_params import params_to_array
            base_arr = params_to_array(entry.params)
            # FIX M6: uniform ±10% perturbation avoids clipping bias from
            # normal distribution tails. Values stay within bounds without
            # distorting the distribution near parameter edges.
            delta = (ubs - lbs) * 0.10
            noise = _rng.uniform(-delta, delta)
            new_arr = np.clip(base_arr + noise, lbs, ubs)
            p = clamp_params(array_to_params(new_arr))
            injected.append(p)

        # Strategy 2: Extreme energy band candidates
        n_extreme = n_inject - n_perturb
        extreme_configs = [
            {'w_E': (7.0, 10.0), 'w_p': (2.0, 5.0)},   # Very high energy weight
            {'w_E': (0.01, 0.5), 'w_p': (0.001, 0.1)},  # Very low energy weight
        ]
        for i in range(n_extreme):
            arr = _rng.uniform(lbs, ubs)
            cfg = extreme_configs[i % len(extreme_configs)]
            for name, (lo, hi) in cfg.items():
                if name in PARAM_NAMES:
                    idx = PARAM_NAMES.index(name)
                    lo_c = max(lo, PARAM_BOUNDS[name][0])
                    hi_c = min(hi, PARAM_BOUNDS[name][1])
                    arr[idx] = _rng.uniform(lo_c, hi_c)
            p = clamp_params(array_to_params(arr))
            injected.append(p)

        return injected[:n_inject]
