"""Stratified population initialization for EP²-MPC evolution."""
import numpy as np
from typing import List, Optional, Dict, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpc.mpc_params import MPCParams, PARAM_NAMES, PARAM_BOUNDS, clamp_params, get_bounds_array


# Energy weight bands for diverse initialization
ENERGY_BANDS = {
    'very_low': {'w_E': (0.01, 0.5),   'w_p': (0.001, 0.2)},   # Disregard energy
    'low':      {'w_E': (0.5,  2.0),   'w_p': (0.1,   1.0)},   # Small energy weight
    'medium':   {'w_E': (2.0,  5.0),   'w_p': (0.5,   2.5)},   # Medium energy focus
    'high':     {'w_E': (5.0,  10.0),  'w_p': (1.0,   5.0)},   # High energy weight
}


def random_params_in_band(band: str, rng: np.random.Generator) -> MPCParams:
    """Sample MPCParams with w_E and w_p constrained to an energy band.
    
    All other parameters are sampled uniformly from their full bounds.
    """
    lbs, ubs = get_bounds_array()
    arr = rng.uniform(lbs, ubs)
    
    # Override energy-related params with band constraints
    band_cfg = ENERGY_BANDS.get(band, ENERGY_BANDS['medium'])
    
    for param_name, (lo, hi) in band_cfg.items():
        if param_name in PARAM_NAMES:
            idx = PARAM_NAMES.index(param_name)
            # Clamp to valid bounds
            lo_c = max(lo, PARAM_BOUNDS[param_name][0])
            hi_c = min(hi, PARAM_BOUNDS[param_name][1])
            arr[idx] = rng.uniform(lo_c, hi_c)
    
    # N_base must be integer in [8, 18]
    n_idx = PARAM_NAMES.index('N_base')
    arr[n_idx] = float(int(round(arr[n_idx])))
    
    params = MPCParams(**{name: float(arr[i]) for i, name in enumerate(PARAM_NAMES)})
    return clamp_params(params)


class StratifiedPopulation:
    """Creates a stratified initial population across energy weight bands.
    
    Divides pop_size into 4 equal bands to ensure diversity in energy-time tradeoff.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._population: List[MPCParams] = []

    def initialize(self, pop_size: int, energy_bands: int = 4) -> 'StratifiedPopulation':
        """Create diverse initial population.
        
        Args:
            pop_size: total population size
            energy_bands: number of energy bands (default 4)
        
        Returns:
            self for chaining
        """
        self._population = []
        band_names = list(ENERGY_BANDS.keys())
        per_band = pop_size // len(band_names)
        remainder = pop_size % len(band_names)

        for i, band in enumerate(band_names):
            n = per_band + (1 if i < remainder else 0)
            for _ in range(n):
                p = random_params_in_band(band, self.rng)
                self._population.append(p)

        # Shuffle for randomness
        indices = self.rng.permutation(len(self._population))
        self._population = [self._population[i] for i in indices]
        return self

    def get_population(self) -> List[MPCParams]:
        """Return list of MPCParams."""
        return list(self._population)

    def size(self) -> int:
        """Current population size."""
        return len(self._population)

    def add(self, params: MPCParams):
        """Add a parameter set to the population."""
        self._population.append(params)

    def replace(self, idx: int, params: MPCParams):
        """Replace a member at index."""
        self._population[idx] = params

    def to_array(self) -> np.ndarray:
        """Return (pop_size, 10) array of all parameter vectors."""
        from mpc.mpc_params import params_to_array
        return np.array([params_to_array(p) for p in self._population])
