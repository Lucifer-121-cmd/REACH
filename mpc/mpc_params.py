"""10D parameter space for EP²-MPC optimization."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


PARAM_NAMES: List[str] = [
    'w_p', 'w_dP', 'w_E', 'w_omega', 'w_d',
    'd_safe', 'dP_thresh', 'd_thresh', 'N_base', 'w_t'
]

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    'w_p':       (0.001, 5.0),
    'w_dP':      (0.001, 3.0),
    'w_E':       (0.01,  10.0),
    'w_omega':   (0.01,  10.0),
    'w_d':       (10.0,  500.0),
    'd_safe':    (0.26,  0.38),  # collision. Upper bound 0.38m leaves clearance for the robot to pass through
    'dP_thresh': (2.0,   30.0),
    'd_thresh':  (0.3,   2.0),
    'N_base':    (8.0,   18.0),
    'w_t':       (0.1,   5.0),
}

# NOTE on w_goal and w_du:
# These two weights are NOT included in the 10D evolved parameter space.
# w_goal is the terminal position cost. Leaving it fixed prevents the LLM
# from trading accuracy for energy — a physically invalid tradeoff that would
# create solutions that "save energy" by never reaching the goal.
# w_du is a control-rate regularisation term too small to affect Pareto
# objectives meaningfully; fixing it reduces search dimensionality.
DEFAULT_PARAMS: Dict[str, float] = {
    'w_p':       0.3,
    'w_dP':      0.05,
    'w_E':       0.3,
    'w_omega':   0.05,    # allow sharp turns near obstacles
    'w_d':       400.0,   # strong obstacle repulsion
    'd_safe':    0.35,    # must be > collision_radius=0.25m
    'dP_thresh': 10.0,
    'd_thresh':  2.0,     # extend horizon when obstacle within 2m
    'N_base':    14.0,
    'w_t':       0.3,
    'w_goal':    10.0,
    'w_du':      0.05,
}

# d_safe must be > collision_radius (0.25m) so the MPC cost fires before physical
# collision. Upper bound 0.38m leaves clearance for the robot to pass through
# tight gaps (minimum viable gap = 1.75m >> robot_r + d_safe = 0.25 + 0.38 = 0.63m).
D_SAFE_MAX = 0.40   # decoupled from body radius; must allow d_safe > collision_radius (meters)


@dataclass
class MPCParams:
    """10D parameter space for EP²-MPC plus fixed weights."""
    w_p: float = 0.3
    w_dP: float = 0.05
    w_E: float = 0.3
    w_omega: float = 0.05
    w_d: float = 400.0
    d_safe: float = 0.35
    dP_thresh: float = 10.0
    d_thresh: float = 2.0
    N_base: float = 14.0
    w_t: float = 0.3
    # Fixed/secondary weights
    w_goal: float = 10.0
    w_du: float = 0.05

    @property
    def N_base_int(self) -> int:
        """N_base as integer."""
        return int(round(self.N_base))

    def to_dict(self) -> Dict[str, float]:
        """Serialize parameters to a plain float dictionary."""
        return {
            'w_p': self.w_p, 'w_dP': self.w_dP, 'w_E': self.w_E,
            'w_omega': self.w_omega, 'w_d': self.w_d, 'd_safe': self.d_safe,
            'dP_thresh': self.dP_thresh, 'd_thresh': self.d_thresh,
            'N_base': self.N_base, 'w_t': self.w_t,
            'w_goal': self.w_goal, 'w_du': self.w_du,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'MPCParams':
        """Construct an MPCParams instance from a plain dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def validate_params(params: MPCParams) -> Tuple[bool, List[str]]:
    """Check all constraints. Returns (valid, list_of_violations)."""
    violations = []
    # Hard constraint
    if params.d_safe >= D_SAFE_MAX:
        violations.append(f"d_safe={params.d_safe:.3f} must be < {D_SAFE_MAX}")
    # N_base integer in [8, 18]
    n_int = params.N_base_int
    if n_int < 8 or n_int > 18:
        violations.append(f"N_base_int={n_int} must be in [8, 18]")
    # Bounds check for all 10 params
    for name in PARAM_NAMES:
        val = getattr(params, name)
        lo, hi = PARAM_BOUNDS[name]
        if val < lo or val > hi:
            violations.append(f"{name}={val:.4f} out of bounds [{lo}, {hi}]")
    return len(violations) == 0, violations


def params_to_array(params: MPCParams) -> np.ndarray:
    """Convert MPCParams to 10D numpy array (PARAM_NAMES order)."""
    return np.array([getattr(params, name) for name in PARAM_NAMES])


def array_to_params(arr: np.ndarray, w_goal: float = 10.0, w_du: float = 0.05) -> MPCParams:
    """Convert 10D numpy array to MPCParams."""
    if len(arr) != len(PARAM_NAMES):
        raise ValueError(f"Expected {len(PARAM_NAMES)}-element array, got {len(arr)}")
    kwargs = {name: float(arr[i]) for i, name in enumerate(PARAM_NAMES)}
    kwargs['w_goal'] = w_goal
    kwargs['w_du'] = w_du
    return MPCParams(**kwargs)


def get_bounds_array() -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower_bounds, upper_bounds) numpy arrays for 10D space."""
    lbs = np.array([PARAM_BOUNDS[n][0] for n in PARAM_NAMES])
    ubs = np.array([PARAM_BOUNDS[n][1] for n in PARAM_NAMES])
    return lbs, ubs


def clamp_params(params: MPCParams) -> MPCParams:
    """Clamp all parameters to their valid bounds."""
    d = params.to_dict()
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        d[name] = float(np.clip(d[name], lo, hi))
    # Enforce hard constraint
    d['d_safe'] = min(d['d_safe'], D_SAFE_MAX - 1e-6)
    # N_base must round to integer in [8,18]
    d['N_base'] = float(np.clip(round(d['N_base']), 8, 18))
    return MPCParams(**d)


def random_params(rng: Optional[np.random.Generator] = None) -> MPCParams:
    """Sample a random valid MPCParams."""
    if rng is None:
        rng = np.random.default_rng()
    lbs, ubs = get_bounds_array()
    arr = rng.uniform(lbs, ubs)
    params = array_to_params(arr)
    return clamp_params(params)
