"""Adaptive horizon computation for EP²-MPC (Patent Claim 2).

Core formula:
    N_k = N_base + round((N_max - N_base) · φ(d_obs, ΔP))
    φ   = max(φ_prox, φ_energy)

N_base  : evolved parameter in [8, 18] — resting horizon when φ=0.
N_max   : fixed system ceiling = 24 — maximum horizon under full danger.

Behaviour:
    φ = 0  (no danger, no anomaly)  →  N = N_base   (use evolved resting horizon)
    φ = 1  (obstacle near OR power spike)  →  N = N_max = 24   (full lookahead)
    φ ∈ (0,1)  →  N interpolates between N_base and N_max

This means N_base is a genuine optimizable parameter: a small N_base (e.g. 8)
minimises solver time in open space; a large N_base (e.g. 16) gives richer
lookahead everywhere. The LLM evolves N_base to trade these off per scenario.
"""
import numpy as np
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pure functions (stateless, unit-testable)
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_phi_prox(d_obs: float, d_thresh: float, beta_d: float) -> float:
    """Proximity signal — increases as the robot approaches obstacles.

    φ_prox(d) = sigmoid(β_d · (d_thresh − d_obs))

    High when d_obs < d_thresh (danger zone).
    Low when d_obs >> d_thresh (open space).
    """
    return sigmoid(beta_d * (d_thresh - d_obs))


def compute_phi_energy(dP: float, dP_thresh: float, beta_e: float) -> float:
    """Energy-anomaly signal — increases with power deviation from baseline.

    φ_energy(ΔP) = sigmoid(β_e · (|ΔP| − ΔP_thresh))

    High when |ΔP| > dP_thresh (unexpected power spike or drop).
    Low when power tracks the expected forward-motion profile.
    """
    return sigmoid(beta_e * (abs(dP) - dP_thresh))


def compute_phi(
    d_obs: float,
    dP: float,
    d_thresh: float,
    dP_thresh: float,
    beta_d: float,
    beta_e: float,
) -> float:
    """Combined adaptive signal φ = max(φ_prox, φ_energy).

    Either proximity OR energy anomaly independently triggers horizon
    extension. Taking the max ensures neither signal is masked by the other.
    This is the core of Patent Claim 2.
    """
    phi_prox = compute_phi_prox(d_obs, d_thresh, beta_d)
    phi_energy = compute_phi_energy(dP, dP_thresh, beta_e)
    return max(phi_prox, phi_energy)


def compute_power_anomaly(
    v: float,
    omega: float,
    prev_power: float,
    coeffs: Dict[str, float],
    c0: float = 3.0,
    w: float = 0.7,
) -> float:
    """Compute ΔP — blended anomaly between model prediction and temporal change.

    P_expected(v)  = c1·v² + c4·|v| + c0   (forward-motion baseline at omega=0)
    P_actual(v, ω) = full polynomial model
    dP_model       = P_actual − P_expected  (rotation-induced anomaly)
    dP_temporal    = P_actual − P_prev       (step-to-step change)
    ΔP             = w·dP_model + (1−w)·dP_temporal

    The blend (default w=0.7) keeps straight-line ΔP near zero while still
    reacting to sudden rotation-induced power spikes.

    Args:
        v:          current linear velocity (m/s)
        omega:      current angular velocity (rad/s)
        prev_power: power at previous timestep (W)
        coeffs:     energy model coefficients {c1..c5}
        c0:         idle power offset for expected model (W), default 3.0
        w:          blend weight [0, 1]; 0=temporal only, 1=model only

    Returns:
        ΔP scalar (Watts) — positive = higher-than-expected power
    """
    c1 = coeffs.get('c1', 2.5)
    c2 = coeffs.get('c2', 0.8)
    c3 = coeffs.get('c3', 0.3)
    c4 = coeffs.get('c4', 1.2)
    c5 = coeffs.get('c5', 3.0)

    P_expected = c1 * v**2 + c4 * abs(v) + c0
    P_actual = (c1 * v**2 + c2 * omega**2
                + c3 * abs(v) * abs(omega) + c4 * abs(v) + c5)

    dP_model = P_actual - P_expected
    dP_temporal = P_actual - prev_power
    return w * dP_model + (1.0 - w) * dP_temporal


# ---------------------------------------------------------------------------
# AdaptiveHorizon class
# ---------------------------------------------------------------------------

class AdaptiveHorizon:
    """Computes the adaptive prediction horizon N_k at each MPC step.

    Patent Claim 2 formula:
        N_k = N_base + round((N_max − N_base) · φ(d_obs, ΔP))
        φ   = max(φ_prox(d_obs), φ_energy(ΔP))

    N_base comes from the LLM-evolved 10D parameter vector (params.N_base_int).
    N_max is a fixed system constant (default 24).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        # Hard system bounds (not evolved)
        self.N_min: int = cfg.get('N_min', 8)
        self.N_max: int = cfg.get('N_max', 24)
        # Sigmoid sensitivity parameters (fixed, tuned for Pioneer 3-DX scale)
        self.beta_d: float = cfg.get('beta_d', 3.0)
        self.beta_e: float = cfg.get('beta_e', 0.15)
        # Default threshold fallbacks when params not provided
        self.d_thresh_default: float = cfg.get('d_thresh_default', 1.0)
        self.dP_thresh_default: float = cfg.get('dP_thresh_default', 10.0)
        # Ablation control flags: set True to isolate one signal.
        # disable_energy_signal → φ_energy forced to 0; proximity-only mode.
        # disable_prox_signal   → φ_prox   forced to 0; energy-only mode.
        # Both False (default)  → full dual-signal φ = max(φ_prox, φ_energy).
        self.disable_energy_signal: bool = cfg.get('disable_energy_signal', False)
        self.disable_prox_signal:   bool = cfg.get('disable_prox_signal',   False)
        # Per-trial history for paper Figure 3
        self._history: List[Dict] = []

    def compute(
        self,
        d_obs: float,
        v: float,
        omega: float,
        prev_power: float,
        params: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Compute N_k for the current timestep.

        Args:
            d_obs:      minimum LiDAR range to nearest obstacle (metres)
            v:          current linear velocity (m/s)
            omega:      current angular velocity (rad/s)
            prev_power: power computed at the previous step (Watts)
            params:     MPCParams — provides N_base, d_thresh, dP_thresh
            config:     optional override dict (used in unit tests)

        Returns:
            N_k as an integer in [N_min, N_max]
        """
        cfg = config or {}

        # --- Evolved thresholds (from params if available) -----------------
        d_thresh = (
            params.d_thresh
            if params is not None and hasattr(params, 'd_thresh')
            else cfg.get('d_thresh', self.d_thresh_default)
        )
        dP_thresh = (
            params.dP_thresh
            if params is not None and hasattr(params, 'dP_thresh')
            else cfg.get('dP_thresh', self.dP_thresh_default)
        )

        # --- N_base: the LLM-evolved resting horizon (Patent Claim 2) ------
        # When phi=0 (no danger), N = N_base.
        # When phi=1 (full danger), N = N_max.
        if params is not None and hasattr(params, 'N_base_int'):
            N_base = int(params.N_base_int)
        else:
            N_base = int(cfg.get('N_base', self.N_min))
        # Safety clamp: N_base must stay within system bounds
        N_base = int(np.clip(N_base, self.N_min, self.N_max))

        # --- Power anomaly and combined phi --------------------------------
        coeffs: Dict[str, float] = cfg.get(
            'coeffs', {'c1': 2.5, 'c2': 0.8, 'c3': 0.3, 'c4': 1.2, 'c5': 3.0}
        )
        c0: float = cfg.get('c0', 3.0)
        w_power: float = cfg.get('w_power', 0.7)

        dP = compute_power_anomaly(v, omega, prev_power, coeffs, c0, w_power)
        # Ablation flags allow isolating one signal
        phi_prox   = 0.0 if self.disable_prox_signal   else compute_phi_prox(d_obs, d_thresh, self.beta_d)
        phi_energy = 0.0 if self.disable_energy_signal else compute_phi_energy(dP, dP_thresh, self.beta_e)
        phi = max(phi_prox, phi_energy)

        # --- Core Patent Claim 2 formula ------------------------------------
        N = N_base + round((self.N_max - N_base) * phi)
        N = int(np.clip(N, self.N_min, self.N_max))

        # --- Log per-step data for paper Figure 3 --------------------------
        c1, c2, c3, c4, c5 = (
            coeffs.get('c1', 2.5), coeffs.get('c2', 0.8), coeffs.get('c3', 0.3),
            coeffs.get('c4', 1.2), coeffs.get('c5', 3.0),
        )
        P_actual = c1 * v**2 + c2 * omega**2 + c3 * abs(v) * abs(omega) + c4 * abs(v) + c5
        P_expected_log = c1 * v**2 + c4 * abs(v) + c0

        self._history.append({
            'phi': phi,
            'phi_prox': phi_prox,
            'phi_energy': phi_energy,
            'N': N,
            'N_base': N_base,
            'd_obs': d_obs,
            'dP': dP,
            'P_actual': P_actual,
            'P_expected': P_expected_log,
            'P_prev': prev_power,
            'v': v,
            'omega': omega,
        })
        return N

    def reset(self) -> None:
        """Clear per-trial history. Call at the start of every trial."""
        self._history.clear()

    def get_history(self) -> List[Dict]:
        """Return per-step history list for paper figures and diagnostics."""
        return list(self._history)
