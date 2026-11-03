"""EP²-MPC - Energy-efficient Model Predictive Control for Pioneer 3-DX."""
from mpc.energy_model import EnergyModel, compute_power, load_default_coefficients
from mpc.mpc_params import MPCParams, DEFAULT_PARAMS, validate_params
from mpc.adaptive_horizon import AdaptiveHorizon
from mpc.ep2_mpc import EP2MPC
