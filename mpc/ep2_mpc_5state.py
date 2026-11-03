"""EP²-MPC 5-state variant (no cumulative energy state).

State:   x = [x_pos, y_pos, theta, v, omega]^T
Control: u = [v_cmd, omega_cmd]^T

Keeps per-step power cost but removes the energy state and terminal energy cost.
"""
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple

import casadi as ca

from mpc.mpc_params import MPCParams, DEFAULT_PARAMS
from mpc.energy_model import compute_power, load_default_coefficients
from mpc.adaptive_horizon import AdaptiveHorizon
from mpc.solver_opts import build_ipopt_opts

# Maximum number of obstacles in the NLP formulation.
# Unused slots are padded with far-away dummy obstacles.
MAX_OBS = 6

# Dummy obstacle placed far away so it contributes zero cost.
_DUMMY_OBS = (1000.0, 1000.0, 0.01)


class EP2MPC5State:
    """Energy-Efficient Economic MPC (5-state, no energy accumulation)."""

    def __init__(self, params: Optional[MPCParams] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else MPCParams(**DEFAULT_PARAMS)
        self.config = config or {}
        self.dt = self.config.get('dt', 0.1)
        self.coeffs = self.config.get('energy_coeffs', load_default_coefficients())

        # Kinematic limits
        self.v_max = self.config.get('max_v', 1.2)
        self.omega_max = self.config.get('max_omega', 3.14159)

        # Motor lag time constants (Pioneer 3-DX first-order response)
        self.tau_v = self.config.get('tau_v', 0.3)
        self.tau_omega = self.config.get('tau_omega', 0.2)
        self.alpha_v = 1.0 - np.exp(-self.dt / self.tau_v)
        self.alpha_omega = 1.0 - np.exp(-self.dt / self.tau_omega)

        # Adaptive horizon
        self.adaptive_horizon = AdaptiveHorizon(config)

        # Solver infrastructure
        self._solvers: Dict[int, Dict] = {}
        self._prev_sol: Optional[Dict] = None
        self._prev_u = np.zeros(2)
        self._prev_power = 0.0
        self._last_solver_status = 'unknown'
        N_default = self.config.get('N_default', 12)
        self.setup_solver(N_default)

    @property
    def last_solver_status(self) -> str:
        """Last normalized solver status."""
        return self._last_solver_status

    @staticmethod
    def _normalize_solver_status(stats: Optional[Dict[str, Any]]) -> str:
        """Map raw IPOPT/CasADi solver status into stable taxonomy."""
        if not stats:
            return 'unknown'
        if stats.get('success') is True:
            return 'success'

        raw_status = str(stats.get('return_status', '')).lower()
        if ('infeasible' in raw_status
                or 'restoration_failed' in raw_status):
            return 'infeasible'
        if 'max' in raw_status and 'iter' in raw_status:
            return 'max_iter'
        if 'time' in raw_status:
            return 'timeout'
        if ('nan' in raw_status
                or 'diverg' in raw_status
                or 'numeric' in raw_status):
            return 'numerical_error'
        return 'unknown'

    def setup_solver(self, N: int) -> None:
        """Build CasADi NLP for horizon *N* and cache it.

        NLP parameters layout (total = 31):
            x0          5   initial state
            goal        2   goal position
            U_prev      2   previous control (for Δu cost)
            obs         18  obstacles (MAX_OBS × 3: cx, cy, r)
            walls       4   boundary (xmin, ymin, xmax, ymax)
        """
        if N in self._solvers:
            return

        dt = self.dt
        nx, nu = 5, 2
        alpha_v = self.alpha_v
        alpha_omega = self.alpha_omega

        # Decision variables
        X = ca.MX.sym('X', nx, N + 1)
        U = ca.MX.sym('U', nu, N)

        # Parameters
        x0_sym = ca.MX.sym('x0', nx)
        goal_sym = ca.MX.sym('goal', 2)
        U_prev_sym = ca.MX.sym('U_prev', nu)
        obs_sym = ca.MX.sym('obs', 3, MAX_OBS)   # each col = (cx, cy, r)
        walls_sym = ca.MX.sym('walls', 4)          # xmin, ymin, xmax, ymax

        # Energy model coefficients (captured at build time)
        c1 = self.coeffs['c1']
        c2 = self.coeffs['c2']
        c3 = self.coeffs['c3']
        c4 = self.coeffs['c4']
        c5 = self.coeffs['c5']

        # Smooth absolute value for IPOPT differentiability
        _eps = 1e-4

        def smooth_abs(x):
            """Differentiable approximation of |x| safe for IPOPT (sqrt(x²+ε))."""
            return ca.sqrt(x ** 2 + _eps)

        def _P(v_sym, w_sym):
            """Instantaneous power as CasADi expression."""
            return (c1 * v_sym ** 2
                    + c2 * w_sym ** 2
                    + c3 * smooth_abs(v_sym) * smooth_abs(w_sym)
                    + c4 * smooth_abs(v_sym)
                    + c5)

        # ---- Build cost and constraints ----
        J = 0.0
        g = []

        # Initial state
        g.append(X[:, 0] - x0_sym)

        P_prev = _P(U_prev_sym[0], U_prev_sym[1])

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            v_cmd = uk[0]
            w_cmd = uk[1]
            v_k = xk[3]     # actual velocity (state)
            w_k = xk[4]     # actual angular velocity (state)
            pos_k = xk[:2]  # predicted position

            P_k = _P(v_k, w_k)
            P_km1 = P_prev if k == 0 else _P(X[3, k - 1], X[4, k - 1])

            # --- Weights ---
            w_p = self.params.w_p
            w_dP = self.params.w_dP
            w_t = self.params.w_t
            w_omega = self.params.w_omega
            w_d = self.params.w_d
            w_du = self.params.w_du
            d_safe = self.params.d_safe

            # --- Per-step obstacle cost ---
            obs_cost_k = 0.0
            for j in range(MAX_OBS):
                cx_j = obs_sym[0, j]
                cy_j = obs_sym[1, j]
                r_j = obs_sym[2, j]
                d_surface = (ca.sqrt((pos_k[0] - cx_j) ** 2
                                     + (pos_k[1] - cy_j) ** 2)
                             - r_j)
                obs_cost_k += ca.fmax(0.0, d_safe - d_surface) ** 2

            # --- Wall boundary cost ---
            wall_margin = d_safe
            xmin_w = walls_sym[0]
            ymin_w = walls_sym[1]
            xmax_w = walls_sym[2]
            ymax_w = walls_sym[3]
            wall_cost_k = (ca.fmax(0.0, wall_margin - (pos_k[0] - xmin_w)) ** 2
                           + ca.fmax(0.0, wall_margin - (xmax_w - pos_k[0])) ** 2
                           + ca.fmax(0.0, wall_margin - (pos_k[1] - ymin_w)) ** 2
                           + ca.fmax(0.0, wall_margin - (ymax_w - pos_k[1])) ** 2)

            # --- Control rate cost ---
            if k == 0:
                du = uk - U_prev_sym
            else:
                du = uk - U[:, k - 1]

            # --- Stage cost ---
            J += (w_p * P_k
                  + w_dP * smooth_abs(P_k - P_km1)
                  + w_t * 1.0
                  + w_d * obs_cost_k
                  + w_d * 0.5 * wall_cost_k
                  + w_omega * w_k ** 2
                  + w_du * ca.dot(du, du))

            # --- Dynamics with motor lag ---
            theta_k = xk[2]
            v_next = v_k + alpha_v * (v_cmd - v_k)
            w_next = w_k + alpha_omega * (w_cmd - w_k)
            x_next = ca.vertcat(
                xk[0] + v_k * ca.cos(theta_k) * dt,
                xk[1] + v_k * ca.sin(theta_k) * dt,
                theta_k + w_k * dt,
                v_next,
                w_next,
            )
            g.append(X[:, k + 1] - x_next)

        # ---- Terminal cost ----
        x_N = X[:, N]
        J += (self.params.w_goal
              * ca.dot(x_N[:2] - goal_sym, x_N[:2] - goal_sym))

        # ---- Pack NLP ----
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params_nlp = ca.vertcat(
            x0_sym,                           # 5
            goal_sym,                         # 2
            U_prev_sym,                       # 2
            ca.reshape(obs_sym, -1, 1),       # 3 * MAX_OBS = 18
            walls_sym,                        # 4
        )                                     # total = 31
        g_expr = ca.vertcat(*g)

        # Variable bounds
        n_opt = nx * (N + 1) + nu * N
        lbx = -1e10 * np.ones(n_opt)
        ubx = 1e10 * np.ones(n_opt)

        for k in range(N + 1):
            base = k * nx
            lbx[base + 3] = -self.v_max
            ubx[base + 3] = self.v_max
            lbx[base + 4] = -self.omega_max
            ubx[base + 4] = self.omega_max

        u_start = nx * (N + 1)
        for k in range(N):
            base = u_start + k * nu
            lbx[base + 0] = -self.v_max
            ubx[base + 0] = self.v_max
            lbx[base + 1] = -self.omega_max
            ubx[base + 1] = self.omega_max

        nlp = {'x': opt_vars, 'f': J, 'g': g_expr, 'p': params_nlp}

        opts = build_ipopt_opts(self.config, name='ep2mpc5')

        solver = ca.nlpsol('ep2mpc5', 'ipopt', nlp, opts)

        self._solvers[N] = {
            'solver': solver,
            'N': N,
            'n_opt': n_opt,
            'lbx': lbx,
            'ubx': ubx,
            'n_g': g_expr.shape[0],
            'nx': nx,
            'nu': nu,
        }

    def _dynamics_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """5-state dynamics with motor lag in numpy."""
        v_k, w_k = x[3], x[4]
        v_cmd, w_cmd = u[0], u[1]
        theta = x[2]

        v_next = v_k + self.alpha_v * (v_cmd - v_k)
        w_next = w_k + self.alpha_omega * (w_cmd - w_k)

        return np.array([
            x[0] + v_k * np.cos(theta) * self.dt,
            x[1] + v_k * np.sin(theta) * self.dt,
            x[2] + w_k * self.dt,
            v_next,
            w_next,
        ])

    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        d_obs: float,
        obstacles: List,
        params: Optional[MPCParams] = None,
        wall_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """Run one MPC step (returns E_N_predicted as 0.0 for compatibility)."""
        if params is not None:
            self.params = params

        x0_local = np.asarray(x0, dtype=float)
        if x0_local.shape[0] > 5:
            x0_local = x0_local[:5]

        # Adaptive horizon (still uses instantaneous d_obs for the trigger)
        N = self.adaptive_horizon.compute(
            d_obs, x0_local[3], x0_local[4], self._prev_power, self.params,
            {'coeffs': self.coeffs},
        )

        t_start = time.time()

        v_cmd, omega_cmd = self._solve_casadi(
            x0_local, goal, N, obstacles, wall_bounds)

        solve_ms = (time.time() - t_start) * 1000.0
        self._prev_u = np.array([v_cmd, omega_cmd])
        self._prev_power = compute_power(x0_local[3], x0_local[4], self.coeffs)
        return float(v_cmd), float(omega_cmd), float(solve_ms), 0.0

    def _solve_casadi(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        N: int,
        obstacles: List,
        wall_bounds: Optional[Tuple],
    ) -> Tuple[float, float]:
        """Solve the NLP with CasADi / IPOPT."""
        if N not in self._solvers:
            self.setup_solver(N)

        info = self._solvers[N]
        solver = info['solver']
        nx, nu = info['nx'], info['nu']

        # ---- Pad obstacles to MAX_OBS ----
        obs_flat = np.zeros(3 * MAX_OBS)
        for j in range(MAX_OBS):
            if j < len(obstacles):
                obs = obstacles[j]
                obs_flat[j * 3 + 0] = obs[0]
                obs_flat[j * 3 + 1] = obs[1]
                obs_flat[j * 3 + 2] = obs[2] if len(obs) >= 3 else 0.3
            else:
                obs_flat[j * 3 + 0] = _DUMMY_OBS[0]
                obs_flat[j * 3 + 1] = _DUMMY_OBS[1]
                obs_flat[j * 3 + 2] = _DUMMY_OBS[2]

        # ---- Wall bounds ----
        if wall_bounds is not None:
            walls = np.array(wall_bounds, dtype=float)
        else:
            walls = np.array([-100.0, -100.0, 100.0, 100.0])

        # ---- Parameter vector ----
        p = np.concatenate([
            x0,                     # 5
            goal[:2],               # 2
            self._prev_u,           # 2
            obs_flat,               # 18
            walls,                  # 4
        ])                          # total = 31

        # ---- Initial guess ----
        if (self._prev_sol is not None
                and self._prev_sol.get('N') == N):
            X_g = np.roll(self._prev_sol['X_opt'], -1, axis=1)
            X_g[:, -1] = X_g[:, -2]
            U_g = np.roll(self._prev_sol['U_opt'], -1, axis=1)
            U_g[:, -1] = U_g[:, -2]
        else:
            X_g, U_g = self._build_initial_guess(x0, goal, N, nx, nu)

        x_init = np.concatenate([X_g.ravel(order='F'),
                                 U_g.ravel(order='F')])

        n_g = info['n_g']
        lbg = np.zeros(n_g)
        ubg = np.zeros(n_g)

        try:
            sol = solver(
                x0=x_init, p=p,
                lbx=info['lbx'], ubx=info['ubx'],
                lbg=lbg, ubg=ubg,
            )
            self._last_solver_status = self._normalize_solver_status(
                solver.stats())
            x_opt = np.array(sol['x']).ravel()
            X_opt = x_opt[:nx * (N + 1)].reshape(nx, N + 1, order='F')
            U_opt = x_opt[nx * (N + 1):].reshape(nu, N, order='F')

            v_cmd = float(np.clip(U_opt[0, 0], -self.v_max, self.v_max))
            omega_cmd = float(np.clip(U_opt[1, 0],
                                      -self.omega_max, self.omega_max))

            self._prev_sol = {'N': N, 'X_opt': X_opt, 'U_opt': U_opt}
            return v_cmd, omega_cmd

        except Exception as exc:
            self._last_solver_status = self._normalize_solver_status(
                getattr(solver, 'stats', lambda: {})())
            raise RuntimeError(
                f"CasADi/IPOPT solve failed for horizon {N}: {exc}"
            ) from exc

    def _build_initial_guess(self, x0, goal, N, nx, nu):
        """Simple heading controller roll-out for warm start."""
        X = np.zeros((nx, N + 1))
        U = np.zeros((nu, N))
        X[:, 0] = x0

        for k in range(N):
            dx = goal[0] - X[0, k]
            dy = goal[1] - X[1, k]
            dist = np.hypot(dx, dy)
            theta_goal = np.arctan2(dy, dx)
            dtheta = theta_goal - X[2, k]
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

            v_cmd = min(0.5, dist / max(1, N * self.dt))
            w_cmd = np.clip(dtheta / self.dt,
                            -self.omega_max, self.omega_max)
            U[:, k] = [v_cmd, w_cmd]
            X[:, k + 1] = self._dynamics_numpy(X[:, k], U[:, k])

        return X, U

    def reset(self):
        """Clear warm-start cache."""
        self._prev_sol = None
        self._prev_u = np.zeros(2)
        self._prev_power = 0.0
        self.adaptive_horizon.reset()