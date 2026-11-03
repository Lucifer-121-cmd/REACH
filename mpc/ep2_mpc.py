"""EP²-MPC: Economic MPC with 6-state augmented model for Pioneer 3-DX.

Key improvements over initial version:
  - Per-step obstacle distance prediction across the full horizon
  - First-order motor lag dynamics for v and omega
  - Wall boundary prediction
  - Smooth approximations for IPOPT compatibility

Requires CasADi + IPOPT.
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

# --- Obstacle geometry (post-repair) -----------------------------------------
#: Robot body radius. Must match config['robot']['collision_radius'].
R_BODY = 0.25

#: Smoothing parameter for the obstacle/wall hinge.
HINGE_EPS = 1e-6


def smooth_hinge_sq(z, eps: float = HINGE_EPS):
    """C-two smooth stand-in for ``max(0, z)**2``.

    ``phi(z) = ((z + sqrt(z^2 + eps)) / 2)^2``

    The previous ``ca.fmax(0, z)**2`` is only C-one at z = 0: its second
    derivative jumps, which is precisely where IPOPT's Hessian lives. Right at
    the activation boundary -- the states that matter most for obstacle
    avoidance -- the curvature was discontinuous.

    Properties, all verifiable by inspection:
      * z << 0  ->  phi ~ (eps / (4|z|))^2, i.e. vanishes quadratically fast
      * z = 0   ->  phi = eps / 4 = 2.5e-7, a negligible constant offset
      * z >> 0  ->  phi -> z^2, matching the original penalty
    """
    return ((z + ca.sqrt(z * z + eps)) / 2.0) ** 2


class EP2MPC:
    """Energy-Efficient Economic MPC for Pioneer 3-DX.

    State:   x = [x_pos, y_pos, theta, v, omega, E_cumulative]^T
    Control: u = [v_cmd, omega_cmd]^T

    Dynamics include first-order motor lag so that actual velocity states
    track the commanded velocities with time constants tau_v and tau_omega.
    """

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

        # --- per-solve instrumentation --------------------------------------
        # _last_solver_status used to be a scalar overwritten every solve, so a
        # trial's 250-odd statuses collapsed to whichever one happened last.
        # These lists record every solve; the evaluator drains them per trial.
        self.solver_ms_all: List[float] = []
        self.solver_status_all: List[str] = []
        self.iter_count_all: List[int] = []
        self.N_all: List[int] = []
        self.built_all: List[bool] = []      # did this step construct a new NLP?
        self.warm_all: List[bool] = []       # did this step warm-start?
        self.deadline_misses = 0

        # --- deadline --------------------------------------------------------
        # The control period is the deadline. IPOPT only checks max_cpu_time
        # between iterations, so it bounds a runaway solve but cannot itself
        # measure compliance -- deadline_misses does that, from the wall clock.
        self.deadline_s = float(self.config.get('deadline_s', self.dt))

        self.N_min = int(self.config.get('N_min', 8))
        self.N_max = int(self.config.get('N_max', 50))
        N_default = int(self.config.get('N_default', 12))

        # --- solver bank ------------------------------------------------------
        # Pre-building moves NLP construction out of the timed window entirely.
        # Without it, first-visit construction is charged to solve time at every
        # new N -- measured at 388.8/660.2/751.2 ms median on the three maps,
        # which is the whole of the REALTIME null result.
        if self.config.get('prebuild_solver_bank', False):
            self.prebuild_solver_bank()
        else:
            self.setup_solver(N_default)

    def prebuild_solver_bank(self, N_values=None) -> int:
        """Construct every solver up front so no solve pays construction cost.

        Returns the number of solvers built. Costs 17-32 s once per scenario for
        the full N in [8, 50] range; that is the trade the REALTIME contribution
        is about.
        """
        if N_values is None:
            N_values = range(self.N_min, self.N_max + 1)
        built = 0
        for N in N_values:
            if N not in self._solvers:
                self.setup_solver(int(N))
                built += 1
        return built

    def reset_instrumentation(self) -> None:
        """Clear the per-solve records. Call at the start of each trial."""
        self.solver_ms_all = []
        self.solver_status_all = []
        self.iter_count_all = []
        self.N_all = []
        self.built_all = []
        self.warm_all = []
        self.deadline_misses = 0

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

    # ------------------------------------------------------------------
    # CasADi NLP construction
    # ------------------------------------------------------------------

    def setup_solver(self, N: int) -> None:
        """Build CasADi NLP for horizon *N* and cache it.

        NLP parameters layout (total = 32):
            x0          6   initial state
            goal        2   goal position
            U_prev      2   previous control (for Δu cost)
            obs         18  obstacles (MAX_OBS × 3: cx, cy, r)
            walls       4   boundary (xmin, ymin, xmax, ymax)
        """
        if N in self._solvers:
            return

        dt = self.dt
        nx, nu = 6, 2
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

            # --- Per-step obstacle cost (horizon-wide prediction + geometry repair) ---
            # d_surface is the clearance between the ROBOT BODY and the obstacle
            # surface, so it must subtract r_body as well as the obstacle radius.
            # The original code omitted r_body, which made the penalty's
            # activation set a strict subset of the collision set: the chicane
            # trace reached 0.2425 m from an obstacle surface against a 0.25 m
            # body radius -- a 7.5 mm penetration in a trial reported as
            # success. This moves the activation point by r_body on every step
            # of every trial, which is exactly why it was bundled as one
            # invalidation event after the old results were frozen.
            obs_cost_k = 0.0
            for j in range(MAX_OBS):
                cx_j = obs_sym[0, j]
                cy_j = obs_sym[1, j]
                r_j = obs_sym[2, j]
                d_surface = (ca.sqrt((pos_k[0] - cx_j) ** 2
                                     + (pos_k[1] - cy_j) ** 2)
                             - r_j - R_BODY)
                obs_cost_k += smooth_hinge_sq(d_safe - d_surface)

            # --- Wall boundary cost ---
            # Walls get r_body too: the body, not the centre, is what contacts.
            wall_margin = d_safe
            xmin_w = walls_sym[0]
            ymin_w = walls_sym[1]
            xmax_w = walls_sym[2]
            ymax_w = walls_sym[3]
            wall_cost_k = (smooth_hinge_sq(wall_margin - (pos_k[0] - xmin_w - R_BODY))
                           + smooth_hinge_sq(wall_margin - (xmax_w - pos_k[0] - R_BODY))
                           + smooth_hinge_sq(wall_margin - (pos_k[1] - ymin_w - R_BODY))
                           + smooth_hinge_sq(wall_margin - (ymax_w - pos_k[1] - R_BODY)))

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
                xk[5] + P_k * dt,          # cumulative energy
            )
            g.append(X[:, k + 1] - x_next)

        # ---- Terminal cost ----
        x_N = X[:, N]
        J += (self.params.w_goal
              * ca.dot(x_N[:2] - goal_sym, x_N[:2] - goal_sym))
        J += self.params.w_E * x_N[5]  # terminal cumulative energy

        # ---- Pack NLP ----
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params_nlp = ca.vertcat(
            x0_sym,                           # 6
            goal_sym,                         # 2
            U_prev_sym,                       # 2
            ca.reshape(obs_sym, -1, 1),       # 3 * MAX_OBS = 18
            walls_sym,                        # 4
        )                                     # total = 32
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
            lbx[base + 5] = 0.0           # E >= 0

        u_start = nx * (N + 1)
        for k in range(N):
            base = u_start + k * nu
            lbx[base + 0] = -self.v_max
            ubx[base + 0] = self.v_max
            lbx[base + 1] = -self.omega_max
            ubx[base + 1] = self.omega_max

        nlp = {'x': opt_vars, 'f': J, 'g': g_expr, 'p': params_nlp}

        opts = build_ipopt_opts(self.config, name='ep2mpc')

        solver = ca.nlpsol('ep2mpc', 'ipopt', nlp, opts)

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

    # ------------------------------------------------------------------
    # Dynamics (numpy — for warm-start roll-out)
    # ------------------------------------------------------------------

    def _dynamics_numpy(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """6-state dynamics with motor lag in numpy."""
        v_k, w_k = x[3], x[4]
        v_cmd, w_cmd = u[0], u[1]
        theta = x[2]
        P = compute_power(v_k, w_k, self.coeffs)

        v_next = v_k + self.alpha_v * (v_cmd - v_k)
        w_next = w_k + self.alpha_omega * (w_cmd - w_k)

        return np.array([
            x[0] + v_k * np.cos(theta) * self.dt,
            x[1] + v_k * np.sin(theta) * self.dt,
            x[2] + w_k * self.dt,
            v_next,
            w_next,
            x[5] + P * self.dt,
        ])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        d_obs: float,
        obstacles: List,
        params: Optional[MPCParams] = None,
        wall_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """Run one MPC step.

        The NLP now predicts obstacle distances at every horizon step using
        the predicted robot trajectory, rather than using a single current
        LiDAR snapshot (fixes the static-d_obs limitation).

        Motor lag is modelled so that v/omega states track commands with
        first-order time constants tau_v and tau_omega.

        Args:
            x0: current 6-D state  [x, y, theta, v, omega, E]
            goal: 2-D goal position
            d_obs: minimum LiDAR distance (used for adaptive horizon only)
            obstacles: list of (cx, cy, radius) tuples
            params: MPC parameters (overrides self.params)
            wall_bounds: (xmin, ymin, xmax, ymax) or None

        Returns:
            (v_cmd, omega_cmd, solve_ms, E_N_predicted)
        """
        if params is not None:
            self.params = params

        # Adaptive horizon (still uses instantaneous d_obs for the trigger)
        N = self.adaptive_horizon.compute(
            d_obs, x0[3], x0[4], self._prev_power, self.params,
            {'coeffs': self.coeffs},
        )

        # Construct OUTSIDE the timed window. Previously setup_solver was
        # called from _solve_casadi (i.e. after t_start), so a first visit to any
        # N charged its whole NLP construction to that step's "solve time". That
        # single mistake is what made horizon changes look 5-8x more expensive
        # than they are.
        built = N not in self._solvers
        if built:
            self.setup_solver(N)

        warm = (self._prev_sol is not None)

        t_start = time.perf_counter()

        v_cmd, omega_cmd, E_N = self._solve_casadi(
            x0, goal, N, obstacles, wall_bounds)

        solve_ms = (time.perf_counter() - t_start) * 1000.0

        # record every solve, not just the last one.
        stats = self._solvers[N]['solver'].stats()
        self.solver_ms_all.append(float(solve_ms))
        self.solver_status_all.append(self._last_solver_status)
        self.iter_count_all.append(int(stats.get('iter_count') or -1))
        self.N_all.append(int(N))
        self.built_all.append(bool(built))
        self.warm_all.append(bool(warm))
        if solve_ms > self.deadline_s * 1000.0:
            self.deadline_misses += 1

        self._prev_u = np.array([v_cmd, omega_cmd])
        self._prev_power = compute_power(x0[3], x0[4], self.coeffs)
        return float(v_cmd), float(omega_cmd), float(solve_ms), float(E_N)

    # ------------------------------------------------------------------
    # CasADi solve
    # ------------------------------------------------------------------

    def _solve_casadi(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        N: int,
        obstacles: List,
        wall_bounds: Optional[Tuple],
    ) -> Tuple[float, float, float]:
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
            x0,                     # 6
            goal[:2],               # 2
            self._prev_u,           # 2
            obs_flat,               # 18
            walls,                  # 4
        ])                          # total = 32

        # ---- Initial guess (warm-start repair) ----
        # Three defects fixed here, all of which degraded EVERY warm start:
        #   1. X_g[:,0] was left at the *predicted* state instead of the measured
        #      x0, so the guess disagreed with the initial-condition constraint.
        #   2. The energy row was carried over absolutely, so the guess claimed
        #      last step's accumulated energy at this step's k=0.
        #   3. A horizon change threw the whole solution away and cold-started.
        X_g = U_g = None
        lam_x0 = lam_g0 = None
        if self._prev_sol is not None:
            N_prev = int(self._prev_sol['N'])
            X_s, U_s = self._shift_solution(
                self._prev_sol['X_opt'], self._prev_sol['U_opt'])
            if N_prev == N:
                X_g, U_g = X_s, U_s
                lam_x0, lam_g0 = self._shift_duals(
                    self._prev_sol.get('lam_x'), self._prev_sol.get('lam_g'),
                    N_prev, nx, nu)
            else:
                X_g, U_g = self._remap_horizon(X_s, U_s, N_prev, N, x0, goal, nx, nu)
                lam_x0, lam_g0 = self._remap_duals(
                    self._prev_sol.get('lam_x'), self._prev_sol.get('lam_g'),
                    N_prev, N, nx, nu)

        if X_g is None:
            X_g, U_g = self._build_initial_guess(x0, goal, N, nx, nu)

        # Anchor the guess to reality and re-base the energy row.
        X_g = np.array(X_g, dtype=float, copy=True)
        X_g[:, 0] = x0
        X_g[5, :] = x0[5] + (X_g[5, :] - X_g[5, 0])

        x_init = np.concatenate([X_g.ravel(order='F'),
                                 U_g.ravel(order='F')])

        n_g = info['n_g']
        lbg = np.zeros(n_g)
        ubg = np.zeros(n_g)

        call = {'x0': x_init, 'p': p, 'lbx': info['lbx'], 'ubx': info['ubx'],
                'lbg': lbg, 'ubg': ubg}
        # Only pass duals whose dimensions match this solver, so a remap bug can
        # never turn into a CasADi shape error mid-campaign.
        if lam_x0 is not None and len(lam_x0) == info['n_opt']:
            call['lam_x0'] = lam_x0
        if lam_g0 is not None and len(lam_g0) == n_g:
            call['lam_g0'] = lam_g0

        try:
            sol = solver(**call)
            self._last_solver_status = self._normalize_solver_status(
                solver.stats())
            x_opt = np.array(sol['x']).ravel()
            X_opt = x_opt[:nx * (N + 1)].reshape(nx, N + 1, order='F')
            U_opt = x_opt[nx * (N + 1):].reshape(nu, N, order='F')

            v_cmd = float(np.clip(U_opt[0, 0], -self.v_max, self.v_max))
            omega_cmd = float(np.clip(U_opt[1, 0],
                                      -self.omega_max, self.omega_max))
            E_N = float(X_opt[5, -1])

            self._prev_sol = {
                'N': N, 'X_opt': X_opt, 'U_opt': U_opt,
                'lam_x': np.array(sol['lam_x']).ravel(),
                'lam_g': np.array(sol['lam_g']).ravel(),
            }
            return v_cmd, omega_cmd, E_N

        except Exception as exc:
            self._last_solver_status = self._normalize_solver_status(
                getattr(solver, 'stats', lambda: {})())
            raise RuntimeError(
                f"CasADi/IPOPT solve failed for horizon {N}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Warm-start transfer
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_solution(X_opt, U_opt):
        """Receding-horizon shift: drop step 0, duplicate the last column."""
        X = np.roll(np.asarray(X_opt, dtype=float), -1, axis=1)
        X[:, -1] = X[:, -2]
        U = np.roll(np.asarray(U_opt, dtype=float), -1, axis=1)
        U[:, -1] = U[:, -2]
        return X, U

    @staticmethod
    def _shift_duals(lam_x, lam_g, N, nx, nu):
        """Shift duals one step, mirroring _shift_solution's block layout.

        Decision vector is [X (nx*(N+1)), U (nu*N)] in Fortran order, so both
        halves are contiguous per-stage blocks and shift block-wise.
        """
        if lam_x is None or lam_g is None:
            return None, None
        lam_x = np.asarray(lam_x, dtype=float).ravel()
        lam_g = np.asarray(lam_g, dtype=float).ravel()
        nX = nx * (N + 1)
        if lam_x.size != nX + nu * N or lam_g.size != nx * (N + 1):
            return None, None
        X = lam_x[:nX].reshape(nx, N + 1, order='F')
        U = lam_x[nX:].reshape(nu, N, order='F')
        G = lam_g.reshape(nx, N + 1, order='F')
        Xs, Us = EP2MPC._shift_solution(X, U)
        Gs = np.roll(G, -1, axis=1)
        Gs[:, -1] = Gs[:, -2]
        return (np.concatenate([Xs.ravel(order='F'), Us.ravel(order='F')]),
                Gs.ravel(order='F'))

    def _remap_horizon(self, X_s, U_s, N_prev, N_new, x0, goal, nx, nu):
        """Map a shifted solution from horizon N_prev onto horizon N_new.

        Down (N_new < N_prev): truncate -- the retained prefix is already optimal
        for the shorter problem's first N_new stages.
        Up (N_new > N_prev): keep the whole prefix and roll the tail forward with
        the same heading controller that builds a cold guess, so the extension is
        dynamically consistent rather than a flat hold.
        """
        X_s = np.asarray(X_s, dtype=float)
        U_s = np.asarray(U_s, dtype=float)
        if N_new < N_prev:
            return X_s[:, :N_new + 1].copy(), U_s[:, :N_new].copy()

        X_c, U_c = self._build_initial_guess(x0, goal, N_new, nx, nu)
        X = np.array(X_c, dtype=float)
        U = np.array(U_c, dtype=float)
        # prefix-pad: overwrite the cold guess's first N_prev stages with the
        # transferred ones, leaving the rolled-out tail in place.
        X[:, :N_prev + 1] = X_s
        U[:, :N_prev] = U_s
        return X, U

    @staticmethod
    def _remap_duals(lam_x, lam_g, N_prev, N_new, nx, nu):
        """Truncate or zero-pad duals block-wise across a horizon change.

        Zero is the correct pad: a stage that did not exist has no active
        multiplier to inherit, and IPOPT treats a zero initial multiplier as
        "unknown" rather than as an assertion.
        """
        if lam_x is None or lam_g is None:
            return None, None
        lam_x = np.asarray(lam_x, dtype=float).ravel()
        lam_g = np.asarray(lam_g, dtype=float).ravel()
        nXp = nx * (N_prev + 1)
        if lam_x.size != nXp + nu * N_prev or lam_g.size != nx * (N_prev + 1):
            return None, None
        Xp = lam_x[:nXp].reshape(nx, N_prev + 1, order='F')
        Up = lam_x[nXp:].reshape(nu, N_prev, order='F')
        Gp = lam_g.reshape(nx, N_prev + 1, order='F')

        def _fit(M, cols):
            if M.shape[1] >= cols:
                return M[:, :cols].copy()
            out = np.zeros((M.shape[0], cols))
            out[:, :M.shape[1]] = M
            return out

        X = _fit(Xp, N_new + 1)
        U = _fit(Up, N_new)
        G = _fit(Gp, N_new + 1)
        return (np.concatenate([X.ravel(order='F'), U.ravel(order='F')]),
                G.ravel(order='F'))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
