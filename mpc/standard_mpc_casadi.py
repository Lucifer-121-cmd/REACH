"""Standard MPC with 5-state model using CasADi + IPOPT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import casadi as ca
import numpy as np

from mpc.solver_opts import build_ipopt_opts
from mpc.ep2_mpc import R_BODY, smooth_hinge_sq

# Maximum number of obstacles in the NLP formulation.
MAX_OBS = 6

# Dummy obstacle placed far away so it contributes zero cost.
_DUMMY_OBS = (1000.0, 1000.0, 0.01)


@dataclass
class StandardMPCWeights:
    """Weights for the comparator MPC.

    ``w_v`` is gone. It weighted ``(v - v_ref)^2``, a velocity-tracking
    term with no counterpart in EP2-MPC, and on its own it predicts a 1.23%
    energy penalty -- larger than the 0.88% corridor advantage the rejected paper
    claimed for EP2. Keeping it would mean the headline margin was smaller than a
    formulation asymmetry we introduced ourselves.

    ``d_safe`` default moves 0.24 -> 0.30. The old value sat below the 0.25 m
    collision radius, so with clearance measured centre-to-surface the obstacle
    penalty was identically zero at every collision-free state. 0.30 is inside
    the [0.26, 0.38] box EP2 uses, so the arms are matched.
    """
    w_obs: float = 360.0      # obstacle penalty weight
    w_goal: float = 10.0      # terminal goal weight
    w_du: float = 0.04        # control smoothness (Δu)
    R_v: float = 0.0          # energy proxy on v_cmd
    R_w: float = 0.0          # energy proxy on ω_cmd
    d_safe: float = 0.30      # safety margin (m), must exceed collision_radius


class StandardMPCCasadi:
    """Standard 5-state MPC for Pioneer 3-DX (no energy state).

    State:   x = [x, y, theta, v, omega]^T
    Control: u = [v_cmd, omega_cmd]^T
    """

    def __init__(self, N: int = 12, weights: Optional[StandardMPCWeights] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self.N = int(N)
        self.weights = weights if weights is not None else StandardMPCWeights()
        self.config = config or {}
        self.dt = self.config.get('dt', 0.1)
        self.v_max = self.config.get('max_v', 1.2)
        self.omega_max = self.config.get('max_omega', 3.14159)
        self.tau_v = self.config.get('tau_v', 0.3)
        self.tau_omega = self.config.get('tau_omega', 0.2)
        self.alpha_v = 1.0 - np.exp(-self.dt / self.tau_v)
        self.alpha_omega = 1.0 - np.exp(-self.dt / self.tau_omega)

        self._solver: Optional[Dict[str, Any]] = None
        self._prev_sol: Optional[Dict[str, Any]] = None
        self._prev_u = np.zeros(2)
        self._last_solver_status = 'unknown'
        self.setup_solver()

    @property
    def last_solver_status(self) -> str:
        return self._last_solver_status

    @staticmethod
    def _normalize_solver_status(stats: Optional[Dict[str, Any]]) -> str:
        if not stats:
            return 'unknown'
        if stats.get('success') is True:
            return 'success'
        raw_status = str(stats.get('return_status', '')).lower()
        if ('infeasible' in raw_status or 'restoration_failed' in raw_status):
            return 'infeasible'
        if 'max' in raw_status and 'iter' in raw_status:
            return 'max_iter'
        if 'time' in raw_status:
            return 'timeout'
        if ('nan' in raw_status or 'diverg' in raw_status or 'numeric' in raw_status):
            return 'numerical_error'
        return 'unknown'

    def setup_solver(self) -> None:
        """Build CasADi NLP for fixed horizon N."""
        if self._solver is not None:
            return

        N = self.N
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
        v_ref_sym = ca.MX.sym('v_ref')
        U_prev_sym = ca.MX.sym('U_prev', nu)
        obs_sym = ca.MX.sym('obs', 3, MAX_OBS)  # (cx, cy, r)
        # Walls are a parameter now. The baseline previously had no wall
        # term at all, and the LiDAR is blind to walls too, so nothing anywhere
        # told this controller a boundary existed.
        walls_sym = ca.MX.sym('walls', 4)       # (xmin, ymin, xmax, ymax)

        _eps = 1e-4

        def smooth_abs(x):
            return ca.sqrt(x ** 2 + _eps)

        J = 0.0
        g = []

        g.append(X[:, 0] - x0_sym)

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            v_cmd = uk[0]
            w_cmd = uk[1]
            v_k = xk[3]
            w_k = xk[4]
            pos_k = xk[:2]

            # obstacle cost. r_body is included (clearance is measured
            # body-to-surface, as in EP2MPC) and the hinge is the same C2-smooth
            # phi(z) = ((z + sqrt(z^2 + eps))/2)^2 the EP2 controller uses, so the
            # two arms differ in formulation, not in numerics.
            obs_cost_k = 0.0
            for j in range(MAX_OBS):
                cx_j = obs_sym[0, j]
                cy_j = obs_sym[1, j]
                r_j = obs_sym[2, j]
                d_surface = (ca.sqrt((pos_k[0] - cx_j) ** 2
                                     + (pos_k[1] - cy_j) ** 2) - r_j - R_BODY)
                obs_cost_k += smooth_hinge_sq(self.weights.d_safe - d_surface)

            # wall cost. Ported from EP2MPC: the baseline had NO wall term at
            # all, so on maps where the robot collides with a boundary rather than
            # an obstacle it was flying blind -- and the LiDAR does not see walls
            # either, so nothing anywhere told it a wall was there.
            wall_cost_k = 0.0
            m = self.weights.d_safe
            wall_cost_k = (
                smooth_hinge_sq(m - (pos_k[0] - walls_sym[0] - R_BODY))
                + smooth_hinge_sq(m - (walls_sym[2] - pos_k[0] - R_BODY))
                + smooth_hinge_sq(m - (pos_k[1] - walls_sym[1] - R_BODY))
                + smooth_hinge_sq(m - (walls_sym[3] - pos_k[1] - R_BODY)))

            # control rate cost
            if k == 0:
                du = uk - U_prev_sym
            else:
                du = uk - U[:, k - 1]

            # stage cost. The w_v * (v - v_ref)^2 velocity-tracking term is
            # DELETED: EP2-MPC has no velocity reference, so that term was
            # a pure formulation handicap worth a predicted 1.23% in energy --
            # larger than the 0.88% corridor advantage the rejected paper claimed.
            J += (
                self.weights.w_obs * obs_cost_k
                + self.weights.w_obs * 0.5 * wall_cost_k
                + self.weights.w_du * ca.dot(du, du)
                + self.weights.R_v * v_cmd ** 2
                + self.weights.R_w * w_cmd ** 2
            )

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

        # terminal cost
        x_N = X[:, N]
        J += self.weights.w_goal * ca.dot(x_N[:2] - goal_sym, x_N[:2] - goal_sym)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params_nlp = ca.vertcat(
            x0_sym,                     # 5
            goal_sym,                   # 2
            v_ref_sym,                  # 1  (shapes the initial guess only)
            U_prev_sym,                 # 2
            ca.reshape(obs_sym, -1, 1), # 18
            walls_sym,                  # 4
        )                               # total = 32
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

        opts = build_ipopt_opts(self.config, name='standard_mpc', extra={
            # Tight warm-start pushes: this controller keeps the same N forever,
            # so its warm start is always same-N and worth trusting.
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_bound_frac': 1e-6,
            'ipopt.warm_start_slack_bound_push': 1e-6,
            'ipopt.warm_start_slack_bound_frac': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
        })

        solver = ca.nlpsol('standard_mpc', 'ipopt', nlp, opts)

        self._solver = {
            'solver': solver,
            'N': N,
            'n_opt': n_opt,
            'lbx': lbx,
            'ubx': ubx,
            'n_g': g_expr.shape[0],
            'nx': nx,
            'nu': nu,
        }

    def _build_initial_guess(self, x0: np.ndarray, goal: np.ndarray, v_ref: float):
        N = self.N
        nx, nu = 5, 2
        X = np.zeros((nx, N + 1))
        U = np.zeros((nu, N))
        X[:, 0] = x0

        for k in range(N):
            dx = goal[0] - X[0, k]
            dy = goal[1] - X[1, k]
            dist = np.hypot(dx, dy)
            theta_goal = np.arctan2(dy, dx)
            dtheta = np.arctan2(np.sin(theta_goal - X[2, k]), np.cos(theta_goal - X[2, k]))

            v_cmd = min(v_ref, dist / max(1.0, N * self.dt))
            w_cmd = np.clip(dtheta / self.dt, -self.omega_max, self.omega_max)
            U[:, k] = [v_cmd, w_cmd]

            v_k = X[3, k]
            w_k = X[4, k]
            v_next = v_k + self.alpha_v * (v_cmd - v_k)
            w_next = w_k + self.alpha_omega * (w_cmd - w_k)
            X[:, k + 1] = np.array([
                X[0, k] + v_k * np.cos(X[2, k]) * self.dt,
                X[1, k] + v_k * np.sin(X[2, k]) * self.dt,
                X[2, k] + w_k * self.dt,
                v_next,
                w_next,
            ])

        return X, U

    def solve(
        self,
        x0: np.ndarray,
        goal: np.ndarray,
        obstacles: List,
        v_ref: Optional[float] = None,
        wall_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """Solve one step.

        ``v_ref`` no longer appears in the cost (the ``w_v (v - v_ref)^2``
        term was deleted); it only shapes the cold-start guess, and
        defaults to the robot's own ``v_max`` so the guess is not handicapped
        either. ``wall_bounds`` is the arena; omitting it places the walls far
        away so the term contributes nothing, matching the old behaviour for
        callers that have no arena to pass.
        """
        if self._solver is None:
            self.setup_solver()

        info = self._solver
        solver = info['solver']
        nx, nu = info['nx'], info['nu']
        if v_ref is None:
            v_ref = self.v_max

        x0_local = np.asarray(x0, dtype=float)
        if x0_local.shape[0] > nx:
            x0_local = x0_local[:nx]

        # Pad obstacles
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

        walls = (np.asarray(wall_bounds, dtype=float) if wall_bounds is not None
                 else np.array([-1000.0, -1000.0, 1000.0, 1000.0]))

        p = np.concatenate([
            x0_local,
            goal[:2],
            np.array([v_ref]),
            self._prev_u,
            obs_flat,
            walls,
        ])

        if self._prev_sol is not None and self._prev_sol.get('N') == self.N:
            X_g = np.roll(self._prev_sol['X_opt'], -1, axis=1)
            X_g[:, -1] = X_g[:, -2]
            U_g = np.roll(self._prev_sol['U_opt'], -1, axis=1)
            U_g[:, -1] = U_g[:, -2]
        else:
            X_g, U_g = self._build_initial_guess(x0_local, goal, v_ref)

        x_init = np.concatenate([X_g.ravel(order='F'), U_g.ravel(order='F')])

        n_g = info['n_g']
        lbg = np.zeros(n_g)
        ubg = np.zeros(n_g)

        t_start = time.time()
        try:
            sol = solver(
                x0=x_init,
                p=p,
                lbx=info['lbx'], ubx=info['ubx'],
                lbg=lbg, ubg=ubg,
            )
            self._last_solver_status = self._normalize_solver_status(solver.stats())
            x_opt = np.array(sol['x']).ravel()
            X_opt = x_opt[:nx * (self.N + 1)].reshape(nx, self.N + 1, order='F')
            U_opt = x_opt[nx * (self.N + 1):].reshape(nu, self.N, order='F')

            v_cmd = float(np.clip(U_opt[0, 0], -self.v_max, self.v_max))
            omega_cmd = float(np.clip(U_opt[1, 0], -self.omega_max, self.omega_max))

            self._prev_sol = {'N': self.N, 'X_opt': X_opt, 'U_opt': U_opt}
            self._prev_u = np.array([v_cmd, omega_cmd])
            solve_ms = (time.time() - t_start) * 1000.0
            return v_cmd, omega_cmd, solve_ms

        except Exception as exc:
            self._last_solver_status = self._normalize_solver_status(
                getattr(solver, 'stats', lambda: {})())
            raise RuntimeError(
                f"CasADi/IPOPT solve failed for horizon {self.N}: {exc}"
            ) from exc

    def reset(self) -> None:
        self._prev_sol = None
        self._prev_u = np.zeros(2)
        self._last_solver_status = 'unknown'
