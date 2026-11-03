#!/usr/bin/env python3
"""REACH: maximise outbound progress subject to a return-to-dock energy reserve.

The problem
-----------
    maximise   reach(p_0..p_N)
    subject to E_k + Ebar_dock(z_k) <= B                          (R)
               dynamics, actuator bounds, obstacle/wall cost

**Why this escapes the defect that killed the earlier min-energy-to-goal
formulation.** There the objective and the constraint pointed the same way --
both wanted the short cheap path -- so the constraint selected nothing and every
reported "budget effect" was the unconstrained optimum. In REACH the objective
wants to spend *more* energy than the reserve allows, so **the optimum sits on
the constraint boundary by construction**, at every reserve fraction and on
every map. `E_0` becomes physically meaningful: a partly depleted pack.

Two design points that are not free choices
-------------------------------------------
1. **(R) is imposed at the TERMINAL NODE ONLY.** Every-node is strictly stronger
   and manufactures spurious infeasibility on any manoeuvre where geodesic
   distance temporarily increases -- which is every obstacle detour on these maps.

2. **An explicit `P = 0` shutdown mode.** With `c5 = 3 W` idle draw there is
   otherwise no finite-energy invariant set, so no infinite-horizon feasibility
   statement about this system is true. The robot must be allowed to stop drawing.

The certificate
---------------
`Ebar_dock(z) = a + lambda_min * D*(x, y)`, fitted by
`scripts/19_reach_certificate.py` over ~10,000 states per map to a declared 95%
coverage, with the slope pinned at the physical floor
`lambda_min = 2*sqrt(c1*c5) + c4 = 6.677226` J/m.

A position-only `Ebar` is heading-blind and fails by 21-30x near the goal
(a 180 deg realignment costs 10.9 J with zero `D*` progress). The
affine form answers that objection rather than ignoring it: the worst-case
realignment cost is *bounded*, and the fitted intercept (23.73-27.97 J) is roughly
2x it, so the constant absorbs the heading term. Measured, the affine form's waste at
maximum misalignment is **0.75 J** -- it is tightest exactly where a proportional
certificate is loosest (32 J). That is why a position-only certificate is admissible
here, and it is a measurement, not an assumption.

Scalar comparison, deliberately kept
------------------------------------
`--certificate scalar` runs the same controller with `Ebar = lambda_q * D*` at the
same 95% coverage. Reach under the two certificates is the experiment: the affine
form reserves 25.2-28.8% less energy, and reach is what that converts into.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

#: Robot body radius, matching mpc/ep2_mpc.py R_BODY.
R_BODY = 0.25

#: Smoothing for the obstacle/wall hinge, matching mpc/ep2_mpc.py HINGE_EPS.
HINGE_EPS = 1e-6


def smooth_hinge_sq(z, eps: float = HINGE_EPS):
    """C2 hinge, identical to mpc/ep2_mpc.py.

    Imported by value rather than by reference so this module can be read on its
    own, but it MUST stay identical -- a different hinge here would make REACH's
    obstacle geometry incomparable with EP2-MPC's.
    """
    import casadi as ca
    return ((z + ca.sqrt(z * z + eps)) / 2.0) ** 2


@dataclass
class ReachWeights:
    """Cost weights. `w_reach` is negative-progress, i.e. the objective."""
    #: Reward on outbound geodesic progress. The objective, so it dominates.
    w_reach: float = 100.0
    #: Obstacle/wall penalty. Matches EP2-MPC's role, half weight on walls.
    #: SHAPING only since 2026-09-11 -- clearance is enforced by HARD
    #: inequality rows (see ReachMPC._build): the closed-loop smoke test
    #: measured a plan clearing an obstacle by 0.18 m (inside its own
    #: d_safe band) because a quadratic hinge is cheap near activation
    #: (w_obs·z² → 0 as z → 0) and the 20,000-scale progress reward buys
    #: corner-cutting at any finite weight. No collision-safety claim
    #: follows from it: the hard row is an experimental control
    #: (planning-level collisions are eliminated so the campaign's collision
    #: rate measures execution effects -- noise and mismatch -- not optimizer
    #: arithmetic), not a guarantee for the real robot.
    w_obs: float = 400.0
    d_safe: float = 0.30
    #: Hard clearance margin beyond r + R_BODY enforced as inequality rows.
    #: Must exceed the open-loop execution wobble between replans
    #: (measured ~0.1-0.2 m at a 2.5 s replan cadence on the generated maps).
    #: 2026-09-12: 0.20 put every plan exactly ON the row (plan clearances
    #: measured 0.200-0.218 m) and left zero margin for plant-vs-model
    #: wobble, so a return leg riding the bound hit the top wall. 0.35 keeps
    #: ~0.15 m of slack for execution between replans.
    hard_clearance: float = 0.35
    #: Control-rate penalty, purely for numerical conditioning of the arc.
    w_du: float = 0.05
    #: Penalty on |omega|, so "progress" is not bought with pirouettes.
    w_omega: float = 0.5


@dataclass
class ReachSpec:
    """Everything about the mission that is not the controller."""
    #: Total usable pack energy, joules.
    B: float
    #: Energy already spent when the horizon starts.
    E0: float = 0.0
    #: Certificate: 'affine' (a + lam*D*) or 'scalar' (lam_q*D*).
    certificate: str = 'affine'
    a_q: float = 0.0
    lam: float = 6.677226


class ReachMPC:
    """Maximum-reach NMPC with a terminal return-to-dock energy reserve.

    State is the same 6-vector EP2-MPC uses -- (x, y, theta, v, omega, E) -- so the
    dynamics, the motor lag and the energy accumulation are identical and the two
    controllers are directly comparable. The differences are entirely in the
    objective and the terminal constraint.
    """

    def __init__(self, scenario_name: str, spec: ReachSpec,
                 weights: Optional[ReachWeights] = None,
                 N: int = 20, dt: float = 0.1,
                 config: Optional[dict] = None,
                 v_max_override: Optional[float] = None):
        import casadi as ca
        from env.scenarios import get_scenario
        from mpc.energy_model import load_default_coefficients
        from planning.energy_dijkstra import build_field, build_interpolant

        self.ca = ca
        self.spec = spec
        self.w = weights or ReachWeights()
        self.N = int(N)
        self.dt = float(dt)
        cfg = config or {}
        mpc_cfg = cfg.get('mpc', {})
        robot = cfg.get('robot', {})

        # A ScenarioConfig passes through directly (generated REACH maps load
        # from YAML paths, not the name registry); a name resolves through
        # the registry as before.
        from env.scenarios import ScenarioConfig as _SC
        self.scenario = (scenario_name if isinstance(scenario_name, _SC)
                         else get_scenario(scenario_name))
        # v_max_override: the "v_max <- v*" clipping baseline --
        # the two-line change that may reproduce the whole measured effect on
        # homogeneous terrain, and is expected to do nothing there (λ(v) is
        # flat across the band; that flatness is WHY the speed lever is dead).
        self.v_max = float(v_max_override if v_max_override is not None
                           else robot.get('max_v', 1.2))
        self.w_max = float(robot.get('max_omega', 3.14159))
        # Planning-model motor lag. PLANNING-MODEL SIMPLIFICATION
        # (measured 2026-09-22; closed by the matched-tau control
        # 2026-09-24, scripts/32): the plant integrates
        # tau_v=0.3 / tau_omega=0.2 (configs/global_config.yaml robot block);
        # this NLP uses build-time defaults 0.15/0.10. At dt_plan=0.5 the plan
        # believes alpha_v=0.964 vs the plant's 0.811 — one node from rest the
        # plan expects v=1.056, the plant reaches 0.889 (~16% transient
        # optimism). CONTROL RESULT (240-mission main grid, tau matched,
        # pre-registered): breaches hold 0/240 (invariant); mean reach
        # 9.43 -> 9.22 m (-2.2%, the optimism was worth ~0.2 m of reach);
        # docking 10.0% -> 13.8% (+3.8 pp, the matched plan is slightly more
        # conservative — the same compliance-docking-reach trade, compliance
        # unchanged). The paper's method section declares the simplification
        # and cites the control.
        tau_v = float(mpc_cfg.get('tau_v', 0.15))
        tau_w = float(mpc_cfg.get('tau_omega', 0.10))
        self.alpha_v = 1.0 - math.exp(-self.dt / tau_v)
        self.alpha_w = 1.0 - math.exp(-self.dt / tau_w)

        c = load_default_coefficients()
        self.c1, self.c2 = float(c['c1']), float(c['c2'])
        self.c3, self.c4 = float(c['c3']), float(c['c4'])
        self.c5 = float(c['c5'])

        # D* field and its interpolant. `extend_into_blocked` keeps the gradient
        # bounded inside obstacles -- a constant fill gave |dD*/dp| ~ 2e7 one cell
        # from a boundary, which no NLP can descend.
        #
        # On generated terrain maps the field is built WITH the terrain, so
        # D* is an ENERGY distance (J) to the dock. The objective then
        # maximises energy-reach (the honest currency under heterogeneous
        # terrain -- metres would treat a carpeted metre and a tiled metre as
        # equal), and the reserve prices the return leg with the same terrain
        # the plant pays. Legacy maps (terrain None) keep the metric field
        # and every existing number unchanged.
        self.field = build_field(self.scenario, terrain=self.scenario.terrain)
        self.Dstar = build_interpolant(self.field, name='Dstar_reach')

        # Terrain multipliers for the NLP power term, so the controller's
        # model of what energy costs matches the plant's (PioneerEnv pays
        # m1·c1·v² + m4·c4·|v| at the robot's position). Clamped like _D:
        # an unclamped call extrapolates.
        self._terr_i1 = self._terr_i4 = None
        self._terr_ext = None
        if self.scenario.terrain is not None:
            self._terr_i1, self._terr_i4, self._terr_ext = \
                self.scenario.terrain.build_interpolants(
                    self.scenario.bounds, h=0.25, name="terrain_reach")

        self._solver = None
        self._nlp_N = None
        self._last_lam_x = None
        self._last_lam_g = None

    # -- certificate -------------------------------------------------------
    def _D(self, pos):
        """D* with its argument clamped into the grid.

        `ca.interpolant(..., 'linear', ...)` EXTRAPOLATES outside its grid rather
        than clamping, so an unclamped call makes D* unbounded below. Measured: the
        solver then bought progress by leaving the arena and reported D*_N = -7.70 m,
        a negative geodesic distance. Clamping is not cosmetic -- it is what makes
        the objective bounded on the feasible set.
        """
        ca = self.ca
        X, Y = self.field.X, self.field.Y
        px = ca.fmin(ca.fmax(pos[0], float(X[0])), float(X[-1]))
        py = ca.fmin(ca.fmax(pos[1], float(Y[0])), float(Y[-1]))
        return self.Dstar(ca.vertcat(px, py))

    def Ebar_at(self, x: float, y: float) -> float:
        """Numeric Ebar_dock at a position (the supervisor's reserve gauge).

        Same slope semantics as :meth:`_Ebar` -- one source of truth, so the
        margin the closed loop reports and the constraint the NLP enforces
        can never quietly diverge.
        """
        d = self.field.dstar(x, y)
        if not math.isfinite(d):
            return float('inf')
        if self.spec.certificate == 'affine':
            slope = (1.0 if self.field.denomination == 'J' else self.spec.lam)
            return self.spec.a_q + slope * d
        return self.spec.lam * d

    def _Ebar(self, pos):
        """Ebar_dock at a symbolic position. The reserve the pack must retain.

        Denomination matters: on a terrain map the field is already in
        joules, so the affine slope pins at 1.0 -- the floor along the route
        IS the field value, and multiplying by λ_min again would double-count
        the terrain already integrated into the edges. On a legacy metric
        field the slope is λ_min as before. The scalar form keeps
        ``spec.lam`` in both cases: on a J-field that slope is the fallback
        inefficiency factor κ (V_fb/D_J at the declared coverage); on a
        metric field it is λ_q.
        """
        d = self._D(pos)
        if self.spec.certificate == 'affine':
            slope = (1.0 if self.field.denomination == 'J' else self.spec.lam)
            return self.spec.a_q + slope * d
        if self.spec.certificate == 'scalar':
            return self.spec.lam * d
        raise ValueError(f"unknown certificate {self.spec.certificate!r}")

    # -- power -------------------------------------------------------------
    def _power(self, v, w, on, pos=None):
        """Instantaneous draw, with an explicit shutdown gate.

        `on` in [0, 1] is a decision variable, not a mode switch: with c5 = 3 W of
        idle draw a controller that cannot stop drawing has no finite-energy
        invariant set, so the shutdown mode has to exist. Gating the
        WHOLE power (not just c5) is what makes `on = 0` a true stop -- and the
        actuator bounds below force v = omega = 0 whenever it is off, so a shut-down
        robot cannot also be moving.

        On a terrain map, c1 and c4 carry the position-dependent
        multipliers of the field the plant actually pays. ``pos`` (a 2-vector
        symbolic) is REQUIRED when terrain is present; the clamp mirrors _D
        because ca.interpolant extrapolates outside its grid.
        """
        ca = self.ca
        if self._terr_i1 is not None:
            if pos is None:
                raise ValueError("terrain map: _power needs the node position")
            tx0, ty0, tx1, ty1 = self._terr_ext
            px = ca.fmin(ca.fmax(pos[0], tx0), tx1)
            py = ca.fmin(ca.fmax(pos[1], ty0), ty1)
            m1 = self._terr_i1(ca.vertcat(px, py))
            m4 = self._terr_i4(ca.vertcat(px, py))
        else:
            m1 = m4 = 1.0
        return on * (m1 * self.c1 * v * v + self.c2 * w * w
                     + self.c3 * ca.fabs(v) * ca.fabs(w)
                     + m4 * self.c4 * ca.fabs(v) + self.c5)

    # -- NLP ---------------------------------------------------------------
    def _build(self, N: int):
        """One NLP per horizon length, cached. Obstacles are parameters."""
        ca = self.ca
        n_obs = max(1, len(self.scenario.obstacles or []))

        X = ca.MX.sym('X', 6, N + 1)          # x y theta v omega E
        U = ca.MX.sym('U', 3, N)              # v_cmd omega_cmd on
        x0_p = ca.MX.sym('x0', 6)
        obs_p = ca.MX.sym('obs', 3 * n_obs)
        walls_p = ca.MX.sym('walls', 4)
        B_p = ca.MX.sym('B')
        # The comparison arms, as PARAMETERS so one build serves every
        # arm -- the campaign swaps arms without rebuilding the NLP:
        #   price_p: a linear energy price on E_N. Nonzero REPLACES the hard
        #     reserve: solve() then passes B_p = 1e9 (constraint inert) and
        #     the controller must trade progress against energy by weight
        #     alone. Fixed price = the constant-weight sweep arm; a price
        #     updated online by ascent on the reserve error = the
        #     feedback-w_p arm (the reviewer's one-line counter-proposal).
        #   dir_p: +1 outbound (maximise D*), -1 return (minimise D*) -- the
        #     supervisory mission has two phases and the objective is the
        #     same expression either way.
        price_p = ca.MX.sym('price')
        dir_p = ca.MX.sym('dir')

        g = [X[:, 0] - x0_p]
        J = 0.0
        d0 = self.Dstar(x0_p[0:2])
        J += price_p * X[5, N]

        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            v_cmd, w_cmd, on = uk[0], uk[1], uk[2]
            v_k, w_k, E_k = xk[3], xk[4], xk[5]

            # --- objective: OUTBOUND progress, i.e. away from the dock ------
            # Geodesic distance to the dock is the natural reach measure on a map
            # with obstacles: euclidean distance rewards driving into a wall.
            # Maximising D* is minimising -D*. dir_p flips the sign for the
            # return phase, where the objective is to CLOSE the geodesic gap.
            J += -dir_p * self.w_reach_k(k, N) * self._D(X[0:2, k + 1])

            # --- keep progress honest ---------------------------------------
            J += self.w.w_omega * w_k * w_k
            if k > 0:
                du = U[:, k] - U[:, k - 1]
                J += self.w.w_du * ca.dot(du[0:2], du[0:2])

            # --- obstacle and wall cost, identical in form to EP2-MPC -------
            pos = X[0:2, k + 1]
            obs_cost = 0.0
            for j in range(n_obs):
                ox, oy, r = obs_p[3*j], obs_p[3*j+1], obs_p[3*j+2]
                d_surf = ca.sqrt((pos[0]-ox)**2 + (pos[1]-oy)**2 + 1e-9) - r - R_BODY
                obs_cost += smooth_hinge_sq(self.w.d_safe - d_surf)
                # HARD clearance (added 2026-09-11): the soft hinge is
                # tradeable at any weight -- a quadratic penalty is cheap
                # near activation and the J-scale progress reward buys
                # corner-cutting (measured: a plan clearing by 0.18 m and a
                # closed-loop collision 8 s in). The wall box is already
                # hard; obstacles now are too. This is an experimental
                # control, not a safety claim: under plant noise and
                # power-model mismatch the robot can still collide.
                g.append(r + R_BODY + self.w.hard_clearance
                         - ca.sqrt((pos[0]-ox)**2 + (pos[1]-oy)**2 + 1e-9))
            m = self.w.d_safe
            wall_cost = (smooth_hinge_sq(m - (pos[0] - walls_p[0] - R_BODY))
                         + smooth_hinge_sq(m - (walls_p[2] - pos[0] - R_BODY))
                         + smooth_hinge_sq(m - (pos[1] - walls_p[1] - R_BODY))
                         + smooth_hinge_sq(m - (walls_p[3] - pos[1] - R_BODY)))
            J += self.w.w_obs * obs_cost + self.w.w_obs * 0.5 * wall_cost

            # --- dynamics with motor lag, matching EP2-MPC -------------------
            v_next = v_k + self.alpha_v * (v_cmd - v_k)
            w_next = w_k + self.alpha_w * (w_cmd - w_k)
            th = xk[2]
            g.append(X[0, k+1] - (xk[0] + v_next * ca.cos(th) * self.dt))
            g.append(X[1, k+1] - (xk[1] + v_next * ca.sin(th) * self.dt))
            g.append(X[2, k+1] - (th + w_next * self.dt))
            g.append(X[3, k+1] - v_next)
            g.append(X[4, k+1] - w_next)
            g.append(X[5, k+1] - (E_k + self._power(v_next, w_next, on,
                                                    pos=X[0:2, k+1]) * self.dt))

            # --- shutdown consistency: off means stopped --------------------
            # Without these two the solver buys free progress by switching the
            # power off while still moving.
            g.append(v_cmd - on * self.v_max)      # <= 0
            g.append(-v_cmd - on * self.v_max)     # <= 0  => |v_cmd| <= on*v_max
            g.append(w_cmd - on * self.w_max)
            g.append(-w_cmd - on * self.w_max)

        # --- (R), TERMINAL NODE ONLY ---------------------------------------
        # every-node is strictly stronger and manufactures spurious
        # infeasibility on any manoeuvre where geodesic distance temporarily
        # increases, which is every obstacle detour on these maps.
        g.append(X[5, N] + self._Ebar(X[0:2, N]) - B_p)

        nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
               'f': J,
               'g': ca.vertcat(*g),
               'p': ca.vertcat(x0_p, obs_p, walls_p, B_p, price_p, dir_p)}
        from mpc.solver_opts import build_ipopt_opts
        # 'planning', not the tracking default: see PLANNING_PROFILE. The tracking
        # values capped this solve at 200 iterations and let it bail after 5, which
        # is why every REACH solve came back infeasible.
        opts = build_ipopt_opts({}, name='reach', profile='planning')
        self._solver = ca.nlpsol('reach', 'ipopt', nlp, opts)
        self._nlp_N = N
        self._n_obs = n_obs
        # 6 equality rows for x0, then per step: 6 dynamics + 4 shutdown rows.
        self._n_g_eq = 6 + N * 6
        self._n_g_shut = N * 4
        self._n_g_obs = N * n_obs
        return self._solver

    def w_reach_k(self, k: int, N: int) -> float:
        """Progress reward, terminal-weighted.

        Rewarding only D*(x_N) lets the arc wander; rewarding every node equally
        makes the controller greedy early and it turns back before the budget binds.
        A linear ramp is the cheapest thing that does neither.
        """
        return self.w.w_reach * (k + 1) / N

    def _geodesic_guess(self, x0: np.ndarray, N: int, outbound: bool = True):
        """Roll the vehicle along ±grad D* at v*, as a warm start.

        Outbound (default): the guess ASCENDS D* -- REACH starts at the dock
        and the objective is to get AWAY from it. Return phase
        (``outbound=False``): DESCENDS it, closing the gap to the dock. Uses
        the same motor-lag integrator as the NLP so the guess satisfies the
        dynamics rows to machine precision and IPOPT starts from a feasible
        point.
        """
        v_star = 1.095445           # argmin of lambda(v) = c1 v + c4 + c5/v
        sgn = 1.0 if outbound else -1.0
        X = np.zeros((6, N + 1))
        U = np.zeros((3, N))
        X[:, 0] = x0
        h = 0.05
        for k in range(N):
            x, y, th, v, w, E = X[:, k]
            # Ascend (or descend) D* by finite difference: the direction that
            # changes geodesic distance fastest is the corridor.
            d = lambda px, py: float(np.array(self.Dstar([px, py])).ravel()[0])
            gx = sgn * (d(x + h, y) - d(x - h, y)) / (2 * h)
            gy = sgn * (d(x, y + h) - d(x, y - h)) / (2 * h)
            n = math.hypot(gx, gy)
            bearing = math.atan2(gy, gx) if n > 1e-9 else th
            err = math.atan2(math.sin(bearing - th), math.cos(bearing - th))
            w_cmd = max(-self.w_max, min(self.w_max, 2.0 * err))
            v_cmd = v_star * max(0.0, math.cos(err))     # slow while realigning
            U[:, k] = (v_cmd, w_cmd, 1.0)
            v_n = v + self.alpha_v * (v_cmd - v)
            w_n = w + self.alpha_w * (w_cmd - w)
            # the guess pays terrain like the plant does, at the NEXT
            # position (the same convention as the NLP's energy row).
            m1 = m4 = 1.0
            if self.scenario.terrain is not None:
                m1, m4 = self.scenario.terrain.multipliers(x + v_n * math.cos(th) * self.dt,
                                                           y + v_n * math.sin(th) * self.dt)
            P = (m1 * self.c1 * v_n * v_n + self.c2 * w_n * w_n
                 + self.c3 * abs(v_n) * abs(w_n) + m4 * self.c4 * abs(v_n) + self.c5)
            nx_, ny_ = x + v_n * math.cos(th) * self.dt, y + v_n * math.sin(th) * self.dt
            wx0, wy0, wx1, wy1 = (float(b) for b in self.scenario.bounds)
            nx_ = min(max(nx_, wx0 + R_BODY), wx1 - R_BODY)
            ny_ = min(max(ny_, wy0 + R_BODY), wy1 - R_BODY)
            X[:, k + 1] = (nx_, ny_, th + w_n * self.dt, v_n, w_n, E + P * self.dt)
        return X, U

    # -- solve -------------------------------------------------------------
    def _shift_guess(self, X_prev: np.ndarray, U_prev: np.ndarray,
                     executed_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Shift the previous solution to the next replan's time origin.

        ``executed_nodes`` plan nodes were flown since that solve; the new
        guess drops them, shifts the rest forward, and pads the tail by
        re-applying the last input while rolling the dynamics out with the
        same lag model the NLP uses. The guess therefore satisfies the
        dynamics rows to machine precision (same construction as
        :meth:`_geodesic_guess`) and starts from the ACTUAL current state --
        unlike the geodesic guess, which starts every solve from a standing
        heading and throws away everything the previous solve learned about
        the corridor.
        """
        n = min(executed_nodes, self.N)
        X = np.zeros((6, self.N + 1))
        U = np.zeros((3, self.N))
        if n >= self.N:                       # everything executed: tail only
            X[:, 0] = X_prev[:, self.N]
            x, y, th, v, w, E = X[:, 0]
            for k in range(self.N):
                u = U_prev[:, self.N - 1]     # hold the last input
                U[:, k] = u
                v_n = v + self.alpha_v * (u[0] - v)
                w_n = w + self.alpha_w * (u[1] - w)
                m1 = m4 = 1.0
                if self.scenario.terrain is not None:
                    m1, m4 = self.scenario.terrain.multipliers(
                        x + v_n * math.cos(th) * self.dt,
                        y + v_n * math.sin(th) * self.dt)
                P = (m1 * self.c1 * v_n * v_n + self.c2 * w_n * w_n
                     + self.c3 * abs(v_n) * abs(w_n)
                     + m4 * self.c4 * abs(v_n) + self.c5)
                x = x + v_n * math.cos(th) * self.dt
                y = y + v_n * math.sin(th) * self.dt
                th = th + w_n * self.dt
                E = E + P * self.dt
                X[:, k + 1] = (x, y, th, v_n, w_n, E)
                v, w = v_n, w_n
            return X, U
        X[:, :self.N + 1 - n] = X_prev[:, n:]
        U[:, :self.N - n] = U_prev[:, n:]
        # pad the tail: hold the last input and roll the model forward,
        # seeding from the LAST CARRIED NODE (X[:, N-n], i.e. the previous
        # plan's final state), so the guess stays continuous.
        x, y, th, v, w, E = X[:, self.N - n]
        for k in range(self.N - n, self.N):
            u = U_prev[:, -1]
            U[:, k] = u
            v_n = v + self.alpha_v * (u[0] - v)
            w_n = w + self.alpha_w * (u[1] - w)
            m1 = m4 = 1.0
            if self.scenario.terrain is not None:
                m1, m4 = self.scenario.terrain.multipliers(
                    x + v_n * math.cos(th) * self.dt,
                    y + v_n * math.sin(th) * self.dt)
            P = (m1 * self.c1 * v_n * v_n + self.c2 * w_n * w_n
                 + self.c3 * abs(v_n) * abs(w_n)
                 + m4 * self.c4 * abs(v_n) + self.c5)
            x = x + v_n * math.cos(th) * self.dt
            y = y + v_n * math.sin(th) * self.dt
            th = th + w_n * self.dt
            E = E + P * self.dt
            X[:, k + 1] = (x, y, th, v_n, w_n, E)
            v, w = v_n, w_n
        return X, U

    def solve(self, x0, B: Optional[float] = None, price: float = 0.0,
              outbound: bool = True,
              use_constraint: bool = True,
              warm: Optional[Tuple[np.ndarray, np.ndarray, int]] = None,
              ) -> Dict[str, Any]:
        """One open-loop REACH solve. Returns the plan and the binding diagnostics.

        Comparison arms (hard reserve / fixed price / dual-ascent price),
        all served by the same compiled NLP:

        * default: hard terminal reserve ``E_N + Ebar(x_N) <= B``.
        * ``price > 0, use_constraint=False``: the reserve becomes a LINEAR
          ENERGY PRICE on E_N -- the constant-weight arm when the price is
          fixed, the feedback-w_p arm when the caller updates it by ascent
          on the reserve error between solves. The constraint is made inert
          by passing B_p = 1e9 (removing the row would need a second build;
          a bound of +inf is the same statement and keeps one code path).
        * ``outbound=False``: the return phase -- same objective with the
        sign flipped, i.e. close the geodesic gap to the dock.
        * ``warm=(X_prev, U_prev, executed_nodes)``: warm start -- shift the
        previous solution as the guess (see :meth:`_shift_guess`) and tell
        IPOPT to trust the returned primal/dual point. Cuts the mean solve
        from ~40 s (cold geodesic guess every replan, measured 2026-09-11)
        to a fraction of it, and removes the unconverged-solve cases that
        the supervisor was misreading as "no certified progress". Omit it
        for the first solve or after a phase change, where there is no
        relevant previous solution.
        """
        ca = self.ca
        N = self.N
        if self._solver is None or self._nlp_N != N:
            self._build(N)

        obs = list(self.scenario.obstacles or [])
        if not obs:                       # the NLP always has >= 1 obstacle slot
            obs = [(-1e3, -1e3, 0.1)]
        while len(obs) < self._n_obs:
            obs.append((-1e3, -1e3, 0.1))
        obs_flat = [float(v) for o in obs[:self._n_obs] for v in (o[0], o[1], o[2])]
        walls = [float(v) for v in self.scenario.bounds]
        Bv = float(self.spec.B if B is None else B)

        nx = 6 * (N + 1)
        nu = 3 * N
        # Initial guess: walk OUTWARD along the geodesic at v*.
        #
        # "Hold still, power on" is the obvious guess and it is a bad one. Measured
        # with it: N=150/dt=0.1 converged to a 1.80 m reach while N=60/dt=0.5 found
        # 11.51 m on the same budget -- a longer horizon finding a six times worse
        # answer is the signature of a local minimum, not of a harder problem. The
        # objective rewards outbound D* progress, and from a standing start every
        # direction looks locally flat in D* except the one the heading already
        # points along, so the solver never discovers the corridor.
        #
        # The geodesic gives a guess that is both feasible and near-optimal in the
        # unconstrained limit, which is exactly what an interior-point method needs.
        x0 = np.asarray(x0, dtype=float)
        if warm is not None:
            X_prev, U_prev, n_exec = warm
            x_guess, u_guess = self._shift_guess(np.asarray(X_prev, dtype=float),
                                                 np.asarray(U_prev, dtype=float),
                                                 int(n_exec))
            # anchor node 0 to the MEASURED state (the shift's node-0 is the
            # old plan's belief about that time, which the plant has since
            # diverged from by wobble; the equality row pins it anyway, but
            # the guess should agree with it)
            x_guess[:, 0] = x0
        else:
            x_guess, u_guess = self._geodesic_guess(x0, N, outbound=outbound)
        w0 = np.concatenate([x_guess.reshape(-1, order='F'),
                             u_guess.reshape(-1, order='F')])

        lbx = np.full(nx + nu, -np.inf)
        ubx = np.full(nx + nu, np.inf)
        wx0, wy0, wx1, wy1 = (float(v) for v in self.scenario.bounds)
        for k in range(N + 1):
            # Hard arena box on position, inset by the body radius PLUS the
            #: same execution buffer as hard_clearance: leaving the map was
            #: otherwise a legal way to buy unbounded outbound progress
            #: against a soft wall penalty, and a plan sitting exactly on
            #: the wall bound (measured 2026-09-12) lets plant wobble cross
            #: it between replans.
            lbx[6*k + 0], ubx[6*k + 0] = (wx0 + R_BODY + self.w.hard_clearance,
                                          wx1 - R_BODY - self.w.hard_clearance)
            lbx[6*k + 1], ubx[6*k + 1] = (wy0 + R_BODY + self.w.hard_clearance,
                                          wy1 - R_BODY - self.w.hard_clearance)
            lbx[6*k + 3], ubx[6*k + 3] = -self.v_max, self.v_max
            lbx[6*k + 4], ubx[6*k + 4] = -self.w_max, self.w_max
            lbx[6*k + 5] = 0.0                       # energy is non-negative
        for k in range(N):
            b = nx + 3*k
            lbx[b + 0], ubx[b + 0] = -self.v_max, self.v_max
            lbx[b + 1], ubx[b + 1] = -self.w_max, self.w_max
            lbx[b + 2], ubx[b + 2] = 0.0, 1.0        # the shutdown gate
        # x0 is pinned by an equality row, so widen its box to whatever was passed --
        # a start state on the arena edge would otherwise make the NLP infeasible
        # before it began.
        for i in (0, 1, 3, 4, 5):
            lbx[i], ubx[i] = -np.inf, np.inf

        # NOTE the row order must match _build exactly: x0 eq (6), then per
        # step n_obs hard-obstacle ineq + 6 dynamics eq + 4 shutdown ineq
        # (the obstacle rows are appended in the cost section, which precedes
        # the dynamics section), then the terminal reserve.
        n_obs = self._n_obs
        lbg = np.zeros(6)
        ubg = np.zeros(6)
        for _ in range(N):
            lbg = np.concatenate([lbg, np.full(n_obs, -np.inf),
                                  np.zeros(6), np.full(4, -np.inf)])
            ubg = np.concatenate([ubg, np.zeros(n_obs),
                                  np.zeros(6), np.zeros(4)])
        lbg = np.concatenate([lbg, [-np.inf]])
        ubg = np.concatenate([ubg, [0.0]])

        p = np.concatenate([np.asarray(x0, dtype=float), obs_flat, walls,
                            [Bv if use_constraint else 1e9,
                             float(price), 1.0 if outbound else -1.0]])
        if warm is not None:
            # A warm solve TRUSTS the previous primal point and, when the
            # previous solve returned duals, pushes them too, so IPOPT
            # resumes rather than re-derives. The PLANNING_PROFILE's
            # warm_start_init_point='no' stays correct for the cold path --
            # there, trusting multipliers that do not exist is what made
            # every early solve return infeasible (solver_opts.py). Which
            # warm kwargs this casadi build accepts is probed from the
            # solver's OWN input list (name_in), never by running a throwaway
            # solve: a probe solve costs a full IPOPT run.
            if getattr(self, '_warm_kwargs', None) is None:
                inputs = set()
                try:
                    inputs = {n for n in self._solver.name_in()}
                except (AttributeError, RuntimeError):
                    pass
                self._warm_kwargs = {'lam_x0', 'lam_g0'} <= inputs
            if (self._warm_kwargs and self._last_lam_x is not None
                    and len(self._last_lam_x) == len(w0)):
                sol = self._solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                                   p=p, lam_x0=self._last_lam_x,
                                   lam_g0=self._last_lam_g)
            else:
                # primal-only warm start: the shifted solution as the guess
                # is already most of the benefit
                sol = self._solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        else:
            sol = self._solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
        st = self._solver.stats()
        # cache the duals for the NEXT warm solve (shape-stable: same N)
        try:
            self._last_lam_x = np.array(sol['lam_x']).ravel()
            self._last_lam_g = np.array(sol['lam_g']).ravel()
        except (KeyError, TypeError):
            pass

        wopt = np.array(sol['x']).ravel()
        Xo = wopt[:nx].reshape(6, N + 1, order='F')
        Uo = wopt[nx:].reshape(3, N, order='F')
        pos_N = Xo[0:2, N]
        d_N = float(np.array(self.Dstar(pos_N.tolist())).ravel()[0])
        d_0 = float(np.array(self.Dstar(np.asarray(x0[0:2]).tolist())).ravel()[0])
        # same slope semantics as _Ebar (a J-field's affine slope is 1.0)
        if self.spec.certificate == 'affine':
            slope = (1.0 if self.field.denomination == 'J' else self.spec.lam)
            ebar_N = self.spec.a_q + slope * d_N
        else:
            ebar_N = self.spec.lam * d_N
        return {
            'success': bool(st.get('success', False)),
            'return_status': st.get('return_status'),
            'X': Xo, 'U': Uo,
            'dstar_0': d_0, 'dstar_N': d_N,
            # reach in the field's own denomination: metres on legacy maps,
            # JOULES of energy-to-dock on terrain maps. reach_gain_m is
            # only metres when the field is metric -- read 'denomination'
            # before quoting units.
            'reach_gain': d_N - d_0,
            'reach_gain_m': (d_N - d_0) if self.field.denomination == 'm' else None,
            'denomination': self.field.denomination,
            'E_N': float(Xo[5, N]),
            'Ebar_N': ebar_N,
            'budget': Bv,
            # How much of the pack the terminal constraint actually consumed. At the
            # boundary this is ~0, which is the whole point of REACH: the optimum
            # sits ON the constraint rather than ignoring it.
            'slack_J': Bv - float(Xo[5, N]) - ebar_N,
            'on_min': float(Uo[2].min()),
            'certificate': self.spec.certificate,
        }
