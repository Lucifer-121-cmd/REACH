"""Closed-loop REACH evaluation: depart the dock, maximise reach, return.

The mission has two phases and the SUPERVISOR owns them:

* **outbound** -- replan with the progress objective every ``replan_every_s``.
  The turn-back is not scripted: it is whatever the arm's reserve mechanism
  makes of the constraint. The hard-constraint arm keeps going while
  ``E + Ebar(x) <= B`` admits progress and switches to the return phase the
  moment a replan can no longer buy reach (``reach_gain`` of the new plan
  stalls). The priced arms turn back when their energy price makes further
  progress unattractive -- earlier or later than optimal, which is the
  measurement.
* **return** -- same NLP, objective sign flipped (close the geodesic gap to
  the dock), replanning as usual.

Outcomes (mutually exclusive, first match wins):

* ``docked``   -- back at the dock (within DOCK_TOL) after having gone out
                 (``reach_m > MIN_OUTBOUND``). The success outcome.
* ``stranded`` -- the pack ran out (``E >= B``) before docking. The reserve
                 violation.
* ``collision``-- the plant hit an obstacle or a wall.
* ``timeout``  -- neither of the above by ``t_mission_s``.

``min_margin_J`` and ``collision`` are recorded on the same record
deliberately: a hard reserve bought with unreported collisions is a reserve
not worth having, so violation rate and collision rate are reported JOINTLY
across reserve fractions by the campaign that consumes these records.

The four comparison arms differ only in what they hand to
``ReachMPC.solve``:

===============  ================================  =========================
arm              mechanism                         failure mode it exposes
===============  ================================  =========================
reach            hard terminal reserve             none (the contribution)
const_weight     fixed linear energy price         over/under-reserves by
                                                    position-blind margin
dual             price updated by ascent on the    reacts only AFTER the
                  reserve error (the reviewer's    error appears -- the
                  one-line counter-proposal)       step-change experiment
vmax_clip        hard reserve, v_max pinned to v*  speed shaping does not
                                                    substitute for the
                                                    constraint (λ(v) flat)
===============  ================================  =========================

Every trial is fully seeded: LiDAR, map perturbation and actuation noise all
derive from the trial seed (the 2026-09-11 LiDAR seeding fix, extended to
this evaluator).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DOCK_TOL = 0.30          # m from the dock that counts as "home"
MIN_OUTBOUND = 1.0       # m of reach before "docked" is a mission, not a no-op
LIDAR_SEED_OFFSET = 1_000_003   # same convention as EP2Evaluator/baselines
NOISE_SEED_OFFSET = 2_000_003
#: Stall-retry gate (2026-09-14 diagnostic): a warm outbound solve with
#: gain < 0.5 J but slack above this is a warm-basin artifact (the cold
#: solve from the same state found +65 J where the warm found -1 J,
#: measured map 0912/B268/seed 31001); slack below it is a TRUE boundary
#: stall (cold and 5000-iteration solves reproduce it) and is not retried.
STALL_RETRY_SLACK = 10.0


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ReachTrialResult:
    map_name: str
    arm: str
    B: float
    seed: int
    outcome: str                 # docked | stranded | collision | timeout
    returned: bool
    reach_m: Optional[float]     # max metric geodesic distance from dock (m)
    reach_J: Optional[float]     # max energy-to-dock along the mission (J)
    energy_spent: float
    mission_time: float
    min_margin_J: float          # min over t of B - E_t - Ebar(x_t)
    margin_at_turnback_J: Optional[float]
    turnback_time: Optional[float]
    turnback_reach_m: Optional[float]
    collision_cause: Optional[str]
    replans: int
    solve_ms_mean: Optional[float]
    solve_ms_max: Optional[float]
    final_price: Optional[float] # the dual arm's terminal multiplier
    power_mismatch: float = 0.0
    lidar_seed: Optional[int] = None
    #: which certificate intercept the reserve used: 'script20_fit' or the
    #: 'LEGACY_PLACEHOLDER_24J'. A placeholder trial is measuring a
    #: placeholder and the record must say so.
    a_q: float = 24.0
    a_q_source: str = 'LEGACY_PLACEHOLDER_24J'
    #: The priced arms' energy price: the FIXED price the const_weight arm
    #: ran at (its arm identity in the sweep), or the dual arm's starting
    #: price. 0.0 for the hard-constraint arms.
    price0: float = 0.0

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

@dataclass
class ReachArm:
    """One comparison arm: what the supervisor hands to ReachMPC.solve each replan.

    ``turnback_rule`` (added after the 2026-09-12 sweeps): how the
    supervisor DECIDES to turn back. The hard-constraint arms turn back
    through the NLP itself -- a violating outbound plan IS the signal
    (that is the contribution). The priced arms have no reserve row, so
    without an explicit rule they ride to the wall and strand: measured
    across two sweeps (prices 0.25-8.0 mis-scaled, 25-400 correctly
    scaled), 80% violations at EVERY price with turn-backs firing at the
    positional stall, not on energy. A priced arm that cannot even try to
    comply is a straw man, so the priced arms get an explicit energy
    turn-back: leave when the remaining budget no longer covers the
    certified return cost -- the rule a real scalar-priced supervisor
    would implement -- while the NLP price stays their ONLY in-loop
    mechanism. The comparison then isolates what the terminal CONSTRAINT
    adds over an explicit scalar rule.
    """
    name: str
    use_constraint: bool = True          # the hard terminal reserve
    price0: float = 0.0                  # fixed price (const_weight)
    dual: bool = False                   # price adapts by ascent
    dual_eta: float = 0.02               # J of reserve error -> price step
    v_max_override: Optional[float] = None
    #: 'nlp' (constraint arms) | 'energy' (priced arms: budget vs
    #: certified return cost) | 'stall' (positional only -- kept only so
    #: the pre-fix sweeps remain interpretable; not a fair comparator)
    turnback_rule: str = 'nlp'
    #: 'nlp' (default: return legs are NLP plans) | 'geodesic' (hybrid,
    #: 2026-09-16: on turn-back, track the J-geodesic home with the
    #: certificate's own pi_fb gains -- the return-leg policy the
    #: threshold baseline and oracle use, attached to the constrained
    #: outbound. The measured motivation: the NLP return strands
    #: near-dock (10% docking) while the SAME tracker on the SAME
    #: geodesic docks 84-90% (oracle / threshold baseline) -- the
    #: outbound constraint is the contribution; the return leg is
    #: execution, and execution should use the efficient policy)
    return_policy: str = 'nlp'


REACH_ARM = ReachArm('reach')
REACH_HYBRID_ARM = ReachArm('reach_hybrid', return_policy='geodesic')
CONST_WEIGHT_ARM = ReachArm('const_weight', use_constraint=False, price0=1.0,
                            turnback_rule='energy')
DUAL_ARM = ReachArm('dual', use_constraint=False, dual=True, price0=0.0,
                    turnback_rule='energy')
VMAX_CLIP_ARM = ReachArm('vmax_clip', use_constraint=True, v_max_override=1.095445)

ARMS = {'reach': REACH_ARM, 'reach_hybrid': REACH_HYBRID_ARM,
        'const_weight': CONST_WEIGHT_ARM, 'dual': DUAL_ARM,
        'vmax_clip': VMAX_CLIP_ARM}


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class ReachSupervisor:
    """Two-phase mission control over one ReachMPC.

    ``command(state_6d, mission_t)`` is called once per PLANT step; the
    supervisor replans on its own clock and holds each plan node's command
    for ``plan_dt`` of plant time. The plan is executed open-loop between
    replans -- the REACH controller is a supervisory planner by design
    (a fixed horizon solves the tracking problem -- settled in the
    horizon-isolation gate -- so the real-time framing IS supervisory).
    """

    def __init__(self, scenario, spec, arm: ReachArm, config: dict,
                 N: int = 60, dt_plan: float = 0.5,
                 replan_every_s: float = 5.0):
        from mpc.reach_mpc import ReachMPC
        self.arm = arm
        self.spec = spec
        self.replan_every_s = float(replan_every_s)
        self.dock = np.asarray(scenario.dock
                               if getattr(scenario, 'dock', None) is not None
                               else scenario.goal, dtype=float)
        self.mpc = ReachMPC(scenario, spec, N=N, dt=dt_plan, config=config,
                            v_max_override=arm.v_max_override)
        self._scenario = scenario            # for the lazy routing field
        self._geo_return = None              # {'path', 'idx'} when active
        self.price = float(arm.price0)
        self.phase = 'outbound'
        self.turnback_t: Optional[float] = None   # mission time of the flip
        self.replans = 0
        self.solve_ms: List[float] = []
        self._plan_U: Optional[np.ndarray] = None
        self._plan_X: Optional[np.ndarray] = None      # for warm starts
        self._executed_nodes = 0                        # nodes flown of _plan_U
        self._plan_phase = None                         # phase _plan_U was built for
        self._last_replan_t = -math.inf
        self._steps_per_node = max(1, int(round(dt_plan / float(config.get('dt', 0.1)))))
        self._node_step = 0

    # -- reserve gauge -----------------------------------------------------

    def margin(self, pos, E) -> float:
        """B - E - Ebar(pos): the certified slack at a plant state."""
        return float(self.spec.B - E - self.mpc.Ebar_at(pos[0], pos[1]))

    # -- the loop interface -------------------------------------------------

    def command(self, state6, mission_t: float):
        """(v_cmd, omega_cmd) for this plant step, replanning as needed."""
        # Return phase and already home: STOP. The shutdown gate (on=0) is
        # the whole point of the P=0 mode -- with c5 = 3 W of idle draw a
        # robot that keeps replanning while sitting at the dock burns its
        # pack and STRANDS itself at home (measured 2026-09-12: 35 replans,
        # 220 J spent pacing a 0.6 m orbit around the dock). The mission
        # ends when the evaluator's docked-check sees it next tick.
        if (self.phase == 'return'
                and np.hypot(state6[0] - self.dock[0],
                             state6[1] - self.dock[1]) <= DOCK_TOL):
            return 0.0, 0.0
        if self._geo_return is not None:
            return self._geodesic_track_step(state6)
        if (self._plan_U is None
                or mission_t - self._last_replan_t >= self.replan_every_s):
            self._replan(state6, mission_t)
        if self._plan_U is None:
            # every solve so far failed: stop in place rather than command
            # from a plan that does not exist
            return 0.0, 0.0
        u = self._plan_U[:, min(self._executed_nodes, self._plan_U.shape[1] - 1)]
        self._node_step += 1
        if self._node_step >= self._steps_per_node:
            self._node_step = 0
            self._executed_nodes += 1
        return float(u[0]), float(u[1])

    def _warm(self, outbound: bool):
        """(X, U, executed) of the previous plan if it is a valid warm start.

        The previous solution is only a warm start for a solve of the SAME
        phase (an outbound arc is a terrible guess for a return arc), and
        only if the solver actually returned multipliers last time.
        """
        if (self._plan_X is None or self._plan_U is None
                or self._plan_phase != outbound):
            return None
        return self._plan_X, self._plan_U, self._executed_nodes

    #: pi_fb tracker constants -- IDENTICAL to the certificate rollout
    #: and the oracle (scripts/24_reach_oracle.py, which documents why
    #: La=0.3: pure pursuit cuts corners by up to La/2, and La=0.5 plus
    #: the +-5 cm obstacle perturbation consumed the routing buffer)
    _TRK_VSTAR = math.sqrt(3.0 / 2.5)
    _TRK_WMAX = math.pi
    _TRK_KP = 2.0
    _TRK_LA = 0.3
    _TRK_ROUTE_BUFFER = 0.35      # = ReachWeights.hard_clearance

    def _build_geodesic_return(self, state6) -> Optional[np.ndarray]:
        """The buffered J-geodesic from the current position to the dock.

        Built lazily ONCE per mission (at turn-back) on a routing field
        inflated by r_body + the execution buffer -- the same field the
        oracle routes on, so the two return legs are the same policy.
        """
        from planning.energy_dijkstra import build_field, geodesic_path
        terrain = getattr(self._scenario, 'terrain', None)
        field = build_field(
            self._scenario, terrain=terrain,
            r_body=0.25 + self._TRK_ROUTE_BUFFER)
        path = geodesic_path(field, (float(state6[0]), float(state6[1])))
        if path is None or len(path) < 2:
            return None
        return np.asarray(path, dtype=float)

    def _geodesic_track_step(self, state6):
        """(v, w) for one plant step of pi_fb pure pursuit on the return
        polyline; final approach heads straight for the dock (the
        oracle's rule)."""
        p = self._geo_return['path']
        x, y, th = float(state6[0]), float(state6[1]), float(state6[2])
        idx = self._geo_return['idx']
        while (idx < len(p) - 1
               and math.hypot(p[idx][0] - x, p[idx][1] - y) < self._TRK_LA):
            idx += 1
        self._geo_return['idx'] = idx
        tx, ty = p[idx]
        err = math.atan2(math.sin(math.atan2(ty - y, tx - x) - th),
                         math.cos(math.atan2(ty - y, tx - x) - th))
        w_cmd = max(-self._TRK_WMAX,
                    min(self._TRK_WMAX, self._TRK_KP * err))
        v_cmd = self._TRK_VSTAR * max(0.0, math.cos(err))
        if (idx >= len(p) - 1
                and math.hypot(x - self.dock[0], y - self.dock[1]) < 1.0):
            # polyline exhausted near the dock: head straight in
            err = math.atan2(
                math.sin(math.atan2(self.dock[1] - y, self.dock[0] - x) - th),
                math.cos(math.atan2(self.dock[1] - y, self.dock[0] - x) - th))
            w_cmd = max(-self._TRK_WMAX,
                        min(self._TRK_WMAX, self._TRK_KP * err))
            v_cmd = self._TRK_VSTAR * max(0.0, math.cos(err))
        return float(v_cmd), float(w_cmd)

    def _replan(self, state6, mission_t: float) -> None:
        t0 = time.perf_counter()
        outbound = self.phase == 'outbound'
        res = self._solve(state6, outbound, warm=self._warm(outbound))
        self._last_replan_t = mission_t

        if outbound:
            # Turn-back signals, by arm design:
            #   'nlp' (constraint arms): a valid plan whose reach_gain has
            #     stalled (< 0.5), a solver-FAILED plan, or a plan that
            #     VIOLATES the reserve -- the planner cannot buy progress
            #     inside what remains, which IS the turn-back. The third
            #     case must NOT keep executing the old outbound plan
            #     (measured: it drove the robot further out while every
            #     return plan was refused, stranding it with -116 J).
            #   'energy' (priced arms, 2026-09-12 fix): no reserve row
            #     exists, so the supervisor applies an explicit rule --
            #     leave when the remaining budget no longer covers the
            #     certified return cost. Without it the priced arms rode
            #     to the wall and stranded at EVERY swept price (two
            #     sweeps, 0.25-8.0 mis-scaled and 25-400 correctly
            #     scaled: 80% violations, turn-backs firing at the
            #     positional stall): a baseline that cannot even try to
            #     comply is a straw man, not a comparator.
            stalled = (res is not None
                       and (res.get('reach_gain') or 0.0) < 0.5)
            violating = (res is not None and res.get('_violates'))
            turn_now = stalled or violating or res is None
            if (not turn_now
                    and self.arm.turnback_rule == 'energy'):
                remaining = self.spec.B - float(state6[5])
                ebar_here = self.mpc.Ebar_at(float(state6[0]),
                                             float(state6[1]))
                turn_now = remaining < ebar_here
            if turn_now:
                self.phase = 'return'
                if self.turnback_t is None:
                    self.turnback_t = mission_t
                if self.arm.return_policy == 'geodesic':
                    # hybrid: hand the return leg to the pi_fb tracker on
                    # the buffered J-geodesic (no further NLP solves). If
                    # the geodesic is unavailable (position inside the
                    # inflation), fall back to the NLP return.
                    path = self._build_geodesic_return(state6)
                    if path is not None:
                        self._geo_return = {'path': path, 'idx': 0}
                        self._plan_U = None       # no plan to execute
                        self._plan_X = None
                        return
                res2 = self._solve(state6, outbound=False, warm=None)
                self._adopt(res2)
            else:
                self._adopt(res)
        else:
            # Return phase: adopt any solver-valid plan even if the reserve
            # is already unrecoverable -- the return attempt is the best
            # effort and the violation magnitude is recorded by the metrics.
            # Refusing here (the first draft) kept the OUTBOUND plan running
            # and guaranteed the stranding it was trying to avoid.
            self._adopt(res)

        if self.arm.dual:
            # Feedback w_p (the dual-ascent baseline): scalar dual ascent on the
            # reserve error. It can only raise the price AFTER the error has
            # appeared -- which is exactly its documented weakness, and the
            # thing the step-change experiment measures.
            err = -self.margin(state6[0:2], state6[5])   # >0 = over-committed
            self.price = max(0.0, self.price + self.arm.dual_eta * err)

    def _adopt(self, res) -> None:
        """Take a solver-valid plan as the executing reference."""
        if res is None:
            return
        self._plan_U = np.asarray(res['U'])
        self._plan_X = np.asarray(res['X'])       # warm-start source
        self._executed_nodes = 0                   # it starts now
        self._plan_phase = (self.phase == 'outbound')
        self._node_step = 0

    def _solve(self, state6, outbound: bool, warm=None):
        """One solve, timed; returns the result dict or None on solver failure.

        A solver FAILURE (infeasible / restoration failure / bad numbers)
        returns None and the previous plan continues -- the supervisor
        degrades to a longer open-loop leg rather than commanding a garbage
        arc. A solver-VALID plan whose reserve row is violated is still
        returned (with ``_violates = True``): the caller decides what it
        means -- in the outbound phase it is the turn-back signal; in the
        return phase it is the best-effort attempt whose violation the
        metrics record. Both classes are counted.

        A WARM solve that comes back violating gets one COLD retry before
        being reported: measured 2026-09-12, the shifted previous solution
        can park IPOPT in a worse basin than the geodesic guess (warm
        slack -5.0 J where the cold solve had +0.0), and mistaking that
        for "no certified progress" triggered a premature turn-back at
        2.0 m reach. The retry costs one solve only when it is needed.

        A warm solve that comes back STALLED with slack still on the
        table gets the same cold retry (added 2026-09-14, stall
        diagnostic): measured on map 0912/B268/seed 31001, the warm solve
        returned gain = -1.05 J with slack = +87.1 J -- a local optimum
        that leaves 87 J of certified margin unused -- while the COLD
        solve from the identical state found gain = +65.0 J, slack +0.08.
        The supervisor read the warm result as a stall and turned back
        with the margin in hand. The violation retry below cannot catch
        this (the plan does not violate), so the stall retry has its own
        gate: gain < 0.5 J while slack > STALL_RETRY_SLACK (10 J) on a
        constraint arm, outbound only. A stall at slack ~ 0 is a TRUE
        boundary stall (diagnostic cases 0915/0916: cold and 5000-iter
        solves reproduce it) and is NOT retried -- that is the honest
        price of certified compliance, not a solver artifact.
        """
        t0 = time.perf_counter()
        res = self.mpc.solve(np.asarray(state6, dtype=float), B=self.spec.B,
                             price=self.price, outbound=outbound,
                             use_constraint=self.arm.use_constraint,
                             warm=warm)
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.replans += 1
        bad = ('Infeasible_Problem_Detected', 'Restoration_Failed',
               'Invalid_Number_Detected', 'Error')
        # Acceptance is the PLAN's usability, not IPOPT's label -- the
        # documented philosophy (solver_opts.py: a linear-interpolant cost
        # can leave IPOPT reporting infeasibility while the returned
        # iterate is fine). Measured on the step-change map: the first
        # outbound solve came back 'Infeasible_Problem_Detected' with a
        # plan that reaches x=16.7 ON the reserve boundary -- and the
        # supervisor threw it away, flipped to return at t=0, and the
        # dock-stop froze the robot at home for the whole mission.
        #
        # Usability is FINITENESS ONLY. slack_J is computed against the
        # REAL B even when the constraint was made inert (B_p=1e9 in the
        # NLP), so a priced arm's unconstrained plan reports a very
        # negative slack simply because it ignored a row it was never
        # asked to respect -- that number is a SIGNAL for the constraint
        # arms, not an adoption gate, and gating on it froze the priced
        # arms at their dock (measured: first plan reaches 17.6 m, slack
        # -97 against a budget it doesn't carry).
        if res.get('return_status') in bad:
            usable = (np.all(np.isfinite(np.asarray(res['X'])))
                      and np.all(np.isfinite(np.asarray(res['U']))))
            if usable:
                self.relabelled_replans = getattr(self, 'relabelled_replans', 0) + 1
                logger.info("replan %s: solver said %s but the iterate is "
                            "finite -- adopting", self.arm.name,
                            res.get('return_status'))
            else:
                self.failed_replans = getattr(self, 'failed_replans', 0) + 1
                logger.warning("replan %s failed (%s); previous plan continues",
                               self.arm.name, res.get('return_status'))
                return None
        # _violates is meaningful ONLY for the constraint arms: it is the
        # turn-back signal inside the plan. The priced arms' slack is
        # informational (their turn-back is the supervisor's energy rule).
        res['_violates'] = bool(self.arm.use_constraint
                                and res.get('slack_J', 0.0) < -1.0)
        if res['_violates'] and warm is not None:
            # the warm basin is worse than nothing: one cold retry before
            # reporting the violation (see docstring)
            t1 = time.perf_counter()
            retry = self.mpc.solve(np.asarray(state6, dtype=float),
                                   B=self.spec.B, price=self.price,
                                   outbound=outbound,
                                   use_constraint=self.arm.use_constraint,
                                   warm=None)
            self.solve_ms.append((time.perf_counter() - t1) * 1e3)
            self.replans += 1
            if retry.get('return_status') not in bad and \
                    retry.get('slack_J', -1.0) >= res.get('slack_J', -1.0):
                retry['_violates'] = bool(
                    self.arm.use_constraint and retry.get('slack_J', 0.0) < -1.0)
                return retry
        # STALL retry (2026-09-14 diagnostic): a warm solve that reports
        # no progress while certified slack remains is a basin artifact,
        # not a boundary -- the cold solve from the same state found
        # +65 J where the warm one found -1 J (measured, map 0912/B268/
        # seed 31001). Gate on the slack so TRUE boundary stalls
        # (slack ~ 0, reproduced by cold and 5000-iter solves alike) are
        # not retried: they are the honest price of compliance.
        if (warm is not None and outbound and self.arm.use_constraint
                and not res['_violates']
                and (res.get('reach_gain') or 0.0) < 0.5
                and res.get('slack_J', 0.0) > STALL_RETRY_SLACK):
            t1 = time.perf_counter()
            retry = self.mpc.solve(np.asarray(state6, dtype=float),
                                   B=self.spec.B, price=self.price,
                                   outbound=True,
                                   use_constraint=True,
                                   warm=None)
            self.solve_ms.append((time.perf_counter() - t1) * 1e3)
            self.replans += 1
            self.stall_retries = getattr(self, 'stall_retries', 0) + 1
            retry_usable = (np.all(np.isfinite(np.asarray(retry['X'])))
                             and np.all(np.isfinite(np.asarray(retry['U']))))
            if retry_usable and (retry.get('reach_gain') or 0.0) > \
                    (res.get('reach_gain') or 0.0):
                retry['_violates'] = bool(retry.get('slack_J', 0.0) < -1.0)
                logger.info("replan %s: stall retry recovered %.1f J of "
                            "gain (warm had %.1f J, slack %.1f)",
                            self.arm.name, retry['reach_gain'],
                            res['reach_gain'], res['slack_J'])
                return retry
        if res['_violates']:
            self.violating_replans = getattr(self, 'violating_replans', 0) + 1
        return res


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------

def run_reach_trial(scenario, arm: ReachArm, B: float, seed: int, *,
                    config: dict,
                    N: int = 60, dt_plan: float = 0.5,
                    replan_every_s: float = 5.0,
                    t_mission_s: float = 240.0,
                    power_mismatch: float = 0.0,
                    actuation_noise: Optional[tuple] = None,
                    mission_E0: float = 0.0,
                    a_q: Optional[float] = None) -> ReachTrialResult:
    """One closed-loop out-and-back mission. Fully seeded from *seed*.

    ``a_q``: the map's fitted certificate intercept (script 20's coverage
    construction on the J-field). When omitted, the supervisor falls back
    to the 24 J corridor placeholder AND the trial record says so -- a
    campaign running the placeholder is measuring a placeholder, and the
    record must not hide that.
    """
    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer
    from mpc.reach_mpc import ReachSpec
    from mpc.energy_model import load_default_coefficients
    from planning.energy_dijkstra import build_field

    # env config follows EP2Evaluator's convention: the robot block plus dt
    # and the energy coefficients (the plant is the same Pioneer 3-DX).
    env_cfg = dict(config.get('robot', {}))
    env_cfg['dt'] = float(config.get('dt', 0.1))
    env_cfg.setdefault('energy_coeffs', load_default_coefficients())
    env = PioneerEnv(env_cfg)

    randomizer = TrialRandomizer()
    perturbed, v0, w0 = randomizer.randomize_trial(scenario, seed=seed)
    env.set_scenario(perturbed.obstacles, perturbed.bounds,
                     terrain=scenario.terrain)
    if power_mismatch:
        env.set_power_mismatch(power_mismatch)
    if actuation_noise is not None:
        env.set_actuation_noise(actuation_noise[0], actuation_noise[1],
                                seed=seed + NOISE_SEED_OFFSET)
    lidar_seed = seed + LIDAR_SEED_OFFSET
    env.reset(perturbed.start, lidar_seed=lidar_seed)
    env._v, env._omega = v0, w0

    # the certificate intercept: fitted per-map by script 20 (coverage
    # construction on the J-field) when the caller passes a_q; the 24.0 J
    # fallback is the corridor LEGACY placeholder and is recorded as such
    # in the result -- measured 2026-09-12, the fitted values are
    # 28.6-54.0 J on the generated maps, so the placeholder under-reserves
    # on every one of them by up to 2.25x.
    a_q_used = float(a_q) if a_q is not None else 24.0
    a_q_source = ('script20_fit' if a_q is not None
                  else 'LEGACY_PLACEHOLDER_24J')
    spec = ReachSpec(B=float(B), certificate='affine', a_q=a_q_used, lam=1.0)

    sup = ReachSupervisor(scenario, spec, arm, config, N=N, dt_plan=dt_plan,
                          replan_every_s=replan_every_s)
    # metric field for metres-of-reach reporting (the J field prices, the
    # metric field reports)
    metric_field = build_field(scenario)

    dock = np.asarray(scenario.dock, dtype=float)
    dt = float(config.get('dt', 0.1))
    reach_m, reach_J = 0.0, 0.0
    min_margin = math.inf
    turnback_time = turnback_reach = turnback_margin = None
    outcome, collision_cause = 'timeout', None
    t = 0.0
    E0 = float(mission_E0)
    max_steps = int(round(t_mission_s / dt))

    for _ in range(max_steps):
        st = env.get_state_vector()
        E = E0 + float(st[5])
        pos = st[0:2]
        if env.check_collision():
            from evaluation.ep2_evaluator import EP2Evaluator
            collision_cause = EP2Evaluator._collision_cause(env, scenario)
            outcome = 'collision'
            break
        if float(np.hypot(*(pos - dock))) <= DOCK_TOL and reach_m > MIN_OUTBOUND:
            outcome = 'docked'
            break
        if E >= spec.B:
            outcome = 'stranded'
            break

        margin = sup.margin(pos, E)
        min_margin = min(min_margin, margin)
        dm = metric_field.dstar(float(pos[0]), float(pos[1]))
        if math.isfinite(dm):
            reach_m = max(reach_m, dm)
        dj = sup.mpc.field.dstar(float(pos[0]), float(pos[1]))
        if math.isfinite(dj):
            reach_J = max(reach_J, dj)
        if turnback_time is None and sup.phase == 'return':
            turnback_time, turnback_reach, turnback_margin = t, reach_m, margin

        v_cmd, w_cmd = sup.command(st, t)
        env.step(v_cmd, w_cmd)
        t += dt

    E_final = E0 + float(env.get_state_vector()[5])
    if not math.isfinite(min_margin):
        min_margin = sup.margin(env.get_state_vector()[0:2], E_final)
    return ReachTrialResult(
        map_name=scenario.name, arm=arm.name, B=float(B), seed=seed,
        outcome=outcome, returned=(outcome == 'docked'),
        reach_m=(reach_m if reach_m > 0 else None),
        reach_J=(reach_J if reach_J > 0 else None),
        energy_spent=E_final, mission_time=t,
        min_margin_J=float(min_margin),
        margin_at_turnback_J=turnback_margin,
        turnback_time=turnback_time, turnback_reach_m=turnback_reach,
        collision_cause=collision_cause,
        replans=sup.replans,
        solve_ms_mean=(float(np.mean(sup.solve_ms)) if sup.solve_ms else None),
        solve_ms_max=(float(np.max(sup.solve_ms)) if sup.solve_ms else None),
        final_price=(sup.price if arm.dual else None),
        power_mismatch=float(power_mismatch),
        lidar_seed=lidar_seed,
        a_q=a_q_used,
        a_q_source=a_q_source,
        price0=float(arm.price0),
    )


# ---------------------------------------------------------------------------
# Campaign sink
# ---------------------------------------------------------------------------

def reach_trial_sink(run_dir: str, tag: str) -> str:
    """Open (append) the per-trial JSONL sink, mirroring EP2Evaluator's."""
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, f"reach_trials_{tag}.jsonl")


def emit_reach_trial(path: str, result: ReachTrialResult) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result.to_record()) + "\n")
