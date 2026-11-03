"""Baseline controllers for EP²-MPC comparison.

Includes reactive potential-field baselines and a real CasADi Standard MPC
baseline. All baselines run in the same environment with motor lag dynamics,
ensuring a fair comparison with EP²-MPC.
"""
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import sys
import os


from evaluation.ep2_evaluator import EvalResult, TrialResult
from env.pioneer_env import PioneerEnv
from env.scenarios import get_scenario, ScenarioConfig
from env.trial_randomizer import TrialRandomizer
from mpc.energy_model import load_default_coefficients, compute_power

logger = logging.getLogger(__name__)

DT = 0.1

#: Parity fix, added 2026-09-11: EP2Evaluator seeds the LiDAR with
#: ``trial_seed + LIDAR_SEED_OFFSET`` so its trials are bit-reproducible, but
#: the two baseline runners below called ``env.reset(start)`` with no
#: ``lidar_seed`` -- only the geometry randomisation was seeded. Two
#: consequences, both measured on run_20260910T174433Z: baseline energies
#: differed between otherwise identical runs (mean ~0, per-trial +-0.5-3 J --
#: the EP2 arm was bit-identical across the same comparison), and a "paired"
#: trial did not see the same scan noise on the two arms. Using the SAME
#: offset as the evaluator makes the baselines reproducible and the pairing
#: exact, trial for trial.
LIDAR_SEED_OFFSET = 1_000_003
MAX_STEPS = 500
SCENARIO_MPC_HORIZONS = {'corridor': 12, 's_curve': 20, 'chicane': 14}

# COMPARATOR REPAIR. Two changes, both of which the rejected paper's baseline needed:
#
#   d_safe: 0.23 / 0.245 / 0.245  ->  0.30 across the board.
#     The old values sat BELOW the 0.25 m collision radius, and the baseline
#     measured clearance centre-to-surface, so its obstacle penalty was
#     identically zero at every collision-free state. It drove into the first
#     obstacle on s_curve at (4.580, 2.381) after 4.0 s and on chicane at
#     (6.212, 2.476) after 5.2 s, with failure time near-invariant to N
#     (4.186 s at N=20 vs 4.256 s at N=24). That is not a lookahead failure, and
#     Table I's "fixed horizons fail" conclusion rested on it. 0.30 is inside the
#     [0.26, 0.38] box EP2 itself uses, so the arms are now matched.
#
#   w_v deleted.
#     `w_v * (v - v_ref)^2` is a velocity-tracking forcing term. EP2-MPC has no
#     velocity reference at all, so this term is pure handicap: it predicts a
#     1.23% energy penalty on its own, which is larger than the 0.88% corridor
#     advantage the paper claimed for EP2. Keeping it would mean the headline
#     margin was smaller than a formulation asymmetry we introduced ourselves.
# FITTED VALUES, adopted 2026-09-06 from scripts/17_tune_baselines.py --arm std_mpc.
# Declared box 4 x 5 x 6 = 120 points, 4,320 trials. w_obs and w_goal were swept as
# MULTIPLIERS on the values below so the relative tuning between maps was preserved
# and only their scale was fitted; d_safe was swept over EP2's OWN search bound,
# PARAM_BOUNDS['d_safe'] = (0.26, 0.38), so neither arm is held to a stricter margin
# than the other. Winner: w_obs x0.25, w_goal x4.0, d_safe 0.26. Validated 250/map:
#
#   corridor  1.000  127.08 J
#   s_curve   1.000  137.19 J      <- was 0.00 with walls active and d_safe 0.30
#   chicane   1.000  133.56 J
#
# Two things the grid shows that a single winner hides:
#   * ALL 18 fully-successful points have d_safe = 0.26. Not one point at 0.28,
#     0.30, 0.34 or 0.38 solves all three maps -- the margin is the entire binding
#     constraint. 0.26 is on the box edge and STAYS there: it is the floor of EP2's
#     own search bound, so it grants the comparator exactly the freedom EP2 had,
#     and going lower would give the baseline a margin EP2 was never allowed.
#   * w_obs_scale spans 8x across those 18 points while energy spans 3.1%, so the
#     obstacle/wall weight is nearly inert once the margin is right.
#
# Why w_goal had to be fitted at all: EVERY goal on EVERY map sits INSIDE the wall
# margin. All three are at x = 19.5 in a 20 m arena, 0.50 m from the right wall,
# while d_safe + r_body demands 0.51-0.63 m of centre clearance. At d_safe = 0.30
# this arm threaded the whole s_curve with NO collision and then parked at
# (19.13, 0.73) for the full 60 s timeout, held off by its own wall penalty, because
# w_goal = 11-14 against w_obs = 420-560 means clearance outweighed the goal ~40x.
# Reporting that as "the standard MPC cannot solve s_curve" would have been a
# fabricated comparator failure -- the goal placement is the cause, not the
# controller.
SCENARIO_STD_MPC_WEIGHTS = {
    'corridor': {'w_obs': 105.0, 'w_goal': 44.0, 'w_du': 0.05, 'd_safe': 0.26},
    's_curve':  {'w_obs': 135.0, 'w_goal': 56.0, 'w_du': 0.08, 'd_safe': 0.26},
    'chicane':  {'w_obs': 140.0, 'w_goal': 56.0, 'w_du': 0.09, 'd_safe': 0.26},
}
#: Pre-tuning values, kept so the tuning-parity table can quote both.
SCENARIO_STD_MPC_WEIGHTS_PRE_WALLS = {
    'corridor': {'w_obs': 420.0, 'w_goal': 11.0, 'w_du': 0.05, 'd_safe': 0.30},
    's_curve':  {'w_obs': 540.0, 'w_goal': 14.0, 'w_du': 0.08, 'd_safe': 0.30},
    'chicane':  {'w_obs': 560.0, 'w_goal': 14.0, 'w_du': 0.09, 'd_safe': 0.30},
}

# The slowdown governor is deleted with w_v: it existed only to keep the
# velocity-tracking term from driving the robot into obstacles at v_ref, and with
# no v_ref there is nothing to govern. Retained as names so a reader grepping the
# old constants finds this note instead of silence.
_DELETED_SLOWDOWN_GOVERNOR = (
    'STD_MPC_SLOWDOWN_DISTANCE, STD_MPC_SLOWDOWN_OFFSET, '
    'STD_MPC_SLOWDOWN_MIN_SCALE, STD_MPC_SLOWDOWN_DENOM, '
    'STD_MPC_GOAL_NEAR_DIST, STD_MPC_GOAL_NEAR_MIN_VREF -- removed with w_v. '
    'See git show 0a3c254:evaluation/baselines.py'
)

# APF gains, RE-TUNED against the corrected clearance convention.
#
# The geometry repair added ``r_body`` to the potential field's clearance, so obstacles are now
# measured body-to-surface as they are in the MPC. The pre-repair gains (2.2, 2.2)
# were fitted to the old centre-to-surface convention, and inflating every
# obstacle by 0.25 m without re-fitting them turned a working s_curve slalom into
# a guaranteed crash (s_curve 1.00 -> 0.00). Shipping that as "the corrected
# baseline" would repeat the exact error that killed the original submission -- a
# comparator crippled by its own geometry -- with the sign flipped.
#
# Fitted by scripts/15_tune_apf_baseline.py over a declared 7x5 box
# (k_rep in 1.0..7.0, d_influence in 1.5..5.0), selection rule fixed in advance:
# max mean success across the three maps, ties broken by lowest mean energy over
# solved maps only. 6 of 35 points tie at 2/3 maps; this one has the lowest
# energy of the six. Validated at 50 trials/map:
#
#            tuned (1.0, 2.2)        untuned (2.2, 2.2)
#   corridor   1.00   149.30 J         1.00   165.00 J
#   s_curve    1.00   171.56 J         0.00    52.12 J
#   chicane    0.00    49.04 J         0.00    47.83 J
#
# RETRACTED 2026-09-06. This block used to conclude: "chicane is 0.00 at EVERY point
# of the grid ... a property of a reactive controller in a chicane (local minimum),
# not a tuning artefact, and it must be reported as such rather than tuned away."
#
# That was wrong, and wrong in the direction that flatters EP2-MPC. The chicane
# failures were WALL collisions -- all 250 ended at y = 3.750-3.796 in an arena
# bounded at y = 4.0, attributed 'wall' by EP2Evaluator._collision_cause -- and this
# force law had no wall term at all while EP2-MPC has carried four wall terms all
# along. It was
# our missing term, not a property of reactive control. With the wall term active and
# the gains re-fitted over a declared box, the APF reaches 1.000 success on ALL THREE
# maps (validated 250 trials/map).
#
# The lesson generalises and is the reason scripts/17_tune_baselines.py exists: every
# geometry change invalidates every gain fitted before it, and an arm that looks
# broken is usually untuned.
# FITTED VALUES, adopted 2026-09-06 from scripts/17_tune_baselines.py --arm apf.
# Declared box 10 x 5 x 7 = 350 points (k_rep 0.02..4.0, d_influence 1.0..4.0,
# d_wall_influence 0.20..1.20), 42,000 trials, rule fixed in advance: max mean
# success across all three maps, ties broken by lowest mean energy over the maps
# reaching 100% only. 129 of 350 points reach 1.000 on all three maps, so the good
# region is broad rather than a knife-edge; this point has the lowest energy of the
# 129. Validated at 50 trials x 5 seeds = 250 per map:
#
#   corridor  1.000  135.22 J
#   s_curve   1.000  158.55 J
#   chicane   1.000  157.58 J      <- was 0.00 before the wall term existed
#
# k_rep = 0.02 is not a degenerate field: the repulsion is ~12% of goal attraction
# at 0.5 m clearance and 2.3x at 0.2 m, i.e. a tight-hugging aggressive APF, which
# is exactly why it is cheap. Energy falls monotonically toward low k_rep
# (150.63 J at 0.02 vs 262.33 J at 1.5) while the count of fully-successful points
# rises (2 -> 35), so weak short-range repulsion is both cheaper and more reliable
# once the wall term supplies containment.
#
# The first sweep's winner sat on the box's lower edge (k_rep = 0.25 of 0.25..4.0)
# and was DISCARDED rather than adopted: a boundary winner is evidence the box was
# placed wrong, not a fitted optimum. It is kept as
# runs/<RUN_ID>/tuning/apf_gain_sweep.BOX1_EDGE_WINNER.json so the widening is
# auditable.
APF_K_REP = 0.02
APF_D_INFLUENCE = 2.2

# Wall influence is a SEPARATE distance from the obstacle one, and much smaller.
#
# Reusing d_influence = 2.2 m for the walls looks like parsimony and is physically
# wrong: an obstacle is a 0.3-1.7 m disc, a wall is an infinite plane, so the same
# influence radius integrates to a vastly larger force. In a 20x4 m arena, 2.2 m
# reaches 2.2 m in from each of the top and bottom walls -- i.e. the whole arena,
# twice over -- so the field never releases and the robot cannot traverse. Measured
# with walls at d_influence = 2.2: every map 0.00 success at 543-598 J, which is
# ~4x a successful run's energy, i.e. timeout rather than collision.
#
# EP2-MPC's wall term is a hinge that is exactly zero beyond d_safe of the surface
# (mpc/ep2_mpc.py: smooth_hinge_sq(d_safe - (pos - wall - R_BODY))), so the parity
# target is a thin margin, not an arena-wide field. Fitted by
# scripts/15_tune_apf_baseline.py alongside k_rep.
APF_D_WALL_INFLUENCE = 1.20
APF_GAINS_PRE_C2 = (2.2, 2.2)  # kept so the tuning-parity table can quote both
APF_TUNING_RECORD = 'runs/<RUN_ID>/tuning/apf_gain_sweep.json'


class BaseController(ABC):
    """One interface for every baseline arm.

    ``robot`` is the full ``config['robot']`` dict and is REQUIRED. It is read
    with ``[]`` rather than ``.get(..., default)`` on purpose: the arms used to
    carry their own ``v_max=1.0, omega_max=2.5`` literals against the MPC's
    1.2 / 3.14159, so every APF arm was speed-handicapped by 17% in linear and
    20% in angular authority before the comparison began. A missing key must now
    raise rather than silently reinstate a handicap.
    """

    def __init__(self, robot: dict, dt: float = DT):
        if not isinstance(robot, dict) or not robot:
            raise ValueError(
                "BaseController requires config['robot']; passing None or {} is "
                "how the v_max=1.0 / omega_max=2.5 handicap got in")
        self.robot = robot
        self.dt = float(dt)
        # KeyError here is the intended behaviour, not a bug to guard.
        self.v_max = float(robot['max_v'])
        self.omega_max = float(robot['max_omega'])
        self.r_body = float(robot['collision_radius'])

    @abstractmethod
    def reset(self):
        """Reset controller state for a new trial."""

    @abstractmethod
    def step(self, state, goal, obstacles, d_obs, scan, bounds=None) -> tuple:
        """Returns (v_cmd, omega_cmd, solve_ms).

        ``bounds`` is the arena (x_min, y_min, x_max, y_max). It is part of the
        interface because a controller that cannot see the walls cannot avoid them,
        and EP2-MPC can.
        """


def _potential_field_command(
    state, goal, obstacles, d_obs,
    v_max, omega_max, r_body,
    k_rep=APF_K_REP, d_influence=APF_D_INFLUENCE, energy_scale=1.0,
    tangent_gain=0.0, turn_bias=1.0,
    min_safe_speed=0.05, omega_gain=3.0,
    bounds=None, d_wall_influence=None,
):
    """Compute (v_cmd, omega_cmd, solve_ms) via potential field.

    ``r_body`` is explicit: clearance is measured body-to-surface, as
    it is in the MPC, instead of centre-to-surface.

    ``bounds`` is the arena ``(x_min, y_min, x_max, y_max)``. PARITY FIX:
    this force law had NO wall term, while EP2-MPC carries four (mpc/ep2_mpc.py,
    ``smooth_hinge_sq`` with ``R_BODY``). The consequence was measurable and total:
    all 250 chicane APF trials ended at y = 3.750-3.796 in an arena bounded at
    y = 4.0 -- inside the 0.25 m body radius of the top wall, in a 5 cm band, at
    x = 6.3-6.7 where the robot swerves past the second obstacle. Attributed by
    ``EP2Evaluator._collision_cause``: 'wall', every trial. Reporting that 0.00 as
    a property of reactive control would have been reporting our own missing term.

    The wall repulsion deliberately reuses ``k_rep`` and ``d_influence`` rather
    than introducing separate wall gains: a second tuning knob on the comparator
    is a second opportunity to hand-tune it into or out of a result. Each wall is
    a plane, clearance measured body-to-surface exactly as for obstacles.
    """
    t0 = time.time()
    x, y, theta = state['x'], state['y'], state['theta']

    dx_g = goal[0] - x
    dy_g = goal[1] - y
    dist_goal = np.hypot(dx_g, dy_g)
    if dist_goal < 0.05:
        return 0.0, 0.0, (time.time() - t0) * 1000

    fx, fy = dx_g / dist_goal, dy_g / dist_goal

    closest = None
    for obs in obstacles:
        ox, oy = obs[0], obs[1]
        r_obs = obs[2] if len(obs) >= 3 else 0.3
        ddx, ddy = x - ox, y - oy
        dist_center = np.hypot(ddx, ddy)
        d_surface = max(dist_center - r_obs - r_body, 0.01)
        if closest is None or d_surface < closest[0]:
            closest = (d_surface, ddx, ddy, dist_center)
        if d_surface < d_influence:
            mag = k_rep * (1.0 / d_surface - 1.0 / d_influence) / d_surface ** 2
            norm = dist_center + 1e-6
            fx += mag * ddx / norm
            fy += mag * ddy / norm

    # Walls, same law and same gains as obstacles. Each of the four planes
    # contributes a repulsion along its inward normal.
    if bounds is not None:
        d_wi = (APF_D_WALL_INFLUENCE if d_wall_influence is None
                else float(d_wall_influence))
        x_min, y_min, x_max, y_max = (float(b) for b in bounds)
        for d_wall, nx, ny in ((x - x_min, 1.0, 0.0),      # left,  push +x
                               (x_max - x, -1.0, 0.0),     # right, push -x
                               (y - y_min, 0.0, 1.0),      # bottom, push +y
                               (y_max - y, 0.0, -1.0)):    # top,   push -y
            d_surface = max(d_wall - r_body, 0.01)
            if d_surface < d_wi:
                mag = k_rep * (1.0 / d_surface - 1.0 / d_wi) / d_surface ** 2
                fx += mag * nx
                fy += mag * ny
            if closest is None or d_surface < closest[0]:
                # Keep the tangent-escape logic aware of walls too, otherwise it
                # slides along an obstacle straight into one.
                closest = (d_surface, nx * d_surface, ny * d_surface,
                           d_surface + r_body)

    if tangent_gain > 0.0 and closest is not None:
        d_surface, ddx, ddy, dist_center = closest
        if d_surface < d_influence:
            norm = dist_center + 1e-6
            tx, ty = -ddy / norm, ddx / norm
            # Keep the tangential push consistent with turn_bias while choosing
            # the side that still projects toward the goal direction.
            bias = 1.0 if turn_bias >= 0.0 else -1.0
            if (tx * dx_g + ty * dy_g) * bias < 0.0:
                bias *= -1.0
            ramp = max(0.0, (d_influence - d_surface) / d_influence)
            fx += bias * tangent_gain * ramp * tx
            fy += bias * tangent_gain * ramp * ty

    target_angle = np.arctan2(fy, fx)
    dtheta = np.arctan2(np.sin(target_angle - theta),
                        np.cos(target_angle - theta))

    v_desired = min(v_max, dist_goal) * energy_scale
    if d_obs < 0.8:
        v_safe = max(min_safe_speed, (d_obs - r_body) * v_max / 0.8)
        v_desired = min(v_desired, v_safe)
    alignment = max(0.0, np.cos(dtheta))
    v_cmd = float(np.clip(v_desired * alignment, 0.0, v_max))
    omega_cmd = float(np.clip(dtheta * omega_gain, -omega_max, omega_max))
    return v_cmd, omega_cmd, (time.time() - t0) * 1000


# ---------------------------------------------------------------------------
# energy_aware_apf and reactive_standard are DELETED.
#
# `energy_aware_apf` was `reactive_standard` with its speed scaled by
# max(0.2, 1 - 0.35) = 0.65 and nothing else changed, i.e. a strictly-worse
# instance of the same controller: same forces, same gains, same geometry, just
# slower. Reporting it as a separate arm and then quoting "21.8% better than the
# best baseline" against it is comparing against a handicapped copy of another
# arm. The abstract's 38.2% was measured against this one.
#
# `reactive_standard` goes too: with the v_max handicap removed it is
# `APFBaseline` without the stuck-escape logic, so it adds no distinct behaviour
# and its name invited the reading that it was an MPC. APFBaseline is the one
# reactive arm that survives.
#
# Recover either from git history if a reviewer asks:
#   git show 0a3c254:evaluation/baselines.py
# ---------------------------------------------------------------------------


class APFBaseline(BaseController):
    """Pure Artificial Potential Field (reactive, no prediction)."""
    STUCK_DISTANCE_OBS_THRESHOLD = 1.4
    STUCK_PROGRESS_THRESHOLD = 0.015  # meters of goal-distance improvement per step
    STUCK_STEPS_FLIP = 6
    TANGENT_NEAR_THRESH = 0.9
    TANGENT_GAIN_NEAR = 1.1
    TANGENT_GAIN_FAR = 0.8

    def __init__(self, robot: dict, dt: float = DT,
                 k_rep=APF_K_REP, d_influence=APF_D_INFLUENCE,
                 d_wall_influence=APF_D_WALL_INFLUENCE):
        super().__init__(robot, dt)
        self.k_rep, self.d_influence = k_rep, d_influence
        self.d_wall_influence = d_wall_influence
        self._turn_bias = 1.0
        self._prev_dist_goal: Optional[float] = None
        self._stuck_steps = 0

    def reset(self):
        """Reset controller state (no-op for stateless controllers)."""
        self._turn_bias = 1.0
        self._prev_dist_goal = None
        self._stuck_steps = 0

    def step(self, state, goal, obstacles, d_obs, scan, bounds=None):
        """Return (v_cmd, omega_cmd, solve_ms) via artificial potential field."""
        dist_goal = float(np.hypot(goal[0] - state['x'], goal[1] - state['y']))
        if self._prev_dist_goal is not None:
            progress = self._prev_dist_goal - dist_goal
            if d_obs < self.STUCK_DISTANCE_OBS_THRESHOLD and progress < self.STUCK_PROGRESS_THRESHOLD:
                self._stuck_steps += 1
            else:
                self._stuck_steps = max(0, self._stuck_steps - 1)
        self._prev_dist_goal = dist_goal
        if self._stuck_steps >= self.STUCK_STEPS_FLIP:
            self._turn_bias *= -1.0
            self._stuck_steps = 0

        tangent_gain = self.TANGENT_GAIN_NEAR if d_obs < self.TANGENT_NEAR_THRESH else self.TANGENT_GAIN_FAR
        return _potential_field_command(state, goal, obstacles, d_obs,
                                        self.v_max, self.omega_max, self.r_body,
                                        k_rep=self.k_rep,
                                        d_influence=self.d_influence,
                                        tangent_gain=tangent_gain,
                                        turn_bias=self._turn_bias,
                                        min_safe_speed=0.0,
                                        omega_gain=3.6,
                                        bounds=bounds,
                                        d_wall_influence=self.d_wall_influence)


def _ep2_collision_cause(env, scenario):
    """EP2Evaluator._collision_cause, imported lazily to keep the mpc chain out of
    module import time."""
    from evaluation.ep2_evaluator import EP2Evaluator
    try:
        return EP2Evaluator._collision_cause(env, scenario)
    except Exception as exc:            # never let attribution kill a trial
        logger.debug("collision attribution failed: %s", exc)
        return 'unknown'


def _emit_baseline_trial(run_dir, tag, rec, **context):
    """Append one JSONL record per baseline trial.

    Only ``EP2Evaluator`` had a trial sink, so success-conditioned energy could be
    recomputed for EP2 but not for the arms it is compared against -- the same gap
    the EP2 sink closes, on the other side of the comparison.
    """
    if not run_dir or not tag:
        return
    import json as _json
    from dataclasses import asdict as _asdict
    os.makedirs(run_dir, exist_ok=True)
    payload = _asdict(rec)
    payload.update(context)
    with open(os.path.join(run_dir, f'trials_{tag}.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(_json.dumps(payload, default=float) + '\n')


def run_baseline(
    controller: BaseController,
    scenario: ScenarioConfig,
    n_trials: int = 50,
    config: Optional[Dict] = None,
    master_seed: int = 0,
    run_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> EvalResult:
    """Run baseline controller for n_trials and aggregate results.

    Args:
        master_seed: base seed for trial randomisation. Trial k uses
                     seed = master_seed + k, matching EP2Evaluator's protocol
                     so paired Wilcoxon tests are valid.
    """
    config = config or {}
    dt = config.get('dt', DT)
    timeout = config.get('evaluation', {}).get('timeout_per_trial', 30.0)
    max_steps = int(timeout / dt)
    randomizer = TrialRandomizer()
    env = PioneerEnv(config)
    results = []

    for trial in range(n_trials):
        perturbed, v0, omega0 = randomizer.randomize_trial(
            scenario, seed=master_seed + trial)
        env.set_scenario(perturbed.obstacles, perturbed.bounds)
        # Same LiDAR seed derivation as EP2Evaluator, so this
        # trial is bit-reproducible and sees exactly the scan noise the EP2
        # arm sees on the same trial index.
        env.reset(perturbed.start, lidar_seed=master_seed + trial + LIDAR_SEED_OFFSET)
        env._v, env._omega = v0, omega0
        controller.reset()

        solver_times = []
        success = collision = False
        state = env.get_state()

        for step in range(max_steps):
            scan = env.get_lidar_scan()
            d_obs = float(np.min(scan))
            try:
                v_cmd, omega_cmd, solve_ms = controller.step(
                    state, perturbed.goal, perturbed.obstacles, d_obs, scan,
                    bounds=perturbed.bounds)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning("Baseline controller fallback to zero command at trial=%d step=%d: %s", trial, step, e)
                v_cmd, omega_cmd, solve_ms = 0.0, 0.0, 0.0
            solver_times.append(solve_ms)
            state, _, _, _ = env.step(v_cmd, omega_cmd)
            if env.check_collision():
                collision = True
                break
            if env.is_goal_reached(perturbed.goal, tolerance=0.2):
                success = True
                break

        final = env.get_state()
        _tr = TrialResult(
            energy=final['cumulative_energy'],
            time=(step + 1) * dt,
            solver_ms=float(np.mean(solver_times)) if solver_times else 0.0,
            success=success,
            collision=collision,
            path_distance=final.get('path_distance', 0.0),
            n_steps=step + 1,
            solver_ms_all=[float(x) for x in solver_times],
            terminal_pose=[float(final['x']), float(final['y']),
                           float(final['theta'])],
            seed=int(master_seed + trial),
            # Parity: EP2Evaluator recorded WHY a trial crashed and the
            # baselines did not, so the APF's 250 chicane failures could not be
            # split into obstacle-vs-wall from the data. Reusing EP2Evaluator's own
            # static classifier rather than reimplementing it -- the point is that
            # both arms attribute a collision identically.
            collision_cause=(_ep2_collision_cause(env, perturbed)
                             if collision else None),
        )
        results.append(_tr)
        _emit_baseline_trial(run_dir, tag, _tr, trial_index=trial,
                             master_seed=master_seed)

    energies = np.array([r.energy for r in results])
    times = np.array([r.time for r in results])
    solver_ms = np.array([r.solver_ms for r in results])
    successes = np.array([r.success for r in results], dtype=float)
    dists = np.array([r.path_distance for r in results])

    return EvalResult(
        energy_mean=float(np.mean(energies)),
        energy_std=float(np.std(energies)),
        time_mean=float(np.mean(times)),
        time_std=float(np.std(times)),
        solver_ms_mean=float(np.mean(solver_ms)),
        solver_ms_std=float(np.std(solver_ms)),
        success_rate=float(np.mean(successes)),
        path_distance_mean=float(np.mean(dists)),
        path_distance_std=float(np.std(dists)),
        n_trials=n_trials,
        trial_results=results,
    )


def run_standard_mpc_casadi_baseline(
    scenario_name: str,
    n_trials: int = 50,
    config: Optional[Dict] = None,
    v_ref: float = 0.9,
    master_seed: int = 0,
    N: Optional[int] = None,
    R_v: float = 0.0,
    R_w: float = 0.0,
    run_dir: Optional[str] = None,
    tag: Optional[str] = None,
    weight_overrides: Optional[Dict[str, float]] = None,
) -> EvalResult:
    """Run StandardMPCCasadi on a scenario and return aggregated EvalResult.

    This wrapper handles the signature difference between StandardMPCCasadi.solve()
    (which takes x0, goal, obstacles, v_ref) and the general EP2Evaluator trial loop.
    The 6D state vector from PioneerEnv is trimmed to 5D before being passed to the
    solver, consistent with EP2MPC5State.

    Args:
        scenario_name: One of 'corridor', 's_curve', 'chicane'.
        n_trials: Number of randomized trials to run.
        config: Full config dict (robot, mpc sub-dicts are extracted).
        v_ref: Reference speed for the MPC velocity-tracking cost.

    Returns:
        EvalResult aggregating energy, time, solver latency, and success rate.
    """
    from mpc.standard_mpc_casadi import StandardMPCCasadi, StandardMPCWeights

    config = config or {}
    robot_cfg = config.get('robot', {})
    mpc_cfg = config.get('mpc', {})
    dt = mpc_cfg.get('dt', DT)
    timeout = config.get('evaluation', {}).get('timeout_per_trial', 30.0)
    max_steps = int(timeout / dt)
    

    scenario = get_scenario(scenario_name)
    randomizer = TrialRandomizer()
    env = PioneerEnv(robot_cfg)
    scenario_weights_cfg = dict(SCENARIO_STD_MPC_WEIGHTS.get(scenario_name) or {})
    if weight_overrides:
        # Used by scripts/17_tune_baselines.py to fit w_obs and d_safe against the
        # wall-active geometry. Unknown keys must raise rather than be dropped: a
        # silently ignored override reports the sweep of a parameter that never
        # changed.
        unknown = set(weight_overrides) - set(StandardMPCWeights().__dict__)
        if unknown:
            raise ValueError(f"unknown StandardMPCWeights fields: {sorted(unknown)}")
        scenario_weights_cfg.update(weight_overrides)
    if scenario_weights_cfg:
        scenario_weights = StandardMPCWeights(**scenario_weights_cfg, R_v=R_v, R_w=R_w)
    else:
        scenario_weights = StandardMPCWeights(R_v=R_v, R_w=R_w)
    # Use explicitly supplied N, or fall back to the per-scenario default.
    horizon = N if N is not None else SCENARIO_MPC_HORIZONS.get(
        scenario_name, mpc_cfg.get('N_default', 12))
    ctrl = StandardMPCCasadi(
        N=horizon,
        weights=scenario_weights,
        config=mpc_cfg,
    )
    results = []

    for trial in range(n_trials):
        perturbed, v0, omega0 = randomizer.randomize_trial(scenario, seed=master_seed + trial)
        env.set_scenario(perturbed.obstacles, perturbed.bounds)
        # Parity, 2026-09-11: same LiDAR seed derivation as EP2Evaluator
        # (see run_baseline above for the defect record).
        env.reset(perturbed.start, lidar_seed=master_seed + trial + LIDAR_SEED_OFFSET)
        env._v, env._omega = v0, omega0
        ctrl.reset()

        solver_times = []
        success = collision = False
        step = 0

        for step in range(max_steps):
            state = env.get_state()
            scan = env.get_lidar_scan()
            d_obs = float(np.min(scan))
            # Trim 6D state [x,y,θ,v,ω,E] to 5D [x,y,θ,v,ω] for StandardMPCCasadi
            x0_5d = np.array([
                state['x'], state['y'], state['theta'],
                state['v'], state['omega'],
            ])
            try:
                # no v_ref, no slowdown governor. The velocity-tracking
                # term and its governor are both gone, so there is nothing to
                # scale here -- the controller is asked for a command and the
                # obstacle/wall cost is what keeps it off the geometry, exactly
                # as in EP2-MPC.
                # wall_bounds was never passed, so StandardMPCCasadi's wall
                # term sat inert (its docstring: "omitting it places the walls far
                # away so the term contributes nothing"). EP2-MPC's four wall terms
                # were always live. Same parity defect as the APF's missing wall
                # repulsion, on the other comparator.
                v_cmd, omega_cmd, solve_ms = ctrl.solve(
                    x0_5d, perturbed.goal, perturbed.obstacles,
                    wall_bounds=perturbed.bounds
                )
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning("StandardMPCCasadi fallback to zero command at trial=%d step=%d: %s", trial, step, e)
                v_cmd, omega_cmd, solve_ms = 0.0, 0.0, 0.0
            solver_times.append(solve_ms)
            state, _, _, _ = env.step(v_cmd, omega_cmd)
            if env.check_collision():
                collision = True
                break
            if env.is_goal_reached(perturbed.goal, tolerance=0.2):
                success = True
                break

        final = env.get_state()
        _tr = TrialResult(
            energy=final['cumulative_energy'],
            time=(step + 1) * dt,
            solver_ms=float(np.mean(solver_times)) if solver_times else 0.0,
            success=success,
            collision=collision,
            path_distance=final.get('path_distance', 0.0),
            n_steps=step + 1,
            solver_ms_all=[float(x) for x in solver_times],
            terminal_pose=[float(final['x']), float(final['y']),
                           float(final['theta'])],
            seed=int(master_seed + trial),
            # Parity: EP2Evaluator recorded WHY a trial crashed and the
            # baselines did not, so the APF's 250 chicane failures could not be
            # split into obstacle-vs-wall from the data. Reusing EP2Evaluator's own
            # static classifier rather than reimplementing it -- the point is that
            # both arms attribute a collision identically.
            collision_cause=(_ep2_collision_cause(env, perturbed)
                             if collision else None),
        )
        results.append(_tr)
        _emit_baseline_trial(run_dir, tag, _tr, trial_index=trial,
                             master_seed=master_seed)

    energies = np.array([r.energy for r in results])
    times = np.array([r.time for r in results])
    solver_ms_arr = np.array([r.solver_ms for r in results])
    successes = np.array([r.success for r in results], dtype=float)
    dists = np.array([r.path_distance for r in results])

    result = EvalResult(
        energy_mean=float(np.mean(energies)),
        energy_std=float(np.std(energies)),
        time_mean=float(np.mean(times)),
        time_std=float(np.std(times)),
        solver_ms_mean=float(np.mean(solver_ms_arr)),
        solver_ms_std=float(np.std(solver_ms_arr)),
        success_rate=float(np.mean(successes)),
        path_distance_mean=float(np.mean(dists)),
        path_distance_std=float(np.std(dists)),
        n_trials=n_trials,
        trial_results=results,
    )
    logger.info(
        "standard_mpc (N=%d): E=%.1fJ  T=%.1fs  D=%.1fm  success=%.0f%%",
        horizon,
        result.energy_mean, result.time_mean,
        result.path_distance_mean, result.success_rate * 100,
    )
    return result


def collect_all_baselines(
    scenario_name: str,
    n_trials: int = 50,
    config: Optional[Dict] = None,
    master_seed: int = 0,
) -> Dict[str, EvalResult]:
    """Run all baseline controllers on a scenario.

    Args:
        master_seed: passed to each baseline so trial perturbations match
                     EP2Evaluator when master_seed is the same value. This
                     enables valid paired Wilcoxon tests.

    Raises:
        ValueError: if ``config['robot']`` is absent. The robot limits must be
            injected rather than defaulted, so that no arm can quietly run at a
            different v_max than the controller it is compared against.
    """
    scenario = get_scenario(scenario_name)
    robot = (config or {}).get('robot')
    if not robot:
        raise ValueError(
            "collect_all_baselines requires config['robot'] with max_v, "
            "max_omega and collision_radius. Defaulting these is how the APF "
            "arms ended up at v_max=1.0 against the MPC's 1.2.")
    controllers = {
        # One reactive arm. energy_aware_apf and reactive_standard are deleted --
        # see the note above APFBaseline for why.
        'apf': APFBaseline(robot=robot, dt=config.get('mpc', {}).get('dt', DT)),
    }
    results = {}
    for name, ctrl in controllers.items():
        logger.info("Running baseline: %s (v_max=%.3f omega_max=%.5f r_body=%.3f)",
                    name, ctrl.v_max, ctrl.omega_max, ctrl.r_body)
        results[name] = run_baseline(ctrl, scenario, n_trials=n_trials,
                                     config=config, master_seed=master_seed)
        r = results[name]
        logger.info("  %s: E=%.1fJ  T=%.1fs  D=%.1fm  success=%.0f%%",
                    name, r.energy_mean, r.time_mean,
                    r.path_distance_mean, r.success_rate * 100)

    logger.info("Running baseline: standard_mpc_casadi (per-scenario N)")
    results['standard_mpc_casadi'] = run_standard_mpc_casadi_baseline(
        scenario_name=scenario_name,
        n_trials=n_trials,
        config=config,
        master_seed=master_seed,
        N=None,   # uses per-scenario default (corridor=12, s_curve=20, chicane=14)
    )


    # Energy-penalized MPC: standard MPC with R_v/R_w proxy energy terms.
    # This is the closest prior-art to EP2-MPC. Proves that scalar energy
    # penalty in standard MPC is insufficient — a full energy state is needed.
    logger.info("Running baseline: energy_penalized_mpc")
    results['energy_penalized_mpc'] = run_standard_mpc_casadi_baseline(
        scenario_name=scenario_name,
        n_trials=n_trials,
        config=config,
        master_seed=master_seed,
        N=None,
        R_v=0.5,
        R_w=0.3,
    )

    # N=24 only on s_curve — proves adaptive horizon is essential, not just
    # longer fixed horizon. On corridor N=12 already succeeds
    # comparison is not needed and weights tuned for N=12 produce invalid results.
    if scenario_name == 's_curve':
        logger.info("Running baseline: standard_mpc_casadi (N=24, s_curve only)")
        results['standard_mpc_casadi[N=24]'] = run_standard_mpc_casadi_baseline(
            scenario_name=scenario_name,
            n_trials=n_trials,
            config=config,
            master_seed=master_seed,
            N=24,
        )

    if scenario_name == 'chicane':
        logger.info("Running baseline: standard_mpc_casadi (N=24, chicane only)")
        results['standard_mpc_casadi[N=24]'] = run_standard_mpc_casadi_baseline(
            scenario_name=scenario_name,
            n_trials=n_trials,
            config=config,
            master_seed=master_seed,
            N=24,
        )
        logger.info("Running baseline: standard_mpc_casadi (N=50, chicane only)")
        results['standard_mpc_casadi[N=50]'] = run_standard_mpc_casadi_baseline(
            scenario_name=scenario_name,
            n_trials=n_trials,
            config=config,
            master_seed=master_seed,
            N=50,
        )

    return results