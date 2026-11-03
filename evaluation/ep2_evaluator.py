"""50-trial evaluation framework for EP²-MPC with path distance tracking."""
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Type
import sys
import os
from collections import Counter


from mpc.mpc_params import MPCParams
from mpc.energy_model import load_default_coefficients
from env.pioneer_env import PioneerEnv
from env.scenarios import get_scenario, ScenarioConfig
from env.trial_randomizer import TrialRandomizer

logger = logging.getLogger(__name__)

@dataclass
class TrialResult:
    """Results from a single evaluation trial.

    The fields below the divider exist because `scripts/04_ablation.py`
    discarded `trial_results` outright, which is why success-conditioned energy
    cannot be recomputed for the six ablation cells where it matters and why
    Reviewer 9's request for IPOPT iteration counts cannot be answered from any
    tracked file. Every field here is per-step or terminal detail that is
    unrecoverable once the trial ends.
    """
    energy: float          # total energy consumed (J)
    time: float            # mission time (s)
    solver_ms: float       # MEAN over this trial's steps -- NOT a per-step sample
    success: bool          # reached goal without collision
    collision: bool        # hit obstacle
    path_distance: float = 0.0   # actual path length (m)
    n_steps: int = 0
    solver_status: str = 'unknown'
    trajectory: List[List[float]] = field(default_factory=list)
    adaptive_horizon_history: List[Dict[str, Any]] = field(default_factory=list)

    # ---- per-step and terminal detail ---------------------------------------
    solver_ms_all: List[float] = field(default_factory=list)
    solver_status_all: List[str] = field(default_factory=list)
    iter_count_all: List[int] = field(default_factory=list)
    N_all: List[int] = field(default_factory=list)
    built_all: List[bool] = field(default_factory=list)
    warm_all: List[bool] = field(default_factory=list)
    terminal_pose: List[float] = field(default_factory=list)   # [x, y, theta]
    collision_cause: Optional[str] = None   # 'obstacle' | 'wall' | None
    energy_trace: List[float] = field(default_factory=list)
    path_trace: List[float] = field(default_factory=list)
    deadline_misses: int = 0
    seed: Optional[int] = None
    lidar_seed: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        """Flat JSONL-ready dict. Traces stay as lists; nothing is summarised."""
        from dataclasses import asdict
        return asdict(self)

@dataclass
class EvalResult:
    """Aggregated results from multiple trial evaluations."""
    energy_mean: float
    energy_std: float
    time_mean: float
    time_std: float
    solver_ms_mean: float
    solver_ms_std: float
    success_rate: float
    path_distance_mean: float = 0.0
    path_distance_std: float = 0.0
    n_trials: int = 0
    trial_results: List[TrialResult] = field(default_factory=list)
    solver_status_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def objectives(self) -> np.ndarray:
        """4-D objective vector (all minimise)."""
        return np.array([
            self.energy_mean,
            self.time_mean,
            self.solver_ms_mean,
            1.0 - self.success_rate,
        ])

class EP2Evaluator:
    """Runs multi-trial evaluation of EP²-MPC on navigation scenarios.

    Each trial applies 3-source randomisation and runs the MPC until goal,
    collision, or timeout.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 mpc_class: Optional[Type] = None):
        self.config = config or {}
        self.dt = self.config.get('mpc', {}).get('dt', 0.1)
        timeout = self.config.get('evaluation', {}).get('timeout_per_trial', 30.0)
        self.max_steps = int(timeout / self.dt)
        self.progress_every = int(self.config.get('evaluation', {}).get('progress_every_trials', 10))
        self.goal_tolerance = 0.2
        self.randomizer = TrialRandomizer()
        self._env: Optional[PioneerEnv] = None
        self._mpc = None
        self.mpc_class = mpc_class
        #: Trial sink. When set, every trial appends one JSONL record.
        self.trial_jsonl_path: Optional[str] = None
        #: Added to the trial seed to derive the LiDAR seed. Offset so it
        #: cannot alias the randomiser's own sub-seeds. Set to None to leave the
        #: scan RNG unseeded -- diagnostics only, since that makes trials
        #: irreproducible.
        self.lidar_seed_offset: Optional[int] = 1_000_003

    def set_trial_sink(self, run_dir: str, tag: str) -> str:
        """Route per-trial records to ``<run_dir>/trials_<tag>.jsonl``.

        One JSON object per line, appended as trials complete, so an interrupted
        campaign keeps everything up to the interruption. Called by the harness
        with the run-id directory; if never called, per-trial detail is still on
        the TrialResult objects but is not persisted.
        """
        os.makedirs(run_dir, exist_ok=True)
        self.trial_jsonl_path = os.path.join(run_dir, f'trials_{tag}.jsonl')
        return self.trial_jsonl_path

    def _emit_trial(self, rec: 'TrialResult', **context) -> None:
        if not self.trial_jsonl_path:
            return
        import json
        payload = rec.to_record()
        payload.update(context)
        with open(self.trial_jsonl_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, default=float) + '\n')

    def _get_env(self) -> PioneerEnv:
        if self._env is None:
            cfg = dict(self.config.get('robot', {}))
            cfg['dt'] = self.dt
            cfg['energy_coeffs'] = load_default_coefficients()
            self._env = PioneerEnv(cfg)
        return self._env

    def _get_mpc(self, params: MPCParams):
        """Return the scenario's MPC, constructing it at most once.

        This used to build a brand-new EP2MPC on every call, and it is
        called once per trial -- so the compiled solver bank was thrown away and
        rebuilt 250 times per cell, and every first visit to an N re-paid NLP
        construction. Caching is safe because `solve()` takes `params` per call
        and `reset()` clears the warm-start state between trials; only the
        compiled NLPs persist, which is the whole point.
        """
        if self._mpc is None:
            from mpc.ep2_mpc import EP2MPC
            mpc_cfg: Dict[str, Any] = dict(self.config.get('mpc', {}))
            mpc_cfg['energy_coeffs'] = load_default_coefficients()
            mpc_cfg.update(self.config.get('robot', {}))
            mpc_cfg['dt'] = self.dt
            mpc_cls = self.mpc_class or EP2MPC
            self._mpc = mpc_cls(params=params, config=mpc_cfg)
        return self._mpc

    def _run_trial(
        self,
        params: MPCParams,
        scenario_config: ScenarioConfig,
        seed: int,
    ) -> TrialResult:
        perturbed, v0, omega0 = self.randomizer.randomize_trial(
            scenario_config, seed=seed)

        env = self._get_env()
        env.set_scenario(perturbed.obstacles, perturbed.bounds)
        # Seed the LiDAR from the trial seed so the scan stream is
        # reproducible. lidar_seed_offset=None leaves it unseeded.
        lidar_seed = (None if self.lidar_seed_offset is None
                      else seed + self.lidar_seed_offset)
        env.reset(perturbed.start, lidar_seed=lidar_seed)
        env._v = v0
        env._omega = omega0

        x0 = env.get_state_vector()
        goal = perturbed.goal

        mpc = self._get_mpc(params)
        mpc.reset()
        if hasattr(mpc, 'reset_instrumentation'):
            mpc.reset_instrumentation()

        solver_times: List[float] = []
        energy_trace: List[float] = []
        path_trace: List[float] = []
        success = False
        collision = False
        collision_cause = None
        n_steps = 0

        for step in range(self.max_steps):
            scan = env.get_lidar_scan()
            d_obs = float(np.min(scan))

            try:
                v_cmd, omega_cmd, solve_ms, _ = mpc.solve(
                    x0, goal, d_obs,
                    perturbed.obstacles, params,
                    wall_bounds=perturbed.bounds,
                )
            except Exception as e:
                logger.debug("MPC solve failed at step %d: %s", step, e)
                v_cmd, omega_cmd, solve_ms = 0.0, 0.0, 0.0

            solver_times.append(solve_ms)
            _, _, done, _ = env.step(v_cmd, omega_cmd)
            x0 = env.get_state_vector()
            energy_trace.append(float(x0[5]))
            path_trace.append(float(env.get_path_distance()))
            n_steps = step + 1

            if env.check_collision():
                collision = True
                collision_cause = self._collision_cause(env, perturbed)
                break
            if env.is_goal_reached(goal, tolerance=self.goal_tolerance):
                success = True
                break

        trajectory = env.get_trajectory().tolist()
        final = env.get_state()
        horizon = getattr(mpc, 'adaptive_horizon', None)
        history = horizon.get_history() if (horizon is not None and hasattr(horizon, 'get_history')) else []

        return TrialResult(
            energy=final['cumulative_energy'],
            time=n_steps * self.dt,
            solver_ms=float(np.mean(solver_times)) if solver_times else 0.0,
            success=success,
            collision=collision,
            path_distance=final.get('path_distance', 0.0),
            n_steps=n_steps,
            solver_status=getattr(mpc, 'last_solver_status', 'unknown'),
            trajectory=trajectory,
            adaptive_horizon_history=history,
            # ---- per-step detail ----
            solver_ms_all=list(getattr(mpc, 'solver_ms_all', [])),
            solver_status_all=list(getattr(mpc, 'solver_status_all', [])),
            iter_count_all=list(getattr(mpc, 'iter_count_all', [])),
            N_all=list(getattr(mpc, 'N_all', [])),
            built_all=list(getattr(mpc, 'built_all', [])),
            warm_all=list(getattr(mpc, 'warm_all', [])),
            terminal_pose=[float(final['x']), float(final['y']), float(final['theta'])],
            collision_cause=collision_cause,
            energy_trace=energy_trace,
            path_trace=path_trace,
            deadline_misses=int(getattr(mpc, 'deadline_misses', 0)),
            seed=int(seed),
            lidar_seed=getattr(env, 'lidar_seed', None),
        )

    @staticmethod
    def _collision_cause(env, scenario) -> str:
        """Distinguish an obstacle hit from a wall hit.

        Worth recording because the LiDAR is blind to walls
        (env/lidar_sim.py iterates obstacles only), so `d_obs` never extends the
        horizon for the surfaces the robot actually collides with on some maps.
        Attributing collisions is how that mechanism becomes visible.
        """
        r = float(env.config.get('collision_radius', 0.25))
        x, y = env._x, env._y
        for obs in (scenario.obstacles or []):
            if len(obs) >= 3 and np.hypot(x - obs[0], y - obs[1]) <= obs[2] + r:
                return 'obstacle'
        b = scenario.bounds
        if b is not None and (x - b[0] < r or b[2] - x < r
                              or y - b[1] < r or b[3] - y < r):
            return 'wall'
        return 'unknown'

    def evaluate(
        self,
        params: MPCParams,
        scenario_name: str,
        n_trials: int = 50,
        master_seed: int = 0,
    ) -> EvalResult:
        """Run *n_trials* and aggregate.

        Args:
            params:        MPC parameters to evaluate.
            scenario_name: one of corridor, s_curve, z_maze.
            n_trials:      number of trials to run.
            master_seed:   base seed for trial randomisation. Trial k uses
                           seed = master_seed + k. Changing master_seed across
                           outer seeds ensures genuinely different perturbations
                           for the 5-seed statistical evaluation.

        Returns:
            EvalResult with all metrics and per-trial detail.
        """
        try:
            scenario = get_scenario(scenario_name)
        except ValueError:
            return EvalResult(
                energy_mean=5000.0, energy_std=0.0,
                time_mean=30.0, time_std=0.0,
                solver_ms_mean=800.0, solver_ms_std=0.0,
                success_rate=0.0, n_trials=0)

        results: List[TrialResult] = []
        start = time.time()
        for trial in range(n_trials):
            trial_seed = master_seed + trial   # unique seed per (outer seed, trial)
            try:
                r = self._run_trial(params, scenario, seed=trial_seed)
                results.append(r)
                self._emit_trial(r, scenario=scenario_name, trial_index=trial,
                                 master_seed=master_seed,
                                 params=params.to_dict())
                if self.progress_every > 0 and (trial + 1) % self.progress_every == 0:
                    elapsed = time.time() - start
                    logger.info(
                        "Eval progress: %d/%d trials (%.1fs elapsed)",
                        trial + 1, n_trials, elapsed)
            except Exception as e:
                logger.warning("Trial %d failed: %s", trial, e)
                results.append(TrialResult(
                    energy=5000.0, time=30.0, solver_ms=800.0,
                    success=False, collision=True, trajectory=[]))

        energies = np.array([r.energy for r in results])
        times = np.array([r.time for r in results])
        solver_mss = np.array([r.solver_ms for r in results])
        successes = np.array([r.success for r in results], dtype=float)
        dists = np.array([r.path_distance for r in results])
        status_counts = dict(Counter(r.solver_status for r in results))

        return EvalResult(
            energy_mean=float(np.mean(energies)),
            energy_std=float(np.std(energies)),
            time_mean=float(np.mean(times)),
            time_std=float(np.std(times)),
            solver_ms_mean=float(np.mean(solver_mss)),
            solver_ms_std=float(np.std(solver_mss)),
            success_rate=float(np.mean(successes)),
            path_distance_mean=float(np.mean(dists)),
            path_distance_std=float(np.std(dists)),
            n_trials=n_trials,
            trial_results=results,
            solver_status_counts=status_counts,
        )
