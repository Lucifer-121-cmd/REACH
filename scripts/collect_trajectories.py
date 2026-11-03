#!/usr/bin/env python3
"""
Collect trajectory data for Fig 8 by running actual simulations.

Saves real x,y,theta positions from env.get_state() at every step —
NOT kinematic reconstruction from v,omega history (which drifts).

Output: results/figures/trajectory_data.json

Usage:
    python scripts/collect_trajectories.py
    python scripts/collect_trajectories.py --n_trials 1
"""
import sys
import os
import json
import argparse
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from utils.config_loader import load_config


# ---------------------------------------------------------------------------
# Load best evolved Pareto params (same loader as 04_ablation.py)
# ---------------------------------------------------------------------------

def load_best_evolved_params(scenario_name: str, results_dir: str):
    """Load the best energy-minimising Pareto solution from evolution results.

    Uses shared checkpoint_utils which picks the highest-generation pareto file
    via recursive glob (not mtime sort).
    """
    from utils.checkpoint_utils import load_best_evolved_params as _load
    return _load(scenario_name, results_dir)


# ---------------------------------------------------------------------------
# EP²-MPC trajectory — actual env positions at every step
# ---------------------------------------------------------------------------

def collect_ep2_trajectory(scenario_name: str, config: dict,
                            results_dir: str, master_seed: int = 0) -> dict:
    """Run EP²-MPC for one trial and collect actual x,y,theta,N_k,energy."""
    from env.scenarios import get_scenario
    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer
    from mpc.ep2_mpc import EP2MPC

    params       = load_best_evolved_params(scenario_name, results_dir)
    logger.info("EP2 genome source for %s: %s", scenario_name, results_dir)
    scenario_cfg = get_scenario(scenario_name)
    randomizer   = TrialRandomizer()
    env          = PioneerEnv(config.get('robot', {}))
    mpc_cfg      = dict(config.get('mpc', {}))
    mpc          = EP2MPC(params=params, config=mpc_cfg)

    dt        = mpc_cfg.get('dt', 0.1)
    max_steps = int(config.get('evaluation', {}).get('timeout_per_trial', 60.0) / dt)

    perturbed, v0, omega0 = randomizer.randomize_trial(
        scenario_cfg, seed=master_seed)
    env.set_scenario(perturbed.obstacles, perturbed.bounds)
    # LiDAR parity, 2026-09-11: seed the scan RNG like the evaluator does
    # (trial seed + offset) so the figure-8 trajectories are reproducible.
    from evaluation.baselines import LIDAR_SEED_OFFSET
    env.reset(perturbed.start, lidar_seed=master_seed + LIDAR_SEED_OFFSET)
    env._v, env._omega = v0, omega0
    mpc.reset()

    xs, ys, thetas, Nks, energies = [], [], [], [], []
    solver_ms, vs, omegas = [], [], []   # per-step solver time + commands (for Figs 2/3/7)
    success = False

    for step in range(max_steps):
        state = env.get_state()
        xs.append(float(state['x']))
        ys.append(float(state['y']))
        thetas.append(float(state['theta']))
        energies.append(float(state.get('cumulative_energy', 0.0)))
        vs.append(float(state['v']))
        omegas.append(float(state['omega']))

        x0 = np.array([
            state['x'], state['y'], state['theta'],
            state['v'], state['omega'],
            state.get('cumulative_energy', 0.0)
        ])

        d_obs = float(np.min(env.get_lidar_scan()))

        try:
            v_cmd, omega_cmd, solve_ms, _ = mpc.solve(
                x0, perturbed.goal, d_obs,
                perturbed.obstacles, params)
        except Exception as exc:
            logger.warning("EP2 solve failed at step %d: %s", step, exc)
            v_cmd, omega_cmd, solve_ms = 0.0, 0.0, 0.0

        # Keep per-step solver time (real data for Fig 7 — previously discarded)
        solver_ms.append(float(solve_ms))

        # Get N_k from adaptive horizon history AFTER solve
        history = mpc.adaptive_horizon.get_history()
        N = int(history[-1]['N']) if history else 8
        Nks.append(N)

        env.step(v_cmd, omega_cmd)

        if env.check_collision():
            logger.info("%s EP2: collision at step %d", scenario_name, step)
            break
        if env.is_goal_reached(perturbed.goal, tolerance=0.2):
            # Add final state
            final = env.get_state()
            xs.append(float(final['x']))
            ys.append(float(final['y']))
            thetas.append(float(final['theta']))
            energies.append(float(final.get('cumulative_energy', 0.0)))
            Nks.append(Nks[-1] if Nks else 8)
            vs.append(float(final['v']))
            omegas.append(float(final['omega']))
            solver_ms.append(solver_ms[-1] if solver_ms else 0.0)
            success = True
            logger.info("%s EP2: goal reached at step %d  energy=%.1fJ",
                        scenario_name, step, final.get('cumulative_energy', 0))
            break

    return {
        'x':        xs,
        'y':        ys,
        'theta':    thetas,
        'N_k':      Nks,
        'energy':   energies,
        'v':        vs,
        'omega':    omegas,
        'solver_ms': solver_ms,      # real per-step solver time (ms) for Fig 7
        'dt':       dt,
        'success':  success,
        'scenario': scenario_name,
        'method':   'ep2mpc',
        'goal':     list(perturbed.goal),
        'start':    list(perturbed.start),
        'obstacles': [[float(o[0]), float(o[1]), float(o[2])]
                      for o in perturbed.obstacles],
        'bounds':   list(perturbed.bounds),
    }


# ---------------------------------------------------------------------------
# Standard MPC trajectory — actual env positions at every step
# ---------------------------------------------------------------------------

def collect_stdmpc_trajectory(scenario_name: str, config: dict,
                               master_seed: int = 0, N: int = 12) -> dict:
    """Run Standard MPC for one trial and collect actual x,y,theta positions."""
    from env.scenarios import get_scenario
    from env.pioneer_env import PioneerEnv
    from env.trial_randomizer import TrialRandomizer
    from mpc.standard_mpc_casadi import StandardMPCCasadi

    scenario_cfg = get_scenario(scenario_name)
    randomizer   = TrialRandomizer()
    env          = PioneerEnv(config.get('robot', {}))
    ctrl         = StandardMPCCasadi(N=N, config=config.get('mpc', {}))

    dt        = config.get('mpc', {}).get('dt', 0.1)
    max_steps = int(30.0 / dt)   # Standard MPC timeout

    perturbed, v0, omega0 = randomizer.randomize_trial(
        scenario_cfg, seed=master_seed)
    env.set_scenario(perturbed.obstacles, perturbed.bounds)
    # LiDAR parity, 2026-09-11 (see EP2 collector above).
    from evaluation.baselines import LIDAR_SEED_OFFSET
    env.reset(perturbed.start, lidar_seed=master_seed + LIDAR_SEED_OFFSET)
    env._v, env._omega = v0, omega0
    ctrl.reset()

    xs, ys, thetas = [], [], []
    collision_point = None
    success = False

    for step in range(max_steps):
        state = env.get_state()
        xs.append(float(state['x']))
        ys.append(float(state['y']))
        thetas.append(float(state['theta']))

        x0_5d = np.array([
            state['x'], state['y'], state['theta'],
            state['v'], state['omega']
        ])

        try:
            v_cmd, omega_cmd, _ = ctrl.solve(
                x0_5d, perturbed.goal, perturbed.obstacles)
        except Exception:
            v_cmd, omega_cmd = 0.0, 0.0

        env.step(v_cmd, omega_cmd)

        if env.check_collision():
            final = env.get_state()
            xs.append(float(final['x']))
            ys.append(float(final['y']))
            thetas.append(float(final['theta']))
            collision_point = [float(final['x']), float(final['y'])]
            logger.info("%s StdMPC N=%d: collision at step %d",
                        scenario_name, N, step)
            break
        if env.is_goal_reached(perturbed.goal, tolerance=0.2):
            final = env.get_state()
            xs.append(float(final['x']))
            ys.append(float(final['y']))
            thetas.append(float(final['theta']))
            success = True
            logger.info("%s StdMPC N=%d: goal at step %d",
                        scenario_name, N, step)
            break

    return {
        'x':              xs,
        'y':              ys,
        'theta':          thetas,
        'success':        success,
        'collision_point': collision_point,
        'scenario':       scenario_name,
        'method':         f'standard_mpc_N{N}',
        'N':              N,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Collect trajectory data for Fig 8')
    parser.add_argument('--run-id', dest='run_id', default=None,
                        help='when given (and --output-dir is not), output goes to '
                             '<runs>/<run-id>/trajectories/ on /data')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Explicit output directory. There is NO fallback to '
                             'results/figures, which is where the tracked '
                             'trajectory_data.json lives (a git-tracked file this '
                             'script refuses to overwrite). Give either '
                             '--run-id or --output-dir.')
    parser.add_argument('--force', action='store_true',
                        help='Permit overwriting a git-tracked output file.')
    parser.add_argument('--config',     type=str, default=None)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--evolution-root', type=str, default=None,
                        help='Directory containing evolution/<scenario>/ from which '
                             'the EP2 genome loads (e.g. runs/<id>/reevolution). '
                             'Default: config results_dir, which is the TRACKED '
                             'pre-reevolution archive. Added 2026-09-11: this '
                             'script previously had no way to plot the re-evolved '
                             'genome.')
    args = parser.parse_args()

    config      = load_config(args.config)
    results_dir = (args.evolution_root if args.evolution_root is not None
                   else config.get('results_dir', 'results'))
    if args.evolution_root is None:
        logger.warning(
            "Genome source: %s (config fallback -- the TRACKED pre-reevolution "
            "archive; pass --evolution-root to plot the re-evolved genome)",
            results_dir)

    # Resolve the output directory BEFORE doing any work. This block used to sit
    # at the end of main(), after `os.makedirs(args.output_dir)`, so --run-id on
    # its own died with `TypeError: expected str, bytes or os.PathLike object,
    # not NoneType` -- and a bad --run-id would only have been caught after all
    # six trajectories had been collected.
    if args.output_dir is None:
        if not args.run_id:
            raise SystemExit("give either --run-id or --output-dir; there is no "
                             "default, because the old one overwrote the tracked "
                             "trajectory_data.json")
        from utils.paths import run_dir as _run_dir
        args.output_dir = _run_dir(args.run_id, 'trajectories')
    os.makedirs(args.output_dir, exist_ok=True)

    SCENARIOS = ['corridor', 's_curve', 'chicane']
    all_data  = {}

    for scenario in SCENARIOS:
        logger.info("=" * 60)
        logger.info("Collecting trajectories for: %s", scenario)
        logger.info("=" * 60)

        # EP²-MPC with evolved params
        ep2 = collect_ep2_trajectory(
            scenario, config, results_dir, master_seed=args.seed)

        # Standard MPC N=12
        std12 = collect_stdmpc_trajectory(
            scenario, config, master_seed=args.seed, N=12)

        all_data[scenario] = {
            'ep2mpc':      ep2,
            'standard_mpc': std12,
        }

        logger.info("%s done — EP2 success=%s, StdMPC success=%s",
                    scenario, ep2['success'], std12['success'])

    if args.output_dir is None:
        raise AssertionError('output_dir is resolved at the top of main()')
    out_path = os.path.join(args.output_dir, 'trajectory_data.json')
    from utils.safe_write import safe_write_json
    safe_write_json(out_path, all_data, force=args.force)
    logger.info("Trajectory data saved to %s", out_path)


if __name__ == '__main__':
    main()