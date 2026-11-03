"""Script 21 — the REACH campaign.

The experiment the REACH paper rests on: the four arms (hard reserve / fixed
price / dual-ascent price / v_max-clip) over the generated out-and-back maps
across a budget grid, with JOINT violation/collision reporting.

Built on the run_campaign.py pattern, deliberately: spawn pool (never fork),
one single-threaded process per task with the thread limits proven in each
result, seeds from the plan (never the worker's identity), per-trial JSONL
sinks that append (so an interrupted campaign keeps everything), a manifest
recording the whole plan + per-task results, --dry-run.

What is different from run_campaign.py, and why:

* The unit of work is a MISSION (one closed-loop out-and-back trial), not a
  seed-block of tracking trials. Each mission runs its own supervisor; there
  is no per-trial controller state worth caching across missions.
* Every map's CERTIFICATE comes from script 20's per-map fit
  (certificates.json beside the maps); a missing fit is a hard error, not a
  silent 24 J placeholder -- that placeholder under-reserved every map by
  up to 2.25x (fitted intercepts run 28.6-54.0 J on the generated maps).
* Budgets: the grid is a fraction of each map's round-trip energy (from the
  manifest's generation criterion), so arms are compared at comparable
  difficulty across maps with different terrain.
* Violation/collision aggregation: violation rate and collision rate are
  reported JOINTLY per (arm, map, budget) -- a reserve bought with unreported
  collisions is the exact failure joint reporting exists to catch.

Stage layout (--stage):
  pilot     1 map x 1 budget x 2 seeds x 4 arms -- pipeline validation
  main      train maps x budget grid x seeds x arms -- the headline grid
  sweep     const_weight price sweep on train maps (selection rule below)
  heldout   held-out maps x budget grid x seeds x arms -- generalisation

const_weight sweep selection rule, FIXED BEFORE THE DATA:
the smallest price with ZERO reserve violations across all training maps
and budgets; ties broken by highest mean reach. A price that violates once
is not a comparator, it is a failure -- the hard-constraint arm is being
compared against an arm that is SUPPOSED to respect the same budget.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SINGLE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    'MPLBACKEND': 'Agg',
}

ARMS = ('reach', 'const_weight', 'dual', 'vmax_clip')
#: Budget grid as fractions of each map's round-trip energy E_rt (the
#: manifest's turn-back-interior criterion guarantees E_rt >= 1.3 x b_max
#: at generation, so fractions of E_rt are comparable across maps).
BUDGET_FRACTIONS = (0.30, 0.45, 0.60, 0.75)
#: const_weight price box. The first sweep (2026-09-12) ran 0.25-8.0 and
#: the price NEVER bound: the objective trades -w_reach*D_J + price*E_N,
#: and on a J-field one joule of energy buys ~one joule of D_J progress,
#: so the crossover is at price ~ w_reach = 100. A box 12x below the
#: crossover is a straw man, and the fix is the box, not the claim. The
#: corrected box brackets the crossover 4x either side.
SWEEP_PRICES = (25.0, 50.0, 100.0, 200.0, 400.0)


@dataclass
class ReachTask:
    """One unit of parallel work: a single closed-loop mission."""
    stage: str
    map_file: str            # absolute path to the map YAML
    map_name: str
    split: str               # train | heldout
    arm: str
    B: float
    seed: int
    a_q: float
    out_dir: str             # directory for the JSONL sink
    replan_every_s: float = 2.5
    t_mission_s: float = 240.0
    price0: float = 0.0      # const_weight arm (sweep overrides)
    config_path: Optional[str] = None
    #: Plant/model mismatch: plant runs with coefficients scaled by 1+delta
    #: while the controller plans nominal — the condition under which a
    #: reserve defends against something real.
    power_mismatch: float = 0.0
    #: Actuation noise (sigma_v, sigma_omega), seeded per trial.
    actuation_noise: Optional[tuple] = None

    @property
    def tag(self) -> str:
        return f"{self.map_name}_{self.arm}_B{int(self.B)}_seed{self.seed}"


def _apply_single_thread_env() -> None:
    for k, v in SINGLE_THREAD_ENV.items():
        os.environ[k] = v


def _renice() -> None:
    try:
        os.nice(19 - os.nice(0))
    except (OSError, AttributeError):
        pass


def run_reach_task(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entry: one mission, one JSONL record. Module-level and
    dict-argumented so it pickles under spawn."""
    _apply_single_thread_env()
    _renice()
    t0 = time.perf_counter()
    task = ReachTask(**task_dict)
    out: Dict[str, Any] = {'tag': task.tag, 'pid': os.getpid(),
                           'map': task.map_name, 'arm': task.arm,
                           'B': task.B, 'seed': task.seed, 'split': task.split}
    out['blas_threads'] = {k: os.environ.get(k) for k in SINGLE_THREAD_ENV}
    try:
        from env.scenarios import load_scenario_from_yaml
        from utils.config_loader import load_config
        from evaluation.reach_evaluator import (ARMS as ARM_DEFS,
                                                ReachArm, run_reach_trial,
                                                reach_trial_sink,
                                                emit_reach_trial)
        sc = load_scenario_from_yaml(task.map_file)
        cfg = load_config(task.config_path)
        base_arm = ARM_DEFS[task.arm]
        arm = ReachArm(**{**dataclasses.asdict(base_arm),
                          'price0': task.price0 or base_arm.price0})
        res = run_reach_trial(sc, arm, B=task.B, seed=task.seed, config=cfg,
                              replan_every_s=task.replan_every_s,
                              t_mission_s=task.t_mission_s,
                              a_q=task.a_q,
                              power_mismatch=task.power_mismatch,
                              actuation_noise=task.actuation_noise)
        rec = res.to_record()
        rec['split'] = task.split
        # price0 rides IN the result dataclass (recorded by the sink); the
        # dict overlay here would never reach the JSONL -- caught on the
        # first sweep, whose 180 records carried no price.
        sink = reach_trial_sink(task.out_dir, task.split)
        emit_reach_trial(sink, res)
        out.update({'ok': True,
                    'outcome': rec['outcome'],
                    'reach_m': rec['reach_m'],
                    'min_margin_J': rec['min_margin_J'],
                    'energy_spent': rec['energy_spent'],
                    'a_q': rec['a_q'], 'a_q_source': rec['a_q_source'],
                    'wall_s': round(time.perf_counter() - t0, 1)})
    except Exception as exc:                     # noqa: BLE001 - worker guard
        out.update({'ok': False, 'error': f"{type(exc).__name__}: {exc}",
                    'wall_s': round(time.perf_counter() - t0, 1)})
    return out


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _load_maps(maps_root: str, split: str) -> List[Dict[str, Any]]:
    """Maps + their fitted certificates + round-trip energy, one entry each."""
    import glob
    import numpy as np
    from env.scenarios import load_scenario_from_yaml
    from mpc.energy_model import load_default_coefficients
    from planning.energy_dijkstra import build_field

    split_dir = os.path.join(maps_root, split)
    cert_path = os.path.join(split_dir, 'certificates.json')
    if not os.path.isfile(cert_path):
        raise SystemExit(
            f"missing {cert_path}: run scripts/20_reach_certificate_terrain "
            "first -- the campaign must NOT run on the 24 J placeholder "
            "(it under-reserves every map by up to 2.25x)")
    certs = {c['map']: c for c in json.load(open(cert_path))}

    entries = []
    for path in sorted(glob.glob(os.path.join(split_dir, 'reach_*.yaml'))):
        sc = load_scenario_from_yaml(path)
        if sc.name not in certs:
            raise SystemExit(f"{sc.name}: no certificate fit in {cert_path}")
        # round-trip energy to the far end on the J-field: the budget grid
        # is fractions of THIS, so arms face comparable difficulty across
        # maps with different terrain
        f = build_field(sc, coeffs=load_default_coefficients(),
                        terrain=sc.terrain)
        far_x = sc.bounds[2] - 1.0
        e_far = f.dstar(float(far_x), float(sc.dock[1]))
        if not math.isfinite(e_far) or e_far <= 0:
            raise SystemExit(f"{sc.name}: far end unreachable on the J-field")
        entries.append({'path': os.path.abspath(path), 'name': sc.name,
                        'E_rt': 2.0 * float(e_far),
                        'a_q': float(certs[sc.name]['a_q'])})
    if not entries:
        raise SystemExit(f"no reach_*.yaml maps in {split_dir}")
    return entries


def build_plan(stage: str, out_dir: str, maps_root: str, *,
               n_seeds: int = 5, seed0: int = 31_000) -> List[ReachTask]:
    """The task list for a stage. Seeds are assigned by the planner."""
    tasks: List[ReachTask] = []
    if stage == 'pilot':
        maps = _load_maps(maps_root, 'train')[:1]
        seeds = list(range(seed0, seed0 + 2))
        for m in maps:
            B = round(0.45 * m['E_rt'], 1)
            for arm in ARMS:
                for s in seeds:
                    tasks.append(ReachTask('pilot', m['path'], m['name'],
                                           'train', arm, B, s, m['a_q'],
                                           out_dir))
    elif stage == 'main':
        maps = _load_maps(maps_root, 'train')
        for m in maps:
            for frac in BUDGET_FRACTIONS:
                B = round(frac * m['E_rt'], 1)
                for arm in ARMS:
                    for s in range(seed0, seed0 + n_seeds):
                        # 2026-10-04 (audit Defect #7): const_weight at the
                        # sweep-4 selected price 50, matching heldout/mismatch.
                        # The original main stage (main2/main3) omitted this
                        # override -- the arm ran at the evaluator fallback
                        # price0=1.0, 100x below the w_reach crossover. The
                        # heldout/mismatch blocks already set 50.0. Found by
                        # the pre-submission raw-data audit; the fix is the
                        # re-run, not the label.
                        tasks.append(ReachTask('main', m['path'], m['name'],
                                               'train', arm, B, s, m['a_q'],
                                               out_dir,
                                               price0=(50.0 if arm == 'const_weight'
                                                       else 0.0)))
    elif stage == 'sweep':
        maps = _load_maps(maps_root, 'train')
        for m in maps:
            for frac in (0.60,):                    # the binding budget
                B = round(frac * m['E_rt'], 1)
                for price in SWEEP_PRICES:
                    for s in range(seed0, seed0 + n_seeds):
                        tasks.append(ReachTask('sweep', m['path'], m['name'],
                                               'train', 'const_weight', B, s,
                                               m['a_q'], out_dir,
                                               price0=price))
    elif stage == 'heldout':
        maps = _load_maps(maps_root, 'heldout')
        for m in maps:
            for frac in BUDGET_FRACTIONS:
                B = round(frac * m['E_rt'], 1)
                for arm in ARMS:
                    for s in range(seed0, seed0 + n_seeds):
                        # const_weight at the sweep-4 selected price (50):
                        # the paper's comparator, chosen on TRAINING maps only
                        tasks.append(ReachTask(
                            'heldout', m['path'], m['name'], 'heldout', arm,
                            B, s, m['a_q'], out_dir,
                            price0=(50.0 if arm == 'const_weight' else 0.0)))
    elif stage == 'mismatch':
        # The key comparison under plant/model mismatch.
        # The budget is the SWEEP-4 budget (0.60 * E_rt -- the binding one
        # the price was calibrated at), so the priced arms face exactly the
        # conditions their price was tuned for, and only the PLANT changes.
        # const_weight at price 50 (the sweep-4 selection) and the dual arm
        # carry their nominal calibration; that staleness is the point.
        # Prediction registered BEFORE this
        # stage ran: the constrained arm's breach-rate advantage WIDENS.
        maps = _load_maps(maps_root, 'train')
        noise = (0.05, 0.10)                     # mismatch spec: sigma_v, sigma_w
        for m in maps:
            B = round(0.60 * m['E_rt'], 1)
            for delta in (-0.15, 0.15):
                for arm in ARMS:
                    for s in range(seed0, seed0 + n_seeds):
                        tasks.append(ReachTask(
                            'mismatch', m['path'], m['name'], 'train', arm,
                            B, s, m['a_q'], out_dir,
                            price0=(50.0 if arm == 'const_weight' else 0.0),
                            power_mismatch=delta,
                            actuation_noise=noise))
    else:
        raise SystemExit(f"unknown stage {stage!r}: "
                         "pilot | main | sweep | heldout | mismatch")
    # duplicate (map, arm, B, seed, price, mismatch) would be the same
    # mission twice — the mismatch stage needs delta in the key: same map,
    # arm, B and seed run at BOTH -15% and +15%
    seen = set()
    for t in tasks:
        key = (t.map_name, t.arm, t.B, t.seed, t.price0, t.power_mismatch)
        if key in seen:
            raise SystemExit(f"duplicate task {t.tag} price={t.price0} "
                             f"mismatch={t.power_mismatch}")
        seen.add(key)
    return tasks


# ---------------------------------------------------------------------------
# R4 aggregation
# ---------------------------------------------------------------------------

def aggregate(jsonl_path: str) -> List[Dict[str, Any]]:
    """Violation and collision rates reported JOINTLY per condition.

    A reserve bought with unreported collisions is a reserve not worth
    having, so each (map, arm, B) cell carries BOTH rates plus the outcome
    mix -- never one without the other.
    """
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    cells: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r['map_name'], r['arm'], r['B'], r.get('split', 'train'),
               r.get('price0', 0.0))
        c = cells.setdefault(key, {'n': 0, 'docked': 0, 'stranded': 0,
                                   'collision': 0, 'timeout': 0,
                                   'violations': 0, 'reach_sum': 0.0,
                                   'reach_n': 0, 'margin_min': math.inf,
                                   'price': r.get('price0', 0.0),
                                   'a_q': r.get('a_q'),
                                   'a_q_source': r.get('a_q_source')})
        c['n'] += 1
        c[r['outcome']] += 1
        if r['min_margin_J'] < 0:
            c['violations'] += 1
        if r['reach_m'] is not None:
            c['reach_sum'] += r['reach_m']
            c['reach_n'] += 1
        c['margin_min'] = min(c['margin_min'], r['min_margin_J'])
    out = []
    for (m, a, B, split, _price), c in sorted(cells.items()):
        n = c['n']
        out.append({
            'map': m, 'arm': a, 'B': B, 'split': split, 'n': n,
            'price0': c['price'],
            'a_q': c['a_q'], 'a_q_source': c['a_q_source'],
            'dock_rate': c['docked'] / n,
            'stranded_rate': c['stranded'] / n,
            'collision_rate': c['collision'] / n,
            'timeout_rate': c['timeout'] / n,
            'violation_rate': c['violations'] / n,
            'reach_mean': (c['reach_sum'] / c['reach_n']
                           if c['reach_n'] else None),
            'margin_min_J': (None if c['margin_min'] == math.inf
                             else c['margin_min']),
            'r4_note': 'violation_rate and collision_rate are reported '
                       'JOINTLY: a reserve bought with unreported '
                       'collisions is a reserve not worth having',
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--stage', required=True,
                    choices=['pilot', 'main', 'sweep', 'heldout', 'mismatch'])
    ap.add_argument('--maps-root', default=None,
                    help='directory containing {train,heldout}/reach_*.yaml '
                         'with certificates.json (default: this run-id\'s '
                         'reach_maps stage)')
    ap.add_argument('--max-workers', type=int, default=8)
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--seed0', type=int, default=31_000)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='bulk stages only; never valid for timing')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='skip execution; re-aggregate the existing JSONL')
    args = ap.parse_args(argv)

    from utils.paths import run_dir as _run_dir
    from utils.safe_write import safe_write_json
    from utils.provenance import git_commit_info

    maps_root = args.maps_root
    if maps_root is None:
        maps_root = _run_dir(args.run_id, 'reach_maps', create=False)
    out_dir = _run_dir(args.run_id, 'reach')
    os.makedirs(out_dir, exist_ok=True)

    jsonl = os.path.join(out_dir, 'reach_trials.jsonl')
    if args.aggregate_only:
        if not os.path.isfile(jsonl):
            raise SystemExit(f"no trials to aggregate: {jsonl}")
        cells = aggregate(jsonl)
        safe_write_json(os.path.join(out_dir, 'r4_aggregate.json'), cells)
        print(f"{len(cells)} cells -> r4_aggregate.json")
        return 0

    tasks = build_plan(args.stage, out_dir, maps_root,
                       n_seeds=args.n_seeds, seed0=args.seed0)
    print(f"stage {args.stage}: {len(tasks)} missions "
          f"({len(ARMS)} arms x budgets x maps x seeds)")
    for t in tasks[:12]:
        print(f"  {t.tag}  a_q={t.a_q:.1f}")
    if len(tasks) > 12:
        print(f"  ... +{len(tasks) - 12} more")

    if args.dry_run:
        print("\n--dry-run: plan is valid and nothing was executed.")
        return 0

    from utils import preflight
    pf = preflight.check(mode='bulk', max_workers=args.max_workers)
    if not pf['ok'] and not args.skip_preflight:
        print(f"\nRefusing to start: preflight: {pf['reason']}")
        return 2

    ctx = mp.get_context('spawn')
    payloads = [asdict(t) for t in tasks]
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ctx.Pool(processes=args.max_workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_reach_task, payloads):
            results.append(r)
            state = 'ok' if r.get('ok') else f"FAILED {r.get('error')}"
            extra = (f" -> {r['outcome']}, reach {r['reach_m']:.1f} m"
                     if r.get('ok') else '')
            print(f"  [{len(results)}/{len(tasks)}] {r['tag']:<46} "
                  f"{r['wall_s']:7.1f}s  {state}{extra}")

    safe_write_json(os.path.join(out_dir, f'campaign_manifest_{args.stage}.json'), {
        'run_id': args.run_id, 'stage': args.stage,
        'max_workers': args.max_workers,
        'wall_s': time.perf_counter() - t0,
        'git': git_commit_info(),
        'preflight': {k: v for k, v in pf.items() if k != 'foreign_python'},
        'maps_root': maps_root,
        'budget_fractions': list(BUDGET_FRACTIONS),
        'sweep_prices': list(SWEEP_PRICES),
        'tasks': payloads, 'results': results,
    })

    cells = aggregate(jsonl) if os.path.isfile(jsonl) else []
    if cells:
        safe_write_json(os.path.join(out_dir, 'r4_aggregate.json'), cells)
        print(f"\nR4 aggregate: {len(cells)} cells -> r4_aggregate.json")
        # headline table for the log
        print(f"\n{'map':<22}{'arm':<14}{'B':>7}{'dock':>6}{'viol':>6}"
              f"{'coll':>6}{'reach':>8}")
        for c in cells:
            if c['split'] != ('heldout' if args.stage == 'heldout' else 'train'):
                continue
            print(f"{c['map']:<22}{c['arm']:<14}{c['B']:>7.0f}"
                  f"{c['dock_rate']:>6.2f}{c['violation_rate']:>6.2f}"
                  f"{c['collision_rate']:>6.2f}"
                  f"{(c['reach_mean'] or 0):>8.2f}")

    n_bad = sum(1 for r in results if not r.get('ok'))
    print(f"\n{len(results) - n_bad}/{len(results)} missions ok in "
          f"{time.perf_counter() - t0:.1f}s")
    return 1 if n_bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
